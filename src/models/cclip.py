"""
C-CLIP: Continual CLIP model with LoRA and Contrastive Knowledge Consolidation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List
import copy

from .clip_wrapper import CLIPWrapper
from .lora import inject_lora, merge_all_lora_weights, get_lora_parameters, count_lora_parameters


class Projector(nn.Module):
    """
    Projector layer for CKC knowledge consolidation.
    Initialized as near-identity so projected features start close to the
    original encoder features, giving CKC loss a stable starting point.
    """
    
    def __init__(self, input_dim: int, output_dim: Optional[int] = None):
        super().__init__()
        if output_dim is None:
            output_dim = input_dim
        self.projection = nn.Linear(input_dim, output_dim)
        
        # Initialize as identity + small noise so projected ≈ original
        nn.init.eye_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projection(x)


class CCLIP(nn.Module):
    """
    C-CLIP: Continual CLIP with LoRA integration and CKC loss.
    
    Args:
        clip_model_name: Name of the CLIP model (e.g., 'ViT-B-16')
        pretrained: Pretrained weights to use (e.g., 'openai')
        lora_r: Rank for LoRA adaptation
        lora_alpha: Scaling factor for LoRA
        lora_dropout: Dropout probability for LoRA
        lora_target_modules: List of module names to apply LoRA to
        integration_coeff: Coefficient for merging LoRA weights (alpha in paper)
        device: Device to use
    """
    
    def __init__(
        self,
        clip_model_name: str = "ViT-B-16",
        pretrained: str = "openai",
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.1,
        lora_target_modules: Optional[List[str]] = None,
        integration_coeff: float = 0.5,
        device: str = "cuda",
    ):
        super().__init__()
        
        # Default LoRA target modules for vision transformer attention
        if lora_target_modules is None:
            lora_target_modules = ['q_proj', 'v_proj', 'in_proj_weight']
        
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.lora_target_modules = lora_target_modules
        self.integration_coeff = integration_coeff
        self.device = device
        
        # Load base CLIP model
        self.clip = CLIPWrapper(
            model_name=clip_model_name,
            pretrained=pretrained,
            device=device,
        )
        
        # Get embedding dimension
        self.embed_dim = self.clip.embed_dim
        
        # Create projectors for vision and text (on the correct device)
        self.vision_projector = Projector(self.embed_dim, self.embed_dim).to(device)
        self.text_projector = Projector(self.embed_dim, self.embed_dim).to(device)
        
        # Store old model for CKC (will be set during continual learning)
        self.old_clip = None
        
        # LoRA layers dictionary
        self.lora_layers = {}
        
        # Current task index
        self.current_task = 0
        
        print(f"Initialized C-CLIP with embedding dim: {self.embed_dim}")
        
    def inject_lora_for_new_task(self):
        """
        Inject LoRA layers for a new continual learning task.
        This should be called at the beginning of each new task.
        """
        print(f"\n=== Starting Task {self.current_task + 1} ===")

        # Save old model for CKC loss (deep copy)
        if self.current_task > 0:
            self.old_clip = copy.deepcopy(self.clip)
            self.old_clip.eval()
            for param in self.old_clip.parameters():
                param.requires_grad = False
            print("Saved old model for CKC")

            # Re-initialize projectors to identity for the new task.
            # This ensures projected features start ≈ original features,
            # giving CKC loss a stable alignment target each task.
            nn.init.eye_(self.vision_projector.projection.weight)
            nn.init.zeros_(self.vision_projector.projection.bias)
            nn.init.eye_(self.text_projector.projection.weight)
            nn.init.zeros_(self.text_projector.projection.bias)
            print("Re-initialized projectors to identity for CKC")

        # Freeze base model
        self.clip.freeze_base_model()

        # Inject LoRA into vision encoder
        vision_lora = inject_lora(
            model=self.clip.model.visual,
            target_modules=self.lora_target_modules,
            r=self.lora_r,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
        )

        # Inject LoRA into text encoder (self.clip.model.transformer)
        text_lora = inject_lora(
            model=self.clip.model.transformer,
            target_modules=self.lora_target_modules,
            r=self.lora_r,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
        )

        # Store all LoRA layers with paths relative to self.clip.model
        # so merge_all_lora_weights(model=self.clip.model, ...) works correctly.
        self.lora_layers = (
            {f"visual.{k}": v for k, v in vision_lora.items()}
            | {f"transformer.{k}": v for k, v in text_lora.items()}
        )

        # Count trainable parameters
        lora_params = count_lora_parameters(self.clip.model)
        print(f"LoRA parameters: {lora_params:,}")

        self.current_task += 1

        
    def merge_lora_after_task(self):
        """
        Merge LoRA weights into base model after completing a task.
        This implements the LoRA integration step from the paper.
        """
        print(f"\n=== Finishing Task {self.current_task} ===")
        
        # Merge LoRA weights
        self.clip.model = merge_all_lora_weights(
            model=self.clip.model,
            lora_layers=self.lora_layers,
            integration_coeff=self.integration_coeff,
        )
        
        # Clear LoRA layers
        self.lora_layers = {}
        
        # Update references and ensure model stays on correct device
        self.clip.visual = self.clip.model.visual
        self.clip.model = self.clip.model.to(self.device)
        
        print("LoRA weights merged into base model")
        
    def encode_image(self, images: torch.Tensor, normalize: bool = True) -> torch.Tensor:
        """Encode images using current model."""
        return self.clip.encode_image(images, normalize=normalize)
    
    def encode_text(self, text: torch.Tensor, normalize: bool = True) -> torch.Tensor:
        """Encode text using current model."""
        return self.clip.encode_text(text, normalize=normalize)
    
    def forward(
        self,
        images: torch.Tensor,
        text: torch.Tensor,
        return_old_features: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for C-CLIP.
        
        Args:
            images: Image tensor (B, C, H, W)
            text: Text tokens (B, seq_len)
            return_old_features: Whether to compute features from old model for CKC
            
        Returns:
            Dictionary containing:
                - image_features: Current model image features
                - text_features: Current model text features
                - projected_image_features: Projected image features
                - projected_text_features: Projected text features
                - old_image_features: Old model image features (if return_old_features=True)
                - old_text_features: Old model text features (if return_old_features=True)
        """
        # Encode with current model
        image_features = self.encode_image(images, normalize=True)
        text_features = self.encode_text(text, normalize=True)
        
        # Project features
        projected_image_features = self.vision_projector(image_features)
        projected_text_features = self.text_projector(text_features)
        
        # Normalize projected features
        projected_image_features = F.normalize(projected_image_features, p=2, dim=-1)
        projected_text_features = F.normalize(projected_text_features, p=2, dim=-1)
        
        output = {
            'image_features': image_features,
            'text_features': text_features,
            'projected_image_features': projected_image_features,
            'projected_text_features': projected_text_features,
        }
        
        # Compute old features for CKC if needed
        if return_old_features and self.old_clip is not None:
            with torch.no_grad():
                old_image_features = self.old_clip.encode_image(images, normalize=True)
                old_text_features = self.old_clip.encode_text(text, normalize=True)
                
                output['old_image_features'] = old_image_features
                output['old_text_features'] = old_text_features
        
        return output
    
    def get_trainable_parameters(self) -> List[nn.Parameter]:
        """
        Get all trainable parameters (LoRA + projectors).
        """
        params = []
        
        # LoRA parameters
        params.extend(get_lora_parameters(self.clip.model))
        
        # Projector parameters
        params.extend(self.vision_projector.parameters())
        params.extend(self.text_projector.parameters())
        
        return params
    
    def save_checkpoint(self, path: str):
        """
        Save model checkpoint.
        """
        checkpoint = {
            'clip_state_dict': self.clip.model.state_dict(),
            'vision_projector_state_dict': self.vision_projector.state_dict(),
            'text_projector_state_dict': self.text_projector.state_dict(),
            'current_task': self.current_task,
            'lora_layers': list(self.lora_layers.keys()),
        }
        torch.save(checkpoint, path)
        print(f"Saved checkpoint to {path}")
    
    def load_checkpoint(self, path: str):
        """
        Load model checkpoint.
        Note: If checkpoint was saved with LoRA structure, need to inject LoRA first.
        """
        checkpoint = torch.load(path, map_location=self.device)
        
        # Check if checkpoint has LoRA structure
        has_lora_structure = any('lora_A' in key or 'original_layer' in key 
                                  for key in checkpoint['clip_state_dict'].keys())
        
        if has_lora_structure:
            print("Checkpoint has LoRA structure, injecting LoRA...")
            self.inject_lora_for_new_task()
            # Load with strict=False to handle any structure differences
            self.clip.model.load_state_dict(checkpoint['clip_state_dict'], strict=False)
        else:
            self.clip.model.load_state_dict(checkpoint['clip_state_dict'])
        
        # Load projectors (handle both old and new key names)
        if 'vision_projector_state_dict' in checkpoint:
            self.vision_projector.load_state_dict(checkpoint['vision_projector_state_dict'])
            self.text_projector.load_state_dict(checkpoint['text_projector_state_dict'])
        elif 'projector_state_dict' in checkpoint:
            self.image_projector.load_state_dict(checkpoint['projector_state_dict']['image'])
            self.text_projector.load_state_dict(checkpoint['projector_state_dict']['text'])
        
        self.current_task = checkpoint.get('current_task', 'unknown')
        
        # Update references
        self.clip.visual = self.clip.model.visual
        
        print(f"Loaded checkpoint from {path}")
        print(f"Current task: {self.current_task}")
