"""
CLIP model wrapper for C-CLIP.
Handles loading and interfacing with CLIP models.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple
import open_clip


class CLIPWrapper(nn.Module):
    """
    Wrapper around OpenCLIP models for C-CLIP.
    """
    
    def __init__(
        self,
        model_name: str = "ViT-B-16",
        pretrained: str = "openai",
        device: str = "cuda",
    ):
        super().__init__()

        # OpenAI pretrained weights were trained with QuickGELU activation.
        # open_clip's default "ViT-B-16" and "ViT-L-14" use standard GELU,
        # causing a weight mismatch.  Auto-upgrade to the "-quickgelu" variant
        # when loading openai weights so activations match the pretrained run.
        OPENAI_QUICKGELU_MAP = {
            "ViT-B-32":  "ViT-B-32-quickgelu",
            "ViT-B-16":  "ViT-B-16-quickgelu",
            "ViT-L-14":  "ViT-L-14-quickgelu",
        }
        if pretrained == "openai" and model_name in OPENAI_QUICKGELU_MAP:
            resolved_name = OPENAI_QUICKGELU_MAP[model_name]
            print(f"[CLIPWrapper] Resolving '{model_name}' -> '{resolved_name}' "
                  f"to match OpenAI QuickGELU activation")
        else:
            resolved_name = model_name

        # Load CLIP model
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            resolved_name,
            pretrained=pretrained,
            device=device,
        )
        
        # Get tokenizer
        self.tokenizer = open_clip.get_tokenizer(resolved_name)
        
        # Extract encoders (OpenCLIP uses 'visual' and calling model.encode_text)
        self.visual = self.model.visual
        # For text encoding, we'll use the full model's encode_text method
        
        # Get feature dimensions
        if hasattr(self.model, 'embed_dim'):
            self.embed_dim = self.model.embed_dim
        else:
            # Fallback: check visual encoder output
            self.embed_dim = self.visual.output_dim if hasattr(self.visual, 'output_dim') else 512
        
        # Temperature parameter
        self.logit_scale = self.model.logit_scale
        
        print(f"Loaded CLIP model: {resolved_name} ({pretrained})")
        print(f"Embedding dimension: {self.embed_dim}")
        
    def encode_image(self, images: torch.Tensor, normalize: bool = True) -> torch.Tensor:
        """
        Encode images to feature vectors.
        
        Args:
            images: Image tensor (B, C, H, W)
            normalize: Whether to L2-normalize features
            
        Returns:
            Image features (B, embed_dim)
        """
        features = self.visual(images)
        if normalize:
            features = features / features.norm(dim=-1, keepdim=True)
        return features
    
    def encode_text(self, text: torch.Tensor, normalize: bool = True) -> torch.Tensor:
        """
        Encode text to feature vectors.
        
        Args:
            text: Text tokens (B, seq_len)
            normalize: Whether to L2-normalize features
            
        Returns:
            Text features (B, embed_dim)
        """
        features = self.model.encode_text(text, normalize=normalize)
        return features
    
    def forward(
        self,
        images: Optional[torch.Tensor] = None,
        text: Optional[torch.Tensor] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Forward pass through both encoders.
        
        Args:
            images: Image tensor (B, C, H, W)
            text: Text tokens (B, seq_len)
            
        Returns:
            Tuple of (image_features, text_features)
        """
        image_features = None
        text_features = None
        
        if images is not None:
            image_features = self.encode_image(images)
        
        if text is not None:
            text_features = self.encode_text(text)
        
        return image_features, text_features
    
    def get_vision_parameters(self):
        """Get parameters of vision encoder."""
        return self.visual.parameters()
    
    def get_text_parameters(self):
        """Get parameters of text encoder."""
        # OpenCLIP architecture: text encoder is part of the model
        # We need to get text-related parameters
        text_params = []
        for name, param in self.model.named_parameters():
            if 'text' in name.lower() or 'token' in name.lower() or 'positional' in name.lower():
                if 'visual' not in name.lower():
                    text_params.append(param)
        return iter(text_params)
    
    def freeze_base_model(self):
        """Freeze all parameters in the base model."""
        for param in self.parameters():
            param.requires_grad = False
        print("Froze all base CLIP parameters")
