"""
LoRA (Low-Rank Adaptation) implementation for C-CLIP.
Based on: "LoRA: Low-Rank Adaptation of Large Language Models" (Hu et al., 2021)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional
import math


class LoRALayer(nn.Module):
    """
    LoRA layer that adds low-rank adaptation to a linear layer.
    
    Args:
        original_layer: The original linear layer to adapt
        r: Rank of the low-rank matrices (default: 16)
        lora_alpha: Scaling factor (default: 32, typically 2*r)
        lora_dropout: Dropout probability (default: 0.1)
    """
    
    def __init__(
        self,
        original_layer: nn.Linear,
        r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.1,
    ):
        super().__init__()
        self.r = r
        self.lora_alpha = lora_alpha
        self.lora_dropout = nn.Dropout(p=lora_dropout) if lora_dropout > 0 else nn.Identity()
        
        # Store original layer
        self.original_layer = original_layer
        self.in_features = original_layer.in_features
        self.out_features = original_layer.out_features
        
        # Freeze original weights
        for param in self.original_layer.parameters():
            param.requires_grad = False
        
        # Initialize LoRA matrices
        # A: (r, in_features) - initialized with Kaiming uniform
        # B: (out_features, r) - initialized with zeros
        self.lora_A = nn.Parameter(torch.zeros(r, self.in_features))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, r))
        
        # Scaling factor
        self.scaling = self.lora_alpha / self.r
        
        # Initialize A with Kaiming uniform, B with zeros
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
    
    @property
    def weight(self):
        """Expose weight attribute for compatibility with MultiheadAttention."""
        return self.original_layer.weight
    
    @property
    def bias(self):
        """Expose bias attribute for compatibility with MultiheadAttention."""
        return self.original_layer.bias if hasattr(self.original_layer, 'bias') else None
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: output = W_original @ x + (B @ A @ x) * scaling
        """
        # Original output
        result = self.original_layer(x)
        
        # LoRA adaptation: x -> A -> B -> scale
        # x: (..., in_features)
        # lora_out: (..., out_features)
        lora_out = self.lora_dropout(x)
        lora_out = F.linear(lora_out, self.lora_A)  # (..., r)
        lora_out = F.linear(lora_out, self.lora_B)  # (..., out_features)
        lora_out = lora_out * self.scaling
        
        return result + lora_out
    
    def merge_lora_weights(self, integration_coeff: float = 0.5) -> nn.Linear:
        """
        Merge LoRA weights into the original layer.
        Returns a new linear layer with merged weights.
        
        Args:
            integration_coeff: Coefficient for merging (default: 0.5)
        """
        # Compute LoRA weight matrix: B @ A
        lora_weight = self.lora_B @ self.lora_A  # (out_features, in_features)
        
        # Merge with original weights: W_new = W_old + alpha * scaling * (B @ A)
        merged_weight = self.original_layer.weight.data + integration_coeff * self.scaling * lora_weight
        
        # Create new layer
        merged_layer = nn.Linear(self.in_features, self.out_features, bias=self.original_layer.bias is not None)
        merged_layer.weight.data = merged_weight
        if self.original_layer.bias is not None:
            merged_layer.bias.data = self.original_layer.bias.data.clone()
        
        return merged_layer


def inject_lora(
    model: nn.Module,
    target_modules: List[str],
    r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.1,
) -> Dict[str, LoRALayer]:
    """
    Inject LoRA layers into specified modules of the model.
    
    Args:
        model: The model to inject LoRA into
        target_modules: List of module names to replace (e.g., ['q_proj', 'v_proj'])
        r: LoRA rank
        lora_alpha: LoRA scaling factor
        lora_dropout: LoRA dropout probability
        
    Returns:
        Dictionary mapping module paths to LoRA layers
    """
    lora_layers = {}
    
    for name, module in model.named_modules():
        # Check if this module should be replaced
        if any(target in name for target in target_modules):
            if isinstance(module, nn.Linear):
                # Get parent module and attribute name
                parent_name = '.'.join(name.split('.')[:-1])
                attr_name = name.split('.')[-1]
                
                if parent_name:
                    parent = model.get_submodule(parent_name)
                else:
                    parent = model
                
                # Create LoRA layer
                lora_layer = LoRALayer(
                    original_layer=module,
                    r=r,
                    lora_alpha=lora_alpha,
                    lora_dropout=lora_dropout,
                )
                
                # Replace the module
                setattr(parent, attr_name, lora_layer)
                lora_layers[name] = lora_layer
                
                print(f"Injected LoRA into: {name}")
    
    return lora_layers


def merge_all_lora_weights(
    model: nn.Module,
    lora_layers: Dict[str, LoRALayer],
    integration_coeff: float = 0.5,
) -> nn.Module:
    """
    Merge all LoRA weights back into the base model.
    
    Args:
        model: Model with LoRA layers
        lora_layers: Dictionary of LoRA layers to merge
        integration_coeff: Integration coefficient (default: 0.5)
        
    Returns:
        Model with merged weights
    """
    for name, lora_layer in lora_layers.items():
        # Get parent module and attribute name
        parent_name = '.'.join(name.split('.')[:-1])
        attr_name = name.split('.')[-1]
        
        if parent_name:
            parent = model.get_submodule(parent_name)
        else:
            parent = model
        
        # Merge and replace
        merged_layer = lora_layer.merge_lora_weights(integration_coeff)
        setattr(parent, attr_name, merged_layer)
        
        print(f"Merged LoRA weights in: {name}")
    
    return model


def get_lora_parameters(model: nn.Module) -> List[nn.Parameter]:
    """
    Get all LoRA parameters from the model.
    
    Args:
        model: Model with LoRA layers
        
    Returns:
        List of LoRA parameters
    """
    lora_params = []
    for module in model.modules():
        if isinstance(module, LoRALayer):
            lora_params.extend([module.lora_A, module.lora_B])
    return lora_params


def count_lora_parameters(model: nn.Module) -> int:
    """
    Count the number of trainable LoRA parameters.
    
    Args:
        model: Model with LoRA layers
        
    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in get_lora_parameters(model))
