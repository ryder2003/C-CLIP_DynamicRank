"""
LoRA (Low-Rank Adaptation) implementation for C-CLIP.
Based on: "LoRA: Low-Rank Adaptation of Large Language Models" (Hu et al., 2021)

OpenCLIP-specific note
----------------------
PyTorch's MultiheadAttention packs Q, K, V into a single `in_proj_weight`
of shape [3*embed_dim, embed_dim].  There are NO separate 'q_proj'/'v_proj'
nn.Linear submodules.  This file handles both cases:

  * LoRALayer        – wraps any plain nn.Linear (e.g. out_proj, c_proj)
  * LoRAForAttn      – wraps nn.MultiheadAttention and injects LoRA deltas
                        directly into the Q and V slices of in_proj_weight,
                        exactly as described in the C-CLIP paper.
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
        
        # Move to same device as LoRA params
        merged_layer = merged_layer.to(self.lora_A.device)
        
        return merged_layer


# ── LoRA for packed QKV (MultiheadAttention with in_proj_weight) ────────────

class LoRAForAttn(nn.Module):
    """
    Wraps nn.MultiheadAttention and applies LoRA to the Q and V projections
    from the packed in_proj_weight, matching the C-CLIP paper exactly.

    in_proj_weight layout (PyTorch convention):
        rows   0 :   embed_dim  → Q  weight
        rows   embed_dim : 2*embed_dim  → K  weight  (unchanged)
        rows 2*embed_dim : 3*embed_dim  → V  weight
    """

    def __init__(
        self,
        original_attn: nn.MultiheadAttention,
        r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.1,
    ):
        super().__init__()
        self.original_attn = original_attn
        self.embed_dim = original_attn.embed_dim
        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / r
        self.lora_dropout = nn.Dropout(p=lora_dropout) if lora_dropout > 0 else nn.Identity()

        # Freeze all original attention parameters
        for p in original_attn.parameters():
            p.requires_grad = False

        # Determine device from existing parameters
        device = next(original_attn.parameters()).device
        d = self.embed_dim

        # Q LoRA:  delta_Q = (lora_q_B @ lora_q_A) * scaling
        self.lora_q_A = nn.Parameter(torch.zeros(r, d, device=device))
        self.lora_q_B = nn.Parameter(torch.zeros(d, r, device=device))
        # V LoRA
        self.lora_v_A = nn.Parameter(torch.zeros(r, d, device=device))
        self.lora_v_B = nn.Parameter(torch.zeros(d, r, device=device))

        nn.init.kaiming_uniform_(self.lora_q_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_q_B)
        nn.init.kaiming_uniform_(self.lora_v_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_v_B)

    def _lora_delta(self) -> torch.Tensor:
        """Build the [3*d, d] delta for in_proj_weight: [delta_Q; 0; delta_V].
        This tensor is part of the autograd graph so gradients flow to lora_*."""
        d = self.embed_dim
        zeros = torch.zeros(d, d,
                            device=self.lora_q_A.device,
                            dtype=self.lora_q_A.dtype)
        delta_q = (self.lora_q_B @ self.lora_q_A) * self.scaling  # [d, d]
        delta_v = (self.lora_v_B @ self.lora_v_A) * self.scaling  # [d, d]
        return torch.cat([delta_q, zeros, delta_v], dim=0)         # [3d, d]

    def forward(self, query, key, value, **kwargs):
        """
        Forward pass using F.multi_head_attention_forward with the LoRA-augmented
        in_proj_weight.  Gradients flow correctly to lora_q_A/B and lora_v_A/B.
        Handles both batch_first=True and seq_first layouts.
        """
        attn = self.original_attn
        delta = self._lora_delta().to(attn.in_proj_weight.dtype)
        augmented_w = attn.in_proj_weight + delta   # new tensor; no in-place mod

        # If batch_first, permute (batch, seq, embed) → (seq, batch, embed)
        batch_first = getattr(attn, 'batch_first', False)
        if batch_first:
            query = query.transpose(0, 1)
            key   = key.transpose(0, 1)
            value = value.transpose(0, 1)

        out, weights = F.multi_head_attention_forward(
            query=query, key=key, value=value,
            embed_dim_to_check=attn.embed_dim,
            num_heads=attn.num_heads,
            in_proj_weight=augmented_w,
            in_proj_bias=attn.in_proj_bias,
            bias_k=attn.bias_k,
            bias_v=attn.bias_v,
            add_zero_attn=attn.add_zero_attn,
            dropout_p=attn.dropout if self.training else 0.0,
            out_proj_weight=attn.out_proj.weight,
            out_proj_bias=attn.out_proj.bias,
            training=self.training,
            key_padding_mask=kwargs.get('key_padding_mask', None),
            need_weights=kwargs.get('need_weights', False),
            attn_mask=kwargs.get('attn_mask', None),
        )
        if batch_first:
            out = out.transpose(0, 1)
        return out, weights

    def merge_lora_weights(self, integration_coeff: float = 0.5):
        """Merge LoRA deltas directly into in_proj_weight and return bare MHA."""
        with torch.no_grad():
            delta = self._lora_delta().to(self.original_attn.in_proj_weight.dtype)
            self.original_attn.in_proj_weight.data.add_(integration_coeff * delta)
        return self.original_attn  # unwrapped MHA with merged weights


# ── injection helper ─────────────────────────────────────────────────────────

def inject_lora(
    model: nn.Module,
    target_modules: List[str],
    r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.1,
) -> Dict[str, object]:
    """
    Inject LoRA into the model.

    Handles two cases automatically:
      1. target_modules contains 'q_proj' or 'v_proj':
         → wraps every nn.MultiheadAttention with LoRAForAttn
           (because OpenCLIP packs Q/K/V into in_proj_weight)
      2. any other name (e.g. 'out_proj', 'c_proj'):
         → replaces matching nn.Linear modules with LoRALayer

    Args:
        model:          Model to modify in-place
        target_modules: Names to target (e.g. ['q_proj','v_proj'] or ['out_proj'])
        r, lora_alpha, lora_dropout: LoRA hyper-parameters

    Returns:
        Dict mapping module paths → LoRALayer or LoRAForAttn instances
    """
    lora_layers = {}
    wants_qv = any(t in ('q_proj', 'v_proj') for t in target_modules)
    # Linear targets: everything in target_modules that isn't q_proj/v_proj
    linear_targets = [t for t in target_modules if t not in ('q_proj', 'v_proj')]

    for name, module in list(model.named_modules()):
        # ── GUARD: skip modules that live INSIDE existing LoRA wrappers ──
        # named_modules() recurses into LoRAForAttn.original_attn and
        # LoRALayer.original_layer, which are bare MHA/Linear.  Without
        # this check, inject_lora wraps them again on the next task, creating
        # nested chains (original_attn.original_attn.original_attn...).
        if 'original_attn' in name or 'original_layer' in name:
            continue

        # Also skip if this module IS already a LoRA wrapper
        if isinstance(module, (LoRAForAttn, LoRALayer)):
            continue

        # ── Case 1: packed QKV (wrap MultiheadAttention for Q/V LoRA) ─────
        if wants_qv and isinstance(module, nn.MultiheadAttention):
            parent_name = '.'.join(name.split('.')[:-1])
            attr_name   = name.split('.')[-1]
            parent      = model.get_submodule(parent_name) if parent_name else model

            lora_attn = LoRAForAttn(
                original_attn=module,
                r=r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
            )
            setattr(parent, attr_name, lora_attn)
            lora_layers[name] = lora_attn
            print(f"Injected Q/V LoRA into attn: {name}")

        # ── Case 2: plain nn.Linear (for MLP layers like c_fc, c_proj) ───
        elif (linear_targets
              and isinstance(module, nn.Linear)
              and any(t in name for t in linear_targets)):

            parent_name = '.'.join(name.split('.')[:-1])
            attr_name   = name.split('.')[-1]
            parent      = model.get_submodule(parent_name) if parent_name else model

            lora_layer = LoRALayer(
                original_layer=module,
                r=r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
            )
            setattr(parent, attr_name, lora_layer)
            lora_layers[name] = lora_layer
            print(f"Injected LoRA into linear: {name}")

    return lora_layers


def merge_all_lora_weights(
    model: nn.Module,
    lora_layers: Dict[str, object],
    integration_coeff: float = 0.5,
) -> nn.Module:
    """
    Merge all LoRA weights back into the base model.
    Works for both LoRALayer (nn.Linear) and LoRAForAttn (MultiheadAttention).
    """
    for name, lora_layer in lora_layers.items():
        parent_name = '.'.join(name.split('.')[:-1])
        attr_name   = name.split('.')[-1]
        parent      = model.get_submodule(parent_name) if parent_name else model

        merged = lora_layer.merge_lora_weights(integration_coeff)
        setattr(parent, attr_name, merged)
        print(f"Merged LoRA in: {name}")

    return model


def get_lora_parameters(model: nn.Module) -> List[nn.Parameter]:
    """Return all trainable LoRA parameters (LoRALayer + LoRAForAttn)."""
    params = []
    for module in model.modules():
        if isinstance(module, LoRALayer):
            params.extend([module.lora_A, module.lora_B])
        elif isinstance(module, LoRAForAttn):
            params.extend([module.lora_q_A, module.lora_q_B,
                           module.lora_v_A, module.lora_v_B])
    return params


def count_lora_parameters(model: nn.Module) -> int:
    """Count total trainable LoRA parameters."""
    return sum(p.numel() for p in get_lora_parameters(model))
