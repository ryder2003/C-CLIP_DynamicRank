import sys
sys.path.insert(0, 'c:\\C-CLip_Implementation')

import torch
import open_clip
import torch.nn as nn

# Load model
model, _, _ = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')

# Check the first attention block in detail
attn = model.visual.transformer.resblocks[0].attn
print("=" * 60)
print("MultiheadAttention attributes:")
print("=" * 60)
for name in dir(attn):
    if not name.startswith('_'):
        attr = getattr(attn, name)
        if isinstance(attr, (nn.Parameter, nn.Linear, nn.Module)):
            print(f"{name}: {type(attr).__name__}")
