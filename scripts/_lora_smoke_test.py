import sys, torch
sys.path.insert(0, '.')
from src.models.cclip import CCLIP
from src.models.lora import count_lora_parameters, LoRAForAttn

device = 'cuda'
model = CCLIP('ViT-B-16', 'openai', device=device).to(device)
model.inject_lora_for_new_task()

n_attn = sum(1 for m in model.modules() if isinstance(m, LoRAForAttn))
total_lora = count_lora_parameters(model.clip.model)
print(f"LoRAForAttn modules injected : {n_attn}")
print(f"Total trainable LoRA params  : {total_lora:,}")

# Quick forward pass
dummy_img  = torch.randn(8, 3, 224, 224, device=device)
dummy_text = torch.randint(0, 49408, (8, 77), device=device)
out = model(dummy_img, dummy_text)
print(f"image_features shape : {out['image_features'].shape}")
print("Forward pass OK -- LoRA is working")

# Merge and check no LoRAForAttn remains
model.merge_lora_after_task()
n_remaining = sum(1 for m in model.modules() if isinstance(m, LoRAForAttn))
print(f"LoRAForAttn after merge: {n_remaining}  (expect 0)")
print("All good -- ready to train!")
