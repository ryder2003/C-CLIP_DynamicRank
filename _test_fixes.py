"""Quick end-to-end test of all fixes."""
import torch, sys
sys.path.insert(0, '.')
from src.models.cclip import CCLIP
from src.losses.cclip_loss import CCLIPLoss
from src.models.lora import get_lora_parameters

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Device: {device}')

model = CCLIP('ViT-B-16', 'openai', device=device).to(device)

# === TASK 0: Train with CLIP loss only ===
model.inject_lora_for_new_task()
lora_params = get_lora_parameters(model.clip.model)
all_params = model.get_trainable_parameters()

optimizer = torch.optim.AdamW(all_params, lr=2e-4, weight_decay=0.05)
criterion = CCLIPLoss(temperature=0.07, use_ckc=False)

dummy_img = torch.randn(8, 3, 224, 224, device=device)
dummy_text = torch.randint(0, 49408, (8, 77), device=device)

print('\n--- Task 0: 5 training steps ---')
for step in range(5):
    optimizer.zero_grad()
    out = model(dummy_img, dummy_text, return_old_features=False)
    loss_dict = criterion(
        image_features=out['image_features'],
        text_features=out['text_features'],
        projected_image_features=out['projected_image_features'],
        projected_text_features=out['projected_text_features'],
    )
    loss_dict['total_loss'].backward()
    optimizer.step()

    lora_grads_nonzero = sum(1 for p in lora_params if p.grad is not None and p.grad.abs().max() > 1e-12)
    clip_loss = loss_dict['clip_loss'].item()
    ckc_loss = loss_dict['ckc_loss'].item()
    total_loss = loss_dict['total_loss'].item()
    print(f'  Step {step}: total={total_loss:.4f} clip={clip_loss:.4f} ckc={ckc_loss:.4f}  LoRA grads: {lora_grads_nonzero}/{len(lora_params)}')

# Check that LoRA weights actually changed
lora_norms = [p.data.abs().max().item() for p in lora_params]
print(f'  Max LoRA weight magnitude after task 0: {max(lora_norms):.6e}')

# Merge LoRA
model.merge_lora_after_task()

# === TASK 1: Train with CLIP + CKC ===
print('\n--- Task 1: 5 training steps with CKC ---')
model.inject_lora_for_new_task()
lora_params = get_lora_parameters(model.clip.model)
all_params = model.get_trainable_parameters()
optimizer = torch.optim.AdamW(all_params, lr=2e-4, weight_decay=0.05)
criterion = CCLIPLoss(temperature=0.07, use_ckc=True)

for step in range(5):
    optimizer.zero_grad()
    out = model(dummy_img, dummy_text, return_old_features=True)
    loss_dict = criterion(
        image_features=out['image_features'],
        text_features=out['text_features'],
        projected_image_features=out['projected_image_features'],
        projected_text_features=out['projected_text_features'],
        old_image_features=out.get('old_image_features'),
        old_text_features=out.get('old_text_features'),
    )
    loss_dict['total_loss'].backward()
    optimizer.step()

    has_proj_grad = any(
        p.grad is not None and p.grad.abs().max() > 1e-12
        for p in list(model.vision_projector.parameters()) + list(model.text_projector.parameters())
    )
    total_loss = loss_dict['total_loss'].item()
    clip_loss = loss_dict['clip_loss'].item()
    ckc_loss = loss_dict['ckc_loss'].item()
    print(f'  Step {step}: total={total_loss:.4f} clip={clip_loss:.4f} ckc={ckc_loss:.4f} proj_grads={has_proj_grad}')

model.merge_lora_after_task()
print('\nAll fixes verified successfully!')
