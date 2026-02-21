import torch, sys
sys.path.insert(0, '.')
from src.models.cclip import CCLIP
from src.losses.cclip_loss import CCLIPLoss
from torch.optim import AdamW

device = 'cuda'
torch.cuda.reset_peak_memory_stats()
torch.cuda.empty_cache()

model = CCLIP('ViT-B-16', 'openai', device=device).to(device)
model.inject_lora_for_new_task()
model.inject_lora_for_new_task()   # task 2 → CKC active, old_model set

opt = AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-5)
criterion = CCLIPLoss(temperature=0.07, use_ckc=True)

for bs in [128, 192, 256]:
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    dummy_img  = torch.randn(bs, 3, 224, 224, device=device)
    dummy_text = torch.randint(0, 49408, (bs, 77), device=device)
    opt.zero_grad()
    with torch.autocast('cuda'):
        out  = model(dummy_img, dummy_text, return_old_features=True)
        loss = criterion(
            out['image_features'], out['text_features'],
            out['projected_image_features'], out['projected_text_features'],
            out.get('old_image_features'), out.get('old_text_features'),
        )['total_loss']
    loss.backward()
    peak = torch.cuda.max_memory_allocated() / 1024**3
    status = "SAFE" if peak < 5.5 else "OOM RISK"
    print(f"  batch_size={bs:3d}  peak={peak:.2f} GB  [{status}]")
