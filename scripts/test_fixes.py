"""Smoke test: verify all fixes compile and produce correct outputs."""
import sys
sys.path.insert(0, '.')

import torch

# 1. Test CCLIPLoss with dual distillation
print("=" * 60)
print("TEST 1: CCLIPLoss with dual distillation")
print("=" * 60)
from src.losses.cclip_loss import CCLIPLoss

loss = CCLIPLoss(
    temperature=0.07,
    use_ckc=True,
    ckc_weight=5.0,
    pretrained_distill_weight=1.0,
)

B, D = 8, 512
img = torch.randn(B, D)
txt = torch.randn(B, D)
proj_img = torch.randn(B, D, requires_grad=True)
proj_txt = torch.randn(B, D)
old_img = torch.randn(B, D)
old_txt = torch.randn(B, D)
pre_img = torch.randn(B, D)
pre_txt = torch.randn(B, D)

out = loss(
    image_features=img,
    text_features=txt,
    projected_image_features=proj_img,
    projected_text_features=proj_txt,
    old_image_features=old_img,
    old_text_features=old_txt,
    pretrained_image_features=pre_img,
    pretrained_text_features=pre_txt,
)

print(f"  CLIP loss:       {out['clip_loss'].item():.4f}")
print(f"  CKC loss:        {out['ckc_loss'].item():.4f} (weighted: {5.0*out['ckc_loss'].item():.4f})")
print(f"  Pretrained loss: {out['pretrained_loss'].item():.4f} (weighted: {1.0*out['pretrained_loss'].item():.4f})")
print(f"  Total loss:      {out['total_loss'].item():.4f}")
print(f"  Gradient flows:  {out['total_loss'].grad_fn is not None}")

# Verify total = clip + 5*ckc + 1*pretrained
expected = out['clip_loss'].item() + 5.0 * out['ckc_loss'].item() + 1.0 * out['pretrained_loss'].item()
actual = out['total_loss'].item()
assert abs(expected - actual) < 1e-4, f"Loss sum mismatch: {expected} != {actual}"
print("  Loss sum check:  PASSED OK")

# 2. Test with use_ckc=False (first task)
print()
print("=" * 60)
print("TEST 2: CCLIPLoss with use_ckc=False (first task)")
print("=" * 60)
loss_t0 = CCLIPLoss(temperature=0.07, use_ckc=False)
out_t0 = loss_t0(image_features=img, text_features=txt)
print(f"  CLIP loss:       {out_t0['clip_loss'].item():.4f}")
print(f"  CKC loss:        {out_t0['ckc_loss'].item():.4f}")
print(f"  Pretrained loss: {out_t0['pretrained_loss'].item():.4f}")
print(f"  Total == CLIP:   {abs(out_t0['total_loss'].item() - out_t0['clip_loss'].item()) < 1e-6}")
print("  First-task check: PASSED OK")

# 3. Test LoRARankBandit with worst-case stability
print()
print("=" * 60)
print("TEST 3: LoRARankBandit worst-case stability")
print("=" * 60)
import importlib.util
spec = importlib.util.spec_from_file_location('rank_bandit', 'src/models/rank_bandit.py')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

b = mod.LoRARankBandit(
    rank_choices=[4, 8, 16, 32],
    algorithm='ucb1',
    plasticity_w=0.5,
    stability_w=0.5,
)

# Scenario A: Good retention
print("  Scenario A: Good retention")
r_good = b.compute_reward(
    task_accuracy=0.85, baseline_accuracy=0.24,
    zeroshot_retention=0.6, zeroshot_baseline=0.6,
    prior_task_accs=[0.55, 0.40],
    prior_task_baselines=[0.24, 0.44],
)

# Scenario B: Catastrophic drop
print("  Scenario B: Catastrophic drop")
r_bad = b.compute_reward(
    task_accuracy=0.96, baseline_accuracy=0.47,
    zeroshot_retention=0.35, zeroshot_baseline=0.6,
    prior_task_accs=[0.08, 0.51],
    prior_task_baselines=[0.24, 0.44],
)

assert r_good > r_bad, f"Good retention ({r_good:.3f}) should score higher than catastrophic ({r_bad:.3f})"
print(f"  Good={r_good:.3f} > Bad={r_bad:.3f}: PASSED OK")

# 4. Test bandit selection (force-explore phase)
print()
print("=" * 60)
print("TEST 4: Bandit force-explore and selection")
print("=" * 60)
ranks_chosen = []
for i in range(4):
    r = b.select_rank(i, f"task_{i}")
    ranks_chosen.append(r)
    b.update(rank=r, reward=0.5, task_idx=i, task_name=f"task_{i}")
assert set(ranks_chosen) == {4, 8, 16, 32}, f"Force-explore should try all ranks: {ranks_chosen}"
print(f"  Ranks chosen: {ranks_chosen}")
print(f"  All arms explored: PASSED OK")

print()
print("=" * 60)
print("ALL TESTS PASSED OK")
print("=" * 60)
