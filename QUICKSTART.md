# C-CLIP + Dynamic Rank: Quick Start Guide

Get up and running with C-CLIP and MAB dynamic rank selection in 10 minutes!

## Step 1: Installation (2 minutes)

```bash
# Clone or navigate to the repository
cd C-CLIP_DynamicRank

# Create virtual environment
python -m venv .venv

# Activate (Linux/Mac)
source .venv/bin/activate

# Activate (Windows)
# .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Step 2: Prepare Datasets (5 minutes)

Download datasets to the `datasets/` directory, then generate CSV splits:

```bash
python scripts/prepare_real_datasets.py
```

This creates train/val CSV files in `data/` for all 5 benchmark datasets:

| Dataset | Directory | Classes | ~Images |
|---------|-----------|---------|---------|
| FGVC Aircraft | `datasets/fgvc_aircraft/` | 100 | 10K |
| DTD | `datasets/dtd/` | 47 | 5.6K |
| EuroSAT | `datasets/eurosat/` | 10 | 27K |
| Flowers102 | `datasets/102flowers/` | 102 | 8K |
| Oxford Pets | `datasets/Oxford_IIITPets/` | 37 | 7.4K |

## Step 3: Train with Dynamic Rank Selection

```bash
# Fresh training run with MAB rank selection
python src/train_bandit.py --config bandit_config.yaml --fresh
```

The `--fresh` flag clears any previous bandit state for a clean run.

### What happens during training:

1. **Baselines**: Pretrained CLIP is evaluated zero-shot on all 5 datasets
2. **Task 1–4**: Force-exploration — each rank arm (4, 8, 16, 32) gets one pull
3. **Task 5**: UCB1 exploitation — bandit picks the best-performing rank
4. **After each task**: LoRA merged → zero-shot eval on all learned tasks → reward computed

### Expected output:

```
Computing pretrained CLIP zero-shot baselines...
  fgvc_aircraft: pretrained zero-shot = 23.97%
  dtd: pretrained zero-shot = 43.99%
  eurosat: pretrained zero-shot = 46.86%
  flowers102: pretrained zero-shot = 67.88%
  oxford_pets: pretrained zero-shot = 87.27%

Starting Task 1 — fgvc_aircraft
[RankBandit] Task 0: force_explore, Rank chosen: 4
  ...training 30 epochs...
  fgvc_aircraft: accuracy = 43.59%

...

TRAINING COMPLETE
Final accuracies:
  fgvc_aircraft  :  39.36% (pretrained: 23.97%, Δ=+15.39%)
  dtd            :  66.44% (pretrained: 43.99%, Δ=+22.45%)
  eurosat        :  92.37% (pretrained: 46.86%, Δ=+45.51%)
  flowers102     :  89.87% (pretrained: 67.88%, Δ=+21.99%)
  oxford_pets    :  95.76% (pretrained: 87.27%, Δ=+8.48%)
  Average        :  76.76% (pretrained: 54.00%)
```

## Step 4: Check Results

Checkpoints and metrics are saved to `checkpoints/bandit_run/`:

```
checkpoints/bandit_run/
├── model_after_task_0_r4.pt      # After Aircraft (rank=4)
├── model_after_task_1_r8.pt      # After DTD (rank=8)
├── model_after_task_2_r16.pt     # After EuroSAT (rank=16)
├── model_after_task_3_r32.pt     # After Flowers (rank=32)
├── model_after_task_4_r16.pt     # After Pets (rank=16, UCB1)
├── model_final_bandit.pt         # Final model
├── bandit_history.json           # Full bandit state & reward history
└── cl_metrics.json               # Continual learning metrics
```

## Configuration

Edit `bandit_config.yaml` to customize:

```yaml
# Key parameters to tune:
bandit:
  rank_choices: [4, 8, 16, 32]     # Available LoRA ranks
  algorithm: "ucb1"                  # ucb1 | epsilon_greedy | thompson
  plasticity_w: 0.4                  # Weight for task learning
  stability_w: 0.6                   # Weight for anti-forgetting

training:
  epochs_per_task: 30                # More epochs = better accuracy
  base_lr: 0.00005                   # Vision LoRA learning rate
  text_lr_multiplier: 3              # Text LR = base_lr * multiplier
  ckc_weight: 2.0                    # CKC distillation strength
  pretrained_distill_weight: 1.5     # Anchor distillation strength

model:
  integration_coeff: 0.5             # LoRA merge coefficient
  # lora_alpha is DYNAMIC: alpha = 2 * rank (uniform scaling=2.0)
```

### Key Tuning Tips

| Parameter | Effect | Guidance |
|-----------|--------|----------|
| `epochs_per_task` | More = better accuracy | 30 (recommended), 50 (best) |
| `stability_w` | Higher = less forgetting | 0.6 (default), 0.7 (conservative) |
| `ckc_weight` | Higher = more distillation | 2.0 (default), increase if forgetting |
| `integration_coeff` | Lower = gentler merge | 0.5 (default), 0.3 (very gentle) |
| `base_lr` | Lower = more stable | 5e-5 (default), 1e-5 (very stable) |

## Common Issues & Solutions

### Out of Memory
```yaml
training:
  batch_size: 32            # Reduce from 64
  accumulate_grad_batches: 8  # Keep effective batch at 256
```

### Bandit Always Picks Same Rank
- This is expected after exploration! UCB1 exploits the best arm.
- To force more exploration, increase `ucb_c` (e.g., 3.0 or 4.0)

### High Forgetting on Specific Task
- Increase `pretrained_distill_weight` (e.g., 2.0 or 3.0)
- Increase `stability_w` (e.g., 0.7)
- Lower `integration_coeff` (e.g., 0.3)

## Our Measured Results (30 Epochs, for reference)

### Accuracy Progression

| Stage | Aircraft | DTD | EuroSAT | Flowers | Pets |
|-------|----------|-----|---------|---------|------|
| Pretrained CLIP | 23.97% | 43.99% | 46.86% | 67.88% | 87.27% |
| After Task 0 (r=4) | **43.59%** | — | — | — | — |
| After Task 1 (r=8) | 42.09% | **69.47%** | — | — | — |
| After Task 2 (r=16) | 39.90% | 67.66% | **93.93%** | — | — |
| After Task 3 (r=32) | 40.14% | 67.55% | 92.84% | **90.96%** | — |
| **Final (r=16)** | **39.36%** | **66.44%** | **92.37%** | **89.87%** | **95.76%** |

### Key Metrics

| Metric | Value |
|--------|-------|
| Average Accuracy (A) | **76.76%** |
| Average Forgetting (F) | **2.48%** |
| Backward Transfer (BWT) | **-2.48%** |
| All above baseline | ✅ Yes |
| Avg gain over baseline | **+22.76%** |
| Bandit best rank | r=16 |

> See [METRICS_ANALYSIS.md](METRICS_ANALYSIS.md) for detailed metric definitions and analysis.

## Success Checklist

After training, you should see:

- [x] All 5 pretrained baselines computed (~54% average)
- [x] Bandit force-explores ranks 4, 8, 16, 32 on tasks 0–3
- [x] UCB1 selects r=16 for task 4 (exploitation)
- [x] Every task finishes above its pretrained baseline
- [x] Average accuracy > 70%
- [x] BWT > -5% (minimal forgetting)
- [x] LoRA utilisation (rank entropy) > 0.99

---

**Ready to train?** Run `python src/train_bandit.py --config bandit_config.yaml --fresh`

**Questions?** Check [README.md](README.md) or [METRICS_ANALYSIS.md](METRICS_ANALYSIS.md)
