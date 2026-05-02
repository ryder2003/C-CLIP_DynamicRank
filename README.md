# C-CLIP with Dynamic Rank Selection (CoDyRA)

A complete PyTorch implementation of **C-CLIP (Continual CLIP)** enhanced with **Multi-Armed Bandit (MAB) dynamic LoRA rank selection** — a multimodal continual learning framework that enables vision-language models to continuously learn from new datasets without catastrophic forgetting, while automatically selecting the optimal LoRA rank per task.

Based on: *"C-CLIP: Multimodal Continual Learning"* + our novel **CoDyRA** (Continual Dynamic Rank Allocation) extension.

## 🌟 Key Features

- **Dynamic LoRA Rank Selection**: Multi-Armed Bandit (UCB1) automatically selects optimal LoRA rank ∈ {4, 8, 16, 32} per task
- **Uniform Scaling via Dynamic Alpha**: `alpha = 2 * rank` ensures consistent scaling=2.0 across all ranks, isolating capacity from magnitude
- **Dual Distillation**: CKC loss against previous-task model + anchor distillation against frozen pretrained CLIP
- **Contrastive Knowledge Consolidation (CKC)**: Novel loss function that learns from old model features
- **Multimodal Learning**: Handles both vision and text modalities with asymmetric learning rates
- **Stateless Continual Learning**: No task-ID required at inference
- **Zero-Shot Preservation**: Maintains general zero-shot capabilities across domains
- **Bandit-Guided Exploration**: Force-explores all ranks, then exploits the best via UCB1

## 📊 Our Results (5-Task CoDyRA Benchmark, 30 Epochs)

### Task Sequence & Bandit Rank Choices

| Task | Dataset | Classes | Train Samples | LoRA Rank (MAB) | Strategy |
|------|---------|---------|---------------|-----------------|----------|
| 1 | FGVC Aircraft | 100 | 6,667 | **r=4** | force_explore |
| 2 | DTD (Textures) | 47 | 3,760 | **r=8** | force_explore |
| 3 | EuroSAT | 10 | 22,950 | **r=16** | force_explore |
| 4 | Flowers102 | 102 | 2,040 | **r=32** | force_explore |
| 5 | Oxford Pets | 37 | 6,282 | **r=16** | ucb1 exploit |

### Pretrained CLIP Zero-Shot Baselines

| Dataset | Pretrained Accuracy |
|---------|-------------------|
| FGVC Aircraft | 23.97% |
| DTD | 43.99% |
| EuroSAT | 46.86% |
| Flowers102 | 67.88% |
| Oxford Pets | 87.27% |
| **Average** | **54.00%** |

### Accuracy Matrix R[i,j]

| Stage | Aircraft | DTD | EuroSAT | Flowers | Pets |
|-------|----------|-----|---------|---------|------|
| Pretrained CLIP | 23.97% | 43.99% | 46.86% | 67.88% | 87.27% |
| After Task 0 (Aircraft, r=4) | **43.59%** | — | — | — | — |
| After Task 1 (DTD, r=8) | 42.09% | **69.47%** | — | — | — |
| After Task 2 (EuroSAT, r=16) | 39.90% | 67.66% | **93.93%** | — | — |
| After Task 3 (Flowers, r=32) | 40.14% | 67.55% | 92.84% | **90.96%** | — |
| **After Task 4 (Pets, r=16)** | **39.36%** | **66.44%** | **92.37%** | **89.87%** | **95.76%** |

### Final Results (After All 5 Tasks)

| Dataset | Pretrained | Final (C-CLIP + DynRank) | Gain |
|---------|-----------|--------------------------|------|
| FGVC Aircraft | 23.97% | **39.36%** | **+15.39%** |
| DTD | 43.99% | **66.44%** | **+22.45%** |
| EuroSAT | 46.86% | **92.37%** | **+45.51%** |
| Flowers102 | 67.88% | **89.87%** | **+21.99%** |
| Oxford Pets | 87.27% | **95.76%** | **+8.48%** |
| **Average** | **54.00%** | **76.76%** | **+22.76%** |

### Key Metrics

| Metric | Value |
|--------|-------|
| **Average Accuracy (A)** | **76.76%** |
| **Average Forgetting (F)** | **2.48%** |
| **Backward Transfer (BWT)** | **-2.48%** |
| All Tasks Above Baseline | ✅ Yes |
| Best Single-Task Gain | +45.51% (EuroSAT) |
| Total Training Time | ~164 min |

### Per-Task Forgetting

| Task | Peak Accuracy | Final Accuracy | Forgetting |
|------|--------------|----------------|------------|
| FGVC Aircraft | 43.59% | 39.36% | 4.23% |
| DTD | 69.47% | 66.44% | 3.03% |
| EuroSAT | 93.93% | 92.37% | 1.56% |
| Flowers102 | 90.96% | 89.87% | 1.09% |

### Bandit Reward History

| Rank | Task Used | Reward | Breakdown |
|------|-----------|--------|-----------|
| r=4 | Aircraft | 0.964 | Plasticity=0.909, Stability=1.000 |
| r=8 | DTD | 0.916 | Plasticity=0.790, Stability=1.000 |
| r=16 | EuroSAT | **1.000** | Plasticity=1.000, Stability=1.000 |
| r=32 | Flowers | 0.868 | Plasticity=0.670, Stability=1.000 |
| r=16 | Pets (exploit) | — | UCB1 selected best arm |

**UCB1 convergence:** After exploring all arms, the bandit correctly identified **r=16** as optimal and selected it for the final task.

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────────────┐
│               C-CLIP with Dynamic Rank (CoDyRA)              │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────┐                  ┌──────────────┐         │
│  │    Vision     │   LoRA (rank r) │     Text     │         │
│  │    Encoder    ├─────┬───────────┤    Encoder   │         │
│  │  (ViT-B/16)  │     │           │              │         │
│  └──────┬───────┘     │           └──────┬───────┘         │
│         │             │                  │                  │
│         │      ┌──────▼──────┐           │                  │
│         │      │  MAB Rank   │           │                  │
│         │      │  Selector   │           │                  │
│         │      │  (UCB1)     │           │                  │
│         │      │ r∈{4,8,16,32}│          │                  │
│         │      └──────┬──────┘           │                  │
│         │             │                  │                  │
│  ┌──────▼──────┐      │          ┌──────▼──────┐           │
│  │   Vision    │      │          │    Text     │           │
│  │  Projector  │      │          │  Projector  │           │
│  └──────┬──────┘      │          └──────┬──────┘           │
│         └─────────────┼──────────────────┘                  │
│                       │                                      │
│             ┌─────────▼──────────┐                          │
│             │  CLIP Loss + CKC   │                          │
│             │  + Anchor Distill  │                          │
│             └────────────────────┘                          │
│                                                              │
│   Frozen Models (for distillation):                         │
│   ┌──────────────┐  ┌──────────────┐  ┌──────────────┐    │
│   │  Pretrained  │  │  Old Model   │  │   Bandit     │    │
│   │  CLIP Anchor │  │  (prev task) │  │   State      │    │
│   └──────────────┘  └──────────────┘  └──────────────┘    │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

## 🚀 Installation

### Prerequisites

- Python 3.8+
- CUDA 11.8+ (for GPU support)
- 8GB+ GPU memory (for ViT-B/16)

### Setup

```bash
# Clone the repository
git clone <your-repo-url>
cd C-CLIP_DynamicRank

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate    # Windows

# Install dependencies
pip install -r requirements.txt
```

## 📊 Data Preparation

### Dataset Format (CSV)

```csv
image,caption
images/train/img1.jpg,"A photo of a cat"
images/train/img2.jpg,"A dog playing in the park"
```

### CoDyRA 5-Task Benchmark Datasets

| # | Dataset | Domain | Classes | Images |
|---|---------|--------|---------|--------|
| 0 | FGVC Aircraft | Fine-grained aircraft | 100 | ~10K |
| 1 | DTD | Describable textures | 47 | ~5.6K |
| 2 | EuroSAT | Satellite imagery | 10 | ~27K |
| 3 | Flowers102 | Fine-grained flowers | 102 | ~8K |
| 4 | Oxford Pets | Cat/dog breeds | 37 | ~7.4K |

```bash
# Prepare dataset CSVs from raw downloads
python scripts/prepare_real_datasets.py
```

## 🎯 Training

### Quick Start (Dynamic Rank / MAB)

```bash
# Fresh training run with MAB rank selection
python src/train_bandit.py --config bandit_config.yaml --fresh
```

### Configuration (`bandit_config.yaml`)

```yaml
model:
  clip_model_name: "ViT-B-16"
  pretrained: "openai"
  # lora_alpha is DYNAMIC: alpha = 2 * rank (uniform scaling=2.0)
  lora_dropout: 0.05
  integration_coeff: 0.5

bandit:
  rank_choices: [4, 8, 16, 32]
  algorithm: "ucb1"
  plasticity_w: 0.4      # weight for task accuracy
  stability_w: 0.6       # weight for retention (anti-forgetting)
  ucb_c: 2.0

training:
  batch_size: 64
  accumulate_grad_batches: 4    # effective batch = 256
  epochs_per_task: 30
  base_lr: 0.00005              # 5e-5
  text_lr_multiplier: 3
  ckc_weight: 2.0
  pretrained_distill_weight: 1.5
```

### Training Process

The training automatically handles:
1. **Zero-shot baselines**: Evaluates pretrained CLIP on all tasks
2. **Per-task training**: Bandit selects LoRA rank → inject LoRA → train → merge
3. **Force-exploration**: First N tasks explore each rank arm once
4. **UCB1 exploitation**: Subsequent tasks pick the best rank via UCB1
5. **Evaluation**: Zero-shot classification on all learned tasks after each task
6. **Reward computation**: Plasticity + stability reward updates bandit state

### Bandit Reward Formula

```
Plasticity  = task_acc_gain / max_possible_gain
Stability   = worst_case_retention_ratio
Reward      = plasticity_w * Plasticity + stability_w * Stability
```

## 📈 Evaluation

### Evaluate Final Model

```bash
# The training script evaluates automatically after each task
# For standalone evaluation with CL metrics:
python scripts/eval_bandit.py
```

### Checkpoint Files

```
checkpoints/bandit_run/
├── model_after_task_0_r4.pt     # After Aircraft (rank=4)
├── model_after_task_1_r8.pt     # After DTD (rank=8)
├── model_after_task_2_r16.pt    # After EuroSAT (rank=16)
├── model_after_task_3_r32.pt    # After Flowers (rank=32)
├── model_after_task_4_r16.pt    # After Pets (rank=16, UCB1)
├── model_final_bandit.pt        # Final model
├── bandit_history.json          # Full bandit state & history
└── cl_metrics.json              # Continual learning metrics
```

## 🧪 Key Implementation Details

### Dynamic LoRA Rank Selection

```python
# Alpha scales with rank for uniform scaling magnitude
alpha = 2 * rank    # scaling = alpha/r = 2.0 for ALL ranks
# This isolates rank's effect to capacity (parameter count) only
```

| Rank | Alpha | Scaling | LoRA Params | Description |
|------|-------|---------|-------------|-------------|
| 4 | 8 | 2.0 | 860K | Minimal adaptation |
| 8 | 16 | 2.0 | 1.7M | Light adaptation |
| 16 | 32 | 2.0 | 3.4M | Moderate adaptation |
| 32 | 64 | 2.0 | 6.9M | Full adaptation |

### Loss Functions

#### CLIP Loss (contrastive alignment)
```math
L_CLIP = -1/(2N) Σ[log(exp(z_v·z_c/τ) / Σexp(z_v·z_c'/τ))]
```

#### CKC Loss (knowledge consolidation vs previous task)
```math
L_CKC = -1/(2N) Σ[log(exp(h_new·z_old/τ) / Σexp(h_new·z_old'/τ))]
```

#### Pretrained Anchor Distillation
```math
L_anchor = -1/(2N) Σ[log(exp(h_new·z_pretrained/τ) / Σexp(h_new·z_pretrained'/τ))]
```

#### Total Loss
```math
L_total = L_CLIP + ckc_weight * L_CKC + pretrained_distill_weight * L_anchor
```

### Hyperparameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| Optimizer | AdamW (β₁=0.9, β₂=0.99) | |
| Weight Decay | 0.01 (projectors), 0.0 (LoRA) | Critical for LoRA stability |
| Base LR | 5e-5 | Vision LoRA |
| Text LR | 1.5e-4 (3× multiplier) | Text encoder |
| Scheduler | Cosine decay + 3-epoch warmup | ~10% warmup ratio |
| Batch Size | 64 micro (256 effective) | 4× gradient accumulation |
| Temperature | 0.07 | |
| Epochs/Task | 30 | |
| Precision | 16-mixed (AMP) | |
| CKC Weight | 2.0 | Distillation strength |
| Anchor Weight | 1.5 | Pretrained anchor strength |
| Integration Coeff | 0.5 | LoRA merge coefficient |

## 📁 Project Structure

```
C-CLIP_DynamicRank/
├── src/
│   ├── models/
│   │   ├── lora.py              # LoRA implementation
│   │   ├── clip_wrapper.py      # CLIP model wrapper (OpenCLIP)
│   │   └── cclip.py             # Base C-CLIP model
│   ├── losses/
│   │   └── cclip_loss.py        # CLIP + CKC + anchor losses
│   ├── data/
│   │   ├── datasets.py          # Dataset classes & DataModule
│   │   └── transforms.py        # CLIP image transformations
│   ├── utils/
│   │   ├── config.py            # Configuration management
│   │   └── evaluation.py        # Zero-shot evaluation
│   ├── train.py                 # Base training script
│   └── train_bandit.py          # MAB dynamic rank training
│
├── cclip_bandit.py              # CCLIPWithBandit model
├── rank_bandit.py               # LoRARankBandit (UCB1/ε-greedy/Thompson)
├── bandit_config.yaml           # MAB training configuration
│
├── configs/                     # Additional config files
├── data/                        # Dataset CSV splits
├── datasets/                    # Raw dataset images
├── checkpoints/                 # Model checkpoints
├── results/                     # Evaluation results
├── report.tex                   # LaTeX report
├── requirements.txt
└── README.md
```

## 🔬 Novel Contributions

### 1. Multi-Armed Bandit Rank Selection (CoDyRA)
Instead of using a fixed LoRA rank for all tasks, our system uses UCB1 to dynamically choose the optimal rank per task. The bandit balances:
- **Plasticity** (learning new tasks well)
- **Stability** (not forgetting old tasks)

### 2. Uniform Scaling via Dynamic Alpha
Setting `alpha = 2 * rank` ensures that the LoRA scaling factor is always 2.0, regardless of rank. This lets the bandit evaluate ranks purely on their **capacity** (number of parameters), not on scaling artifacts.

### 3. Dual Distillation Architecture
- **CKC against previous-task model**: Prevents forgetting the most recently learned task
- **Anchor distillation against frozen pretrained CLIP**: Preserves the original zero-shot transfer capability from the very first task

### 4. Bandit Reward with Worst-Case Stability
The reward function uses worst-case retention across all previously seen tasks, not just average retention. This strongly penalizes catastrophic forgetting on any single task.

## 🐛 Troubleshooting

### Out of Memory
- Reduce batch size in config
- Use smaller model (ViT-B-32 instead of ViT-B-16)
- Reduce max rank from 32 to 16

### Bandit Choosing Suboptimal Ranks
- Increase `ucb_c` for more exploration
- Check that rewards are computed correctly
- Ensure enough tasks for exploration (≥ number of rank arms)

### High Forgetting
- Increase `stability_w` (e.g., 0.7)
- Increase `pretrained_distill_weight`
- Lower `integration_coeff` for gentler merging

## 📚 Citation

If you use this implementation, please cite:

```bibtex
@article{cclip2024,
  title={C-CLIP: Continual CLIP for Multimodal Continual Learning},
  author={[Authors]},
  journal={[Conference/Journal]},
  year={2024}
}
```

## 📄 License

This implementation is provided for research purposes. Please check the original paper for licensing information.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

---

**Note**: This implementation extends the C-CLIP paper with a novel Multi-Armed Bandit dynamic rank selection mechanism (CoDyRA). The base C-CLIP method is from the original paper; the dynamic rank allocation is our contribution.
