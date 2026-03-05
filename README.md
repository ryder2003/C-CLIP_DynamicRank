# C-CLIP: Continual CLIP Implementation

A complete PyTorch implementation of **C-CLIP (Continual CLIP)** - a multimodal continual learning framework that enables vision-language models to continuously learn from new datasets without catastrophic forgetting.

Based on the paper: *"C-CLIP: Multimodal Continual Learning"*

## 🌟 Key Features

- **LoRA Integration**: Parameter-efficient adaptation using Low-Rank Adaptation
- **Contrastive Knowledge Consolidation (CKC)**: Novel loss function that learns from old model features
- **Multimodal Learning**: Handles both vision and text modalities with asymmetric learning rates
- **Stateless Continual Learning**: No task-ID required at inference
- **Zero-Shot Preservation**: Maintains general zero-shot capabilities across domains
- **Backward Transfer**: Performance on old tasks often improves as new tasks are learned

## 📋 Results from Paper

- **Image-Text Retrieval**: 40.83% average I2T Recall@1 across 8 datasets (+9.58% over next best)
- **Zero-Shot Classification**: Only 7.42% ImageNet degradation vs 18-27% for competitors
- Often **exceeds full fine-tuning** performance on new tasks
- Demonstrates **positive backward transfer** on previous tasks

## 📊 Our Results (3 Tasks: Flowers102 → Oxford Pets → Simpsons)

### Full Accuracy Progression

The table below tracks zero-shot classification accuracy at every stage, from the pretrained baseline through bug fixes and the final trained model:

| Stage | Flowers102 | Oxford Pets | Simpsons | Notes |
|-------|-----------|-------------|----------|-------|
| Pretrained CLIP (GELU, buggy) | 63.36% | 85.02% | 51.45% | Wrong activation function |
| Pretrained CLIP (QuickGELU, correct) | 69.63% | 88.72% | 61.58% | Correct baseline |
| 1st Training Run (buggy code) | 69.63% | 88.72% | 61.58% | Identical to baseline — bugs found |
| **After Task 0 (Flowers)** | **99.43%** | 85.47% | 51.16% | +29.80% on Flowers, no CKC yet |
| **After Task 1 (Pets)** | **99.19%** | **95.85%** | 53.27% | CKC preserves Flowers at 99%! |
| **Final (After Task 2, Simpsons)** | **84.69%** | **91.88%** | **98.12%** | All tasks above baseline |

### Key Metrics

#### Paper Metrics (X-TAIL Framework)
| Metric | Value | Description |
|--------|-------|-------------|
| **Last** | **91.56%** | Avg accuracy on all domains after final step |
| **Average** | **84.34%** | Avg accuracy across all steps x all domains |
| **Transfer** | **63.30%** | Avg zero-shot on unseen domains during training |

#### Traditional CL Metrics
| Metric | Value |
|--------|-------|
| Backward Transfer (BWT) | **-9.36%** |
| Forward Transfer (FWT) | -5.78% |
| Average Forgetting | 9.36% |
| Avg Incremental Accuracy | **96.17%** |
| Avg Gain Over Baseline | **+18.25%** |
| Best single-task gain | +36.54% (Simpsons) |
| Max forgetting | 14.74% (Flowers) |
| Trainable parameters | 3.44M / 149M (2.3%) |
| Training time | ~48 hours (RTX 3050 6GB Laptop) |

### Gain Over Baseline (Final vs Pretrained)

| Dataset | Pretrained | Final (C-CLIP) | Gain |
|---------|-----------|----------------|------|
| Flowers102 | 69.63% | 84.69% | +15.06% |
| Oxford Pets | 88.72% | 91.88% | +3.16% |
| Simpsons | 61.58% | 98.12% | +36.54% |
| **Average** | **73.31%** | **91.56%** | **+18.25%** |

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                      C-CLIP Model                        │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  ┌─────────────┐                    ┌─────────────┐    │
│  │   Vision    │    LoRA Layers     │    Text     │    │
│  │   Encoder   ├───────┬────────────┤   Encoder   │    │
│  │  (ViT-B/16) │       │            │             │    │
│  └──────┬──────┘       │            └──────┬──────┘    │
│         │              │                   │            │
│         │      Current Model               │            │
│  ┌──────▼──────┐       │            ┌──────▼──────┐    │
│  │  Vision     │       │            │    Text     │    │
│  │  Projector  │       │            │  Projector  │    │
│  └──────┬──────┘       │            └──────┬──────┘    │
│         │              │                   │            │
│         └──────────────┼───────────────────┘            │
│                        │                                │
│              ┌─────────▼─────────┐                      │
│              │   CKC Loss +      │                      │
│              │   CLIP Loss       │                      │
│              └───────────────────┘                      │
│                                                          │
│         Old Model (frozen, for CKC)                     │
│  ┌─────────────┐              ┌─────────────┐          │
│  │   Vision    │              │    Text     │          │
│  │   Encoder   │              │   Encoder   │          │
│  └─────────────┘              └─────────────┘          │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

## 🚀 Installation

### Prerequisites

- Python 3.8+
- CUDA 11.8+ (for GPU support)
- 8GB+ GPU memory (for ViT-B/16), 24GB+ for ViT-L/14

### Setup

```bash
# Clone the repository
git clone <your-repo-url>
cd C-CLip_Implementation

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

## 📊 Data Preparation

### Dataset Format

C-CLIP supports multiple dataset formats:

#### 1. CSV Format (Recommended)
```csv
image,caption
images/train/img1.jpg,"A photo of a cat"
images/train/img2.jpg,"A dog playing in the park"
```

#### 2. JSON Format
```json
[
  {
    "image": "images/train/img1.jpg",
    "caption": "A photo of a cat"
  },
  {
    "image": "images/train/img2.jpg",
    "caption": "A dog playing in the park"
  }
]
```

#### 3. Directory Structure
```
data/
  dataset_name/
    images/
      img1.jpg
      img1.txt  # Contains caption
      img2.jpg
      img2.txt
```

### Example Datasets (from paper)

1. **Flickr30K** - General real-world images
2. **COCO-caption** - General real-world images
3. **Oxford Pets** - Pet domain
4. **Lexica** - AI-generated images
5. **Simpsons** - Animation domain
6. **WikiArt** - Art domain
7. **Kream** - Fashion/clothing domain
8. **Sketch** - Sketch domain

Place your datasets in the `data/` directory following the structure above.

## 🎯 Training

### Quick Start

```bash
python src/train.py --config configs/default_config.yaml
```

### Configuration

Edit `configs/default_config.yaml` to customize:

```yaml
model:
  clip_model_name: "ViT-B-16"  # ViT-B-32, ViT-B-16, ViT-L-14
  lora_r: 16                    # LoRA rank
  lora_alpha: 32                # Scaling factor
  integration_coeff: 0.7        # Merging coefficient (0.7 recommended)
  lora_target_modules:           # LoRA targets (attention + MLP)
    - "q_proj"
    - "v_proj"
    - "c_fc"
    - "c_proj"

training:
  batch_size: 64                # Adjust based on VRAM (64 for 6GB)
  gradient_accumulation_steps: 4 # Effective batch = 256
  epochs_per_task: 40
  base_lr: 0.0002              # 2e-4 for LoRA
  text_lr_multiplier: 5        # Text encoder LR multiplier

datasets:
  - name: "flickr30k"
    train_path: "data/flickr30k/train.csv"
    val_path: "data/flickr30k/val.csv"
    image_dir: "data/flickr30k/images"
```

### Training Process

The training automatically handles:
1. **Task 1**: Trains with CLIP loss only
2. **Task 2+**: Injects LoRA, trains with CLIP + CKC loss
3. **After each task**: Merges LoRA weights into base model
4. **Evaluation**: Evaluates on all learned tasks after each task

### Multi-GPU Training

```bash
# Modify config
hardware:
  devices: 4  # Use 4 GPUs
  
python src/train.py --config configs/default_config.yaml
```

### Learning Rate Guidelines (from paper)

Different datasets require different learning rates:

- **Flickr30K**: `base_lr: 1e-5`, text_multiplier: 10
- **COCO**: `base_lr: 5e-7`, text_multiplier: 80
- **Others**: `base_lr: 3e-5`, text_multiplier: 10

## 📈 Evaluation

### Evaluate on Retrieval Tasks

```bash
python src/evaluate.py \
  --checkpoint checkpoints/model_final.pt \
  --config configs/default_config.yaml \
  --eval_config configs/eval_config.json \
  --output results/evaluation_results.json
```

### Evaluate Zero-Shot Classification (Per-Task Checkpoints)

```bash
# After Task 0 (Flowers)
python scripts/eval_zero_shot.py \
    --checkpoint checkpoints/real_datasets/model_after_task_0.pt \
    --config configs/real_datasets_config.yaml \
    --output results/task0_accuracy.json

# After Task 1 (Pets)
python scripts/eval_zero_shot.py \
    --checkpoint checkpoints/real_datasets/model_after_task_1.pt \
    --config configs/real_datasets_config.yaml \
    --output results/task1_accuracy.json

# After Task 2 / Final Model
python scripts/eval_zero_shot.py \
    --checkpoint checkpoints/real_datasets/model_final.pt \
    --config configs/real_datasets_config.yaml \
    --output results/final_accuracy.json
```

### Compute All Metrics (Paper + Traditional CL)

```bash
# Computes Last, Average, Transfer, BWT, FWT, AF, AIA, and saves to results/comprehensive_metrics.json
python scripts/compute_all_metrics.py
```

### Generate PDF Report

```bash
# Generates comprehensive report at results/CCLIP_Implementation_Report.pdf
python scripts/generate_report.py
```

### Full Evaluation Pipeline (One Shot)

```bash
python scripts/eval_zero_shot.py --checkpoint checkpoints/real_datasets/model_after_task_0.pt --config configs/real_datasets_config.yaml --output results/task0_accuracy.json
python scripts/eval_zero_shot.py --checkpoint checkpoints/real_datasets/model_after_task_1.pt --config configs/real_datasets_config.yaml --output results/task1_accuracy.json
python scripts/eval_zero_shot.py --checkpoint checkpoints/real_datasets/model_final.pt --config configs/real_datasets_config.yaml --output results/final_accuracy.json
python scripts/compute_all_metrics.py
python scripts/generate_report.py
```

### Metrics Computed

**Paper Metrics (X-TAIL Framework, NeurIPS 2024):**
- **Last**: Average accuracy on all domains after the final learning step
- **Average**: Mean accuracy across all learning steps × all domains
- **Transfer**: Average zero-shot accuracy on unseen domains during training

**Traditional Continual Learning Metrics:**
- **Backward Transfer (BWT)**: How much accuracy on old tasks changes after new ones
- **Forward Transfer (FWT)**: Impact on unseen domains from previous learning
- **Average Forgetting (AF)**: Average peak-to-final accuracy drop on old tasks
- **Average Incremental Accuracy (AIA)**: Avg accuracy on learned tasks at each step

**Standard Metrics:**
- **Image-to-Text Retrieval**: Recall@1, Recall@5, Recall@10
- **Text-to-Image Retrieval**: Recall@1, Recall@5, Recall@10
- **Zero-Shot Classification**: Top-1 Accuracy
- **Gain Over Baseline**: Final vs pretrained accuracy per domain

> See [METRICS_ANALYSIS.md](METRICS_ANALYSIS.md) for detailed mathematical definitions, formulas, interpretations, and computed values for every metric.

## 🧪 Key Implementation Details

### LoRA Configuration

```python
lora_r: 16              # Rank (paper tested 8, 16, 32, 64)
lora_alpha: 32          # Scaling factor (2*r)
lora_dropout: 0.1       # Dropout probability
integration_coeff: 0.7  # Merging coefficient (0.7 optimal for our setup)
lora_target_modules:    # Inject into attention AND MLP layers
  - q_proj, v_proj      # Attention Q/V projections (LoRAForAttn)
  - c_fc, c_proj        # MLP layers (LoRALayer)
# Total: 72 LoRA layers, 3.44M trainable params
```

### Loss Functions

#### CLIP Loss
```math
L_CLIP = -1/(2N) Σ[log(exp(z_v·z_c/τ) / Σexp(z_v·z_c'/τ))]
```

#### CKC Loss
```math
L_CKC = -1/(2N) Σ[log(exp(h_new·z_old/τ) / Σexp(h_new·z_old'/τ))]
```

#### Total Loss
```math
L_total = L_CLIP + L_CKC
```

### Hyperparameters (our configuration)

- **Optimizer**: AdamW (β₁=0.9, β₂=0.99)
- **Weight Decay**: 0.01 for projectors, 0.0 for LoRA params (critical!)
- **Scheduler**: Cosine decay with 3-epoch warmup
- **Batch Size**: 64 micro (effective 256 with gradient accumulation 4)
- **Base LR**: 2e-4 (vision LoRA), 1e-3 (text LoRA, 5x multiplier)
- **Temperature**: 0.07
- **Epochs**: 40 per task
- **Precision**: 16-mixed (AMP)

## 📁 Project Structure

```
C-CLip_Implementation/
├── src/
│   ├── models/
│   │   ├── lora.py              # LoRA implementation
│   │   ├── clip_wrapper.py      # CLIP model wrapper
│   │   └── cclip.py            # Main C-CLIP model
│   ├── losses/
│   │   └── cclip_loss.py       # CLIP and CKC losses
│   ├── data/
│   │   ├── datasets.py          # Dataset classes
│   │   └── transforms.py        # Data transformations
│   ├── utils/
│   │   ├── config.py            # Configuration utilities
│   │   └── evaluation.py        # Evaluation metrics
│   ├── train.py                 # Training script
│   └── evaluate.py              # Evaluation script
├── configs/
│   ├── default_config.yaml      # Default configuration
│   └── eval_config.json         # Evaluation configuration
├── data/                        # Dataset directory
├── checkpoints/                 # Model checkpoints
├── requirements.txt
└── README.md
```

## 🔬 Paper Implementation Details

### Key Innovations

1. **LoRA Integration**: Unlike standard LoRA that keeps adapters separate, C-CLIP merges LoRA weights into the backbone after each task with coefficient α=0.5

2. **CKC Loss**: Instead of preserving old features (as in knowledge distillation), CKC treats old features as positive anchors in contrastive learning, creating 2N² pairs per batch

3. **Projector Layer**: Creates a "connected but not identical" feature space, enabling both stability and plasticity

4. **Asymmetric Learning Rates**: Text encoder learns 10-80× faster than vision encoder to handle caption diversity

### Theoretical Foundation

The paper proves that constraining parameter changes (via LoRA) is equivalent to regularization methods through Lipschitz continuity:

```math
||f_θ(v) - f_θ'(v)|| ≤ K_f ||θ - θ'||
```

## 📊 Expected Results (Paper) & Our Measured Results

### Paper Results (8 Tasks, retrieval metrics)

| Dataset | I2T R@1 | T2I R@1 |
|---------|---------|---------||
| Flickr30K | 84.40% | 73.74% |
| COCO | 56.92% | 42.82% |
| Pets | ~40% | ~35% |
| Lexica | 42.65% | 41.47% |

### Paper Zero-Shot Classification

| Dataset | Accuracy (Final) | Degradation |
|---------|-----------------|-------------|
| ImageNet | 60.31% | 7.42% |
| CIFAR-100 | 61.58% | 5.29% |

### Our Measured Results (3 Tasks: Flowers → Pets → Simpsons)

| Dataset | Pretrained Baseline | After Own Task | Final Model | Gain Over Baseline |
|---------|-------------------|----------------|-------------|-------------------|
| Flowers102 | 69.63% | 99.43% | 84.69% | +15.06% |
| Oxford Pets | 88.72% | 95.85% | 91.88% | +3.16% |
| Simpsons | 61.58% | 98.12% | 98.12% | +36.54% |
| **Average** | **73.31%** | **97.80%** | **91.56%** | **+18.25%** |

> **BWT (Backward Transfer)**: -9.36% | Flowers forgetting: -14.74% | Pets forgetting: -3.97%

## 🐛 Troubleshooting

### Out of Memory
- Reduce batch size in config
- Use smaller model (ViT-B-32 instead of ViT-B-16)
- Use gradient checkpointing (add to config)

### Poor Performance
- Check learning rates (very sensitive to dataset)
- Ensure text_lr_multiplier is set correctly (10-80x)
- Verify data loading (check image-caption pairs)

### Slow Training
- Increase num_workers for data loading
- Use mixed precision (16-mixed)
- Ensure drop_last=True in dataloader for contrastive learning

## 📚 Citation

If you use this implementation, please cite the original paper:

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

## 📧 Contact

For questions or issues, please open an issue on GitHub.

---

**Note**: This implementation is based on the C-CLIP paper. Some details may differ from the original implementation. Please refer to the paper for the official method.
