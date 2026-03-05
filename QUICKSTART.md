# C-CLIP Quick Start Guide

Get up and running with C-CLIP in 10 minutes!

## Step 1: Installation (2 minutes)

```bash
# Clone or navigate to the repository
cd C-CLip_Implementation

# Create virtual environment
python -m venv .venv

# Activate (Windows)
.venv\Scripts\activate

# Activate (Linux/Mac)
# source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Step 2: Test the Implementation (3 minutes)

Run the test suite to verify everything works:

```bash
python scripts/test_implementation.py
```

You should see:
```
✓ All tests passed successfully!
```

## Step 3: Try the Minimal Example (5 minutes)

Run a minimal training example with dummy data:

```bash
python examples/minimal_train.py
```

This will:
- Initialize C-CLIP with ViT-B/32
- Train on Task 1 for 2 epochs (CLIP loss only)
- Train on Task 2 for 2 epochs (CLIP + CKC loss)
- Save the model checkpoint

Expected output:
```
Task 1: Training
  Batch [10/100] Loss: 2.1234 I2T: 15.32% T2I: 14.87%
  ...
✓ Task 1 completed and weights merged

Task 2: Training with CKC
  Batch [10/100] Loss: 2.0145 I2T: 18.45% T2I: 17.92%
  ...
✓ Task 2 completed and weights merged
```

## Step 4: Prepare Your Own Data

### Option A: Quick Format

Create a CSV file with your image-caption pairs:

```csv
image,caption
images/img1.jpg,"A photo of a cat"
images/img2.jpg,"A dog in the park"
```

### Option B: Use the Preparation Script

If you have images with paired text files:

```bash
python scripts/prepare_data.py \
  --image_dir data/my_dataset/images \
  --output_csv data/my_dataset/dataset.csv \
  --split \
  --val_ratio 0.2
```

This creates:
- `dataset_train.csv` (80% of data)
- `dataset_val.csv` (20% of data)

## Step 5: Configure Training

Edit `configs/default_config.yaml`:

```yaml
datasets:
  - name: "my_task1"
    train_path: "data/task1/train.csv"
    val_path: "data/task1/val.csv"
    image_dir: "data/task1/images"
  
  - name: "my_task2"
    train_path: "data/task2/train.csv"
    val_path: "data/task2/val.csv"
    image_dir: "data/task2/images"

training:
  batch_size: 256  # Adjust based on GPU memory
  epochs_per_task: 40
  base_lr: 0.00001  # 1e-5
```

## Step 6: Start Training

```bash
python src/train.py --config configs/default_config.yaml
```

## Common Issues & Solutions

### Issue: Out of Memory

**Solution**: Reduce batch size or use smaller model

```yaml
model:
  clip_model_name: "ViT-B-32"  # Smaller than ViT-B-16

training:
  batch_size: 64  # Reduce from 256
```

### Issue: Slow Training

**Solution**: Increase number of data loading workers

```yaml
data:
  num_workers: 8  # Increase from 4
```

### Issue: Poor Performance

**Solution**: Adjust learning rates (most common issue)

Different datasets need different learning rates:

```yaml
# For general datasets (Flickr30K-like)
training:
  base_lr: 0.00001  # 1e-5
  text_lr_multiplier: 10

# For dense caption datasets (COCO-like)
training:
  base_lr: 0.0000005  # 5e-7
  text_lr_multiplier: 80
```

## What's Next?

### Monitor Training

If you enable W&B logging:

```yaml
logging:
  use_wandb: true
  project_name: "my-c-clip-project"
```

Then check your training progress at https://wandb.ai

### Evaluate Your Model

```bash
# Evaluate per-task checkpoints (zero-shot classification)
python scripts/eval_zero_shot.py \
    --checkpoint checkpoints/real_datasets/model_after_task_0.pt \
    --config configs/real_datasets_config.yaml \
    --output results/task0_accuracy.json

python scripts/eval_zero_shot.py \
    --checkpoint checkpoints/real_datasets/model_after_task_1.pt \
    --config configs/real_datasets_config.yaml \
    --output results/task1_accuracy.json

python scripts/eval_zero_shot.py \
    --checkpoint checkpoints/real_datasets/model_final.pt \
    --config configs/real_datasets_config.yaml \
    --output results/final_accuracy.json
```

### Compute All Metrics

```bash
# Compute paper metrics (Last, Average, Transfer) + traditional CL metrics (BWT, FWT, AF, AIA)
python scripts/compute_all_metrics.py
```

Output is saved to `results/comprehensive_metrics.json`.

### Generate PDF Report

```bash
python scripts/generate_report.py
```

Produces `results/CCLIP_Implementation_Report.pdf` with all metrics, analysis, and reproducibility commands.

### Use the Trained Model

```python
from src.models.cclip import CCLIP

# Load model
model = CCLIP(
    clip_model_name="ViT-B-16",
    pretrained="openai",
    device="cuda"
)
model.load_checkpoint("checkpoints/model_final.pt")
model.eval()

# Encode images and text
import torch
from PIL import Image

image = Image.open("test.jpg")
image_tensor = preprocess(image).unsqueeze(0).to("cuda")

image_features = model.encode_image(image_tensor)
# Use for retrieval, classification, etc.
```

## Key Parameters to Tune

### Most Important (tune these first)

1. **Learning Rate** (`base_lr`): Start with 1e-5, adjust per dataset
2. **Text LR Multiplier** (`text_lr_multiplier`): 10-80x, higher for diverse captions
3. **Batch Size** (`batch_size`): Larger is better for contrastive learning

### Less Critical (use defaults)

- LoRA rank (`lora_r`): 16 works well
- Integration coefficient (`integration_coeff`): 0.7 is optimal (was 0.5 in paper)
- Temperature (`temperature`): 0.07 is standard
- LoRA targets: `[q_proj, v_proj, c_fc, c_proj]` (attention + MLP)

## Expected Training Time

### Measured (NVIDIA RTX 3050 6GB Laptop GPU):

| Task | Dataset | Train Samples | Epochs | Time |
|------|---------|---------------|--------|------|
| 0 | Flowers102 | 6,961 | 40 | ~12 hours |
| 1 | Oxford Pets | 6,282 | 40 | ~12 hours |
| 2 | Simpsons | 17,794 | 40 | ~24 hours |
| **Total** | | **31,037** | **120** | **~48 hours** |

Note: Batch=64, gradient_accumulation=4 (effective 256), precision=16-mixed.

### Estimated for higher-end GPUs:

On a single NVIDIA RTX 4090 (estimated):

- **ViT-B/32**: ~2 hours per task (40 epochs, 1K images, batch 256)
- **ViT-B/16**: ~4 hours per task (40 epochs, 1K images, batch 256)
- **ViT-L/14**: ~8 hours per task (40 epochs, 1K images, batch 64)

## Need Help?

1. Check the main [README.md](README.md) for detailed documentation
2. Run the test suite: `python scripts/test_implementation.py`
3. Try the minimal example: `python examples/minimal_train.py`
4. Open an issue on GitHub

## Quick Tips

✅ **DO:**
- Use large batch sizes (256+) for better contrastive learning
- Set `drop_last=True` in dataloader
- Adjust text encoder LR based on caption diversity
- Monitor both CLIP and CKC losses

❌ **DON'T:**
- Use tiny batch sizes (<32) - contrastive learning needs negatives
- Mix different image resolutions in same task
- Skip validation - helps catch issues early
- Set learning rate too high - causes instability

## Success Checklist

After training, you should see:

- [x] CLIP loss decreasing steadily (Task 0: 2.70 → 0.55)
- [x] CKC loss active and decreasing on tasks 2+ (Task 2: 0.97 → 0.72)
- [x] Zero-shot accuracy on trained task > 95% (we got 99.43%, 95.85%, 98.12%)
- [x] Zero-shot accuracy degradation controlled by CKC
- [ ] Monitor backward transfer: some forgetting is expected (-9.36% BWT in our case)

### Our Measured Results (for reference)

| Stage | Flowers | Pets | Simpsons |
|-------|---------|------|----------|
| Pretrained baseline | 69.63% | 88.72% | 61.58% |
| After Task 0 | 99.43% | 85.47% | 51.16% |
| After Task 1 | 99.19% | 95.85% | 53.27% |
| **Final** | **84.69%** | **91.88%** | **98.12%** |
| **Gain over baseline** | **+15.06%** | **+3.16%** | **+36.54%** |

### Key Metrics Summary

| Metric | Value |
|--------|-------|
| Last (final avg accuracy) | 91.56% |
| Average (all steps × domains) | 84.34% |
| Transfer (unseen domains) | 63.30% |
| Backward Transfer (BWT) | -9.36% |
| Forward Transfer (FWT) | -5.78% |
| Avg Incremental Accuracy (AIA) | 96.17% |
| Avg Gain over baseline | +18.25% |

> See [METRICS_ANALYSIS.md](METRICS_ANALYSIS.md) for detailed metric definitions and interpretations.

If your results are significantly worse, check learning rates and data quality first!

---

**Ready to train?** Run `python src/train.py --config configs/default_config.yaml`

**Questions?** Check [README.md](README.md) or open an issue!
