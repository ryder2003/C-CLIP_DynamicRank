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
python src/evaluate.py \
  --checkpoint checkpoints/model_final.pt \
  --config configs/default_config.yaml \
  --eval_config configs/eval_config.json \
  --output results/my_results.json
```

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
- Integration coefficient (`integration_coeff`): 0.5 is optimal
- Temperature (`temperature`): 0.07 is standard

## Expected Training Time

On a single NVIDIA RTX 4090:

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

- [ ] CLIP loss decreasing steadily
- [ ] CKC loss stable (tasks 2+)
- [ ] I2T Recall@1 > 50% on validation
- [ ] Zero-shot accuracy degradation < 10%
- [ ] Backward transfer (old tasks improve)

If not, check learning rates and data quality first!

---

**Ready to train?** Run `python src/train.py --config configs/default_config.yaml`

**Questions?** Check [README.md](README.md) or open an issue!
