# Dataset Guide for C-CLIP

## Current Status

❌ **No datasets downloaded** - You need to add data to train the model  
✅ **Test suite works** - Uses synthetic data (random tensors)  
✅ **Infrastructure ready** - Data loaders and training pipeline complete

---

## Option 1: Quick Test with Sample Data (Recommended First)

### Create synthetic dataset to test the pipeline:

```bash
# Create 2 tasks with 100 samples each (for testing)
.venv\Scripts\python.exe scripts\create_sample_dataset.py --num_tasks 2 --samples_per_task 100
```

This creates:
- `data/task_1/` - First task with 80 train + 20 val images
- `data/task_2/` - Second task with 80 train + 20 val images
- Random colored images with generated captions

### Update your config:

The script will print the config to add to `configs/default_config.yaml`.

### Then train:

```bash
.venv\Scripts\python.exe src\train.py --config configs\default_config.yaml
```

**Purpose**: Verify the training pipeline works before downloading large datasets.

---

## Option 2: Download Real Datasets (For Paper Results)

### Easy to Download

#### 1. **CIFAR-100** (for zero-shot testing)
```bash
# Automatically downloaded by torchvision
# No manual download needed
```

#### 2. **Oxford-IIIT Pets**
```bash
# Download from: https://www.robots.ox.ac.uk/~vgg/data/pets/
# Or use torchvision datasets
```

### Requires Registration

#### 3. **Flickr30K**
- Website: http://shannon.cs.illinois.edu/DenotationGraph/
- Size: ~13GB
- Format: Images + captions JSON
- **Most commonly used for image-text retrieval**

#### 4. **COCO Captions**
- Website: https://cocodataset.org/#download
- Size: ~25GB (2017 train/val)
- Format: Images + annotations JSON
- **Standard benchmark**

### Harder to Find

#### 5. **Lexica** - AI-generated images
- Custom dataset from paper authors
- May need to create similar using Stable Diffusion

#### 6. **WikiArt** - Art images
- Web scraping or existing datasets
- https://github.com/cs-chan/ArtGAN/tree/master/WikiArt%20Dataset

#### 7. **Simpsons/Sketch/Kream** - Domain-specific
- May need to be recreated or substituted

---

## Option 3: Use Your Own Data

### Format Your Dataset

You have 3 options:

#### A. CSV Format (Easiest)
```csv
image,caption
images/cat1.jpg,"A fluffy cat sitting on a couch"
images/dog1.jpg,"A golden retriever playing in the park"
```

#### B. JSON Format
```json
[
  {
    "image": "images/cat1.jpg",
    "caption": "A fluffy cat sitting on a couch"
  },
  {
    "image": "images/dog1.jpg",
    "caption": "A golden retriever playing in the park"
  }
]
```

#### C. Directory with Paired Files
```
data/
  my_dataset/
    cat1.jpg
    cat1.txt  (contains caption)
    dog1.jpg
    dog1.txt  (contains caption)
```

### Prepare Your Data

```bash
# If you have images with separate caption files:
.venv\Scripts\python.exe scripts\prepare_data.py \
  --image_dir path/to/your/images \
  --output_csv data/my_dataset/dataset.csv \
  --split \
  --val_ratio 0.2
```

This creates:
- `dataset_train.csv` (80% of data)
- `dataset_val.csv` (20% of data)

---

## Recommended Approach

### For Testing the Implementation:
```bash
# 1. Create sample data
.venv\Scripts\python.exe scripts\create_sample_dataset.py --num_tasks 2 --samples_per_task 200

# 2. Train on sample data (will complete quickly)
.venv\Scripts\python.exe src\train.py --config configs\default_config.yaml
```

### For Reproducing Paper Results:
```bash
# 1. Download Flickr30K and COCO (these are most important)
# 2. Format them as CSV files
# 3. Update configs/default_config.yaml with paths
# 4. Train for 40 epochs per task
```

### For Your Own Project:
```bash
# 1. Collect your domain-specific image-caption pairs
# 2. Format as CSV
# 3. Create multiple tasks (different domains)
# 4. Train and evaluate
```

---

## Dataset Sizes Reference

| Dataset | Images | Download | Training Time* |
|---------|--------|----------|----------------|
| Sample (synthetic) | 100 | Instant | ~2 min |
| Flickr30K | 31K | ~13GB | ~2 hours |
| COCO | 120K | ~25GB | ~8 hours |
| Pets | 7K | ~800MB | ~30 min |
| Your custom | Variable | - | Depends |

*On single RTX 4090, 40 epochs, batch 256

---

## Quick Start Commands

### 1. Test with Synthetic Data (5 minutes)
```bash
# Create data
.venv\Scripts\python.exe scripts\create_sample_dataset.py

# Update config (paths will be printed)
# Edit configs/default_config.yaml

# Train
.venv\Scripts\python.exe src\train.py --config configs\default_config.yaml
```

### 2. Download Real Dataset (Flickr30K example)
```bash
# 1. Download from http://shannon.cs.illinois.edu/DenotationGraph/
# 2. Extract to data/flickr30k/
# 3. Format captions to CSV
# 4. Update config
# 5. Train
```

### 3. Use Your Own Data
```bash
# Format your data
.venv\Scripts\python.exe scripts\prepare_data.py \
  --image_dir your/images \
  --output_csv data/my_data/dataset.csv \
  --split

# Update config with your paths
# Train
.venv\Scripts\python.exe src\train.py --config configs\default_config.yaml
```

---

## What You Need Minimally

To run C-CLIP, you need **at least 2 datasets** (for continual learning):

### Option A: Synthetic (testing)
```bash
.venv\Scripts\python.exe scripts\create_sample_dataset.py --num_tasks 2
```

### Option B: Real (1 easy + 1 harder)
- **Task 1**: Your own data or Pets (small, easy)
- **Task 2**: Different domain (e.g., art, sketches, different category)

### Option C: Paper Setup (8 datasets)
- Download all 8 from the paper (significant effort)

---

## Next Steps

**Choose your path:**

✅ **Just Testing?** → Create sample data (5 min)  
✅ **Reproduce Paper?** → Download Flickr30K + COCO (start here)  
✅ **Your Project?** → Format your own data  

**Then:**
1. Update `configs/default_config.yaml` with dataset paths
2. Run training: `.venv\Scripts\python.exe src\train.py --config configs\default_config.yaml`

---

## Bottom Line

- **Tests work** because they use dummy data (random tensors)
- **Training needs real data** - either synthetic or downloaded
- **Easiest start**: Run `create_sample_dataset.py` to test the pipeline
- **For real results**: Download Flickr30K and COCO
