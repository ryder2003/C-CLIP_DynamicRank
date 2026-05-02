# Dataset Guide for C-CLIP + Dynamic Rank (CoDyRA)

## Current Status

✅ **5-Task CoDyRA Benchmark**: Aircraft, DTD, EuroSAT, Flowers102, Oxford Pets
✅ **Data preparation script**: `scripts/prepare_real_datasets.py`
✅ **CSV-based data loading**: Train/val splits generated automatically

---

## CoDyRA 5-Task Benchmark Datasets

These are the 5 datasets used in our continual learning benchmark:

| # | Dataset | Domain | Classes | ~Images | Source |
|---|---------|--------|---------|---------|--------|
| 0 | FGVC Aircraft | Fine-grained aircraft | 100 | 10K | [fgvc.org](https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/) |
| 1 | DTD (Textures) | Describable textures | 47 | 5.6K | [robots.ox.ac.uk](https://www.robots.ox.ac.uk/~vgg/data/dtd/) |
| 2 | EuroSAT | Satellite imagery | 10 | 27K | [GitHub](https://github.com/phelber/EuroSAT) |
| 3 | Flowers102 | Fine-grained flowers | 102 | 8K | [robots.ox.ac.uk](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/) |
| 4 | Oxford Pets | Cat/dog breeds | 37 | 7.4K | [robots.ox.ac.uk](https://www.robots.ox.ac.uk/~vgg/data/pets/) |

### Directory Structure

```
datasets/                          # Raw downloaded images
├── fgvc_aircraft/images/          # Aircraft images
├── dtd/images/                    # Texture images
├── eurosat/                       # Satellite images (subdirs per class)
├── 102flowers/jpg/                # Flower images
└── Oxford_IIITPets/images/        # Pet images

data/                              # Generated CSV splits
├── fgvc_aircraft/
│   ├── train.csv                  # 6,667 image-text pairs
│   └── val.csv                    # 3,333 image-text pairs
├── dtd/
│   ├── train.csv                  # 3,760 pairs
│   └── val.csv                    # 1,880 pairs
├── eurosat/
│   ├── train.csv                  # 22,950 pairs
│   └── val.csv                    # 4,050 pairs
├── flowers102/
│   ├── train.csv                  # 2,040 pairs
│   └── val.csv                    # 6,149 pairs
└── oxford_pets/
    ├── train.csv                  # 6,282 pairs
    └── val.csv                    # 1,108 pairs
```

---

## Step 1: Download Datasets

### FGVC Aircraft
```bash
# Download from https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/
# Extract to datasets/fgvc_aircraft/
# Images should be in datasets/fgvc_aircraft/images/
```

### DTD (Describable Textures)
```bash
# Download from https://www.robots.ox.ac.uk/~vgg/data/dtd/
# Extract to datasets/dtd/
# Images should be in datasets/dtd/images/ (with subdirectories per class)
```

### EuroSAT
```bash
# Download from https://github.com/phelber/EuroSAT
# Extract to datasets/eurosat/
# Images should be in datasets/eurosat/ (with subdirectories per class)
```

### Flowers102
```bash
# Download from https://www.robots.ox.ac.uk/~vgg/data/flowers/102/
# Extract to datasets/102flowers/
# Images should be in datasets/102flowers/jpg/
```

### Oxford-IIIT Pets
```bash
# Download from https://www.robots.ox.ac.uk/~vgg/data/pets/
# Extract to datasets/Oxford_IIITPets/
# Images should be in datasets/Oxford_IIITPets/images/
```

---

## Step 2: Generate CSV Splits

```bash
python scripts/prepare_real_datasets.py
```

This script:
1. Scans each dataset directory for images
2. Generates text captions from class names (e.g., "A photo of a Boeing 747")
3. Creates train/val CSV splits
4. Validates all image paths exist

### CSV Format

```csv
image,caption
datasets/fgvc_aircraft/images/0034309.jpg,"A photo of a Boeing 737-200"
datasets/fgvc_aircraft/images/0034310.jpg,"A photo of a Cessna 172"
```

---

## Step 3: Train

```bash
# Verify all CSVs exist
python src/train_bandit.py --config bandit_config.yaml --fresh
```

The training script validates all CSV files at startup and will report any missing files.

---

## Using Your Own Datasets

### Adding a New Dataset

1. **Place images** in `datasets/your_dataset/images/`

2. **Create CSV files** in `data/your_dataset/`:
```csv
image,caption
datasets/your_dataset/images/img1.jpg,"A photo of a your class name"
datasets/your_dataset/images/img2.jpg,"A photo of another class name"
```

3. **Add to `bandit_config.yaml`**:
```yaml
datasets:
  # ... existing datasets ...
  - name: "your_dataset"
    train_path: "data/your_dataset/train.csv"
    val_path: "data/your_dataset/val.csv"
    image_dir: "datasets/your_dataset/images"
```

### Dataset Size Guidelines

| Size | Images | Recommended Epochs | Expected Time |
|------|--------|-------------------|---------------|
| Small | < 5K | 30–50 | 5–15 min |
| Medium | 5K–20K | 20–30 | 15–60 min |
| Large | 20K+ | 10–20 | 60+ min |

---

## Dataset Diversity Tips

For best continual learning results:
- **Mix domains**: Fine-grained, textures, remote sensing, natural images
- **Vary class counts**: From 10 (EuroSAT) to 102 (Flowers) classes
- **Vary dataset sizes**: Small (2K) to large (23K) training sets
- **Order matters**: Start with moderate difficulty, save easiest for last

Our benchmark order (Aircraft → DTD → EuroSAT → Flowers → Pets) was chosen to test across diverse visual domains.

---

## Troubleshooting

### "CSV file not found"
- Run `python scripts/prepare_real_datasets.py` first
- Check that raw images exist in `datasets/` directory

### "Image not found" during training
- Verify image paths in CSV files are correct relative to project root
- Check `image_dir` in `bandit_config.yaml` matches actual directory

### "0 image-text pairs loaded"
- CSV may be empty or malformed
- Check CSV has `image,caption` header
- Verify image files exist at the paths listed

---

**Ready?** Run `python scripts/prepare_real_datasets.py` then `python src/train_bandit.py --config bandit_config.yaml --fresh`
