"""
Download and prepare datasets for the 8-task MAB bandit experiment.

Existing (already prepared):
  - flowers102   (Oxford Flowers-102)
  - oxford_pets  (Oxford-IIIT Pets)
  - simpsons     (Simpsons Characters)

New datasets to download:
  - dtd          (Describable Textures Dataset)
  - food101      (Food-101)
  - stanford_cars (Stanford Cars)
  - fgvc_aircraft (FGVC Aircraft)
  - eurosat       (EuroSAT satellite imagery)

All datasets are downloaded via torchvision.datasets when available,
or via manual URL otherwise. For each we produce:
  - data/<name>/train.csv   (columns: image, caption)
  - data/<name>/val.csv     (columns: image, caption)

Usage:
    python scripts/prepare_bandit_datasets.py
"""

import os
import sys
import csv
import shutil
import random
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def write_csv(path, rows):
    """Write list of (image_relative_path, caption) to CSV."""
    ensure_dir(os.path.dirname(path))
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['image', 'caption'])
        for img, cap in rows:
            writer.writerow([img, cap])
    print(f"  Wrote {len(rows)} entries → {path}")


# ─────────────────────────────────────────────────────────────────────────
# DTD (Describable Textures Dataset)
# ─────────────────────────────────────────────────────────────────────────

def prepare_dtd():
    """Download and prepare DTD using torchvision."""
    print("\n=== Preparing DTD ===")
    try:
        from torchvision.datasets import DTD as DTDDataset
    except ImportError:
        print("  ERROR: torchvision.datasets.DTD not available. Upgrade torchvision >= 0.13")
        return False

    root = "datasets/dtd"
    ensure_dir(root)

    # Download train + test splits
    train_ds = DTDDataset(root=root, split='train', download=True)
    val_ds = DTDDataset(root=root, split='test', download=True)

    # Build CSV entries
    train_rows = []
    for img_path, label_idx in zip(train_ds._image_files, train_ds._labels):
        rel_path = os.path.relpath(str(img_path), start=root)
        class_name = train_ds.classes[label_idx].replace('_', ' ')
        caption = f"a photo of a {class_name} texture"
        train_rows.append((rel_path, caption))

    val_rows = []
    for img_path, label_idx in zip(val_ds._image_files, val_ds._labels):
        rel_path = os.path.relpath(str(img_path), start=root)
        class_name = val_ds.classes[label_idx].replace('_', ' ')
        caption = f"a photo of a {class_name} texture"
        val_rows.append((rel_path, caption))

    write_csv("data/dtd/train.csv", train_rows)
    write_csv("data/dtd/val.csv", val_rows)
    print(f"  DTD: {len(train_rows)} train, {len(val_rows)} val, {len(train_ds.classes)} classes")
    return True


# ─────────────────────────────────────────────────────────────────────────
# Food-101
# ─────────────────────────────────────────────────────────────────────────

def prepare_food101():
    """Download and prepare Food-101 using torchvision."""
    print("\n=== Preparing Food-101 ===")
    try:
        from torchvision.datasets import Food101
    except ImportError:
        print("  ERROR: torchvision.datasets.Food101 not available.")
        return False

    root = "datasets/food101"
    ensure_dir(root)

    train_ds = Food101(root=root, split='train', download=True)
    val_ds = Food101(root=root, split='test', download=True)

    classes = train_ds.classes

    train_rows = []
    for img_path, label_idx in zip(train_ds._image_files, train_ds._labels):
        rel_path = os.path.relpath(str(img_path), start=root)
        class_name = classes[label_idx].replace('_', ' ')
        caption = f"a photo of {class_name}, a type of food"
        train_rows.append((rel_path, caption))

    val_rows = []
    for img_path, label_idx in zip(val_ds._image_files, val_ds._labels):
        rel_path = os.path.relpath(str(img_path), start=root)
        class_name = classes[label_idx].replace('_', ' ')
        caption = f"a photo of {class_name}, a type of food"
        val_rows.append((rel_path, caption))

    write_csv("data/food101/train.csv", train_rows)
    write_csv("data/food101/val.csv", val_rows)
    print(f"  Food-101: {len(train_rows)} train, {len(val_rows)} val, {len(classes)} classes")
    return True


# ─────────────────────────────────────────────────────────────────────────
# Stanford Cars
# ─────────────────────────────────────────────────────────────────────────

def prepare_stanford_cars():
    """Download and prepare Stanford Cars using torchvision."""
    print("\n=== Preparing Stanford Cars ===")
    try:
        from torchvision.datasets import StanfordCars
    except ImportError:
        print("  ERROR: torchvision.datasets.StanfordCars not available.")
        return False

    root = "datasets/stanford_cars"
    ensure_dir(root)

    try:
        train_ds = StanfordCars(root=root, split='train', download=True)
        val_ds = StanfordCars(root=root, split='test', download=True)
    except Exception as e:
        print(f"  WARNING: Stanford Cars download may fail due to Google Drive limits: {e}")
        print("  You may need to manually download from https://www.kaggle.com/datasets/jessicali9530/stanford-cars-dataset")
        return False

    classes = train_ds.classes

    train_rows = []
    for img_path, label_idx in train_ds._samples:
        rel_path = os.path.relpath(str(img_path), start=root)
        class_name = classes[label_idx]
        caption = f"a photo of a {class_name}"
        train_rows.append((rel_path, caption))

    val_rows = []
    for img_path, label_idx in val_ds._samples:
        rel_path = os.path.relpath(str(img_path), start=root)
        class_name = classes[label_idx]
        caption = f"a photo of a {class_name}"
        val_rows.append((rel_path, caption))

    write_csv("data/stanford_cars/train.csv", train_rows)
    write_csv("data/stanford_cars/val.csv", val_rows)
    print(f"  Stanford Cars: {len(train_rows)} train, {len(val_rows)} val, {len(classes)} classes")
    return True


# ─────────────────────────────────────────────────────────────────────────
# FGVC Aircraft
# ─────────────────────────────────────────────────────────────────────────

def prepare_fgvc_aircraft():
    """Download and prepare FGVC Aircraft using torchvision."""
    print("\n=== Preparing FGVC Aircraft ===")
    try:
        from torchvision.datasets import FGVCAircraft
    except ImportError:
        print("  ERROR: torchvision.datasets.FGVCAircraft not available.")
        return False

    root = "datasets/fgvc_aircraft"
    ensure_dir(root)

    train_ds = FGVCAircraft(root=root, split='train', download=True)
    val_ds = FGVCAircraft(root=root, split='test', download=True)

    classes = train_ds.classes

    train_rows = []
    for img_path, label_idx in zip(train_ds._image_files, train_ds._labels):
        rel_path = os.path.relpath(str(img_path), start=root)
        class_name = classes[label_idx]
        caption = f"a photo of a {class_name}, a type of aircraft"
        train_rows.append((rel_path, caption))

    val_rows = []
    for img_path, label_idx in zip(val_ds._image_files, val_ds._labels):
        rel_path = os.path.relpath(str(img_path), start=root)
        class_name = classes[label_idx]
        caption = f"a photo of a {class_name}, a type of aircraft"
        val_rows.append((rel_path, caption))

    write_csv("data/fgvc_aircraft/train.csv", train_rows)
    write_csv("data/fgvc_aircraft/val.csv", val_rows)
    print(f"  FGVC Aircraft: {len(train_rows)} train, {len(val_rows)} val, {len(classes)} classes")
    return True


# ─────────────────────────────────────────────────────────────────────────
# EuroSAT
# ─────────────────────────────────────────────────────────────────────────

def prepare_eurosat():
    """Download and prepare EuroSAT using torchvision."""
    print("\n=== Preparing EuroSAT ===")
    try:
        from torchvision.datasets import EuroSAT
    except ImportError:
        print("  ERROR: torchvision.datasets.EuroSAT not available.")
        return False

    root = "datasets/eurosat"
    ensure_dir(root)

    try:
        full_ds = EuroSAT(root=root, download=True)
    except Exception as e:
        print(f"  WARNING: EuroSAT download failed: {e}")
        return False

    classes = full_ds.classes

    # Split into train (80%) and val (20%)
    all_samples = list(full_ds.samples)
    random.seed(42)
    random.shuffle(all_samples)
    split_idx = int(0.8 * len(all_samples))
    train_samples = all_samples[:split_idx]
    val_samples = all_samples[split_idx:]

    train_rows = []
    for img_path, label_idx in train_samples:
        rel_path = os.path.relpath(str(img_path), start=root)
        class_name = classes[label_idx].replace('_', ' ').lower()
        caption = f"a satellite photo of {class_name}"
        train_rows.append((rel_path, caption))

    val_rows = []
    for img_path, label_idx in val_samples:
        rel_path = os.path.relpath(str(img_path), start=root)
        class_name = classes[label_idx].replace('_', ' ').lower()
        caption = f"a satellite photo of {class_name}"
        val_rows.append((rel_path, caption))

    write_csv("data/eurosat/train.csv", train_rows)
    write_csv("data/eurosat/val.csv", val_rows)
    print(f"  EuroSAT: {len(train_rows)} train, {len(val_rows)} val, {len(classes)} classes")
    return True


# ─────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Preparing datasets for 8-task MAB bandit experiment")
    print("=" * 60)

    results = {}

    # Check existing datasets
    for name in ['flowers102', 'oxford_pets', 'simpsons']:
        train_csv = f"data/{name}/train.csv"
        if os.path.exists(train_csv):
            print(f"\n✓ {name}: already prepared ({train_csv})")
            results[name] = True
        else:
            print(f"\n✗ {name}: NOT FOUND — run original setup first")
            results[name] = False

    # Download new datasets
    results['dtd'] = prepare_dtd()
    results['food101'] = prepare_food101()
    results['stanford_cars'] = prepare_stanford_cars()
    results['fgvc_aircraft'] = prepare_fgvc_aircraft()
    results['eurosat'] = prepare_eurosat()

    # Summary
    print("\n" + "=" * 60)
    print("DATASET PREPARATION SUMMARY")
    print("=" * 60)
    all_ok = True
    for name, ok in results.items():
        status = "✓ READY" if ok else "✗ FAILED"
        print(f"  {name:20s} : {status}")
        if not ok:
            all_ok = False

    if all_ok:
        print("\nAll 8 datasets ready! Run training with:")
        print("  python src/train_bandit.py --config configs/bandit_config.yaml")
    else:
        failed = [n for n, ok in results.items() if not ok]
        print(f"\n{len(failed)} dataset(s) need attention: {', '.join(failed)}")
        print("You can still train with the available datasets by editing bandit_config.yaml")


if __name__ == '__main__':
    main()
