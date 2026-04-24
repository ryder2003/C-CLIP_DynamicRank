"""
Download and prepare the 10 CoDyRA-benchmark datasets for C-CLIP with MAB.

These are the EXACT datasets from the CoDyRA paper (following the CoOp directory
structure): Aircraft, Caltech-101, DTD, EuroSAT, Flowers-102, Food-101, MNIST,
Oxford Pets, Stanford Cars, SUN397.

For each dataset this script:
  1. Downloads raw images (via torchvision or direct URL)
  2. Creates train/val CSV splits (image, caption) for C-CLIP training

All datasets are stored under  datasets/  and CSVs go to  data/<name>/.

Usage:
    python scripts/prepare_bandit_datasets.py
"""

import os
import sys
import csv
import random
import glob
import shutil
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DATASETS_ROOT = "datasets"
DATA_ROOT = "data"


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
    print(f"  Wrote {len(rows)} entries -> {path}")


# ─────────────────────────────────────────────────────────────────────────
# 1. Oxford Flowers-102  (ALREADY HAVE)
# ─────────────────────────────────────────────────────────────────────────

def check_flowers102():
    print("\n=== [1/10] Oxford Flowers-102 ===")
    if os.path.exists("data/flowers102/train.csv"):
        print("  ALREADY PREPARED")
        return True
    print("  NOT FOUND — please use your existing setup")
    return False


# ─────────────────────────────────────────────────────────────────────────
# 2. Oxford Pets  (ALREADY HAVE)
# ─────────────────────────────────────────────────────────────────────────

def check_oxford_pets():
    print("\n=== [2/10] Oxford-IIIT Pets ===")
    if os.path.exists("data/oxford_pets/train.csv"):
        print("  ALREADY PREPARED")
        return True
    print("  NOT FOUND — please use your existing setup")
    return False


# ─────────────────────────────────────────────────────────────────────────
# 3. DTD (Describable Textures Dataset)
# ─────────────────────────────────────────────────────────────────────────

def prepare_dtd():
    print("\n=== [3/10] DTD (Describable Textures) ===")
    if os.path.exists("data/dtd/train.csv"):
        print("  ALREADY PREPARED")
        return True
    try:
        from torchvision.datasets import DTD as DTDDataset
        root = os.path.join(DATASETS_ROOT, "dtd")
        ensure_dir(root)
        train_ds = DTDDataset(root=root, split='train', download=True)
        val_ds = DTDDataset(root=root, split='test', download=True)

        train_rows = []
        for img_path, label_idx in zip(train_ds._image_files, train_ds._labels):
            rel = os.path.relpath(str(img_path), start=root)
            name = train_ds.classes[label_idx].replace('_', ' ')
            train_rows.append((rel, f"a photo of a {name} texture"))

        val_rows = []
        for img_path, label_idx in zip(val_ds._image_files, val_ds._labels):
            rel = os.path.relpath(str(img_path), start=root)
            name = val_ds.classes[label_idx].replace('_', ' ')
            val_rows.append((rel, f"a photo of a {name} texture"))

        write_csv("data/dtd/train.csv", train_rows)
        write_csv("data/dtd/val.csv", val_rows)
        print(f"  DTD: {len(train_rows)} train, {len(val_rows)} val, {len(train_ds.classes)} classes")
        return True
    except Exception as e:
        print(f"  FAILED: {e}")
        return False


# ─────────────────────────────────────────────────────────────────────────
# 4. EuroSAT
# ─────────────────────────────────────────────────────────────────────────

def prepare_eurosat():
    print("\n=== [4/10] EuroSAT ===")
    if os.path.exists("data/eurosat/train.csv"):
        print("  ALREADY PREPARED")
        return True
    try:
        from torchvision.datasets import EuroSAT
        root = os.path.join(DATASETS_ROOT, "eurosat")
        ensure_dir(root)
        full_ds = EuroSAT(root=root, download=True)
        classes = full_ds.classes

        all_samples = list(full_ds.samples)
        random.seed(42)
        random.shuffle(all_samples)
        split_idx = int(0.8 * len(all_samples))

        train_rows = []
        for img_path, label_idx in all_samples[:split_idx]:
            rel = os.path.relpath(str(img_path), start=root)
            name = classes[label_idx].replace('_', ' ').lower()
            train_rows.append((rel, f"a satellite photo of {name}"))

        val_rows = []
        for img_path, label_idx in all_samples[split_idx:]:
            rel = os.path.relpath(str(img_path), start=root)
            name = classes[label_idx].replace('_', ' ').lower()
            val_rows.append((rel, f"a satellite photo of {name}"))

        write_csv("data/eurosat/train.csv", train_rows)
        write_csv("data/eurosat/val.csv", val_rows)
        print(f"  EuroSAT: {len(train_rows)} train, {len(val_rows)} val, {len(classes)} classes")
        return True
    except Exception as e:
        print(f"  FAILED: {e}")
        return False


# ─────────────────────────────────────────────────────────────────────────
# 5. Food-101
# ─────────────────────────────────────────────────────────────────────────

def prepare_food101():
    print("\n=== [5/10] Food-101 ===")
    if os.path.exists("data/food101/train.csv"):
        print("  ALREADY PREPARED")
        return True
    try:
        from torchvision.datasets import Food101
        root = os.path.join(DATASETS_ROOT, "food101")
        ensure_dir(root)
        train_ds = Food101(root=root, split='train', download=True)
        val_ds = Food101(root=root, split='test', download=True)
        classes = train_ds.classes

        train_rows = []
        for img_path, label_idx in zip(train_ds._image_files, train_ds._labels):
            rel = os.path.relpath(str(img_path), start=root)
            name = classes[label_idx].replace('_', ' ')
            train_rows.append((rel, f"a photo of {name}, a type of food"))

        val_rows = []
        for img_path, label_idx in zip(val_ds._image_files, val_ds._labels):
            rel = os.path.relpath(str(img_path), start=root)
            name = classes[label_idx].replace('_', ' ')
            val_rows.append((rel, f"a photo of {name}, a type of food"))

        write_csv("data/food101/train.csv", train_rows)
        write_csv("data/food101/val.csv", val_rows)
        print(f"  Food-101: {len(train_rows)} train, {len(val_rows)} val, {len(classes)} classes")
        return True
    except Exception as e:
        print(f"  FAILED: {e}")
        return False


# ─────────────────────────────────────────────────────────────────────────
# 6. FGVC Aircraft
# ─────────────────────────────────────────────────────────────────────────

def prepare_fgvc_aircraft():
    print("\n=== [6/10] FGVC Aircraft ===")
    if os.path.exists("data/fgvc_aircraft/train.csv"):
        print("  ALREADY PREPARED")
        return True
    try:
        from torchvision.datasets import FGVCAircraft
        root = os.path.join(DATASETS_ROOT, "fgvc_aircraft")
        ensure_dir(root)
        train_ds = FGVCAircraft(root=root, split='train', download=True)
        val_ds = FGVCAircraft(root=root, split='test', download=True)
        classes = train_ds.classes

        train_rows = []
        for img_path, label_idx in zip(train_ds._image_files, train_ds._labels):
            rel = os.path.relpath(str(img_path), start=root)
            name = classes[label_idx]
            train_rows.append((rel, f"a photo of a {name}, a type of aircraft"))

        val_rows = []
        for img_path, label_idx in zip(val_ds._image_files, val_ds._labels):
            rel = os.path.relpath(str(img_path), start=root)
            name = classes[label_idx]
            val_rows.append((rel, f"a photo of a {name}, a type of aircraft"))

        write_csv("data/fgvc_aircraft/train.csv", train_rows)
        write_csv("data/fgvc_aircraft/val.csv", val_rows)
        print(f"  FGVC Aircraft: {len(train_rows)} train, {len(val_rows)} val, {len(classes)} classes")
        return True
    except Exception as e:
        print(f"  FAILED: {e}")
        return False


# ─────────────────────────────────────────────────────────────────────────
# 7. Stanford Cars
# ─────────────────────────────────────────────────────────────────────────

def prepare_stanford_cars():
    print("\n=== [7/10] Stanford Cars ===")
    if os.path.exists("data/stanford_cars/train.csv"):
        print("  ALREADY PREPARED")
        return True
    try:
        from torchvision.datasets import StanfordCars
        root = os.path.join(DATASETS_ROOT, "stanford_cars")
        ensure_dir(root)
        train_ds = StanfordCars(root=root, split='train', download=True)
        val_ds = StanfordCars(root=root, split='test', download=True)
        classes = train_ds.classes

        train_rows = []
        for img_path, label_idx in train_ds._samples:
            rel = os.path.relpath(str(img_path), start=root)
            train_rows.append((rel, f"a photo of a {classes[label_idx]}"))

        val_rows = []
        for img_path, label_idx in val_ds._samples:
            rel = os.path.relpath(str(img_path), start=root)
            val_rows.append((rel, f"a photo of a {classes[label_idx]}"))

        write_csv("data/stanford_cars/train.csv", train_rows)
        write_csv("data/stanford_cars/val.csv", val_rows)
        print(f"  Stanford Cars: {len(train_rows)} train, {len(val_rows)} val, {len(classes)} classes")
        return True
    except Exception as e:
        print(f"  FAILED: {e}")
        print("  >> Download manually from: https://www.kaggle.com/datasets/jessicali9530/stanford-cars-dataset")
        return False


# ─────────────────────────────────────────────────────────────────────────
# 8. SUN397
# ─────────────────────────────────────────────────────────────────────────

def prepare_sun397():
    print("\n=== [8/10] SUN397 ===")
    if os.path.exists("data/sun397/train.csv"):
        print("  ALREADY PREPARED")
        return True
    try:
        from torchvision.datasets import SUN397
        root = os.path.join(DATASETS_ROOT, "sun397")
        ensure_dir(root)
        full_ds = SUN397(root=root, download=True)

        # SUN397 has 397 scene categories
        # Build class list from folder names
        classes = sorted(set(
            str(Path(p).parent.name) for p, _ in full_ds._image_files_labels
        )) if hasattr(full_ds, '_image_files_labels') else []

        # Use _image_files and _labels attributes
        all_samples = list(zip(full_ds._image_files, full_ds._labels))
        random.seed(42)
        random.shuffle(all_samples)
        split_idx = int(0.8 * len(all_samples))

        # Get class names from the dataset
        ds_classes = full_ds.classes if hasattr(full_ds, 'classes') else None

        train_rows = []
        for img_path, label_idx in all_samples[:split_idx]:
            rel = os.path.relpath(str(img_path), start=root)
            if ds_classes:
                name = ds_classes[label_idx].replace('/', ' ').replace('_', ' ').strip()
            else:
                name = Path(img_path).parent.name.replace('_', ' ')
            train_rows.append((rel, f"a photo of a {name}"))

        val_rows = []
        for img_path, label_idx in all_samples[split_idx:]:
            rel = os.path.relpath(str(img_path), start=root)
            if ds_classes:
                name = ds_classes[label_idx].replace('/', ' ').replace('_', ' ').strip()
            else:
                name = Path(img_path).parent.name.replace('_', ' ')
            val_rows.append((rel, f"a photo of a {name}"))

        write_csv("data/sun397/train.csv", train_rows)
        write_csv("data/sun397/val.csv", val_rows)
        n_classes = len(ds_classes) if ds_classes else len(set(l for _, l in all_samples))
        print(f"  SUN397: {len(train_rows)} train, {len(val_rows)} val, {n_classes} classes")
        return True
    except Exception as e:
        print(f"  FAILED: {e}")
        print("  >> Download manually from: http://vision.princeton.edu/projects/2010/SUN/SUN397.tar.gz")
        return False


# ─────────────────────────────────────────────────────────────────────────
# 9. Caltech-101
# ─────────────────────────────────────────────────────────────────────────

def prepare_caltech101():
    print("\n=== [9/10] Caltech-101 ===")
    if os.path.exists("data/caltech101/train.csv"):
        print("  ALREADY PREPARED")
        return True
    try:
        from torchvision.datasets import Caltech101
        root = os.path.join(DATASETS_ROOT, "caltech101")
        ensure_dir(root)
        full_ds = Caltech101(root=root, download=True)

        categories = full_ds.categories
        # Filter out BACKGROUND_Google class
        categories_clean = [c for c in categories if c != 'BACKGROUND_Google']

        all_samples = []
        for idx in range(len(full_ds)):
            img, label = full_ds[idx]
            cat = categories[label]
            if cat == 'BACKGROUND_Google':
                continue
            # Get image path
            img_folder = os.path.join(root, "caltech-101", "101_ObjectCategories", cat)
            if not os.path.isdir(img_folder):
                img_folder = os.path.join(root, "101_ObjectCategories", cat)
            all_samples.append((idx, label, cat))

        random.seed(42)
        random.shuffle(all_samples)
        split_idx = int(0.8 * len(all_samples))

        # We need actual file paths — enumerate the dataset directory
        cat_to_files = {}
        for base_dir in [
            os.path.join(root, "caltech-101", "101_ObjectCategories"),
            os.path.join(root, "101_ObjectCategories"),
        ]:
            if os.path.isdir(base_dir):
                for cat in os.listdir(base_dir):
                    cat_path = os.path.join(base_dir, cat)
                    if os.path.isdir(cat_path) and cat != 'BACKGROUND_Google':
                        files = sorted(glob.glob(os.path.join(cat_path, "*.jpg")))
                        if files:
                            cat_to_files[cat] = files
                break

        # Build indexed list
        all_indexed = []
        for cat, files in sorted(cat_to_files.items()):
            for f in files:
                all_indexed.append((f, cat))

        random.seed(42)
        random.shuffle(all_indexed)
        split_idx = int(0.8 * len(all_indexed))

        train_rows = []
        for img_path, cat in all_indexed[:split_idx]:
            rel = os.path.relpath(img_path, start=root)
            name = cat.replace('_', ' ')
            train_rows.append((rel, f"a photo of a {name}"))

        val_rows = []
        for img_path, cat in all_indexed[split_idx:]:
            rel = os.path.relpath(img_path, start=root)
            name = cat.replace('_', ' ')
            val_rows.append((rel, f"a photo of a {name}"))

        write_csv("data/caltech101/train.csv", train_rows)
        write_csv("data/caltech101/val.csv", val_rows)
        print(f"  Caltech-101: {len(train_rows)} train, {len(val_rows)} val, {len(cat_to_files)} classes")
        return True
    except Exception as e:
        print(f"  FAILED: {e}")
        print("  >> Download manually from: http://www.vision.caltech.edu/Image_Datasets/Caltech101/101_ObjectCategories.tar.gz")
        return False


# ─────────────────────────────────────────────────────────────────────────
# 10. MNIST
# ─────────────────────────────────────────────────────────────────────────

def prepare_mnist():
    print("\n=== [10/10] MNIST ===")
    if os.path.exists("data/mnist/train.csv"):
        print("  ALREADY PREPARED")
        return True
    try:
        from torchvision.datasets import MNIST
        from PIL import Image

        root = os.path.join(DATASETS_ROOT, "mnist")
        ensure_dir(root)
        img_dir = os.path.join(root, "images")
        ensure_dir(img_dir)

        train_ds = MNIST(root=root, train=True, download=True)
        val_ds = MNIST(root=root, train=False, download=True)

        digit_names = ['zero', 'one', 'two', 'three', 'four',
                       'five', 'six', 'seven', 'eight', 'nine']

        # Save train images as JPGs and build CSV
        train_rows = []
        print("  Saving MNIST train images...")
        for idx in range(len(train_ds)):
            img, label = train_ds[idx]
            # Convert to RGB (MNIST is grayscale)
            if not isinstance(img, Image.Image):
                img = Image.fromarray(img.numpy() if hasattr(img, 'numpy') else img)
            img_rgb = img.convert('RGB')
            fname = f"train_{idx:05d}.jpg"
            img_rgb.save(os.path.join(img_dir, fname))
            train_rows.append((fname, f"a photo of the number {digit_names[label]}"))
            if idx % 10000 == 0 and idx > 0:
                print(f"    {idx}/{len(train_ds)}")

        val_rows = []
        print("  Saving MNIST val images...")
        for idx in range(len(val_ds)):
            img, label = val_ds[idx]
            if not isinstance(img, Image.Image):
                img = Image.fromarray(img.numpy() if hasattr(img, 'numpy') else img)
            img_rgb = img.convert('RGB')
            fname = f"val_{idx:05d}.jpg"
            img_rgb.save(os.path.join(img_dir, fname))
            val_rows.append((fname, f"a photo of the number {digit_names[label]}"))

        write_csv("data/mnist/train.csv", train_rows)
        write_csv("data/mnist/val.csv", val_rows)
        print(f"  MNIST: {len(train_rows)} train, {len(val_rows)} val, 10 classes")
        return True
    except Exception as e:
        print(f"  FAILED: {e}")
        return False


# ─────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Preparing CoDyRA 10-dataset benchmark for C-CLIP + MAB")
    print("=" * 60)

    results = {}
    results['flowers102'] = check_flowers102()
    results['oxford_pets'] = check_oxford_pets()
    results['dtd'] = prepare_dtd()
    results['eurosat'] = prepare_eurosat()
    results['food101'] = prepare_food101()
    results['fgvc_aircraft'] = prepare_fgvc_aircraft()
    results['stanford_cars'] = prepare_stanford_cars()
    results['sun397'] = prepare_sun397()
    results['caltech101'] = prepare_caltech101()
    results['mnist'] = prepare_mnist()

    # Summary
    print("\n" + "=" * 60)
    print("DATASET PREPARATION SUMMARY")
    print("=" * 60)
    for name, ok in results.items():
        status = "READY" if ok else "FAILED"
        print(f"  {name:20s} : {status}")

    ready = sum(1 for v in results.values() if v)
    failed = [n for n, ok in results.items() if not ok]

    if not failed:
        print(f"\nAll 10 datasets ready! Run training with:")
        print(f"  python src/train_bandit.py --config configs/bandit_config.yaml")
    else:
        print(f"\n{len(failed)} dataset(s) need attention: {', '.join(failed)}")
        print(f"{ready}/10 ready. You can still train with available datasets by")
        print(f"removing the missing ones from configs/bandit_config.yaml")


if __name__ == '__main__':
    main()
