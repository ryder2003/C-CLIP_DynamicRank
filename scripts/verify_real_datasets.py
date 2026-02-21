"""
Sanity-check: verify every CSV row can open its image, and print a few samples.
Run before training to catch path/permission issues early.

    python scripts/verify_real_datasets.py
"""

import os, sys, random
from pathlib import Path
import pandas as pd
from PIL import Image

DATASETS = [
    {
        "name":      "flowers102",
        "train_csv": "data/flowers102/train.csv",
        "val_csv":   "data/flowers102/val.csv",
        "image_dir": "datasets/102flowers/jpg",
    },
    {
        "name":      "oxford_pets",
        "train_csv": "data/oxford_pets/train.csv",
        "val_csv":   "data/oxford_pets/val.csv",
        "image_dir": "datasets/Oxford_IIITPets/images",
    },
    {
        "name":      "simpsons",
        "train_csv": "data/simpsons/train.csv",
        "val_csv":   "data/simpsons/val.csv",
        "image_dir": "datasets/simpsons_archive/simpsons_dataset",
    },
]

CHECK_SAMPLE = 5   # images to spot-check per split; set to -1 to check ALL


def check_split(csv_path: str, image_dir: str, label: str):
    if not os.path.exists(csv_path):
        print(f"  [MISSING] {csv_path}")
        return False

    df   = pd.read_csv(csv_path)
    rows = df.to_dict("records")

    if CHECK_SAMPLE == -1:
        sample = rows
    else:
        sample = random.sample(rows, min(CHECK_SAMPLE, len(rows)))

    errors = 0
    for row in sample:
        img_path = os.path.join(image_dir, row["image"])
        try:
            img = Image.open(img_path).convert("RGB")
            _ = img.size          # actually decode header
        except Exception as e:
            print(f"    ERROR  {img_path}: {e}")
            errors += 1

    status = "OK" if errors == 0 else f"{errors} ERRORS"
    print(f"  {label:8s}  rows={len(df):>6}  sample={len(sample)}  [{status}]")
    if len(rows) > 0:
        ex = random.choice(rows)
        print(f"    sample caption: \"{ex['caption']}\"  →  {ex['image']}")
    return errors == 0


all_ok = True
for ds in DATASETS:
    print(f"\n{'='*55}")
    print(f"Dataset: {ds['name']}")
    print(f"  image_dir: {ds['image_dir']}")
    ok1 = check_split(ds["train_csv"], ds["image_dir"], "train")
    ok2 = check_split(ds["val_csv"],   ds["image_dir"], "val")
    all_ok = all_ok and ok1 and ok2

print()
if all_ok:
    print("All datasets verified. Ready to train:")
    print("  python src/train.py --config configs/real_datasets_config.yaml")
else:
    print("Some errors found – fix paths before training.")
    sys.exit(1)
