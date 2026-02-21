"""
Prepare the three downloaded datasets for C-CLIP continual training.

Datasets handled:
  1. Oxford 102 Flowers   – datasets/102flowers/jpg/
  2. Oxford-IIIT Pets     – datasets/Oxford_IIITPets/images/
  3. Simpsons Archive     – datasets/simpsons_archive/simpsons_dataset/<character>/

Each dataset is converted to a pair of CSV files (train.csv / val.csv) with
columns:  image (relative path from image_dir)  |  caption

Run from the repo root:
    python scripts/prepare_real_datasets.py
"""

import os
import re
import pandas as pd
from pathlib import Path
import random

RANDOM_SEED = 42
VAL_RATIO   = 0.15          # 15 % validation split
BASE_DATA   = Path("data")  # output root

random.seed(RANDOM_SEED)

# ─── helpers ────────────────────────────────────────────────────────────────

def _split_and_save(records: list[dict], task_name: str, image_dir: Path):
    """Shuffle, split and write train/val CSVs."""
    random.shuffle(records)
    n_val   = max(1, int(len(records) * VAL_RATIO))
    val     = records[:n_val]
    train   = records[n_val:]

    out_dir = BASE_DATA / task_name
    out_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(train).to_csv(out_dir / "train.csv", index=False)
    pd.DataFrame(val).to_csv(  out_dir / "val.csv",   index=False)

    print(f"[{task_name}]  total={len(records)}  train={len(train)}  val={len(val)}")
    print(f"  image_dir : {image_dir}")
    print(f"  CSVs      : {out_dir}/train.csv  /val.csv")
    print()


# ─── 1. Oxford 102 Flowers ──────────────────────────────────────────────────
#
# No label file ships with the plain jpg download, so we use a generic
# single template caption for every image.
# If you have 'imagelabels.mat' (from the Oxford page), the commented-out
# section below loads proper per-image class names.

FLOWER_LABELS = [
    "pink primrose", "hard-leaved pocket orchid", "canterbury bells",
    "sweet pea", "english marigold", "tiger lily", "moon orchid",
    "bird of paradise", "monkshood", "globe thistle", "snapdragon",
    "colt's foot", "king protea", "spear thistle", "yellow iris",
    "globe-flower", "purple coneflower", "peruvian lily", "balloon flower",
    "giant white arum lily", "fire lily", "pincushion flower",
    "fritillary", "red ginger", "grape hyacinth", "corn poppy",
    "prince of wales feathers", "stemless gentian", "artichoke",
    "sweet william", "carnation", "garden phlox", "love in the mist",
    "mexican aster", "alpine sea holly", "ruby-lipped cattleya",
    "cape flower", "great masterwort", "siam tulip", "lenten rose",
    "barbeton daisy", "daffodil", "sword lily", "poinsettia",
    "bolero deep blue", "wallflower", "marigold", "buttercup",
    "oxeye daisy", "common dandelion", "petunia", "wild pansy",
    "primula", "sunflower", "pelargonium", "bishop of llandaff",
    "gaura", "geranium", "orange dahlia", "pink-yellow dahlia",
    "cautleya spicata", "japanese anemone", "black-eyed susan",
    "silverbush", "californian poppy", "osteospermum", "spring crocus",
    "bearded iris", "windflower", "tree poppy", "gazania", "azalea",
    "water lily", "rose", "thorn apple", "morning glory", "passion flower",
    "lotus", "toad lily", "anthurium", "frangipani", "clematis",
    "hibiscus", "columbine", "desert-rose", "tree mallow",
    "magnolia", "cyclamen", "watercress", "canna lily", "hippeastrum",
    "bee balm", "ball moss", "foxglove", "bougainvillea", "camellia",
    "mallow", "mexican petunia", "bromelia", "blanket flower",
    "trumpet creeper", "blackberry lily",
]   # 102 entries  (Oxford categorisation order 1-102)

def prepare_flowers():
    img_dir    = Path("datasets/102flowers/jpg")
    label_file = Path("datasets/102flowers/imagelabels.mat")
    images     = sorted(img_dir.glob("image_*.jpg"))

    if not images:
        print("[flowers] No images found – skipping.")
        return

    # Load official per-image labels from imagelabels.mat (1-indexed)
    if label_file.exists():
        import scipy.io
        mat      = scipy.io.loadmat(str(label_file))
        id_to_label = {i + 1: int(mat["labels"].flatten()[i]) for i in range(len(mat["labels"].flatten()))}
        print(f"  Loaded official imagelabels.mat  ({len(id_to_label)} entries)")
    else:
        # Fallback: distribute classes round-robin (no accurate label info)
        id_to_label = None
        print("  WARNING: imagelabels.mat not found – using round-robin class assignment")

    records = []
    for img in images:
        img_id  = int(re.search(r"(\d+)", img.stem).group(1))   # 1-based
        if id_to_label:
            class_idx = id_to_label[img_id] - 1                 # → 0-based
        else:
            class_idx = (img_id - 1) % len(FLOWER_LABELS)
        label   = FLOWER_LABELS[class_idx]
        caption = f"a photo of a {label}"
        records.append({"image": img.name, "caption": caption})

    _split_and_save(records, "flowers102", img_dir.resolve())


# ─── 2. Oxford-IIIT Pets ────────────────────────────────────────────────────
#
# Filenames like  Abyssinian_34.jpg, saint_bernard_12.jpg
# Breed = everything before the trailing _<number>
# First letter capitalised → "Abyssinian", "Saint Bernard"

def _breed_name(stem: str) -> str:
    name = re.sub(r"_\d+$", "", stem)          # remove trailing _number
    return name.replace("_", " ").title()


def prepare_pets():
    img_dir = Path("datasets/Oxford_IIITPets/images")
    images  = sorted(img_dir.glob("*.jpg"))

    if not images:
        print("[pets] No images found – skipping.")
        return

    records = []
    for img in images:
        breed   = _breed_name(img.stem)
        caption = f"a photo of a {breed}"
        records.append({"image": img.name, "caption": caption})

    _split_and_save(records, "oxford_pets", img_dir.resolve())


# ─── 3. Simpsons Archive ────────────────────────────────────────────────────
#
# Directory: simpsons_dataset/<character_name>/<pic_xxxx.jpg>
# Character name e.g.  bart_simpson  →  "Bart Simpson"

def _char_name(folder: str) -> str:
    return folder.replace("_", " ").title()


def prepare_simpsons():
    base      = Path("datasets/simpsons_archive/simpsons_dataset")
    char_dirs = sorted([d for d in base.iterdir() if d.is_dir()])

    if not char_dirs:
        print("[simpsons] No character folders found – skipping.")
        return

    records = []
    for char_dir in char_dirs:
        char    = _char_name(char_dir.name)
        caption = f"a photo of {char} from The Simpsons"
        for img in sorted(char_dir.glob("*.jpg")):
            # store path relative to simpsons_dataset/ so image_dir = simpsons_dataset
            rel = f"{char_dir.name}/{img.name}"
            records.append({"image": rel, "caption": caption})

    _split_and_save(records, "simpsons", base.resolve())


# ─── main ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("Preparing datasets for C-CLIP continual training")
    print("=" * 60)
    print()
    prepare_flowers()
    prepare_pets()
    prepare_simpsons()
    print("Done!  CSVs written to data/<task>/")
