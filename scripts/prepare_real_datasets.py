"""
Prepare the 5 available datasets for C-CLIP continual training with MAB.

Datasets (CoDyRA benchmark subset):
  0. FGVC Aircraft    – datasets/fgvc_aircraft/
  1. DTD              – datasets/dtd/
  2. EuroSAT          – datasets/eurosat/
  3. Oxford Flowers   – datasets/102flowers/
  4. Oxford-IIIT Pets – datasets/Oxford_IIITPets/

Each dataset produces:
  - data/<name>/train.csv   (columns: image, caption)
  - data/<name>/val.csv     (columns: image, caption)
  - data/<name>/class_names.txt  (one class per line, for zero-shot eval)

Run from repo root:
    python scripts/prepare_real_datasets.py
"""

import os
import re
import csv
import random
from pathlib import Path

RANDOM_SEED = 42
VAL_RATIO = 0.15          # 15% val split (used when dataset has no official split)
BASE_DATA = Path("data")  # output root

random.seed(RANDOM_SEED)


# ─── helpers ────────────────────────────────────────────────────────────────

def write_csv(path: Path, rows: list):
    """Write list of (image_relative_path, caption) to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['image', 'caption'])
        for img, cap in rows:
            writer.writerow([img, cap])
    print(f"  Wrote {len(rows)} entries → {path}")


def write_class_names(path: Path, names: list):
    """Write one class name per line."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        for name in names:
            f.write(name + '\n')
    print(f"  Wrote {len(names)} class names → {path}")


def split_rows(rows: list, val_ratio: float = VAL_RATIO):
    """Shuffle and split rows into (train, val)."""
    rows = list(rows)
    random.shuffle(rows)
    n_val = max(1, int(len(rows) * val_ratio))
    return rows[n_val:], rows[:n_val]


# ─── 1. FGVC Aircraft ──────────────────────────────────────────────────────
#
# Directory:  datasets/fgvc_aircraft/
# Labels:    images_variant_trainval.txt  /  images_variant_test.txt
# Format:    <image_id> <variant name>    (e.g. "0034309 Boeing 737-300")
# Images:    datasets/fgvc_aircraft/images/<image_id>.jpg

def prepare_fgvc_aircraft():
    print("\n=== FGVC Aircraft ===")
    base = Path("datasets/fgvc_aircraft")
    img_dir = base / "images"

    if not img_dir.exists():
        print("  SKIP: images/ not found")
        return False

    def parse_variant_file(filepath):
        """Parse lines like '0034309 Boeing 737-300' → [(id, variant), ...]"""
        records = []
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                # First token is image ID, rest is variant name
                parts = line.split(' ', 1)
                if len(parts) == 2:
                    img_id, variant = parts
                    records.append((img_id, variant))
        return records

    train_rows = []
    val_rows = []
    all_variants = set()

    # trainval split
    trainval_file = base / "images_variant_trainval.txt"
    if trainval_file.exists():
        for img_id, variant in parse_variant_file(trainval_file):
            img_file = f"{img_id}.jpg"
            if (img_dir / img_file).exists():
                caption = f"a photo of a {variant}, a type of aircraft"
                train_rows.append((img_file, caption))
                all_variants.add(variant)

    # test split → our validation
    test_file = base / "images_variant_test.txt"
    if test_file.exists():
        for img_id, variant in parse_variant_file(test_file):
            img_file = f"{img_id}.jpg"
            if (img_dir / img_file).exists():
                caption = f"a photo of a {variant}, a type of aircraft"
                val_rows.append((img_file, caption))
                all_variants.add(variant)

    if not train_rows:
        print("  SKIP: No variant label files found")
        return False

    # If no test file, split trainval
    if not val_rows:
        train_rows, val_rows = split_rows(train_rows)

    out_dir = BASE_DATA / "fgvc_aircraft"
    write_csv(out_dir / "train.csv", train_rows)
    write_csv(out_dir / "val.csv", val_rows)

    class_names = sorted(all_variants)
    write_class_names(out_dir / "class_names.txt", class_names)

    print(f"  FGVC Aircraft: {len(train_rows)} train, {len(val_rows)} val, "
          f"{len(class_names)} variants")
    return True


# ─── 2. DTD (Describable Textures) ─────────────────────────────────────────
#
# Directory:  datasets/dtd/
# Labels:    labels/train1.txt, labels/val1.txt, labels/test1.txt  (split 1)
# Format:    <class>/<filename>.jpg  (e.g. "banded/banded_0001.jpg")
# Images:    datasets/dtd/images/<class>/<filename>.jpg

def prepare_dtd():
    print("\n=== DTD (Describable Textures) ===")
    base = Path("datasets/dtd")
    img_dir = base / "images"
    labels_dir = base / "labels"

    if not img_dir.exists():
        print("  SKIP: images/ not found")
        return False

    def parse_split_file(filepath):
        """Parse lines like 'banded/banded_0001.jpg' → [(rel_path, class), ...]"""
        records = []
        if not filepath.exists():
            return records
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                class_name = line.split('/')[0]
                records.append((line, class_name))
        return records

    train_records = []
    val_records = []
    all_classes = set()

    # Try official split files (split 1)
    if labels_dir.exists():
        for split_file in ['train1.txt', 'val1.txt']:
            for rel, cls in parse_split_file(labels_dir / split_file):
                if (img_dir / rel).exists():
                    train_records.append((rel, cls))
                    all_classes.add(cls)

        for rel, cls in parse_split_file(labels_dir / 'test1.txt'):
            if (img_dir / rel).exists():
                val_records.append((rel, cls))
                all_classes.add(cls)
    else:
        # Fallback: scan image directory structure
        for class_dir in sorted(img_dir.iterdir()):
            if class_dir.is_dir():
                cls = class_dir.name
                all_classes.add(cls)
                for img in sorted(class_dir.glob("*.jpg")):
                    rel = f"{cls}/{img.name}"
                    train_records.append((rel, cls))

    if not train_records:
        print("  SKIP: No images found")
        return False

    # If no separate test split, split the combined set
    if not val_records:
        train_records, val_records = split_rows(train_records)

    # Build CSV rows with captions
    train_rows = [(rel, f"a photo of a {cls.replace('_', ' ')} texture")
                  for rel, cls in train_records]
    val_rows = [(rel, f"a photo of a {cls.replace('_', ' ')} texture")
                for rel, cls in val_records]

    out_dir = BASE_DATA / "dtd"
    write_csv(out_dir / "train.csv", train_rows)
    write_csv(out_dir / "val.csv", val_rows)

    class_names = sorted(all_classes)
    write_class_names(out_dir / "class_names.txt",
                      [c.replace('_', ' ') for c in class_names])

    print(f"  DTD: {len(train_rows)} train, {len(val_rows)} val, "
          f"{len(class_names)} textures")
    return True


# ─── 3. EuroSAT ────────────────────────────────────────────────────────────
#
# Directory:  datasets/eurosat/
# Structure:  Has class subfolders (AnnualCrop, Forest, Highway, ...) with .jpg
#             AND train.csv / validation.csv / test.csv
#             AND label_map.json

def prepare_eurosat():
    print("\n=== EuroSAT ===")
    base = Path("datasets/eurosat")

    if not base.exists():
        print("  SKIP: datasets/eurosat/ not found")
        return False

    # Class name mapping for readable captions
    EUROSAT_NAMES = {
        'AnnualCrop': 'annual crop',
        'Forest': 'forest',
        'HerbaceousVegetation': 'herbaceous vegetation',
        'Highway': 'highway',
        'Industrial': 'industrial area',
        'Pasture': 'pasture',
        'PermanentCrop': 'permanent crop',
        'Residential': 'residential area',
        'River': 'river',
        'SeaLake': 'sea or lake',
    }

    all_classes = set()

    def process_csv(csv_path):
        """Parse EuroSAT CSV (likely: filename, label or similar)."""
        rows = []
        if not csv_path.exists():
            return rows

        import csv as csv_mod
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv_mod.reader(f)
            header = next(reader, None)
            if header is None:
                return rows
            for row in reader:
                if len(row) < 2:
                    continue
                filename = row[0]
                label = row[1] if len(row) > 1 else ''
                # Determine class from label or from filename
                # EuroSAT filenames are like: ClassName_00001.jpg
                class_name = label.strip()
                if not class_name:
                    # Try to infer from filename
                    for cls in EUROSAT_NAMES:
                        if filename.startswith(cls):
                            class_name = cls
                            break

                if class_name:
                    readable = EUROSAT_NAMES.get(class_name, class_name.replace('_', ' ').lower())
                    all_classes.add(readable)

                    # Image path: ClassName/filename.jpg
                    img_rel = f"{class_name}/{filename}"
                    if (base / img_rel).exists():
                        rows.append((img_rel, f"a satellite photo of {readable}"))
                    elif (base / filename).exists():
                        rows.append((filename, f"a satellite photo of {readable}"))
        return rows

    # Try using existing CSVs first
    train_csv = base / "train.csv"
    val_csv = base / "validation.csv"
    test_csv = base / "test.csv"

    train_rows = []
    val_rows = []

    if train_csv.exists():
        train_rows = process_csv(train_csv)
    if val_csv.exists() or test_csv.exists():
        val_rows = process_csv(val_csv if val_csv.exists() else test_csv)

    # Fallback: scan class subfolders if CSVs didn't yield results
    if not train_rows:
        print("  CSVs empty or not parseable, scanning class folders...")
        all_records = []
        for class_dir in sorted(base.iterdir()):
            if class_dir.is_dir() and class_dir.name in EUROSAT_NAMES:
                cls = class_dir.name
                readable = EUROSAT_NAMES[cls]
                all_classes.add(readable)
                for img in sorted(class_dir.glob("*.jpg")):
                    rel = f"{cls}/{img.name}"
                    all_records.append((rel, f"a satellite photo of {readable}"))
                # Also check for .tif files
                for img in sorted(class_dir.glob("*.tif")):
                    rel = f"{cls}/{img.name}"
                    all_records.append((rel, f"a satellite photo of {readable}"))

        if not all_records:
            print("  SKIP: No images found")
            return False

        train_rows, val_rows = split_rows(all_records)

    if not train_rows:
        print("  SKIP: No data generated")
        return False

    out_dir = BASE_DATA / "eurosat"
    write_csv(out_dir / "train.csv", train_rows)
    write_csv(out_dir / "val.csv", val_rows)

    class_names = sorted(all_classes)
    write_class_names(out_dir / "class_names.txt", class_names)

    print(f"  EuroSAT: {len(train_rows)} train, {len(val_rows)} val, "
          f"{len(class_names)} classes")
    return True


# ─── 4. Oxford Flowers-102 ─────────────────────────────────────────────────
#
# Directory:  datasets/102flowers/
# Labels:     imagelabels.mat  (1-indexed class labels for each image)
# Images:     datasets/102flowers/jpg/image_00001.jpg ...

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
]   # 102 entries (Oxford categorisation order 1-102)


def prepare_flowers():
    print("\n=== Oxford Flowers-102 ===")
    img_dir = Path("datasets/102flowers/jpg")
    label_file = Path("datasets/102flowers/imagelabels.mat")
    images = sorted(img_dir.glob("image_*.jpg"))

    if not images:
        print("  SKIP: No images found in datasets/102flowers/jpg/")
        return False

    # Load official per-image labels from imagelabels.mat (1-indexed)
    if label_file.exists():
        import scipy.io
        mat = scipy.io.loadmat(str(label_file))
        labels_arr = mat["labels"].flatten()
        id_to_label = {i + 1: int(labels_arr[i]) for i in range(len(labels_arr))}
        print(f"  Loaded imagelabels.mat ({len(id_to_label)} entries)")
    else:
        id_to_label = None
        print("  WARNING: imagelabels.mat not found — using round-robin class assignment")

    # Also try to use setid.mat for official train/test splits
    setid_file = Path("datasets/102flowers/setid.mat")
    has_official_split = False
    train_ids, val_ids = set(), set()

    if setid_file.exists():
        import scipy.io
        setid = scipy.io.loadmat(str(setid_file))
        # trnid = training, valid = validation, tstid = test
        trnid = set(setid['trnid'].flatten().tolist())
        valid = set(setid['valid'].flatten().tolist())
        tstid = set(setid['tstid'].flatten().tolist())
        train_ids = trnid | valid   # combine train + val for training
        val_ids = tstid             # use test as our validation
        has_official_split = True
        print(f"  Using official splits: train={len(train_ids)}, val={len(val_ids)}")

    train_rows = []
    val_rows = []
    used_labels = set()

    for img in images:
        img_id = int(re.search(r"(\d+)", img.stem).group(1))  # 1-based
        if id_to_label:
            class_idx = id_to_label[img_id] - 1  # → 0-based
        else:
            class_idx = (img_id - 1) % len(FLOWER_LABELS)
        label = FLOWER_LABELS[class_idx]
        used_labels.add(label)
        caption = f"a photo of a {label}"
        row = (img.name, caption)

        if has_official_split:
            if img_id in train_ids:
                train_rows.append(row)
            elif img_id in val_ids:
                val_rows.append(row)
        else:
            train_rows.append(row)

    # If no official split, do random split
    if not has_official_split:
        train_rows, val_rows = split_rows(train_rows)

    out_dir = BASE_DATA / "flowers102"
    write_csv(out_dir / "train.csv", train_rows)
    write_csv(out_dir / "val.csv", val_rows)

    class_names = sorted(used_labels)
    write_class_names(out_dir / "class_names.txt", class_names)

    print(f"  Flowers-102: {len(train_rows)} train, {len(val_rows)} val, "
          f"{len(class_names)} classes")
    return True


# ─── 5. Oxford-IIIT Pets ───────────────────────────────────────────────────
#
# Directory:  datasets/Oxford_IIITPets/
# Images:     datasets/Oxford_IIITPets/images/Abyssinian_34.jpg ...
# Breed = everything before the trailing _<number>

def _breed_name(stem: str) -> str:
    name = re.sub(r"_\d+$", "", stem)  # remove trailing _number
    return name.replace("_", " ").title()


def prepare_pets():
    print("\n=== Oxford-IIIT Pets ===")
    img_dir = Path("datasets/Oxford_IIITPets/images")
    images = sorted(img_dir.glob("*.jpg"))

    if not images:
        print("  SKIP: No images found in datasets/Oxford_IIITPets/images/")
        return False

    all_breeds = set()
    all_rows = []

    for img in images:
        # Skip non-image files (e.g. some datasets have .mat files)
        breed = _breed_name(img.stem)
        all_breeds.add(breed)
        caption = f"a photo of a {breed}, a type of pet"
        all_rows.append((img.name, caption))

    train_rows, val_rows = split_rows(all_rows)

    out_dir = BASE_DATA / "oxford_pets"
    write_csv(out_dir / "train.csv", train_rows)
    write_csv(out_dir / "val.csv", val_rows)

    class_names = sorted(all_breeds)
    write_class_names(out_dir / "class_names.txt", class_names)

    print(f"  Oxford Pets: {len(train_rows)} train, {len(val_rows)} val, "
          f"{len(class_names)} breeds")
    return True


# ─── main ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("Preparing 5 datasets for C-CLIP MAB continual training")
    print("=" * 60)

    results = {}
    results['fgvc_aircraft'] = prepare_fgvc_aircraft()
    results['dtd'] = prepare_dtd()
    results['eurosat'] = prepare_eurosat()
    results['flowers102'] = prepare_flowers()
    results['oxford_pets'] = prepare_pets()

    # Summary
    print("\n" + "=" * 60)
    print("DATASET PREPARATION SUMMARY")
    print("=" * 60)
    for name, ok in results.items():
        status = "✓ READY" if ok else "✗ FAILED"
        print(f"  {name:20s} : {status}")

    ready = sum(1 for v in results.values() if v)
    failed = [n for n, ok in results.items() if not ok]

    if not failed:
        print(f"\nAll 5 datasets ready! Run training with:")
        print(f"  python src/train_bandit.py --config bandit_config.yaml")
    else:
        print(f"\n{len(failed)} dataset(s) need attention: {', '.join(failed)}")
        print(f"{ready}/5 ready.")
