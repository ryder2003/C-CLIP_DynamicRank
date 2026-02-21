"""
Zero-shot classification evaluation for C-CLIP on classification datasets.

The paper uses zero-shot accuracy (not retrieval) for datasets like Oxford 102
Flowers, Oxford-IIIT Pets, etc.  This script:
  1. Encodes every class name with a small set of CLIP prompt templates
  2. Encodes each validation image
  3. Reports top-1 accuracy per dataset and backward transfer after training

Usage:
    # Evaluate after full continual training
    python scripts/eval_zero_shot.py --checkpoint checkpoints/real_datasets/model_final.pt \
                                     --config    configs/real_datasets_config.yaml

    # Evaluate a per-task checkpoint (e.g. right after task 1)
    python scripts/eval_zero_shot.py --checkpoint checkpoints/real_datasets/model_after_task_0.pt \
                                     --config    configs/real_datasets_config.yaml \
                                     --tasks     flowers102
"""

import os, sys, argparse, json
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.cclip import CCLIP
from src.data.datasets import ClassificationDataset
from src.data.transforms import get_clip_transforms
from src.utils.config import load_config

# ── Prompt templates (same as OpenAI's "ensemble of prompts" approach) ──────
TEMPLATES = [
    "a photo of a {}.",
    "a good photo of a {}.",
    "a photo of the {}.",
    "a close-up photo of a {}.",
    "a bright photo of a {}.",
    "a cropped photo of a {}.",
    "a rendition of a {}.",
    "itap of a {}.",
]

# ── Dataset definitions ──────────────────────────────────────────────────────
EVAL_SETS = {
    "flowers102": {
        "val_csv":   "data/flowers102/val.csv",
        "image_dir": "datasets/102flowers/jpg",
        "caption_prefix": "a photo of a ",   # strip this to get clean class name
    },
    "oxford_pets": {
        "val_csv":   "data/oxford_pets/val.csv",
        "image_dir": "datasets/Oxford_IIITPets/images",
        "caption_prefix": "a photo of a ",
    },
    "simpsons": {
        "val_csv":   "data/simpsons/val.csv",
        "image_dir": "datasets/simpsons_archive/simpsons_dataset",
        "caption_prefix": "a photo of ",
    },
}


@torch.no_grad()
def zeroshot_accuracy(model, dataloader, class_names, tokenizer, device):
    """Compute zero-shot top-1 accuracy using ensemble prompts."""
    # Build class-text feature matrix
    all_cls_feats = []
    for cname in tqdm(class_names, desc="  Encoding class prompts", leave=False):
        texts  = [t.format(cname) for t in TEMPLATES]
        tokens = tokenizer(texts).to(device)
        feats  = model.encode_text(tokens, normalize=True)     # (T, D)
        feats  = feats.mean(dim=0)
        feats  = feats / feats.norm()
        all_cls_feats.append(feats)
    cls_matrix = torch.stack(all_cls_feats, dim=0)             # (C, D)

    correct = total = 0
    for images, labels in tqdm(dataloader, desc="  Classifying", leave=False):
        images = images.to(device)
        labels = labels.to(device)
        img_feats = model.encode_image(images, normalize=True) # (B, D)
        sims      = img_feats @ cls_matrix.T                   # (B, C)
        preds     = sims.argmax(dim=1)
        correct  += (preds == labels).sum().item()
        total    += len(labels)

    return correct / total * 100


def load_model(checkpoint_path, config, device):
    model = CCLIP(
        clip_model_name  = config["model"]["clip_model_name"],
        pretrained       = config["model"]["pretrained"],
        lora_r           = config["model"]["lora_r"],
        lora_alpha       = config["model"]["lora_alpha"],
        lora_dropout     = config["model"]["lora_dropout"],
        lora_target_modules = config["model"].get("lora_target_modules"),
        integration_coeff   = config["model"]["integration_coeff"],
        device           = device,
    ).to(device)
    print(f"Loading checkpoint: {checkpoint_path}")
    model.load_checkpoint(checkpoint_path)
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint")
    parser.add_argument("--config",     required=True, help="Path to YAML config")
    parser.add_argument("--tasks",      nargs="+", default=list(EVAL_SETS.keys()),
                        help="Which tasks to evaluate (default: all three)")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers",type=int, default=0)
    parser.add_argument("--output",     default="results/zero_shot_eval.json",
                        help="Where to save JSON results")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    config = load_config(args.config)
    model  = load_model(args.checkpoint, config, device)

    val_transform = get_clip_transforms(
        image_size=config["data"]["image_size"], is_train=False
    )
    tokenizer = model.clip.tokenizer

    results = {}
    for task_name in args.tasks:
        if task_name not in EVAL_SETS:
            print(f"Unknown task '{task_name}' – skipping")
            continue

        cfg = EVAL_SETS[task_name]
        print(f"\n{'='*55}")
        print(f"Task: {task_name}")

        ds = ClassificationDataset(
            data_path = cfg["val_csv"],
            image_dir = cfg["image_dir"],
            transform = val_transform,
        )

        # Derive clean class names by stripping the template prefix
        prefix = cfg["caption_prefix"]
        class_names = [
            c[len(prefix):].rstrip(".")
            if c.startswith(prefix) else c
            for c in ds.class_names
        ]

        dl = DataLoader(
            ds,
            batch_size  = args.batch_size,
            shuffle     = False,
            num_workers = args.num_workers,
            pin_memory  = device == "cuda",
        )

        acc = zeroshot_accuracy(model, dl, class_names, tokenizer, device)
        print(f"  Zero-shot Top-1 Accuracy: {acc:.2f}%  ({ds.__len__()} images, {len(class_names)} classes)")
        results[task_name] = {"accuracy": round(acc, 4), "n_images": len(ds), "n_classes": len(class_names)}

    # Save results
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
