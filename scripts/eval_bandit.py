"""
Post-training evaluation for C-CLIP with MAB dynamic rank selection.

Loads a trained CCLIPWithBandit checkpoint and evaluates zero-shot
classification accuracy on all datasets in the config. Computes:
  - Per-task accuracy
  - Average accuracy
  - Forgetting metrics (backward transfer)
  - Bandit analysis (which ranks worked best)

Usage:
    python scripts/eval_bandit.py \\
        --checkpoint checkpoints/bandit_run/model_final_bandit.pt \\
        --config bandit_config.yaml \\
        --output results/bandit_eval.json
"""

import os
import sys
import argparse
import json
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.cclip_bandit import CCLIPWithBandit, build_cclip_with_bandit
from src.data.datasets import ClassificationDataset
from src.data.transforms import get_clip_transforms
from src.utils.config import load_config
from src.utils.evaluation import evaluate_zero_shot_classification


def load_class_names(dataset_config):
    """Load class names from class_names.txt or auto-detect from CSV."""
    # Try data/<name>/class_names.txt
    name = dataset_config['name']
    auto_path = os.path.join('data', name, 'class_names.txt')
    if os.path.exists(auto_path):
        with open(auto_path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip()]

    # Fallback: extract from val CSV
    val_path = dataset_config.get('val_path', '')
    if os.path.exists(val_path):
        import pandas as pd
        df = pd.read_csv(val_path)
        return sorted(df['caption'].unique().tolist())

    return []


def evaluate_dataset(model, dataset_config, device='cuda'):
    """Evaluate zero-shot classification on a single dataset."""
    val_path = dataset_config.get('val_path', '')
    image_dir = dataset_config.get('image_dir', '')

    if not os.path.exists(val_path):
        print(f"  [SKIP] val_path not found: {val_path}")
        return None

    # Load class names
    class_names = load_class_names(dataset_config)
    if not class_names:
        print(f"  [SKIP] No class names for {dataset_config['name']}")
        return None

    # Build classification dataset
    val_transform = get_clip_transforms(image_size=224, is_train=False)
    cls_dataset = ClassificationDataset(
        data_path=val_path,
        image_dir=image_dir,
        transform=val_transform,
    )

    dataloader = DataLoader(
        cls_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # Clean class names (remove caption prefixes/suffixes for zero-shot prompting)
    clean_names = []
    for cn in cls_dataset.class_names:
        for prefix in ['a photo of a ', 'a photo of ', 'a satellite photo of ']:
            if cn.lower().startswith(prefix):
                cn = cn[len(prefix):]
                break
        for suffix in [', a type of aircraft', ', a type of food', ', a type of pet',
                       ' texture']:
            if cn.lower().endswith(suffix):
                cn = cn[:-len(suffix)]
                break
        clean_names.append(cn.strip())

    metrics = evaluate_zero_shot_classification(
        model=model,
        dataloader=dataloader,
        class_names=clean_names,
        device=device,
    )

    return {
        'accuracy': metrics['accuracy'],
        'correct': metrics['correct'],
        'total': metrics['total'],
        'n_classes': len(clean_names),
    }


def main():
    parser = argparse.ArgumentParser(description='Evaluate C-CLIP Bandit model')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint (model_final_bandit.pt)')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to config YAML')
    parser.add_argument('--output', type=str, default='results/bandit_eval.json',
                        help='Path to save evaluation results JSON')
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # Build and load model
    print(f"Loading checkpoint: {args.checkpoint}")
    model = build_cclip_with_bandit(config, device=device).to(device)
    model.load_checkpoint(args.checkpoint)
    model.eval()

    # Load bandit history if available
    bandit_history_path = os.path.join(
        os.path.dirname(args.checkpoint), 'bandit_history.json')
    bandit_history = None
    if os.path.exists(bandit_history_path):
        with open(bandit_history_path, 'r') as f:
            bandit_history = json.load(f)
        print(f"Loaded bandit history from {bandit_history_path}")

    # Evaluate on all datasets
    print(f"\n{'='*60}")
    print("ZERO-SHOT CLASSIFICATION EVALUATION")
    print(f"{'='*60}\n")

    all_results = {}
    for ds_cfg in config['datasets']:
        ds_name = ds_cfg['name']
        print(f"Evaluating: {ds_name}")
        result = evaluate_dataset(model, ds_cfg, device=device)
        if result:
            all_results[ds_name] = result
            print(f"  Accuracy: {result['accuracy']:.2f}% "
                  f"({result['correct']}/{result['total']}, "
                  f"{result['n_classes']} classes)\n")
        else:
            print(f"  Could not evaluate\n")

    # Summary statistics
    accuracies = [r['accuracy'] for r in all_results.values()]
    avg_accuracy = sum(accuracies) / len(accuracies) if accuracies else 0

    print(f"\n{'='*60}")
    print("RESULTS SUMMARY")
    print(f"{'='*60}")
    for name, result in all_results.items():
        print(f"  {name:20s}: {result['accuracy']:6.2f}%")
    print(f"  {'─'*35}")
    print(f"  {'Average':20s}: {avg_accuracy:6.2f}%")

    # Bandit analysis
    if bandit_history:
        print(f"\n{'='*60}")
        print("BANDIT ANALYSIS")
        print(f"{'='*60}")
        print(f"  Algorithm: {bandit_history.get('algorithm', 'unknown')}")
        print(f"  Best rank: {bandit_history.get('best_rank', 'unknown')}")
        print(f"\n  Arm statistics:")
        for rank_str, arm in bandit_history.get('arms', {}).items():
            print(f"    rank={rank_str:>2s}: pulls={arm['n_pulls']}, "
                  f"mean_reward={arm['mean_reward']:.3f}")

        print(f"\n  Task history:")
        for entry in bandit_history.get('task_history', []):
            print(f"    Task {entry['task_idx']}: {entry['task_name']:20s} → "
                  f"rank={entry['rank_chosen']}, reward={entry['reward']:.3f}")

    # Save results
    output = {
        'per_dataset': all_results,
        'average_accuracy': avg_accuracy,
        'bandit_history': bandit_history,
    }

    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else '.', exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved → {args.output}")


if __name__ == '__main__':
    main()
