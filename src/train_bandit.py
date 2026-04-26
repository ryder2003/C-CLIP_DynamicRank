"""
Training script for C-CLIP with MAB-driven dynamic rank selection.

Key differences from the original train.py:
  1. Model is CCLIPWithBandit instead of CCLIP.
  2. inject_lora_for_new_task() now returns the chosen rank.
  3. After training each task we evaluate → compute reward → update bandit.
  4. Zero-shot classification evaluation is used (not retrieval) since these
     are classification datasets with duplicate captions per class.
  5. Bandit state is printed at the end of every task and saved automatically.

Usage:
    python src/train_bandit.py --config bandit_config.yaml
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, SequentialLR, LinearLR
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from typing import Dict, Any, Optional, List
import json
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.cclip_bandit import CCLIPWithBandit, build_cclip_with_bandit
from src.losses.cclip_loss import CCLIPLoss, compute_retrieval_metrics
from src.utils.config import load_config, get_default_config
from src.utils.evaluation import evaluate_retrieval, evaluate_zero_shot_classification
from src.data.datasets import ClassificationDataset
from src.data.transforms import get_clip_transforms


# ---------------------------------------------------------------------------
# Lightning module (same interface as original CCLIPTrainer)
# ---------------------------------------------------------------------------

class CCLIPBanditTrainer(pl.LightningModule):
    """Lightning module wrapping CCLIPWithBandit for one CL task."""

    def __init__(
        self,
        model: CCLIPWithBandit,
        config: Dict[str, Any],
        current_task_idx: int = 0,
    ):
        super().__init__()
        self.model_cclip = model
        self.config = config
        self.current_task_idx = current_task_idx

        use_ckc = current_task_idx > 0
        self.criterion = CCLIPLoss(
            temperature=config['training']['temperature'],
            use_ckc=use_ckc,
        )

        self.base_lr = config['training']['base_lr']
        self.text_lr_multiplier = config['training']['text_lr_multiplier']
        self.weight_decay = config['training']['weight_decay']
        self.warmup_epochs = config['training']['warmup_epochs']
        self.epochs_per_task = config['training']['epochs_per_task']

        self.save_hyperparameters(ignore=['model'])

    def forward(self, images, text):
        return self.model_cclip(
            images=images,
            text=text,
            return_old_features=(self.current_task_idx > 0),
        )

    def training_step(self, batch, batch_idx):
        images, text = batch
        outputs = self(images, text)
        loss_dict = self.criterion(
            image_features=outputs['image_features'],
            text_features=outputs['text_features'],
            projected_image_features=outputs['projected_image_features'],
            projected_text_features=outputs['projected_text_features'],
            old_image_features=outputs.get('old_image_features'),
            old_text_features=outputs.get('old_text_features'),
        )
        metrics = compute_retrieval_metrics(outputs['image_features'], outputs['text_features'])
        self.log('train/total_loss', loss_dict['total_loss'], on_step=True, on_epoch=True, prog_bar=True)
        self.log('train/clip_loss',  loss_dict['clip_loss'],  on_step=True, on_epoch=True)
        self.log('train/ckc_loss',   loss_dict['ckc_loss'],   on_step=True, on_epoch=True)
        self.log('train/i2t@1', metrics['i2t_recall@1'], on_step=True, on_epoch=True)
        # Log the rank that was chosen for this task
        self.log('train/lora_rank', float(self.model_cclip.current_lora_r), on_step=False, on_epoch=True)
        return loss_dict['total_loss']

    def validation_step(self, batch, batch_idx):
        images, text = batch
        outputs = self(images, text)
        loss_dict = self.criterion(
            image_features=outputs['image_features'],
            text_features=outputs['text_features'],
            projected_image_features=outputs['projected_image_features'],
            projected_text_features=outputs['projected_text_features'],
            old_image_features=outputs.get('old_image_features'),
            old_text_features=outputs.get('old_text_features'),
        )
        metrics = compute_retrieval_metrics(outputs['image_features'], outputs['text_features'])
        self.log('val/total_loss', loss_dict['total_loss'], on_epoch=True, prog_bar=True)
        self.log('val/i2t@1', metrics['i2t_recall@1'], on_epoch=True, prog_bar=True)
        return loss_dict['total_loss']

    def configure_optimizers(self):
        vision_lora, text_lora, proj = [], [], []
        for name, param in self.model_cclip.named_parameters():
            if not param.requires_grad:
                continue
            nl = name.lower()
            if 'projector' in nl:
                proj.append(param)
            elif 'clip.model.transformer' in nl or 'text' in nl:
                text_lora.append(param)
            else:
                vision_lora.append(param)

        param_groups = [g for g in [
            {'params': vision_lora, 'lr': self.base_lr,                              'weight_decay': 0.0, 'name': 'vision_lora'},
            {'params': text_lora,   'lr': self.base_lr * self.text_lr_multiplier,    'weight_decay': 0.0, 'name': 'text_lora'},
            {'params': proj,        'lr': self.base_lr,                              'weight_decay': self.weight_decay, 'name': 'projector'},
        ] if len(g['params']) > 0]

        optimizer = AdamW(
            param_groups,
            betas=(self.config['training']['beta1'], self.config['training']['beta2']),
        )
        warmup = LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=self.warmup_epochs)
        cosine = CosineAnnealingLR(optimizer, T_max=self.epochs_per_task - self.warmup_epochs, eta_min=1e-6)
        scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[self.warmup_epochs])
        return {'optimizer': optimizer, 'lr_scheduler': {'scheduler': scheduler, 'interval': 'epoch'}}


# ---------------------------------------------------------------------------
# Zero-shot classification evaluation helper
# ---------------------------------------------------------------------------

def _load_class_names(dataset_config: Dict) -> List[str]:
    """Load class names from the class_names.txt file for a dataset."""
    # Try explicit path first
    class_names_path = dataset_config.get('class_names_path')
    if class_names_path and os.path.exists(class_names_path):
        with open(class_names_path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip()]

    # Auto-detect: look in data/<name>/class_names.txt
    name = dataset_config['name']
    auto_path = os.path.join('data', name, 'class_names.txt')
    if os.path.exists(auto_path):
        with open(auto_path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip()]

    # Fallback: extract unique class names from val CSV captions
    val_path = dataset_config.get('val_path', '')
    if os.path.exists(val_path):
        import pandas as pd
        df = pd.read_csv(val_path)
        # Extract class name from caption "a photo of a {name}" pattern
        captions = sorted(df['caption'].unique().tolist())
        return captions

    return []


def _evaluate_zero_shot(model, dataset_config, device='cuda'):
    """
    Evaluate zero-shot classification on a single dataset.
    Returns accuracy in [0, 1].
    """
    from torch.utils.data import DataLoader

    val_path = dataset_config.get('val_path', '')
    image_dir = dataset_config.get('image_dir', '')

    if not os.path.exists(val_path):
        print(f"  [SKIP] val_path not found: {val_path}")
        return None

    # Load class names
    class_names = _load_class_names(dataset_config)
    if not class_names:
        print(f"  [SKIP] No class names found for {dataset_config['name']}")
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

    # Use the class_names from the dataset (which are captions)
    # For zero-shot, we need clean class names (not full captions)
    # Extract from captions: "a photo of a {name}" → "{name}"
    clean_names = []
    for cn in cls_dataset.class_names:
        # Try to extract the class name from common caption patterns
        for prefix in ['a photo of a ', 'a photo of ', 'a satellite photo of ',
                       'a photo of a ', 'a photo of the ']:
            if cn.lower().startswith(prefix):
                cn = cn[len(prefix):]
                break
        # Remove trailing suffixes like ", a type of aircraft"
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

    return metrics['accuracy'] / 100.0  # normalise to [0, 1]


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train_with_bandit(config_path: Optional[str] = None):
    torch.set_float32_matmul_precision('medium')

    config = load_config(config_path) if (config_path and os.path.exists(config_path)) else get_default_config()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # ---- Build model with bandit ----
    model = build_cclip_with_bandit(config, device=device).to(device)

    # ---- Validate data files exist ----
    print("\nValidating dataset CSV files...")
    missing_files = []
    for ds_cfg in config['datasets']:
        for key in ['train_path', 'val_path']:
            fpath = ds_cfg.get(key, '')
            if not os.path.exists(fpath):
                missing_files.append(fpath)
    if missing_files:
        print("\n" + "=" * 60)
        print("ERROR: The following data CSV files are missing:")
        for f in missing_files:
            print(f"  ✗ {f}")
        print("\nYou must run the data preparation script first:")
        print("  python scripts/prepare_real_datasets.py")
        print("=" * 60)
        sys.exit(1)
    print("  All CSV files found ✓")

    # ---- Data module ----
    from src.data.datasets import ContinualLearningDataModule
    data_module = ContinualLearningDataModule(
        dataset_configs=config['datasets'],
        tokenizer=model.clip.tokenizer,
        batch_size=config['training']['batch_size'],
        num_workers=config['data']['num_workers'],
        image_size=config['data']['image_size'],
        max_text_length=config['data']['max_text_length'],
    )
    data_module.setup()

    checkpoint_dir = config['logging']['checkpoint_dir']
    os.makedirs(checkpoint_dir, exist_ok=True)

    num_tasks = len(config['datasets'])
    print(f"\nContinual learning with {num_tasks} tasks (MAB rank selection)\n")

    # ---- Compute pretrained zero-shot baselines BEFORE any training ----
    pretrained_zeroshot: Dict[str, float] = {}
    print("=" * 60)
    print("Computing pretrained CLIP zero-shot baselines...")
    print("=" * 60)
    for ds_cfg in config['datasets']:
        ds_name = ds_cfg['name']
        acc = _evaluate_zero_shot(model, ds_cfg, device=device)
        if acc is not None:
            pretrained_zeroshot[ds_name] = acc
            print(f"  {ds_name}: pretrained zero-shot = {acc*100:.2f}%")
        else:
            pretrained_zeroshot[ds_name] = 0.5
            print(f"  {ds_name}: could not evaluate, using 0.5 baseline")
    print()

    # Per-task tracking for bandit reward
    rank_chosen_per_task: Dict[int, int] = {}
    task_accuracy_after: Dict[int, float] = {}
    training_start = time.time()

    for task_idx in range(num_tasks):
        task_name = config['datasets'][task_idx]['name']
        task_start = time.time()

        # ----- 1. Bandit selects rank & injects LoRA -----
        chosen_rank = model.inject_lora_for_new_task(task_idx, task_name)
        rank_chosen_per_task[task_idx] = chosen_rank

        data_module.set_task(task_idx)

        # ----- 2. Train with Lightning -----
        trainer_module = CCLIPBanditTrainer(model=model, config=config, current_task_idx=task_idx)

        logger = None
        if config['logging'].get('use_wandb', False):
            logger = WandbLogger(
                project=config['logging']['project_name'],
                name=f"{config['logging']['experiment_name']}_task{task_idx}_r{chosen_rank}",
            )

        task_ckpt_dir = os.path.join(checkpoint_dir, f'task_{task_idx}')
        callbacks = [
            ModelCheckpoint(
                dirpath=task_ckpt_dir,
                filename=f'r{chosen_rank}-epoch{{epoch:02d}}-loss{{val/total_loss:.4f}}',
                monitor='val/total_loss',
                mode='min',
                save_top_k=2,
            ),
            LearningRateMonitor(logging_interval='epoch'),
        ]

        trainer = pl.Trainer(
            max_epochs=config['training']['epochs_per_task'],
            accelerator=config['hardware']['accelerator'],
            devices=config['hardware']['devices'],
            precision=config['hardware']['precision'],
            logger=logger,
            callbacks=callbacks,
            log_every_n_steps=config['logging']['log_every_n_steps'],
            gradient_clip_val=1.0,
            accumulate_grad_batches=config['training'].get('accumulate_grad_batches', 1),
        )
        trainer.fit(trainer_module, data_module)

        # ----- 3. Compute LoRA utilisation BEFORE merging -----
        # (After merge, LoRA layers are gone → utilisation would be 1.0 always)
        utilisation = model.compute_lora_utilisation()

        # ----- 4. Merge LoRA -----
        model.merge_lora_after_task()

        # ----- 5. Save task checkpoint -----
        ckpt_path = os.path.join(checkpoint_dir, f'model_after_task_{task_idx}_r{chosen_rank}.pt')
        model.save_checkpoint(ckpt_path)

        # ----- 6. Evaluate zero-shot classification on all tasks seen so far -----
        print(f"\n--- Zero-shot evaluation after Task {task_idx + 1} ({task_name}) ---")
        for eval_idx in range(task_idx + 1):
            eval_cfg = config['datasets'][eval_idx]
            eval_name = eval_cfg['name']
            acc = _evaluate_zero_shot(model, eval_cfg, device=device)
            if acc is not None:
                print(f"  {eval_name}: accuracy = {acc*100:.2f}%")
                if eval_idx == task_idx:
                    task_accuracy_after[task_idx] = acc
                else:
                    # Track how previous tasks are doing (for forgetting analysis)
                    task_accuracy_after[eval_idx] = acc
            else:
                print(f"  {eval_name}: evaluation failed")

        # ----- 7. Update bandit with reward -----
        # Stability: average accuracy on ALL prior tasks
        prior_accs = [task_accuracy_after.get(i, 0.5) for i in range(task_idx)]
        zeroshot_retention = float(sum(prior_accs) / len(prior_accs)) if prior_accs else 1.0

        # Baseline: pretrained zero-shot on current task
        baseline_acc = pretrained_zeroshot.get(task_name, 0.5)
        prior_baseline = sum(pretrained_zeroshot.get(config['datasets'][i]['name'], 0.5)
                             for i in range(task_idx)) / max(task_idx, 1)

        model.update_bandit(
            rank=chosen_rank,
            task_idx=task_idx,
            task_name=task_name,
            task_accuracy=task_accuracy_after.get(task_idx, 0.5),
            baseline_accuracy=baseline_acc,
            zeroshot_retention=zeroshot_retention,
            zeroshot_baseline=prior_baseline if task_idx > 0 else 1.0,
            extra_info={"lora_utilisation_pre_merge": utilisation},
        )

        # ----- 8. Print bandit summary -----
        task_elapsed = time.time() - task_start
        total_elapsed = time.time() - training_start
        print(f"\n{'='*50}")
        print(f"BANDIT STATE after task {task_idx + 1}/{num_tasks}")
        print(f"Task time: {task_elapsed/60:.1f}min | Total: {total_elapsed/60:.1f}min")
        print(f"{'='*50}")
        summary = model.bandit.summary()
        for r, arm_data in summary['arms'].items():
            print(f"  rank={r}: pulls={arm_data['n_pulls']}, "
                  f"mean_reward={arm_data['mean_reward']:.3f}")
        print(f"  Best rank so far: {summary['best_rank']}")

    # ---- Final checkpoint ----
    final_path = os.path.join(checkpoint_dir, 'model_final_bandit.pt')
    model.save_checkpoint(final_path)

    # Save full bandit history
    bandit_log_path = os.path.join(checkpoint_dir, 'bandit_history.json')
    with open(bandit_log_path, 'w') as f:
        json.dump(model.bandit.summary(), f, indent=2)

    total_time = time.time() - training_start
    print(f"\n{'='*60}")
    print(f"TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"Total time       : {total_time/60:.1f} minutes")
    print(f"Bandit history   → {bandit_log_path}")
    print(f"Final model      → {final_path}")
    print(f"Ranks chosen     : {rank_chosen_per_task}")

    # Final accuracy summary
    print(f"\nFinal accuracies:")
    for task_idx in range(num_tasks):
        name = config['datasets'][task_idx]['name']
        acc = task_accuracy_after.get(task_idx, 0)
        baseline = pretrained_zeroshot.get(name, 0)
        delta = (acc - baseline) * 100
        print(f"  {name:20s}: {acc*100:6.2f}% (pretrained: {baseline*100:.2f}%, Δ={delta:+.2f}%)")

    avg_final = sum(task_accuracy_after.get(i, 0) for i in range(num_tasks)) / num_tasks
    avg_baseline = sum(pretrained_zeroshot.get(config['datasets'][i]['name'], 0) for i in range(num_tasks)) / num_tasks
    print(f"  {'Average':20s}: {avg_final*100:6.2f}% (pretrained: {avg_baseline*100:.2f}%)")
    print(f"\nContinual learning with MAB rank selection complete!")


def main():
    parser = argparse.ArgumentParser(description='C-CLIP with MAB rank selection')
    parser.add_argument('--config', type=str, default=None)
    args = parser.parse_args()
    train_with_bandit(args.config)


if __name__ == '__main__':
    main()
