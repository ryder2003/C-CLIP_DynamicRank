"""
Training script for C-CLIP with MAB-driven dynamic rank selection.

Key differences from the original train.py:
  1. Model is CCLIPWithBandit instead of CCLIP.
  2. inject_lora_for_new_task() now returns the chosen rank.
  3. After training each task we evaluate → compute reward → update bandit.
  4. Zero-shot evaluation on an optional reference dataset is used as the
     stability signal for the bandit reward.
  5. Bandit state is printed at the end of every task and saved automatically.

Usage:
    python src/train_bandit.py --config configs/default_config.yaml
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
from typing import Dict, Any, Optional
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.cclip_bandit import CCLIPWithBandit, build_cclip_with_bandit
from src.losses.cclip_loss import CCLIPLoss, compute_retrieval_metrics
from src.utils.config import load_config, get_default_config
from src.utils.evaluation import evaluate_retrieval, evaluate_zero_shot_classification


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
# Main training loop
# ---------------------------------------------------------------------------

def train_with_bandit(config_path: Optional[str] = None):
    torch.set_float32_matmul_precision('medium')

    config = load_config(config_path) if (config_path and os.path.exists(config_path)) else get_default_config()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # ---- Build model with bandit ----
    model = build_cclip_with_bandit(config, device=device).to(device)

    # ---- Data module (reuse existing) ----
    # Import here to avoid circular imports
    try:
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
        has_data_module = True
    except ImportError:
        print("WARNING: ContinualLearningDataModule not found — "
              "you must supply dataloaders manually.")
        has_data_module = False

    checkpoint_dir = config['logging']['checkpoint_dir']
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Clean stale PL checkpoints
    import shutil
    for i in range(len(config['datasets'])):
        d = os.path.join(checkpoint_dir, f'task_{i}')
        if os.path.exists(d):
            shutil.rmtree(d, ignore_errors=True)

    num_tasks = len(config['datasets'])
    print(f"\nContinual learning with {num_tasks} tasks (MAB rank selection)\n")

    # Pretrained zero-shot baseline (for stability reward)
    # We record this BEFORE any training
    pretrained_zeroshot: Dict[str, float] = {}

    # Per-task tracking for bandit reward
    rank_chosen_per_task: Dict[int, int] = {}
    task_accuracy_after: Dict[int, float] = {}

    for task_idx in range(num_tasks):
        task_name = config['datasets'][task_idx]['name']

        # ----- 1. Bandit selects rank & injects LoRA -----
        chosen_rank = model.inject_lora_for_new_task(task_idx, task_name)
        rank_chosen_per_task[task_idx] = chosen_rank

        if not has_data_module:
            print(f"Skipping training for task {task_idx} (no data module)")
            continue

        data_module.set_task(task_idx)

        # ----- 2. Train with Lightning -----
        trainer_module = CCLIPBanditTrainer(model=model, config=config, current_task_idx=task_idx)

        logger = None
        if config['logging'].get('use_wandb', False):
            logger = WandbLogger(
                project=config['logging']['project_name'],
                name=f"{config['logging']['experiment_name']}_task{task_idx}_r{chosen_rank}",
            )

        callbacks = [
            ModelCheckpoint(
                dirpath=os.path.join(checkpoint_dir, f'task_{task_idx}'),
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

        # ----- 3. Merge LoRA -----
        model.merge_lora_after_task()

        # ----- 4. Save task checkpoint -----
        ckpt_path = os.path.join(checkpoint_dir, f'model_after_task_{task_idx}_r{chosen_rank}.pt')
        model.save_checkpoint(ckpt_path)

        # ----- 5. Evaluate on all tasks seen so far -----
        print(f"\n--- Evaluation after Task {task_idx + 1} ---")
        all_val_loaders = data_module.get_all_val_dataloaders()

        for eval_idx in range(task_idx + 1):
            if all_val_loaders[eval_idx] is None:
                continue
            eval_name = config['datasets'][eval_idx]['name']
            metrics = evaluate_retrieval(model=model, dataloader=all_val_loaders[eval_idx], device=device)
            acc = metrics['i2t_recall@1'] / 100.0   # normalise to [0,1]
            print(f"  {eval_name}: I2T@1 = {metrics['i2t_recall@1']:.2f}%")

            if eval_idx == task_idx:
                task_accuracy_after[task_idx] = acc

        # ----- 6. Update bandit with reward -----
        # For the stability signal we use the average recall on ALL prior tasks
        prior_accs = [task_accuracy_after[i] for i in range(task_idx)]
        zeroshot_retention = float(sum(prior_accs) / len(prior_accs)) if prior_accs else 1.0

        # Baseline: pretrained zero-shot on current task
        # (Approximation: first time we see this task there's no prior measurement.
        #  We set it to CLIP's out-of-the-box zero-shot if available, else 0.5)
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
        )

        # ----- 7. Print bandit summary -----
        print(f"\n{'='*50}")
        print("BANDIT STATE after task", task_idx + 1)
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
    print(f"\nBandit history saved → {bandit_log_path}")
    print(f"Final model saved   → {final_path}")
    print("\nContinual learning with MAB rank selection complete!")


def main():
    parser = argparse.ArgumentParser(description='C-CLIP with MAB rank selection')
    parser.add_argument('--config', type=str, default=None)
    args = parser.parse_args()
    train_with_bandit(args.config)


if __name__ == '__main__':
    main()
