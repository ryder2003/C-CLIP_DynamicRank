"""
Training script for C-CLIP continual learning.
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
from typing import Dict, Any
import wandb

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.cclip import CCLIP
from src.losses.cclip_loss import CCLIPLoss, compute_retrieval_metrics
from src.data.datasets import ContinualLearningDataModule
from src.utils.config import load_config, save_config, get_default_config
from src.utils.evaluation import evaluate_retrieval


class CCLIPTrainer(pl.LightningModule):
    """
    PyTorch Lightning module for training C-CLIP.
    """
    
    def __init__(
        self,
        model: CCLIP,
        config: Dict[str, Any],
        current_task_idx: int = 0,
    ):
        super().__init__()
        
        self.model_cclip = model
        self.config = config
        self.current_task_idx = current_task_idx
        
        # Loss function
        use_ckc = current_task_idx > 0  # Use CKC from second task onwards
        self.criterion = CCLIPLoss(
            temperature=config['training']['temperature'],
            use_ckc=use_ckc,
        )
        
        # Training config
        self.base_lr = config['training']['base_lr']
        self.text_lr_multiplier = config['training']['text_lr_multiplier']
        self.weight_decay = config['training']['weight_decay']
        self.warmup_epochs = config['training']['warmup_epochs']
        self.epochs_per_task = config['training']['epochs_per_task']
        
        # Metrics
        self.save_hyperparameters(ignore=['model'])
        
    def forward(self, images, text):
        """Forward pass."""
        return self.model_cclip(
            images=images,
            text=text,
            return_old_features=(self.current_task_idx > 0),
        )
    
    def training_step(self, batch, batch_idx):
        """Training step."""
        images, text = batch
        
        # Forward pass
        outputs = self(images, text)
        
        # Compute loss
        loss_dict = self.criterion(
            image_features=outputs['image_features'],
            text_features=outputs['text_features'],
            projected_image_features=outputs['projected_image_features'],
            projected_text_features=outputs['projected_text_features'],
            old_image_features=outputs.get('old_image_features', None),
            old_text_features=outputs.get('old_text_features', None),
        )
        
        # Compute retrieval metrics
        metrics = compute_retrieval_metrics(
            image_features=outputs['image_features'],
            text_features=outputs['text_features'],
        )
        
        # Log metrics
        self.log('train/total_loss', loss_dict['total_loss'], on_step=True, on_epoch=True, prog_bar=True)
        self.log('train/clip_loss', loss_dict['clip_loss'], on_step=True, on_epoch=True)
        self.log('train/ckc_loss', loss_dict['ckc_loss'], on_step=True, on_epoch=True)
        self.log('train/i2t_recall@1', metrics['i2t_recall@1'], on_step=True, on_epoch=True)
        self.log('train/t2i_recall@1', metrics['t2i_recall@1'], on_step=True, on_epoch=True)
        
        return loss_dict['total_loss']
    
    def validation_step(self, batch, batch_idx):
        """Validation step."""
        images, text = batch
        
        # Forward pass
        outputs = self(images, text)
        
        # Compute loss
        loss_dict = self.criterion(
            image_features=outputs['image_features'],
            text_features=outputs['text_features'],
            projected_image_features=outputs['projected_image_features'],
            projected_text_features=outputs['projected_text_features'],
            old_image_features=outputs.get('old_image_features', None),
            old_text_features=outputs.get('old_text_features', None),
        )
        
        # Compute retrieval metrics
        metrics = compute_retrieval_metrics(
            image_features=outputs['image_features'],
            text_features=outputs['text_features'],
        )
        
        # Log metrics
        self.log('val/total_loss', loss_dict['total_loss'], on_step=False, on_epoch=True, prog_bar=True)
        self.log('val/clip_loss', loss_dict['clip_loss'], on_step=False, on_epoch=True)
        self.log('val/ckc_loss', loss_dict['ckc_loss'], on_step=False, on_epoch=True)
        self.log('val/i2t_recall@1', metrics['i2t_recall@1'], on_step=False, on_epoch=True, prog_bar=True)
        self.log('val/t2i_recall@1', metrics['t2i_recall@1'], on_step=False, on_epoch=True, prog_bar=True)
        
        return loss_dict['total_loss']
    
    def configure_optimizers(self):
        """Configure optimizer and scheduler."""
        # Get trainable parameters with different learning rates
        # Vision encoder: base_lr
        # Text encoder: base_lr * text_lr_multiplier
        # Projectors: base_lr
        
        vision_params = []
        text_params = []
        projector_params = []
        
        for name, param in self.model_cclip.named_parameters():
            if not param.requires_grad:
                continue
            
            if 'text' in name.lower():
                text_params.append(param)
            elif 'projector' in name.lower():
                projector_params.append(param)
            else:
                vision_params.append(param)
        
        param_groups = [
            {'params': vision_params, 'lr': self.base_lr, 'name': 'vision'},
            {'params': text_params, 'lr': self.base_lr * self.text_lr_multiplier, 'name': 'text'},
            {'params': projector_params, 'lr': self.base_lr, 'name': 'projector'},
        ]
        
        # Filter out empty groups
        param_groups = [g for g in param_groups if len(g['params']) > 0]
        
        optimizer = AdamW(
            param_groups,
            betas=(self.config['training']['beta1'], self.config['training']['beta2']),
            weight_decay=self.weight_decay,
        )
        
        # Cosine learning rate scheduler with linear warmup
        warmup_scheduler = LinearLR(
            optimizer,
            start_factor=0.01,   # Start at 1% of base_lr
            end_factor=1.0,
            total_iters=self.warmup_epochs,
        )
        cosine_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=self.epochs_per_task - self.warmup_epochs,
            eta_min=1e-6,
        )
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[self.warmup_epochs],
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch',
            }
        }


def train_continual_learning(config_path: str):
    """
    Main function for continual learning training.
    
    Args:
        config_path: Path to configuration file
    """
    # Suppress Tensor Core precision warning and gain a small speedup
    torch.set_float32_matmul_precision('medium')

    # Load config
    if config_path and os.path.exists(config_path):
        config = load_config(config_path)
        print(f"Loaded config from {config_path}")
    else:
        config = get_default_config()
        print("Using default config")
    
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Initialize C-CLIP model
    model = CCLIP(
        clip_model_name=config['model']['clip_model_name'],
        pretrained=config['model']['pretrained'],
        lora_r=config['model']['lora_r'],
        lora_alpha=config['model']['lora_alpha'],
        lora_dropout=config['model']['lora_dropout'],
        lora_target_modules=config['model'].get('lora_target_modules', None),
        integration_coeff=config['model']['integration_coeff'],
        device=device,
    ).to(device)
    
    # Initialize data module
    data_module = ContinualLearningDataModule(
        dataset_configs=config['datasets'],
        tokenizer=model.clip.tokenizer,
        batch_size=config['training']['batch_size'],
        num_workers=config['data']['num_workers'],
        image_size=config['data']['image_size'],
        max_text_length=config['data']['max_text_length'],
    )
    data_module.setup()
    
    # Create checkpoint directory
    checkpoint_dir = config['logging']['checkpoint_dir']
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Continual learning loop
    num_tasks = data_module.get_num_tasks()
    print(f"\n{'='*60}")
    print(f"Starting Continual Learning with {num_tasks} tasks")
    print(f"{'='*60}\n")
    
    for task_idx in range(num_tasks):
        task_name = config['datasets'][task_idx]['name']
        print(f"\n{'='*60}")
        print(f"Task {task_idx + 1}/{num_tasks}: {task_name}")
        print(f"{'='*60}\n")
        
        # Inject LoRA for new task
        model.inject_lora_for_new_task()
        
        # Set current task in data module
        data_module.set_task(task_idx)
        
        # Initialize trainer module
        trainer_module = CCLIPTrainer(
            model=model,
            config=config,
            current_task_idx=task_idx,
        )
        
        # Setup logger
        logger = None
        if config['logging'].get('use_wandb', False):
            logger = WandbLogger(
                project=config['logging']['project_name'],
                name=f"{config['logging']['experiment_name']}_task{task_idx}",
            )
        
        # Setup callbacks
        # NOTE: We monitor val/total_loss rather than val/i2t_recall@1 because
        # classification datasets (many images share the same caption) produce
        # artificially near-zero recall@1 within a batch, even when the model
        # is converging correctly. Loss is the reliable signal.
        callbacks = [
            ModelCheckpoint(
                dirpath=os.path.join(checkpoint_dir, f'task_{task_idx}'),
                filename='cclip-epoch{epoch:02d}-loss{val/total_loss:.4f}',
                monitor='val/total_loss',
                mode='min',
                save_top_k=3,
            ),
            LearningRateMonitor(logging_interval='epoch'),
        ]
        
        # Initialize trainer
        accumulate_grad = config['training'].get('accumulate_grad_batches', 1)
        trainer = pl.Trainer(
            max_epochs=config['training']['epochs_per_task'],
            accelerator=config['hardware']['accelerator'],
            devices=config['hardware']['devices'],
            precision=config['hardware']['precision'],
            logger=logger,
            callbacks=callbacks,
            log_every_n_steps=config['logging']['log_every_n_steps'],
            gradient_clip_val=1.0,
            accumulate_grad_batches=accumulate_grad,
        )
        
        # Train
        trainer.fit(trainer_module, data_module)
        
        # Merge LoRA weights after task
        model.merge_lora_after_task()
        
        # Save checkpoint after merging
        checkpoint_path = os.path.join(checkpoint_dir, f'model_after_task_{task_idx}.pt')
        model.save_checkpoint(checkpoint_path)
        
        # Evaluate on all tasks so far
        print(f"\n{'='*60}")
        print(f"Evaluation after Task {task_idx + 1}")
        print(f"{'='*60}\n")
        
        all_val_loaders = data_module.get_all_val_dataloaders()
        for eval_task_idx in range(task_idx + 1):
            if all_val_loaders[eval_task_idx] is None:
                continue
            
            eval_task_name = config['datasets'][eval_task_idx]['name']
            print(f"\nEvaluating on {eval_task_name}...")
            
            metrics = evaluate_retrieval(
                model=model,
                dataloader=all_val_loaders[eval_task_idx],
                device=device,
            )
            
            print(f"  I2T Recall@1: {metrics['i2t_recall@1']:.2f}%")
            print(f"  T2I Recall@1: {metrics['t2i_recall@1']:.2f}%")
            
            if logger:
                wandb.log({
                    f'eval/{eval_task_name}/i2t_recall@1': metrics['i2t_recall@1'],
                    f'eval/{eval_task_name}/t2i_recall@1': metrics['t2i_recall@1'],
                    'task': task_idx,
                })
    
    print(f"\n{'='*60}")
    print("Continual Learning Complete!")
    print(f"{'='*60}\n")
    
    # Final checkpoint
    final_checkpoint_path = os.path.join(checkpoint_dir, 'model_final.pt')
    model.save_checkpoint(final_checkpoint_path)
    print(f"Saved final model to {final_checkpoint_path}")


def main():
    parser = argparse.ArgumentParser(description='Train C-CLIP with continual learning')
    parser.add_argument('--config', type=str, default=None, help='Path to config file')
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to checkpoint to resume from')
    
    args = parser.parse_args()
    
    train_continual_learning(args.config)


if __name__ == '__main__':
    main()
