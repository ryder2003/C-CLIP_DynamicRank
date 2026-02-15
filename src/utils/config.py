"""
Configuration utilities for C-CLIP.
"""

import yaml
from typing import Dict, Any
import os


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to YAML config file
        
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def save_config(config: Dict[str, Any], save_path: str):
    """
    Save configuration to YAML file.
    
    Args:
        config: Configuration dictionary
        save_path: Path to save YAML file
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    with open(save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"Saved config to {save_path}")


def get_default_config() -> Dict[str, Any]:
    """
    Get default configuration for C-CLIP.
    
    Returns:
        Default configuration dictionary
    """
    config = {
        # Model settings
        'model': {
            'clip_model_name': 'ViT-B-16',
            'pretrained': 'openai',
            'lora_r': 16,
            'lora_alpha': 32,
            'lora_dropout': 0.1,
            'lora_target_modules': ['q_proj', 'v_proj'],
            'integration_coeff': 0.5,
        },
        
        # Training settings
        'training': {
            'batch_size': 256,
            'epochs_per_task': 40,
            'base_lr': 1e-5,
            'text_lr_multiplier': 10,  # Text encoder LR = base_lr * multiplier
            'weight_decay': 0.2,
            'warmup_epochs': 5,
            'optimizer': 'adamw',
            'beta1': 0.9,
            'beta2': 0.99,
            'temperature': 0.07,
        },
        
        # Data settings
        'data': {
            'image_size': 224,
            'max_text_length': 77,
            'num_workers': 4,
        },
        
        # Datasets for continual learning
        'datasets': [
            {
                'name': 'flickr30k',
                'train_path': 'data/flickr30k/train.csv',
                'val_path': 'data/flickr30k/val.csv',
                'image_dir': 'data/flickr30k/images',
            },
            {
                'name': 'coco',
                'train_path': 'data/coco/train.csv',
                'val_path': 'data/coco/val.csv',
                'image_dir': 'data/coco/images',
            },
            # Add more datasets as needed
        ],
        
        # Logging and checkpointing
        'logging': {
            'project_name': 'c-clip',
            'experiment_name': 'default',
            'log_every_n_steps': 50,
            'save_checkpoint_every_n_epochs': 10,
            'checkpoint_dir': 'checkpoints',
        },
        
        # Hardware
        'hardware': {
            'accelerator': 'gpu',
            'devices': 1,
            'precision': '16-mixed',
        },
    }
    
    return config
