"""
Evaluation script for C-CLIP.
Evaluates trained model on retrieval and zero-shot classification tasks.
"""

import os
import sys
import argparse
import torch
from torch.utils.data import DataLoader
import json

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.cclip import CCLIP
from src.data.datasets import ImageTextDataset
from src.data.transforms import get_clip_transforms
from src.utils.config import load_config
from src.utils.evaluation import evaluate_retrieval, evaluate_zero_shot_classification


def evaluate_model(
    checkpoint_path: str,
    config_path: str,
    eval_datasets: list,
    output_path: str = None,
):
    """
    Evaluate a trained C-CLIP model.
    
    Args:
        checkpoint_path: Path to model checkpoint
        config_path: Path to config file
        eval_datasets: List of dataset configs to evaluate on
        output_path: Path to save evaluation results
    """
    # Load config
    config = load_config(config_path)
    
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Initialize model
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
    
    # Load checkpoint
    print(f"Loading checkpoint from {checkpoint_path}")
    model.load_checkpoint(checkpoint_path)
    model.eval()
    
    # Evaluation results
    all_results = {}
    
    # Evaluate on each dataset
    for dataset_config in eval_datasets:
        dataset_name = dataset_config['name']
        eval_type = dataset_config.get('type', 'retrieval')  # 'retrieval' or 'classification'
        
        print(f"\n{'='*60}")
        print(f"Evaluating on {dataset_name} ({eval_type})")
        print(f"{'='*60}\n")
        
        if eval_type == 'retrieval':
            # Create dataset
            val_transform = get_clip_transforms(
                image_size=config['data']['image_size'],
                is_train=False,
            )
            
            dataset = ImageTextDataset(
                data_path=dataset_config['data_path'],
                image_dir=dataset_config.get('image_dir', None),
                transform=val_transform,
                tokenizer=model.clip.tokenizer,
                max_text_length=config['data']['max_text_length'],
            )
            
            dataloader = DataLoader(
                dataset,
                batch_size=config['training']['batch_size'],
                shuffle=False,
                num_workers=config['data']['num_workers'],
                pin_memory=True,
            )
            
            # Evaluate retrieval
            metrics = evaluate_retrieval(
                model=model,
                dataloader=dataloader,
                device=device,
            )
            
            print(f"Results for {dataset_name}:")
            print(f"  I2T Recall@1:  {metrics['i2t_recall@1']:.2f}%")
            print(f"  I2T Recall@5:  {metrics['i2t_recall@5']:.2f}%")
            print(f"  I2T Recall@10: {metrics['i2t_recall@10']:.2f}%")
            print(f"  T2I Recall@1:  {metrics['t2i_recall@1']:.2f}%")
            print(f"  T2I Recall@5:  {metrics['t2i_recall@5']:.2f}%")
            print(f"  T2I Recall@10: {metrics['t2i_recall@10']:.2f}%")
            
            all_results[dataset_name] = metrics
        
        elif eval_type == 'classification':
            # Load class names
            with open(dataset_config['class_names_path'], 'r') as f:
                class_names = [line.strip() for line in f]
            
            # Create dataset (assumes format with image and label)
            # This is simplified - you may need to adapt based on your dataset format
            val_transform = get_clip_transforms(
                image_size=config['data']['image_size'],
                is_train=False,
            )
            
            # For classification, you'd need a different dataset class
            # This is a placeholder - implement based on your needs
            print(f"Zero-shot classification evaluation for {dataset_name}")
            print("Note: You need to implement classification dataset loader")
            
            # metrics = evaluate_zero_shot_classification(
            #     model=model,
            #     dataloader=dataloader,
            #     class_names=class_names,
            #     device=device,
            # )
            
            # all_results[dataset_name] = metrics
    
    # Save results
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"\nSaved evaluation results to {output_path}")
    
    return all_results


def main():
    parser = argparse.ArgumentParser(description='Evaluate C-CLIP model')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--eval_config', type=str, required=True, help='Path to evaluation dataset config')
    parser.add_argument('--output', type=str, default='results/evaluation_results.json', help='Path to save results')
    
    args = parser.parse_args()
    
    # Load evaluation dataset configs
    with open(args.eval_config, 'r') as f:
        eval_datasets = json.load(f)
    
    # Evaluate
    evaluate_model(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        eval_datasets=eval_datasets,
        output_path=args.output,
    )


if __name__ == '__main__':
    main()
