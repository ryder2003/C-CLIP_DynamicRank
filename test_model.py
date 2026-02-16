"""
Quick test script for trained C-CLIP model.
Tests the checkpoint on sample datasets.
"""

import torch
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.cclip import CCLIP
from torch.utils.data import DataLoader
from src.data.datasets import ImageTextDataset
from src.data.transforms import get_clip_transforms
from src.utils.evaluation import evaluate_retrieval

def quick_test():
    """Quick test of the trained model."""
    
    # Configuration
    checkpoint_path = "checkpoints/sample_test/model_after_task_0.pt"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print("="*60)
    print("C-CLIP Model Quick Test")
    print("="*60)
    print(f"Device: {device}")
    print(f"Checkpoint: {checkpoint_path}")
    print()
    
    # Check if checkpoint exists
    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        print("Please train the model first or provide a valid checkpoint path.")
        return
    
    # Initialize model
    print("Loading model...")
    model = CCLIP(
        clip_model_name='ViT-B-32',
        pretrained='openai',
        lora_r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        lora_target_modules=['out_proj'],
        integration_coeff=1.0,
        device=device,
    ).to(device)
    
    # Load checkpoint - it has LoRA structure from training
    print("Loading checkpoint...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Inject LoRA to match the checkpoint structure (LoRA was present during save)
    model.inject_lora_for_new_task()
    
    # Load the state dict (use strict=False to handle any minor mismatches)
    model.clip.model.load_state_dict(checkpoint['clip_state_dict'], strict=False)
    if 'projector_state_dict' in checkpoint:
        model.image_projector.load_state_dict(checkpoint['projector_state_dict']['image'])
        model.text_projector.load_state_dict(checkpoint['projector_state_dict']['text'])
    
    # Note: Checkpoint was saved after LoRA merge in Task 1, so LoRA params are in original_layer
    # For inference, we can use the model as-is with LoRA structure
    
    model.eval()
    print("✅ Model loaded successfully!")
    print()
    
    # Test on both tasks
    tasks = [
        {'name': 'task_1', 'data_path': 'data/task_1/val.csv', 'image_dir': 'data/task_1'},
        {'name': 'task_2', 'data_path': 'data/task_2/val.csv', 'image_dir': 'data/task_2'},
    ]
    
    transform = get_clip_transforms(image_size=224, is_train=False)
    
    for task in tasks:
        print("="*60)
        print(f"Testing on {task['name']}")
        print("="*60)
        
        # Check if data exists
        if not os.path.exists(task['data_path']):
            print(f"⚠️  Data not found: {task['data_path']}")
            print()
            continue
        
        # Create dataset
        dataset = ImageTextDataset(
            data_path=task['data_path'],
            image_dir=task['image_dir'],
            transform=transform,
            tokenizer=model.clip.tokenizer,
            max_text_length=77,
        )
        
        dataloader = DataLoader(
            dataset,
            batch_size=16,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
        )
        
        print(f"Dataset size: {len(dataset)} samples")
        
        # Evaluate
        print("Evaluating...")
        with torch.no_grad():
            metrics = evaluate_retrieval(model, dataloader, device)
        
        # Print results
        print("\n📊 Results:")
        print(f"  Image-to-Text Retrieval:")
        print(f"    Recall@1:  {metrics['i2t_recall@1']:.2f}%")
        print(f"    Recall@5:  {metrics['i2t_recall@5']:.2f}%")
        print(f"    Recall@10: {metrics['i2t_recall@10']:.2f}%")
        print(f"  Text-to-Image Retrieval:")
        print(f"    Recall@1:  {metrics['t2i_recall@1']:.2f}%")
        print(f"    Recall@5:  {metrics['t2i_recall@5']:.2f}%")
        print(f"    Recall@10: {metrics['t2i_recall@10']:.2f}%")
        print()
    
    print("="*60)
    print("Testing completed!")
    print("="*60)

if __name__ == '__main__':
    quick_test()
