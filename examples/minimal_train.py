"""
Minimal training example for C-CLIP.
This shows how to use the C-CLIP API directly without PyTorch Lightning.
"""

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.models.cclip import CCLIP
from src.losses.cclip_loss import CCLIPLoss, compute_retrieval_metrics
from src.data.datasets import ImageTextDataset
from src.data.transforms import get_clip_transforms


def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    total_i2t = 0
    total_t2i = 0
    
    for batch_idx, (images, text) in enumerate(dataloader):
        images = images.to(device)
        text = text.to(device)
        
        # Forward pass
        outputs = model(
            images=images,
            text=text,
            return_old_features=(model.current_task > 1),
        )
        
        # Compute loss
        loss_dict = criterion(
            image_features=outputs['image_features'],
            text_features=outputs['text_features'],
            projected_image_features=outputs['projected_image_features'],
            projected_text_features=outputs['projected_text_features'],
            old_image_features=outputs.get('old_image_features', None),
            old_text_features=outputs.get('old_text_features', None),
        )
        
        loss = loss_dict['total_loss']
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.get_trainable_parameters(), max_norm=1.0)
        optimizer.step()
        
        # Metrics
        metrics = compute_retrieval_metrics(
            outputs['image_features'], outputs['text_features']
        )
        
        total_loss += loss.item()
        total_i2t += metrics['i2t_recall@1']
        total_t2i += metrics['t2i_recall@1']
        
        # Log
        if (batch_idx + 1) % 10 == 0:
            avg_loss = total_loss / (batch_idx + 1)
            avg_i2t = total_i2t / (batch_idx + 1)
            avg_t2i = total_t2i / (batch_idx + 1)
            
            print(f"  Batch [{batch_idx+1}/{len(dataloader)}] "
                  f"Loss: {avg_loss:.4f} "
                  f"I2T: {avg_i2t:.2f}% "
                  f"T2I: {avg_t2i:.2f}%")
    
    return total_loss / len(dataloader)


def main():
    """
    Minimal training example.
    This demonstrates the basic training loop without PyTorch Lightning.
    """
    print("="*60)
    print("C-CLIP Minimal Training Example")
    print("="*60 + "\n")
    
    # Configuration
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size = 32  # Small batch for demo
    num_epochs = 2
    learning_rate = 1e-5
    
    print(f"Device: {device}")
    print(f"Batch size: {batch_size}")
    print(f"Epochs: {num_epochs}\n")
    
    # Initialize model
    print("Initializing model...")
    model = CCLIP(
        clip_model_name="ViT-B-32",
        pretrained="openai",
        lora_r=8,
        lora_alpha=16,
        device=device,
    ).to(device)
    
    # Create dummy dataset (replace with your actual dataset)
    print("Setting up dummy dataset...")
    
    # For this example, we'll create a simple in-memory dataset
    # In practice, use ImageTextDataset with your data
    class DummyDataset(torch.utils.data.Dataset):
        def __init__(self, size=100):
            self.size = size
            self.transform = get_clip_transforms(224, is_train=True)
        
        def __len__(self):
            return self.size
        
        def __getitem__(self, idx):
            # Random image
            image = torch.randn(3, 224, 224)
            # Random text tokens
            text = torch.randint(0, 49408, (77,))
            return image, text
    
    dataset = DummyDataset(size=100)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )
    
    # Task 1
    print("\n" + "="*60)
    print("Task 1: Training")
    print("="*60 + "\n")
    
    # Inject LoRA
    model.inject_lora_for_new_task()
    
    # Setup loss (no CKC for first task)
    criterion = CCLIPLoss(temperature=0.07, use_ckc=False)
    
    # Setup optimizer
    optimizer = AdamW(model.get_trainable_parameters(), lr=learning_rate)
    
    # Train
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        loss = train_one_epoch(model, dataloader, criterion, optimizer, device, epoch)
        print(f"  Average Loss: {loss:.4f}\n")
    
    # Merge LoRA
    model.merge_lora_after_task()
    print("✓ Task 1 completed and weights merged\n")
    
    # Task 2 (with CKC)
    print("="*60)
    print("Task 2: Training with CKC")
    print("="*60 + "\n")
    
    # Inject LoRA for task 2
    model.inject_lora_for_new_task()
    
    # Setup loss (with CKC)
    criterion = CCLIPLoss(temperature=0.07, use_ckc=True)
    
    # Setup optimizer
    optimizer = AdamW(model.get_trainable_parameters(), lr=learning_rate)
    
    # Train
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        loss = train_one_epoch(model, dataloader, criterion, optimizer, device, epoch)
        print(f"  Average Loss: {loss:.4f}\n")
    
    # Merge LoRA
    model.merge_lora_after_task()
    print("✓ Task 2 completed and weights merged\n")
    
    # Save model
    checkpoint_path = "checkpoints/minimal_example.pt"
    os.makedirs("checkpoints", exist_ok=True)
    model.save_checkpoint(checkpoint_path)
    print(f"✓ Model saved to {checkpoint_path}")
    
    print("\n" + "="*60)
    print("Training complete!")
    print("="*60)


if __name__ == '__main__':
    main()
