"""
Quick demo script to test C-CLIP implementation.
Tests basic model functionality without full training.
"""

import torch
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.models.cclip import CCLIP
from src.losses.cclip_loss import CCLIPLoss, compute_retrieval_metrics


def test_model_initialization():
    """Test model initialization."""
    print("="*60)
    print("Testing Model Initialization")
    print("="*60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}\n")
    
    # Initialize model
    model = CCLIP(
        clip_model_name="ViT-B-32",  # Use smaller model for testing
        pretrained="openai",
        lora_r=8,  # Smaller rank for testing
        lora_alpha=16,
        device=device,
    )
    
    print(f"✓ Model initialized successfully")
    print(f"  - Embedding dimension: {model.embed_dim}")
    print(f"  - Current task: {model.current_task}")
    
    return model


def test_lora_injection(model):
    """Test LoRA injection."""
    print("\n" + "="*60)
    print("Testing LoRA Injection")
    print("="*60 + "\n")
    
    # Inject LoRA for first task
    model.inject_lora_for_new_task()
    
    print(f"✓ LoRA injected for Task {model.current_task}")
    print(f"  - Number of LoRA layers: {len(model.lora_layers)}")
    
    # Count trainable parameters
    trainable_params = sum(p.numel() for p in model.get_trainable_parameters())
    total_params = sum(p.numel() for p in model.parameters())
    
    print(f"  - Trainable parameters: {trainable_params:,}")
    print(f"  - Total parameters: {total_params:,}")
    print(f"  - Trainable ratio: {trainable_params/total_params*100:.2f}%")


def test_forward_pass(model):
    """Test forward pass."""
    print("\n" + "="*60)
    print("Testing Forward Pass")
    print("="*60 + "\n")
    
    device = next(model.parameters()).device
    batch_size = 4
    
    # Create dummy data
    images = torch.randn(batch_size, 3, 224, 224).to(device)
    text = torch.randint(0, 49408, (batch_size, 77)).to(device)  # Random tokens
    
    # Forward pass
    with torch.no_grad():
        outputs = model(images, text, return_old_features=False)
    
    print(f"✓ Forward pass successful")
    print(f"  - Image features shape: {outputs['image_features'].shape}")
    print(f"  - Text features shape: {outputs['text_features'].shape}")
    print(f"  - Projected image features shape: {outputs['projected_image_features'].shape}")
    print(f"  - Projected text features shape: {outputs['projected_text_features'].shape}")
    
    return outputs


def test_loss_computation(outputs):
    """Test loss computation."""
    print("\n" + "="*60)
    print("Testing Loss Computation")
    print("="*60 + "\n")
    
    # Test CLIP loss only (first task)
    criterion = CCLIPLoss(temperature=0.07, use_ckc=False)
    
    loss_dict = criterion(
        image_features=outputs['image_features'],
        text_features=outputs['text_features'],
        projected_image_features=outputs['projected_image_features'],
        projected_text_features=outputs['projected_text_features'],
    )
    
    print(f"✓ Loss computation successful")
    print(f"  - Total loss: {loss_dict['total_loss'].item():.4f}")
    print(f"  - CLIP loss: {loss_dict['clip_loss'].item():.4f}")
    print(f"  - CKC loss: {loss_dict['ckc_loss'].item():.4f}")
    
    # Test retrieval metrics
    metrics = compute_retrieval_metrics(
        image_features=outputs['image_features'],
        text_features=outputs['text_features'],
    )
    
    print(f"\n✓ Retrieval metrics computed")
    print(f"  - I2T Recall@1: {metrics['i2t_recall@1']:.2f}%")
    print(f"  - T2I Recall@1: {metrics['t2i_recall@1']:.2f}%")


def test_lora_merging(model):
    """Test LoRA weight merging."""
    print("\n" + "="*60)
    print("Testing LoRA Merging")
    print("="*60 + "\n")
    
    # Save model state before merging
    old_state = {k: v.clone() for k, v in model.clip.visual.state_dict().items() if 'weight' in k}
    
    # Merge LoRA
    model.merge_lora_after_task()
    
    # Check that weights have changed
    new_state = {k: v.clone() for k, v in model.clip.visual.state_dict().items() if 'weight' in k}
    
    # Compare a few weights
    changed = False
    for key in list(old_state.keys())[:3]:
        if not torch.allclose(old_state[key], new_state[key]):
            changed = True
            break
    
    print(f"✓ LoRA merging successful")
    print(f"  - Weights changed: {changed}")
    print(f"  - LoRA layers cleared: {len(model.lora_layers) == 0}")


def test_continual_learning_cycle(model):
    """Test a complete continual learning cycle."""
    print("\n" + "="*60)
    print("Testing Continual Learning Cycle")
    print("="*60 + "\n")
    
    device = next(model.parameters()).device
    
    # Task 2
    print("Starting Task 2...")
    model.inject_lora_for_new_task()
    
    # Create dummy data
    batch_size = 4
    images = torch.randn(batch_size, 3, 224, 224).to(device)
    text = torch.randint(0, 49408, (batch_size, 77)).to(device)
    
    # Forward pass with old features
    with torch.no_grad():
        outputs = model(images, text, return_old_features=True)
    
    print(f"✓ Task 2 forward pass successful")
    print(f"  - Has old features: {'old_image_features' in outputs}")
    
    # Test CKC loss
    criterion = CCLIPLoss(temperature=0.07, use_ckc=True)
    
    loss_dict = criterion(
        image_features=outputs['image_features'],
        text_features=outputs['text_features'],
        projected_image_features=outputs['projected_image_features'],
        projected_text_features=outputs['projected_text_features'],
        old_image_features=outputs['old_image_features'],
        old_text_features=outputs['old_text_features'],
    )
    
    print(f"✓ CKC loss computation successful")
    print(f"  - Total loss: {loss_dict['total_loss'].item():.4f}")
    print(f"  - CLIP loss: {loss_dict['clip_loss'].item():.4f}")
    print(f"  - CKC loss: {loss_dict['ckc_loss'].item():.4f}")
    
    # Merge
    model.merge_lora_after_task()
    print(f"✓ Task 2 completed and merged")


def test_checkpoint_save_load(model):
    """Test checkpoint saving and loading."""
    print("\n" + "="*60)
    print("Testing Checkpoint Save/Load")
    print("="*60 + "\n")
    
    # Save checkpoint
    checkpoint_path = "test_checkpoint.pt"
    model.save_checkpoint(checkpoint_path)
    print(f"✓ Checkpoint saved to {checkpoint_path}")
    
    # Load checkpoint
    device = next(model.parameters()).device
    new_model = CCLIP(
        clip_model_name="ViT-B-32",
        pretrained="openai",
        lora_r=8,
        device=device,
    )
    new_model.load_checkpoint(checkpoint_path)
    print(f"✓ Checkpoint loaded successfully")
    print(f"  - Current task: {new_model.current_task}")
    
    # Clean up
    os.remove(checkpoint_path)
    print(f"✓ Test checkpoint removed")


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("C-CLIP Implementation Test Suite")
    print("="*60 + "\n")
    
    try:
        # Test 1: Model initialization
        model = test_model_initialization()
        
        # Test 2: LoRA injection
        test_lora_injection(model)
        
        # Test 3: Forward pass
        outputs = test_forward_pass(model)
        
        # Test 4: Loss computation
        test_loss_computation(outputs)
        
        # Test 5: LoRA merging
        test_lora_merging(model)
        
        # Test 6: Continual learning cycle
        test_continual_learning_cycle(model)
        
        # Test 7: Checkpoint save/load
        test_checkpoint_save_load(model)
        
        print("\n" + "="*60)
        print("✓ All tests passed successfully!")
        print("="*60 + "\n")
        
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())
