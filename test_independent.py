"""
Test the trained C-CLIP model on independent test dataset.
This dataset was NOT used during training or validation.
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

def test_on_independent_data():
    """Test model on completely independent dataset."""
    
    # Configuration
    checkpoint_path = "checkpoints/sample_test/model_after_task_0.pt"
    test_data_path = "data/test_independent/test.csv"
    test_image_dir = "data/test_independent"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print("="*70)
    print(" "*15 + "C-CLIP INDEPENDENT TEST")
    print("="*70)
    print(f"Device: {device}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Test Data: {test_data_path}")
    print()
    
    # Check if checkpoint exists
    if not os.path.exists(checkpoint_path):
        print(f"[ERROR] Checkpoint not found: {checkpoint_path}")
        print("Please train the model first.")
        return
    
    # Check if test data exists
    if not os.path.exists(test_data_path):
        print(f"[ERROR] Test data not found: {test_data_path}")
        print("Run: python generate_test_dataset.py")
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
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.inject_lora_for_new_task()
    model.clip.model.load_state_dict(checkpoint['clip_state_dict'], strict=False)
    
    if 'projector_state_dict' in checkpoint:
        model.image_projector.load_state_dict(checkpoint['projector_state_dict']['image'])
        model.text_projector.load_state_dict(checkpoint['projector_state_dict']['text'])
    
    model.eval()
    print("[OK] Model loaded successfully!")
    print()
    
    # Load test data
    print("="*70)
    print(" "*20 + "LOADING TEST DATA")
    print("="*70)
    
    transform = get_clip_transforms(image_size=224, is_train=False)
    
    dataset = ImageTextDataset(
        data_path=test_data_path,
        image_dir=test_image_dir,
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
    
    print(f"Total test samples: {len(dataset)}")
    print()
    
    # Show sample data
    print("Sample test captions:")
    with open(test_data_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()[1:6]  # Skip header, show first 5
        for i, line in enumerate(lines, 1):
            _, caption = line.strip().split(',', 1)
            print(f"  {i}. {caption}")
    print()
    
    # Evaluate
    print("="*70)
    print(" "*22 + "RUNNING EVALUATION")
    print("="*70)
    print("Extracting features and computing retrieval metrics...")
    print()
    
    with torch.no_grad():
        metrics = evaluate_retrieval(model, dataloader, device)
    
    # Print results
    print("="*70)
    print(" "*27 + "RESULTS")
    print("="*70)
    print()
    print("IMAGE-TO-TEXT RETRIEVAL:")
    print(f"   Recall@1:  {metrics['i2t_recall@1']:>6.2f}%")
    print(f"   Recall@5:  {metrics['i2t_recall@5']:>6.2f}%")
    print(f"   Recall@10: {metrics['i2t_recall@10']:>6.2f}%")
    print()
    print("TEXT-TO-IMAGE RETRIEVAL:")
    print(f"   Recall@1:  {metrics['t2i_recall@1']:>6.2f}%")
    print(f"   Recall@5:  {metrics['t2i_recall@5']:>6.2f}%")
    print(f"   Recall@10: {metrics['t2i_recall@10']:>6.2f}%")
    print()
    print("="*70)
    
    # Interpretation
    print()
    print("INTERPRETATION:")
    print(f"   - Test set size: {len(dataset)} samples")
    print(f"   - Recall@10: {metrics['i2t_recall@10']:.1f}% (for {len(dataset)} samples, top-10 may not be 100%)")
    print(f"   - This is UNSEEN data - never used in training or validation")
    print(f"   - Performance shows generalization capability")
    print()
    print("[COMPLETED] Testing finished successfully!")
    print("="*70)

if __name__ == '__main__':
    test_on_independent_data()
