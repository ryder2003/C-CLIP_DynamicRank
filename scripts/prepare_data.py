"""
Example script for preparing a simple dataset for C-CLIP training.
This demonstrates how to create a dataset from images and text files.
"""

import os
import pandas as pd
import argparse
from pathlib import Path


def create_csv_dataset(
    image_dir: str,
    output_csv: str,
    caption_file: str = None,
    caption_suffix: str = ".txt",
):
    """
    Create a CSV dataset file from a directory of images.
    
    Args:
        image_dir: Directory containing images
        output_csv: Path to output CSV file
        caption_file: Optional JSON file mapping images to captions
        caption_suffix: Suffix for caption files (if using paired files)
    """
    image_dir = Path(image_dir)
    
    # Find all image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    image_files = []
    for ext in image_extensions:
        image_files.extend(image_dir.glob(f'**/*{ext}'))
    
    print(f"Found {len(image_files)} images in {image_dir}")
    
    # Create data list
    data = []
    missing_captions = 0
    
    for img_path in image_files:
        # Get relative path from image_dir
        rel_path = img_path.relative_to(image_dir)
        
        # Look for corresponding caption file
        caption_path = img_path.with_suffix(caption_suffix)
        
        if caption_path.exists():
            with open(caption_path, 'r', encoding='utf-8') as f:
                caption = f.read().strip()
            
            data.append({
                'image': str(rel_path),
                'caption': caption
            })
        else:
            missing_captions += 1
            # Use filename as caption (fallback)
            caption = img_path.stem.replace('_', ' ').replace('-', ' ')
            data.append({
                'image': str(rel_path),
                'caption': caption
            })
    
    if missing_captions > 0:
        print(f"Warning: {missing_captions} images without caption files")
    
    # Create DataFrame and save
    df = pd.DataFrame(data)
    
    # Create output directory if needed
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    
    # Save CSV
    df.to_csv(output_csv, index=False)
    print(f"Saved {len(df)} image-caption pairs to {output_csv}")
    
    # Print sample
    print("\nSample entries:")
    print(df.head())


def split_dataset(
    input_csv: str,
    train_csv: str,
    val_csv: str,
    val_ratio: float = 0.2,
    random_seed: int = 42,
):
    """
    Split a dataset into train and validation sets.
    
    Args:
        input_csv: Input CSV file
        train_csv: Output training CSV file
        val_csv: Output validation CSV file
        val_ratio: Ratio of validation set (default: 0.2)
        random_seed: Random seed for reproducibility
    """
    import numpy as np
    
    # Load data
    df = pd.read_csv(input_csv)
    print(f"Loaded {len(df)} samples from {input_csv}")
    
    # Shuffle
    df = df.sample(frac=1, random_state=random_seed).reset_index(drop=True)
    
    # Split
    val_size = int(len(df) * val_ratio)
    train_size = len(df) - val_size
    
    train_df = df.iloc[:train_size]
    val_df = df.iloc[train_size:]
    
    # Save
    os.makedirs(os.path.dirname(train_csv), exist_ok=True)
    os.makedirs(os.path.dirname(val_csv), exist_ok=True)
    
    train_df.to_csv(train_csv, index=False)
    val_df.to_csv(val_csv, index=False)
    
    print(f"Training set: {len(train_df)} samples -> {train_csv}")
    print(f"Validation set: {len(val_df)} samples -> {val_csv}")


def main():
    parser = argparse.ArgumentParser(description='Prepare dataset for C-CLIP')
    parser.add_argument('--image_dir', type=str, required=True, help='Directory containing images')
    parser.add_argument('--output_csv', type=str, required=True, help='Output CSV file')
    parser.add_argument('--split', action='store_true', help='Split into train/val')
    parser.add_argument('--val_ratio', type=float, default=0.2, help='Validation ratio')
    
    args = parser.parse_args()
    
    # Create CSV
    create_csv_dataset(
        image_dir=args.image_dir,
        output_csv=args.output_csv,
    )
    
    # Split if requested
    if args.split:
        base_path = args.output_csv.replace('.csv', '')
        train_csv = f"{base_path}_train.csv"
        val_csv = f"{base_path}_val.csv"
        
        split_dataset(
            input_csv=args.output_csv,
            train_csv=train_csv,
            val_csv=val_csv,
            val_ratio=args.val_ratio,
        )


if __name__ == '__main__':
    main()
