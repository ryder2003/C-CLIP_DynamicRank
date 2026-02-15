"""
Create a minimal sample dataset for testing C-CLIP.
This creates synthetic data so you can test training without downloading large datasets.
"""

import os
import pandas as pd
from PIL import Image
import numpy as np

def create_sample_dataset(output_dir='data/sample_dataset', num_samples=100):
    """
    Create a sample dataset with random images and captions.
    
    Args:
        output_dir: Directory to save the dataset
        num_samples: Number of samples to create
    """
    # Create directories
    train_dir = os.path.join(output_dir, 'images', 'train')
    val_dir = os.path.join(output_dir, 'images', 'val')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    
    # Sample captions
    objects = ['cat', 'dog', 'bird', 'car', 'tree', 'house', 'person', 'flower']
    actions = ['sitting', 'running', 'standing', 'flying', 'sleeping', 'playing']
    locations = ['in the park', 'on the beach', 'in the garden', 'by the river', 'in the city']
    
    # Create training data
    train_data = []
    for i in range(int(num_samples * 0.8)):
        # Create random colored image
        img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        img_pil = Image.fromarray(img)
        
        # Save image
        img_name = f'train_img_{i:04d}.jpg'
        img_path = os.path.join(train_dir, img_name)
        img_pil.save(img_path)
        
        # Generate caption
        obj = np.random.choice(objects)
        action = np.random.choice(actions)
        location = np.random.choice(locations)
        caption = f"A {obj} {action} {location}"
        
        train_data.append({
            'image': f'images/train/{img_name}',
            'caption': caption
        })
    
    # Create validation data
    val_data = []
    for i in range(int(num_samples * 0.2)):
        # Create random colored image
        img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        img_pil = Image.fromarray(img)
        
        # Save image
        img_name = f'val_img_{i:04d}.jpg'
        img_path = os.path.join(val_dir, img_name)
        img_pil.save(img_path)
        
        # Generate caption
        obj = np.random.choice(objects)
        action = np.random.choice(actions)
        location = np.random.choice(locations)
        caption = f"A {obj} {action} {location}"
        
        val_data.append({
            'image': f'images/val/{img_name}',
            'caption': caption
        })
    
    # Save CSVs
    train_df = pd.DataFrame(train_data)
    val_df = pd.DataFrame(val_data)
    
    train_csv = os.path.join(output_dir, 'train.csv')
    val_csv = os.path.join(output_dir, 'val.csv')
    
    train_df.to_csv(train_csv, index=False)
    val_df.to_csv(val_csv, index=False)
    
    print(f"✓ Created sample dataset in {output_dir}")
    print(f"  - Training samples: {len(train_data)}")
    print(f"  - Validation samples: {len(val_data)}")
    print(f"  - Train CSV: {train_csv}")
    print(f"  - Val CSV: {val_csv}")
    
    return output_dir


def create_multi_task_sample_datasets(base_dir='data', num_tasks=2, samples_per_task=100):
    """
    Create multiple sample datasets for continual learning testing.
    
    Args:
        base_dir: Base directory for datasets
        num_tasks: Number of tasks to create
        samples_per_task: Samples per task
    """
    print(f"\nCreating {num_tasks} sample datasets for continual learning...\n")
    
    task_configs = []
    
    for task_idx in range(num_tasks):
        task_name = f'task_{task_idx + 1}'
        task_dir = os.path.join(base_dir, task_name)
        
        create_sample_dataset(task_dir, samples_per_task)
        
        # Create config entry
        config = {
            'name': task_name,
            'train_path': os.path.join(task_dir, 'train.csv'),
            'val_path': os.path.join(task_dir, 'val.csv'),
            'image_dir': task_dir,
        }
        task_configs.append(config)
        print()
    
    # Print config for YAML
    print("="*60)
    print("Add this to your config YAML:")
    print("="*60)
    print("\ndatasets:")
    for config in task_configs:
        print(f"  - name: \"{config['name']}\"")
        print(f"    train_path: \"{config['train_path']}\"")
        print(f"    val_path: \"{config['val_path']}\"")
        print(f"    image_dir: \"{config['image_dir']}\"")
        print()
    
    return task_configs


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Create sample datasets for C-CLIP testing')
    parser.add_argument('--num_tasks', type=int, default=2, help='Number of tasks (datasets) to create')
    parser.add_argument('--samples_per_task', type=int, default=100, help='Number of samples per task')
    parser.add_argument('--output_dir', type=str, default='data', help='Output directory')
    
    args = parser.parse_args()
    
    create_multi_task_sample_datasets(
        base_dir=args.output_dir,
        num_tasks=args.num_tasks,
        samples_per_task=args.samples_per_task
    )
    
    print("="*60)
    print("✓ Sample datasets created!")
    print("="*60)
    print("\nNext steps:")
    print("1. Update configs/default_config.yaml with the dataset paths above")
    print("2. Run training: .venv\\Scripts\\python.exe src\\train.py --config configs\\default_config.yaml")
    print("\nNote: These are synthetic datasets for testing the pipeline.")
    print("For real training, download actual datasets (Flickr30K, COCO, etc.)")
