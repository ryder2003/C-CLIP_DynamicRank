"""
Dataset classes for C-CLIP continual learning.
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from PIL import Image
import pandas as pd
from typing import List, Optional, Dict, Tuple
import json

from .transforms import get_clip_transforms


class ImageTextDataset(Dataset):
    """
    Generic image-text dataset for continual learning.
    
    Supports multiple formats:
    1. CSV file with columns: 'image', 'caption'
    2. JSON file with list of {'image': path, 'caption': text}
    3. Directory structure with paired image and text files
    
    Args:
        data_path: Path to data file or directory
        image_dir: Directory containing images (if paths in data are relative)
        transform: Image transformations
        tokenizer: Text tokenizer
        max_text_length: Maximum text sequence length
    """
    
    def __init__(
        self,
        data_path: str,
        image_dir: Optional[str] = None,
        transform=None,
        tokenizer=None,
        max_text_length: int = 77,
    ):
        super().__init__()
        
        self.data_path = data_path
        self.image_dir = image_dir or ""
        self.transform = transform
        self.tokenizer = tokenizer
        self.max_text_length = max_text_length
        
        # Load data
        self.data = self._load_data()
        
        print(f"Loaded {len(self.data)} image-text pairs from {data_path}")
        
    def _load_data(self) -> List[Dict[str, str]]:
        """Load data from file."""
        data = []
        
        # Check file extension
        if self.data_path.endswith('.csv'):
            df = pd.read_csv(self.data_path)
            for _, row in df.iterrows():
                data.append({
                    'image': row['image'],
                    'caption': row['caption']
                })
        
        elif self.data_path.endswith('.json'):
            with open(self.data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        
        elif os.path.isdir(self.data_path):
            # Assume paired files in directory
            image_files = sorted([f for f in os.listdir(self.data_path) 
                                if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
            for img_file in image_files:
                # Look for corresponding text file
                txt_file = os.path.splitext(img_file)[0] + '.txt'
                txt_path = os.path.join(self.data_path, txt_file)
                
                if os.path.exists(txt_path):
                    with open(txt_path, 'r', encoding='utf-8') as f:
                        caption = f.read().strip()
                    
                    data.append({
                        'image': img_file,
                        'caption': caption
                    })
        
        else:
            raise ValueError(f"Unsupported data format: {self.data_path}")
        
        return data
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get image-text pair.
        
        Returns:
            Tuple of (image_tensor, text_tokens)
        """
        item = self.data[idx]
        
        # Load image
        image_path = os.path.join(self.image_dir, item['image'])
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            # Return a blank image
            image = Image.new('RGB', (224, 224), (0, 0, 0))
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        # Tokenize text
        caption = item['caption']
        if self.tokenizer:
            # For open_clip tokenizer
            text_tokens = self.tokenizer(caption)[0]  # Returns tensor
        else:
            # Fallback: just return the caption string
            text_tokens = caption
        
        return image, text_tokens


class ClassificationDataset(Dataset):
    """
    Image classification dataset for zero-shot evaluation.
    Reads a CSV with columns 'image' and 'caption'; derives integer class labels
    from the caption text (unique captions → unique classes).

    Args:
        data_path:  Path to CSV file (columns: image, caption)
        image_dir:  Directory containing images
        transform:  Image transforms
    """

    def __init__(
        self,
        data_path: str,
        image_dir: Optional[str] = None,
        transform=None,
    ):
        super().__init__()
        self.image_dir = image_dir or ""
        self.transform = transform

        df = pd.read_csv(data_path)
        # Build a deterministic class → int mapping from unique captions
        unique_captions = sorted(df["caption"].unique().tolist())
        self.class_names: List[str] = unique_captions            # full caption strings
        self.caption_to_idx: Dict[str, int] = {
            c: i for i, c in enumerate(unique_captions)
        }

        self.records = [
            {"image": row["image"], "label": self.caption_to_idx[row["caption"]]}
            for _, row in df.iterrows()
        ]
        print(
            f"ClassificationDataset: {len(self.records)} images, "
            f"{len(self.class_names)} classes  ({data_path})"
        )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        item = self.records[idx]
        image_path = os.path.join(self.image_dir, item["image"])
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception:
            image = Image.new("RGB", (224, 224), (0, 0, 0))
        if self.transform:
            image = self.transform(image)
        return image, item["label"]


class ContinualLearningDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for continual learning with multiple datasets.
    
    Args:
        dataset_configs: List of dataset configurations, each with:
            - name: Dataset name
            - train_path: Path to training data
            - val_path: Path to validation data
            - image_dir: Directory containing images
        tokenizer: Text tokenizer
        batch_size: Batch size for training
        num_workers: Number of data loading workers
        image_size: Image size
        max_text_length: Maximum text sequence length
    """
    
    def __init__(
        self,
        dataset_configs: List[Dict],
        tokenizer,
        batch_size: int = 256,
        num_workers: int = 4,
        image_size: int = 224,
        max_text_length: int = 77,
    ):
        super().__init__()
        
        self.dataset_configs = dataset_configs
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_size
        self.max_text_length = max_text_length
        
        # Current task index
        self.current_task_idx = 0
        
        # Store datasets
        self.train_datasets = []
        self.val_datasets = []
        
        # Transforms
        self.train_transform = get_clip_transforms(image_size, is_train=True)
        self.val_transform = get_clip_transforms(image_size, is_train=False)
        
    def setup(self, stage: Optional[str] = None):
        """
        Setup datasets for all tasks.
        """
        for config in self.dataset_configs:
            # Training dataset
            train_dataset = ImageTextDataset(
                data_path=config['train_path'],
                image_dir=config.get('image_dir', None),
                transform=self.train_transform,
                tokenizer=self.tokenizer,
                max_text_length=self.max_text_length,
            )
            self.train_datasets.append(train_dataset)
            
            # Validation dataset
            if 'val_path' in config:
                val_dataset = ImageTextDataset(
                    data_path=config['val_path'],
                    image_dir=config.get('image_dir', None),
                    transform=self.val_transform,
                    tokenizer=self.tokenizer,
                    max_text_length=self.max_text_length,
                )
                self.val_datasets.append(val_dataset)
            else:
                self.val_datasets.append(None)
        
        print(f"Setup complete: {len(self.train_datasets)} datasets ready")
    
    def set_task(self, task_idx: int):
        """Set the current task for continual learning."""
        self.current_task_idx = task_idx
        print(f"Switched to task {task_idx}: {self.dataset_configs[task_idx]['name']}")
    
    def train_dataloader(self) -> DataLoader:
        """Get training dataloader for current task."""
        return DataLoader(
            self.train_datasets[self.current_task_idx],
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,  # Important for contrastive learning
        )
    
    def val_dataloader(self) -> Optional[DataLoader]:
        """Get validation dataloader for current task."""
        val_dataset = self.val_datasets[self.current_task_idx]
        if val_dataset is None:
            return None
        
        return DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
    
    def get_all_val_dataloaders(self) -> List[DataLoader]:
        """
        Get validation dataloaders for all tasks (for evaluation).
        """
        dataloaders = []
        for val_dataset in self.val_datasets:
            if val_dataset is not None:
                dataloader = DataLoader(
                    val_dataset,
                    batch_size=self.batch_size,
                    shuffle=False,
                    num_workers=self.num_workers,
                    pin_memory=True,
                )
                dataloaders.append(dataloader)
            else:
                dataloaders.append(None)
        
        return dataloaders
    
    def get_num_tasks(self) -> int:
        """Get total number of tasks."""
        return len(self.dataset_configs)
