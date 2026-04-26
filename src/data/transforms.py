"""
Data transformations for C-CLIP.
"""

from torchvision import transforms
from typing import Tuple


def get_clip_transforms(
    image_size: int = 224,
    is_train: bool = True,
) -> transforms.Compose:
    """
    Get image transformations for CLIP.
    
    Args:
        image_size: Target image size (default: 224)
        is_train: Whether for training (with augmentation) or validation
        
    Returns:
        Composed transformations
    """
    # Mean and std for CLIP (OpenAI)
    mean = (0.48145466, 0.4578275, 0.40821073)
    std = (0.26862954, 0.26130258, 0.27577711)
    
    if is_train:
        transform = transforms.Compose([
            transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
    
    return transform
