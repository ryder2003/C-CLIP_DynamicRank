"""
Evaluation utilities for C-CLIP.
"""

import torch
import torch.nn.functional as F
from typing import Dict, List, Optional
from tqdm import tqdm
import numpy as np


@torch.no_grad()
def evaluate_retrieval(
    model,
    dataloader,
    device: str = 'cuda',
) -> Dict[str, float]:
    """
    Evaluate image-text retrieval performance.
    
    Args:
        model: C-CLIP model
        dataloader: Validation dataloader
        device: Device to use
        
    Returns:
        Dictionary with retrieval metrics:
            - i2t_recall@1, i2t_recall@5, i2t_recall@10
            - t2i_recall@1, t2i_recall@5, t2i_recall@10
    """
    model.eval()
    
    # Collect all features
    all_image_features = []
    all_text_features = []
    
    for images, text in tqdm(dataloader, desc="Extracting features"):
        images = images.to(device)
        text = text.to(device)
        
        # Get features
        image_features = model.encode_image(images, normalize=True)
        text_features = model.encode_text(text, normalize=True)
        
        all_image_features.append(image_features.cpu())
        all_text_features.append(text_features.cpu())
    
    # Concatenate all features
    all_image_features = torch.cat(all_image_features, dim=0)
    all_text_features = torch.cat(all_text_features, dim=0)
    
    # Compute similarity matrix
    similarity = all_image_features @ all_text_features.T  # (N, N)
    
    # Image-to-text retrieval
    i2t_ranks = []
    for i in range(len(all_image_features)):
        # Get sorted indices by similarity
        sorted_indices = similarity[i].argsort(descending=True)
        # Find rank of correct text (i)
        rank = (sorted_indices == i).nonzero(as_tuple=True)[0].item()
        i2t_ranks.append(rank)
    
    i2t_ranks = np.array(i2t_ranks)
    i2t_recall_1 = (i2t_ranks < 1).mean() * 100
    i2t_recall_5 = (i2t_ranks < 5).mean() * 100
    i2t_recall_10 = (i2t_ranks < 10).mean() * 100
    
    # Text-to-image retrieval
    t2i_ranks = []
    for i in range(len(all_text_features)):
        # Get sorted indices by similarity
        sorted_indices = similarity[:, i].argsort(descending=True)
        # Find rank of correct image (i)
        rank = (sorted_indices == i).nonzero(as_tuple=True)[0].item()
        t2i_ranks.append(rank)
    
    t2i_ranks = np.array(t2i_ranks)
    t2i_recall_1 = (t2i_ranks < 1).mean() * 100
    t2i_recall_5 = (t2i_ranks < 5).mean() * 100
    t2i_recall_10 = (t2i_ranks < 10).mean() * 100
    
    metrics = {
        'i2t_recall@1': i2t_recall_1,
        'i2t_recall@5': i2t_recall_5,
        'i2t_recall@10': i2t_recall_10,
        't2i_recall@1': t2i_recall_1,
        't2i_recall@5': t2i_recall_5,
        't2i_recall@10': t2i_recall_10,
    }
    
    return metrics


@torch.no_grad()
def evaluate_zero_shot_classification(
    model,
    dataloader,
    class_names: List[str],
    templates: Optional[List[str]] = None,
    device: str = 'cuda',
) -> Dict[str, float]:
    """
    Evaluate zero-shot image classification.
    
    Args:
        model: C-CLIP model
        dataloader: Dataloader with (image, label) pairs
        class_names: List of class names
        templates: List of prompt templates (e.g., ["a photo of a {}"])
        device: Device to use
        
    Returns:
        Dictionary with classification accuracy
    """
    model.eval()
    
    # Default templates
    if templates is None:
        templates = [
            "a photo of a {}.",
            "a rendering of a {}.",
            "a cropped photo of the {}.",
            "the photo of a {}.",
            "a photo of a clean {}.",
            "a photo of a dirty {}.",
            "a dark photo of the {}.",
            "a photo of my {}.",
            "a photo of the cool {}.",
            "a close-up photo of a {}.",
            "a bright photo of the {}.",
            "a cropped photo of a {}.",
            "a photo of the {}.",
            "a good photo of the {}.",
            "a photo of one {}.",
            "a close-up photo of the {}.",
            "a rendition of the {}.",
            "a photo of the clean {}.",
            "a rendition of a {}.",
            "a photo of a nice {}.",
            "a good photo of a {}.",
            "a photo of the nice {}.",
            "a photo of the small {}.",
            "a photo of the weird {}.",
            "a photo of the large {}.",
            "a photo of a cool {}.",
            "a photo of a small {}.",
        ]
    
    # Generate text features for all classes
    tokenizer = model.clip.tokenizer
    all_class_features = []
    
    for class_name in tqdm(class_names, desc="Encoding class names"):
        # Generate prompts for this class
        texts = [template.format(class_name) for template in templates]
        text_tokens = tokenizer(texts).to(device)
        
        # Encode texts
        text_features = model.encode_text(text_tokens, normalize=True)
        
        # Average across templates
        text_features = text_features.mean(dim=0)
        text_features = text_features / text_features.norm()
        
        all_class_features.append(text_features)
    
    all_class_features = torch.stack(all_class_features, dim=0)  # (num_classes, embed_dim)
    
    # Classify images
    correct = 0
    total = 0
    
    for images, labels in tqdm(dataloader, desc="Classifying images"):
        images = images.to(device)
        labels = labels.to(device)
        
        # Encode images
        image_features = model.encode_image(images, normalize=True)
        
        # Compute similarity with all classes
        similarity = image_features @ all_class_features.T  # (batch_size, num_classes)
        
        # Get predictions
        predictions = similarity.argmax(dim=1)
        
        correct += (predictions == labels).sum().item()
        total += len(labels)
    
    accuracy = correct / total * 100
    
    return {
        'accuracy': accuracy,
        'correct': correct,
        'total': total,
    }


def compute_forgetting_metrics(
    current_accuracies: List[float],
    initial_accuracies: List[float],
) -> Dict[str, float]:
    """
    Compute forgetting metrics for continual learning.
    
    Args:
        current_accuracies: Current accuracies on all tasks
        initial_accuracies: Initial accuracies on tasks when first learned
        
    Returns:
        Dictionary with forgetting metrics
    """
    # Performance degradation
    degradation = [initial - current for initial, current in zip(initial_accuracies, current_accuracies)]
    
    # Average forgetting (only on learned tasks)
    avg_forgetting = np.mean(degradation) if degradation else 0.0
    
    # Maximum forgetting
    max_forgetting = max(degradation) if degradation else 0.0
    
    # Backward transfer (negative forgetting means positive transfer)
    backward_transfer = -avg_forgetting
    
    return {
        'avg_forgetting': avg_forgetting,
        'max_forgetting': max_forgetting,
        'backward_transfer': backward_transfer,
    }
