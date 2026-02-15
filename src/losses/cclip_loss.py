"""
Loss functions for C-CLIP: CLIP Loss and Contrastive Knowledge Consolidation (CKC) Loss.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict


class CLIPLoss(nn.Module):
    """
    Standard CLIP contrastive loss (InfoNCE).
    Computes bidirectional contrastive loss between image and text features.
    
    Args:
        temperature: Temperature parameter for softmax (default: 0.07)
    """
    
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
        
    def forward(
        self,
        image_features: torch.Tensor,
        text_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute CLIP loss.
        
        Args:
            image_features: Normalized image features (B, D)
            text_features: Normalized text features (B, D)
            
        Returns:
            CLIP loss value
        """
        batch_size = image_features.shape[0]
        device = image_features.device
        
        # Compute similarity matrix: (B, B)
        logits = image_features @ text_features.T / self.temperature
        
        # Labels: diagonal elements are positive pairs
        labels = torch.arange(batch_size, device=device)
        
        # Image-to-text loss
        loss_i2t = F.cross_entropy(logits, labels)
        
        # Text-to-image loss
        loss_t2i = F.cross_entropy(logits.T, labels)
        
        # Average bidirectional loss
        loss = (loss_i2t + loss_t2i) / 2.0
        
        return loss


class ContrastiveKnowledgeConsolidationLoss(nn.Module):
    """
    Contrastive Knowledge Consolidation (CKC) Loss.
    
    This loss treats old model features as positive anchors and performs
    contrastive learning between projected new features and old features.
    
    The key innovation is that instead of just preserving old features,
    it learns a better feature space from the old model through contrastive learning.
    
    Args:
        temperature: Temperature parameter for softmax (default: 0.07)
    """
    
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
        
    def forward(
        self,
        projected_image_features: torch.Tensor,
        projected_text_features: torch.Tensor,
        old_image_features: torch.Tensor,
        old_text_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute CKC loss.
        
        According to the paper:
        - Concatenate [image_features, text_features] to create 2N samples
        - Treat corresponding old features as positives
        - All other samples (including cross-modal) as negatives
        - This creates 2N^2 contrastive pairs per batch
        
        Args:
            projected_image_features: Projected image features from new model (B, D)
            projected_text_features: Projected text features from new model (B, D)
            old_image_features: Image features from old model (B, D)
            old_text_features: Text features from old model (B, D)
            
        Returns:
            CKC loss value
        """
        batch_size = projected_image_features.shape[0]
        device = projected_image_features.device
        
        # Concatenate features: [vision, text] -> (2B, D)
        # This creates 2N samples as mentioned in the paper
        h_new = torch.cat([projected_image_features, projected_text_features], dim=0)  # (2B, D)
        z_old = torch.cat([old_image_features, old_text_features], dim=0)  # (2B, D)
        
        # Normalize features
        h_new = F.normalize(h_new, p=2, dim=-1)
        z_old = F.normalize(z_old, p=2, dim=-1)
        
        # Compute similarity matrix: (2B, 2B)
        # Each new feature compared with all old features
        logits = h_new @ z_old.T / self.temperature  # (2B, 2B)
        
        # Labels: diagonal elements are positive pairs
        # i-th new feature should match i-th old feature
        labels = torch.arange(2 * batch_size, device=device)
        
        # Bidirectional contrastive loss
        # Loss for h_new -> z_old
        loss_new_to_old = F.cross_entropy(logits, labels)
        
        # Loss for z_old -> h_new
        loss_old_to_new = F.cross_entropy(logits.T, labels)
        
        # Average bidirectional loss
        loss = (loss_new_to_old + loss_old_to_new) / 2.0
        
        return loss


class CCLIPLoss(nn.Module):
    """
    Combined C-CLIP loss: CLIP Loss + CKC Loss.
    
    Total loss: L = L_CLIP + L_CKC
    
    Args:
        temperature: Temperature parameter for contrastive losses
        use_ckc: Whether to use CKC loss (set False for first task)
    """
    
    def __init__(
        self,
        temperature: float = 0.07,
        use_ckc: bool = True,
    ):
        super().__init__()
        
        self.clip_loss = CLIPLoss(temperature=temperature)
        self.ckc_loss = ContrastiveKnowledgeConsolidationLoss(temperature=temperature)
        self.use_ckc = use_ckc
        
    def forward(
        self,
        image_features: torch.Tensor,
        text_features: torch.Tensor,
        projected_image_features: Optional[torch.Tensor] = None,
        projected_text_features: Optional[torch.Tensor] = None,
        old_image_features: Optional[torch.Tensor] = None,
        old_text_features: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute total C-CLIP loss.
        
        Args:
            image_features: Current model image features (B, D)
            text_features: Current model text features (B, D)
            projected_image_features: Projected image features (B, D)
            projected_text_features: Projected text features (B, D)
            old_image_features: Old model image features (B, D)
            old_text_features: Old model text features (B, D)
            
        Returns:
            Dictionary containing:
                - total_loss: Combined loss
                - clip_loss: CLIP loss component
                - ckc_loss: CKC loss component (0 if not used)
        """
        # Compute CLIP loss
        clip_loss_value = self.clip_loss(image_features, text_features)
        
        # Compute CKC loss if enabled and old features available
        # Use a proper zero tensor that maintains gradient tracking
        ckc_loss_value = image_features.sum() * 0.0
        
        if self.use_ckc and old_image_features is not None:
            if projected_image_features is None or projected_text_features is None:
                raise ValueError("Projected features required for CKC loss")
            
            ckc_loss_value = self.ckc_loss(
                projected_image_features=projected_image_features,
                projected_text_features=projected_text_features,
                old_image_features=old_image_features,
                old_text_features=old_text_features,
            )
        
        # Total loss
        total_loss = clip_loss_value + ckc_loss_value
        
        return {
            'total_loss': total_loss,
            'clip_loss': clip_loss_value,
            'ckc_loss': ckc_loss_value,
        }


def compute_retrieval_metrics(
    image_features: torch.Tensor,
    text_features: torch.Tensor,
) -> Dict[str, float]:
    """
    Compute image-text retrieval metrics (Recall@1).
    
    Args:
        image_features: Image features (B, D)
        text_features: Text features (B, D)
        
    Returns:
        Dictionary with i2t_recall@1 and t2i_recall@1
    """
    batch_size = image_features.shape[0]
    
    # Compute similarity matrix
    similarity = image_features @ text_features.T  # (B, B)
    
    # Image-to-text retrieval (each image retrieves text)
    i2t_ranks = similarity.argsort(dim=1, descending=True)
    i2t_correct = (i2t_ranks[:, 0] == torch.arange(batch_size, device=i2t_ranks.device)).sum().item()
    i2t_recall = i2t_correct / batch_size
    
    # Text-to-image retrieval (each text retrieves image)
    t2i_ranks = similarity.T.argsort(dim=1, descending=True)
    t2i_correct = (t2i_ranks[:, 0] == torch.arange(batch_size, device=t2i_ranks.device)).sum().item()
    t2i_recall = t2i_correct / batch_size
    
    return {
        'i2t_recall@1': i2t_recall * 100.0,
        't2i_recall@1': t2i_recall * 100.0,
    }
