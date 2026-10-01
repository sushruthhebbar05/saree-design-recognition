from __future__ import annotations
import torch
from torch import nn
import torch.nn.functional as F

class SupervisedContrastiveLoss(nn.Module):
    """Supervised Contrastive Loss.
    
    Pulls same-class embeddings together and pushes different-class embeddings apart.
    """
    
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute loss.
        
        Args:
            features: Embeddings, shape (B, D).
            labels: Class labels, shape (B,).
            
        Returns:
            Scalar loss.
        """
        features = F.normalize(features, dim=1)
        logits = features @ features.T / self.temperature
        
        # Positive mask: same class, excluding self
        mask = (labels.unsqueeze(1) == labels.unsqueeze(0)).float()
        mask = mask - torch.eye(labels.shape[0], device=features.device)
        mask = mask.clamp(min=0.0)
        
        # Stability
        logits = logits - logits.max(dim=1, keepdim=True).values.detach()
        
        # Exponentials
        exp_logits = torch.exp(logits) * (1 - torch.eye(labels.shape[0], device=features.device))
        
        # Log probabilities
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-8)
        
        # Mean log probability of positives
        mean_log_prob_pos = (mask * log_prob).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        
        return -mean_log_prob_pos.mean()
