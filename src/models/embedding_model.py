from __future__ import annotations
import torch
import torch.nn.functional as F
from torch import nn
from .backbone import build_backbone

class EmbeddingModel(nn.Module):
    """Metric learning embedding model.
    
    Encodes images into normalized 256-dimensional embeddings.
    """
    
    def __init__(
        self,
        backbone: str = "resnet18",
        pretrained: bool = True,
        embedding_dim: int = 256,
        dropout: float = 0.2,
    ):
        super().__init__()
        if backbone != "resnet18":
            raise ValueError(f"Unsupported backbone: {backbone}. Use 'resnet18'.")
        
        self.encoder, dim = build_backbone(pretrained=pretrained)
        self.projection = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, embedding_dim),
        )
        self.embedding_dim = embedding_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input images, shape (B, 3, H, W).
            
        Returns:
            L2-normalized embeddings, shape (B, 256).
        """
        features = self.projection(self.encoder(x))
        return F.normalize(features, p=2, dim=1)
