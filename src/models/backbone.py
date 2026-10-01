from __future__ import annotations
from torch import nn
from torchvision.models import resnet18, ResNet18_Weights

def build_backbone(pretrained: bool = True) -> tuple[nn.Module, int]:
    """Build ResNet-18 backbone.
    
    Args:
        pretrained: Use ImageNet pretrained weights.
        
    Returns:
        (model, embedding_dim)
    """
    weights = ResNet18_Weights.DEFAULT if pretrained else None
    model = resnet18(weights=weights)
    embedding_dim = model.fc.in_features
    model.fc = nn.Identity()
    return model, embedding_dim
