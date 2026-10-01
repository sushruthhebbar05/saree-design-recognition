from __future__ import annotations
import torch
import numpy as np
from torchvision import transforms
from typing import Any

def color_invariance_score(
    model: torch.nn.Module,
    image_tensor: torch.Tensor,
    device: str = "cuda",
) -> dict[str, Any]:
    """Evaluate color invariance by comparing embeddings under color transforms.
    
    Args:
        model: Embedding model.
        image_tensor: Single image tensor, shape (1, 3, H, W).
        device: Device to run on.
        
    Returns:
        Dictionary with invariance scores for each transform.
    """
    model.eval()
    image_tensor = image_tensor.to(device)
    
    transforms_list = [
        ("grayscale", transforms.Grayscale(num_output_channels=3)),
        ("hue_shift", transforms.ColorJitter(hue=0.3)),
        ("saturation", transforms.ColorJitter(saturation=0.5)),
        ("brightness", transforms.ColorJitter(brightness=0.3)),
    ]
    
    scores = {}
    with torch.no_grad():
        base_emb = model(image_tensor)
        
        for name, transform_fn in transforms_list:
            # Apply transform to PIL image
            pil_img = transforms.ToPILImage()(image_tensor[0].cpu())
            transformed = transform_fn(pil_img)
            transformed_tensor = transforms.ToTensor()(transformed).unsqueeze(0).to(device)
            
            # Normalize for ImageNet (approximate)
            transformed_tensor = transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )(transformed_tensor)
            
            transformed_emb = model(transformed_tensor)
            sim = float((base_emb * transformed_emb).sum(dim=1).mean())
            scores[name] = sim
    
    return {
        "color_invariance_scores": scores,
        "mean_invariance": float(np.mean(list(scores.values()))),
    }
