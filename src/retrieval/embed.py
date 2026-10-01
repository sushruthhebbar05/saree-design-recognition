from __future__ import annotations
import torch
from PIL import Image
from src.data.dataset import eval_transform

def embed_paths(
    model: torch.nn.Module,
    image_paths: list[str],
    root: str = ".",
    size: int = 224,
    batch_size: int = 32,
) -> torch.Tensor:
    """Embed a list of images.
    
    Args:
        model: Embedding model.
        image_paths: List of relative image paths.
        root: Root directory.
        size: Image size.
        batch_size: Batch size for processing.
        
    Returns:
        Embeddings, shape (N, 256).
    """
    model.eval()
    transform = eval_transform(size)
    device = next(model.parameters()).device
    embeddings = []

    with torch.no_grad():
        for i in range(0, len(image_paths), batch_size):
            batch = []
            for path in image_paths[i : i + batch_size]:
                try:
                    with Image.open(f"{root}/{path}") as img:
                        batch.append(transform(img.convert("RGB")))
                except Exception as e:
                    print(f"Warning: Failed to load {path}: {e}")
                    continue
            
            if batch:
                batch_tensor = torch.stack(batch).to(device)
                out = model(batch_tensor)
                embeddings.append(out.cpu())

    if not embeddings:
        return torch.empty((0, 256))

    return torch.cat(embeddings, dim=0)
