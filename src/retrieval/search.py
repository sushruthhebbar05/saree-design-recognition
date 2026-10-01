from __future__ import annotations
import torch

def search(
    query_embedding: torch.Tensor,
    gallery_embeddings: torch.Tensor,
    top_k: int = 10,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rank gallery by similarity to query using cosine distance.
    
    Args:
        query_embedding: Query embedding, shape (256,).
        gallery_embeddings: Gallery embeddings, shape (N, 256).
        top_k: Number of top results to return.
        
    Returns:
        (similarities, indices) of top-k results.
    """
    query_embedding = torch.as_tensor(query_embedding)
    gallery_embeddings = torch.as_tensor(gallery_embeddings)

    # Cosine similarity = dot product for normalized embeddings
    similarities = gallery_embeddings @ query_embedding
    
    top_k = min(top_k, similarities.shape[0])
    values, indices = torch.topk(similarities, k=top_k)
    
    return values, indices
