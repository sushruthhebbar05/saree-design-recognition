import torch
from src.models.embedding_model import EmbeddingModel

def test_embedding_shape_and_norm():
    """Test embedding model output shape and L2 normalization."""
    model = EmbeddingModel(pretrained=False)
    x = torch.randn(2, 3, 224, 224)
    emb = model(x)
    
    assert emb.shape == (2, 256), f"Expected (2, 256), got {emb.shape}"
    assert torch.allclose(
        torch.norm(emb, dim=1), torch.ones(2), atol=1e-5
    ), "Embeddings should be L2-normalized"
    print("✓ test_embedding_shape_and_norm passed")

if __name__ == "__main__":
    test_embedding_shape_and_norm()
