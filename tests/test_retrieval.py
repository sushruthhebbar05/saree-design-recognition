import torch
from src.retrieval.search import search

def test_search():
    """Test search function."""
    query = torch.tensor([1.0, 0.0])
    gallery = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]])
    values, indices = search(query, gallery, top_k=2)
    
    assert values.shape[0] == 2, f"Expected 2 results, got {values.shape[0]}"
    assert indices[0].item() == 0, "First result should be most similar"
    print("✓ test_search passed")

if __name__ == "__main__":
    test_search()
