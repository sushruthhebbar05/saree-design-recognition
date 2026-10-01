from __future__ import annotations
import argparse
import pandas as pd
import torch

from src.models.embedding_model import EmbeddingModel
from src.retrieval.embed import embed_paths
from src.retrieval.search import search
from src.utils.config import load_config

def main():
    parser = argparse.ArgumentParser(description="Retrieve similar designs from gallery.")
    parser.add_argument("--query", required=True, help="Query image path")
    parser.add_argument("--gallery", required=True, help="Gallery CSV path")
    parser.add_argument("--checkpoint", required=True, help="Model checkpoint path")
    parser.add_argument("--root", default=".", help="Root directory for paths")
    parser.add_argument("--top-k", type=int, default=5, help="Number of results")
    args = parser.parse_args()

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    config = ckpt.get("config", load_config("configs/default.yaml"))

    # Load model
    model = EmbeddingModel(
        backbone=config["model"]["backbone"],
        pretrained=config["model"]["pretrained"],
        embedding_dim=config["model"]["embedding_dim"],
        dropout=config["model"]["dropout"]
    )
    model.load_state_dict(ckpt["model"])
    model.eval()

    # Embed gallery and query
    gallery_df = pd.read_csv(args.gallery)
    print(f"Loading gallery ({len(gallery_df)} images)...")
    gallery_emb = embed_paths(
        model,
        gallery_df["image_path"].tolist(),
        root=args.root,
        size=config["image"]["size"]
    )

    print(f"Embedding query...")
    query_emb = embed_paths(
        model,
        [args.query],
        root=args.root,
        size=config["image"]["size"]
    )

    if query_emb.shape[0] == 0:
        print("Error: Failed to embed query image")
        return

    # Search
    scores, indices = search(query_emb[0], gallery_emb, top_k=args.top_k)

    # Display results
    print(f"\n{'Rank':<6} {'Image':<50} {'Design ID':<20} {'Similarity':<12}")
    print("-" * 88)
    for rank, (score, idx) in enumerate(zip(scores, indices), start=1):
        row = gallery_df.iloc[int(idx)]
        print(
            f"{rank:<6} {row['image_path']:<50} {str(row['design_id']):<20} {float(score):.4f}"
        )

if __name__ == "__main__":
    main()
