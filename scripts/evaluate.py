from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
import torch
import numpy as np
from tqdm import tqdm

from src.models.embedding_model import EmbeddingModel
from src.retrieval.embed import embed_paths
from src.retrieval.search import search
from src.evaluation.retrieval_metrics import retrieval_metrics, verification_metrics
from src.utils.config import load_config

def main():
    parser = argparse.ArgumentParser(description="Evaluate retrieval and verification.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", default=".")
    parser.add_argument("--output", default="outputs/eval_results.txt")
    args = parser.parse_args()

    config = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = EmbeddingModel(
        backbone=config["model"]["backbone"],
        pretrained=config["model"]["pretrained"],
        embedding_dim=config["model"]["embedding_dim"],
        dropout=config["model"]["dropout"]
    ).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"])

    # Load splits
    gallery_df = pd.read_csv(Path(config["data"]["splits"]) / "gallery.csv")
    query_df = pd.read_csv(Path(config["data"]["splits"]) / "query.csv")

    print("\n=== Embedding Gallery ===")
    gallery_emb = embed_paths(
        model,
        gallery_df["image_path"].tolist(),
        root=args.data_root,
        size=config["image"]["size"]
    )
    print(f"✓ Gallery: {gallery_emb.shape}")

    print("\n=== Embedding Queries ===")
    query_emb = embed_paths(
        model,
        query_df["image_path"].tolist(),
        root=args.data_root,
        size=config["image"]["size"]
    )
    print(f"✓ Query: {query_emb.shape}")

    # Retrieval
    print("\n=== Retrieval Evaluation ===")
    rankings = []
    for q_emb in tqdm(query_emb):
        _, indices = search(q_emb, gallery_emb, top_k=len(gallery_df))
        rankings.append(indices.cpu().numpy())

    metrics = retrieval_metrics(
        query_df["design_id"].astype(str).values,
        gallery_df["design_id"].astype(str).values,
        rankings
    )
    print(f"Top-1: {metrics['top1']:.4f}")
    print(f"Top-5: {metrics['top5']:.4f}")
    print(f"Top-10: {metrics['top10']:.4f}")
    print(f"MRR: {metrics['mrr']:.4f}")

    # Verification (sample)
    print("\n=== Verification Evaluation (sampled) ===")
    same_scores = []
    diff_scores = []

    query_design_ids = query_df["design_id"].astype(str).values
    gallery_design_ids = gallery_df["design_id"].astype(str).values

    for i, (q_emb, q_id) in enumerate(tqdm(zip(query_emb, query_design_ids), total=len(query_emb))):
        for j, g_id in enumerate(gallery_design_ids):
            sim = float((q_emb * gallery_emb[j]).sum())
            if q_id == g_id:
                same_scores.append(sim)
            else:
                diff_scores.append(sim)

    if same_scores and diff_scores:
        same_scores = np.array(same_scores)
        diff_scores = np.array(diff_scores)
        threshold = (same_scores.mean() + diff_scores.mean()) / 2

        y_true = np.concatenate([np.ones(len(same_scores)), np.zeros(len(diff_scores))])
        scores = np.concatenate([same_scores, diff_scores])

        vmetrics = verification_metrics(y_true, scores, threshold)
        print(f"ROC-AUC: {vmetrics['roc_auc']:.4f}")
        print(f"PR-AUC: {vmetrics['pr_auc']:.4f}")
        print(f"F1: {vmetrics['f1']:.4f}")
        print(f"EER: {vmetrics['eer']:.4f}")
        print(f"Threshold: {vmetrics['threshold']:.4f}")

    # Save results
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        f.write("=== Retrieval Metrics ===\n")
        for k, v in metrics.items():
            f.write(f"{k}: {v:.4f}\n")
        if same_scores and diff_scores:
            f.write("\n=== Verification Metrics ===\n")
            for k, v in vmetrics.items():
                f.write(f"{k}: {v:.4f}\n")
    print(f"\n✓ Results saved to {args.output}")

if __name__ == "__main__":
    main()
