from __future__ import annotations
import numpy as np
from typing import Any

def retrieval_metrics(
    query_ids: np.ndarray,
    gallery_ids: np.ndarray,
    rankings: list[np.ndarray],
) -> dict[str, Any]:
    """Compute retrieval metrics.
    
    Args:
        query_ids: Query design IDs.
        gallery_ids: Gallery design IDs.
        rankings: List of ranked gallery indices per query.
        
    Returns:
        Dictionary with Top-1/5/10, MRR metrics.
    """
    ranks = []
    query_ids = np.asarray(query_ids)
    gallery_ids = np.asarray(gallery_ids)

    for q, order in zip(query_ids, rankings):
        hit_positions = np.where(gallery_ids[order] == q)[0]
        if len(hit_positions) == 0:
            ranks.append(10**9)
        else:
            ranks.append(int(hit_positions[0]) + 1)

    ranks = np.asarray(ranks)

    return {
        "top1": float(np.mean(ranks <= 1)),
        "top5": float(np.mean(ranks <= 5)),
        "top10": float(np.mean(ranks <= 10)),
        "mrr": float(np.mean([1.0 / r if r < 10**9 else 0.0 for r in ranks])),
    }

def verification_metrics(
    y_true: np.ndarray,
    scores: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Compute verification metrics.
    
    Args:
        y_true: Binary labels (1=same design, 0=different).
        scores: Similarity scores.
        threshold: Classification threshold.
        
    Returns:
        Dictionary with ROC-AUC, PR-AUC, F1, accuracy, etc.
    """
    from sklearn.metrics import (
        roc_auc_score,
        average_precision_score,
        precision_recall_curve,
        roc_curve,
        f1_score,
        precision_score,
        recall_score,
        accuracy_score,
    )
    
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)
    y_pred = (scores >= threshold).astype(int)
    
    fpr, tpr, _ = roc_curve(y_true, scores)
    fnr = 1 - tpr
    eer = float(fpr[np.argmin(np.abs(fpr - fnr))])
    
    return {
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "pr_auc": float(average_precision_score(y_true, scores)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "eer": eer,
        "threshold": float(threshold),
    }
