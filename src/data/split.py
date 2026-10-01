from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

def make_splits(
    metadata_path: str,
    output_dir: str,
    seed: int = 42,
    train_ratio: float = 0.6,
) -> dict:
    """Create leakage-free train/val/gallery/query splits by design_id.
    
    Args:
        metadata_path: Path to metadata CSV.
        output_dir: Output directory.
        seed: Random seed.
        train_ratio: Fraction for training.
        
    Returns:
        Split counts.
    """
    df = pd.read_csv(metadata_path)
    if "design_id" not in df.columns or "image_path" not in df.columns:
        raise ValueError("metadata.csv must include image_path and design_id")

    design_ids = df["design_id"].astype(str).unique()
    if len(design_ids) < 2:
        raise ValueError("At least two design IDs required")

    # Split by design_id
    train_designs, test_designs = train_test_split(
        design_ids, test_size=1 - train_ratio, random_state=seed
    )
    val_designs, gallery_designs = train_test_split(
        test_designs, test_size=0.5, random_state=seed
    ) if len(test_designs) > 1 else (test_designs, test_designs)

    # Assign images to splits
    train = df[df["design_id"].astype(str).isin(train_designs)].copy()
    val = df[df["design_id"].astype(str).isin(val_designs)].copy()
    gallery_pool = df[df["design_id"].astype(str).isin(gallery_designs)].copy()

    # For gallery/query: one image per design in query, rest in gallery
    query = gallery_pool.groupby("design_id", group_keys=False).head(1).copy()
    gallery = gallery_pool.drop(query.index).copy()

    if gallery.empty:
        gallery = gallery_pool.copy()
        query = gallery_pool.groupby("design_id", group_keys=False).tail(1).copy()

    # Save splits
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train.to_csv(out_dir / "train.csv", index=False)
    val.to_csv(out_dir / "val.csv", index=False)
    gallery.to_csv(out_dir / "gallery.csv", index=False)
    query.to_csv(out_dir / "query.csv", index=False)

    return {
        "train": len(train),
        "val": len(val),
        "gallery": len(gallery),
        "query": len(query),
        "train_designs": len(train_designs),
        "val_designs": len(val_designs),
        "gallery_designs": len(gallery_designs),
    }

def main():
    parser = argparse.ArgumentParser(description="Create leakage-free data splits.")
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--output-dir", default="data/splits")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.6)
    args = parser.parse_args()

    result = make_splits(args.metadata, args.output_dir, args.seed, args.train_ratio)
    print("\n=== Data Split Summary ===")
    for k, v in result.items():
        print(f"{k}: {v}")

if __name__ == "__main__":
    main()
