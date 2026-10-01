from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

def discover(root: str | Path) -> pd.DataFrame:
    """Discover images and infer design IDs from directory structure.
    
    Args:
        root: Dataset root directory.
        
    Returns:
        DataFrame with columns [image_path, design_id, source, color_group, split].
    """
    root = Path(root).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Dataset root not found: {root}")

    rows = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            rel = path.relative_to(root)
            # Use parent directory name as design_id, or filename if top-level
            design_id = rel.parts[-2] if len(rel.parts) > 1 else path.stem
            rows.append({
                "image_path": str(rel),
                "design_id": design_id,
                "source": root.name,
                "color_group": "",
                "split": ""
            })

    if not rows:
        raise RuntimeError(f"No images found in {root}")

    return pd.DataFrame(rows)

def main():
    parser = argparse.ArgumentParser(description="Discover images and create metadata.")
    parser.add_argument("--root", required=True, help="Dataset root folder")
    parser.add_argument("--output", default="data/metadata.csv", help="Output CSV path")
    args = parser.parse_args()

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    df = discover(args.root)
    df.to_csv(out, index=False)
    print(f"✓ Discovered {len(df)} images with {df['design_id'].nunique()} design IDs")
    print(f"✓ Saved to {out}")

if __name__ == "__main__":
    main()
