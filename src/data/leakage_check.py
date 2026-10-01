from __future__ import annotations
import argparse
import hashlib
from pathlib import Path
import pandas as pd
from PIL import Image

def check_leakage(metadata_path: str, root: str = ".") -> dict:
    """Check for duplicates, invalid images, and missing files.
    
    Args:
        metadata_path: Path to metadata CSV.
        root: Root directory for image paths.
        
    Returns:
        Report dictionary.
    """
    df = pd.read_csv(metadata_path)
    seen = {}
    duplicates = []
    invalid = []

    for row in df.itertuples():
        path = Path(root) / str(row.image_path)
        if not path.exists():
            invalid.append(str(path))
            continue

        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest in seen:
            duplicates.append((seen[digest], str(path)))
        else:
            seen[digest] = str(path)

        try:
            with Image.open(path) as img:
                img.verify()
        except Exception as e:
            invalid.append(f"invalid:{path} ({e})")

    return {
        "images": len(df),
        "designs": df["design_id"].nunique(),
        "duplicates": duplicates,
        "invalid_or_missing": invalid
    }

def main():
    parser = argparse.ArgumentParser(description="Check dataset for leakage.")
    parser.add_argument("--metadata", default="data/metadata.csv")
    parser.add_argument("--root", default=".")
    args = parser.parse_args()

    report = check_leakage(args.metadata, args.root)
    print("\n=== Leakage Report ===")
    print(f"Images: {report['images']}")
    print(f"Designs: {report['designs']}")
    print(f"Duplicates: {len(report['duplicates'])}")
    print(f"Invalid/Missing: {len(report['invalid_or_missing'])}")
    if report["duplicates"]:
        print("\nDuplicate files:")
        for a, b in report["duplicates"][:5]:
            print(f"  {a} <-> {b}")
    if report["invalid_or_missing"]:
        print("\nInvalid/Missing:")
        for p in report["invalid_or_missing"][:5]:
            print(f"  {p}")

if __name__ == "__main__":
    main()
