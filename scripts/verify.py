from __future__ import annotations
import argparse
from PIL import Image
import torch

from src.data.dataset import eval_transform
from src.models.embedding_model import EmbeddingModel
from src.utils.config import load_config

def main():
    parser = argparse.ArgumentParser(description="Verify if two images are the same design.")
    parser.add_argument("--image-a", required=True, help="First image path")
    parser.add_argument("--image-b", required=True, help="Second image path")
    parser.add_argument("--checkpoint", required=True, help="Model checkpoint path")
    parser.add_argument("--threshold", type=float, default=0.5, help="Similarity threshold")
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

    # Embed images
    transform = eval_transform(config["image"]["size"])

    def embed_image(path):
        with Image.open(path) as img:
            tensor = transform(img.convert("RGB")).unsqueeze(0)
        with torch.no_grad():
            return model(tensor)

    emb_a = embed_image(args.image_a)
    emb_b = embed_image(args.image_b)
    sim = (emb_a * emb_b).sum(dim=1).item()
    pred = "SAME" if sim >= args.threshold else "DIFFERENT"

    print(f"\nImage A: {args.image_a}")
    print(f"Image B: {args.image_b}")
    print(f"Similarity: {sim:.4f}")
    print(f"Threshold: {args.threshold:.4f}")
    print(f"Prediction: {pred}\n")

if __name__ == "__main__":
    main()
