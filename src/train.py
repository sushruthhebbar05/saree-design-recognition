from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
import torch
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.dataset import SareeDataset
from src.models.embedding_model import EmbeddingModel
from src.models.losses import SupervisedContrastiveLoss
from src.utils.config import load_config
from src.utils.seed import set_seed

def train_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion,
    optimizer,
    scaler,
    device: str,
) -> float:
    """Train for one epoch.
    
    Returns:
        Average loss.
    """
    model.train()
    running_loss = 0.0
    
    pbar = tqdm(loader, desc="Training")
    for images, labels, _ in pbar:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=scaler.is_enabled()):
            embeddings = model(images)
            loss = criterion(embeddings, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()
        pbar.set_postfix({"loss": loss.item():.4f})

    return running_loss / max(1, len(loader))

def main():
    parser = argparse.ArgumentParser(description="Train embedding model.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--data-root", default=".")
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load data
    train_df = pd.read_csv(Path(config["data"]["splits"]) / "train.csv")
    train_dataset = SareeDataset(
        train_df,
        root=args.data_root,
        size=config["image"]["size"],
        train=True
    )
    loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=config["training"]["workers"],
        drop_last=True
    )

    # Model
    model = EmbeddingModel(
        backbone=config["model"]["backbone"],
        pretrained=config["model"]["pretrained"],
        embedding_dim=config["model"]["embedding_dim"],
        dropout=config["model"]["dropout"]
    ).to(device)

    # Optimizer and loss
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["training"]["lr"],
        weight_decay=config["training"]["weight_decay"]
    )
    criterion = SupervisedContrastiveLoss(temperature=config["loss"]["temperature"])
    scaler = GradScaler(
        enabled=config["training"].get("mixed_precision", False) and device.type == "cuda"
    )

    # Training loop
    ckpt_dir = Path("checkpoints")
    ckpt_dir.mkdir(exist_ok=True)

    best_loss = float("inf")
    patience_counter = 0
    patience = config["training"].get("patience", 5)

    print(f"\nTraining for {config['training']['epochs']} epochs...\n")

    for epoch in range(config["training"]["epochs"]):
        avg_loss = train_epoch(model, loader, criterion, optimizer, scaler, device)
        print(f"Epoch {epoch + 1}/{config['training']['epochs']} - Loss: {avg_loss:.4f}")

        # Save best
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            torch.save(
                {"model": model.state_dict(), "config": config},
                ckpt_dir / "best.pt"
            )
            print(f"✓ Saved best checkpoint: loss={best_loss:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch + 1}")
                break

    print(f"\n✓ Training complete. Best loss: {best_loss:.4f}")

if __name__ == "__main__":
    main()
