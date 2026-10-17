
import os
import sys
import json
import yaml
import argparse
import logging
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

from .dataset import CXRDataset, NIH_15_CLASSES
from .model import DenseNetCustom

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")
logger = logging.getLogger("Trainer")


def calculate_roc_auc(targets, probs, label_cols):
    """Computes per-class and mean ROC-AUC scores."""
    aucs = {}
    for i, label in enumerate(label_cols):
        y_true = targets[:, i]
        y_score = probs[:, i]
        if len(np.unique(y_true)) > 1:
            try:
                aucs[label] = float(roc_auc_score(y_true, y_score))
            except Exception:
                aucs[label] = float("nan")
        else:
            aucs[label] = float("nan")

    valid_vals = [v for v in aucs.values() if not np.isnan(v)]
    mean_auc = float(np.mean(valid_vals)) if valid_vals else 0.0
    return mean_auc, aucs


def train_one_epoch(model, loader, criterion, optimizer, scaler, device):
    """Executes a single training epoch."""
    model.train()
    running_loss = 0.0
    total_samples = 0

    pbar = tqdm(loader, desc="Training", leave=False)
    for images, targets in pbar:
        images = images.to(device)
        targets = targets.to(device)
        optimizer.zero_grad()

        if scaler:
            with torch.cuda.amp.autocast():
                logits = model(images)
                loss = criterion(logits, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(images)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()

        batch_size = images.size(0)
        running_loss += loss.item() * batch_size
        total_samples += batch_size
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    return running_loss / max(total_samples, 1)


@torch.no_grad()
def validate(model, loader, criterion, device, label_cols):
    """Evaluates the model on validation data, reporting loss and mean AUC."""
    model.eval()
    running_loss = 0.0
    total_samples = 0
    all_targets = []
    all_probs = []

    for images, targets in loader:
        images = images.to(device)
        targets = targets.to(device)

        logits = model(images)
        loss = criterion(logits, targets)

        probs = torch.sigmoid(logits)

        batch_size = images.size(0)
        running_loss += loss.item() * batch_size
        total_samples += batch_size

        all_targets.append(targets.cpu().numpy())
        all_probs.append(probs.cpu().numpy())

    val_loss = running_loss / max(total_samples, 1)
    val_targets = np.vstack(all_targets)
    val_probs = np.vstack(all_probs)

    mean_auc, per_class_auc = calculate_roc_auc(val_targets, val_probs, label_cols)
    return val_loss, mean_auc, per_class_auc, val_probs, val_targets


def train(args):
    """Main training orchestration."""
    device = torch.device(args.device if torch.cuda.is_available() and "cuda" in args.device else "cpu")
    logger.info(f"Target execution device: {device}")

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # 1. Datasets & Loaders
    label_cols = args.label_cols or NIH_15_CLASSES
    logger.info(f"Setting up dataset loaders for {len(label_cols)} target classes...")

    train_dataset = CXRDataset(
        df_or_csv_path=args.train_csv,
        images_dir=args.images_dir,
        label_cols=label_cols,
        img_col=args.img_col,
        finding_col=args.finding_col,
        img_size=args.img_size,
        is_train=True
    )
    val_dataset = CXRDataset(
        df_or_csv_path=args.val_csv,
        images_dir=args.images_dir,
        label_cols=label_cols,
        img_col=args.img_col,
        finding_col=args.finding_col,
        img_size=args.img_size,
        is_train=False
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda")
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda")
    )

    logger.info(f"Dataset ready. Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    # 2. Model, Loss, Optimizer
    model = DenseNetCustom(num_classes=len(label_cols), pretrained=args.pretrained).to(device)

    # Optional positive weighting for imbalanced classes
    if args.use_pos_weight:
        # Calculate positive ratio
        train_targets = train_dataset.targets
        pos_counts = train_targets.sum(axis=0)
        neg_counts = len(train_targets) - pos_counts
        pos_weight = torch.tensor(neg_counts / np.maximum(pos_counts, 1.0), dtype=torch.float32).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2, verbose=True
    )
    scaler = torch.cuda.amp.GradScaler() if (args.fp16 and device.type == "cuda") else None

    best_metric = -1.0
    best_checkpoint_path = os.path.join(args.checkpoint_dir, "best_nih_model.pth")

    # 3. Epoch Loop
    import time
    logger.info(f"Commencing training across {args.epochs} epochs...")
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        current_lr = optimizer.param_groups[0]["lr"]
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device)
        val_loss, val_mean_auc, per_class_auc, _, _ = validate(model, val_loader, criterion, device, label_cols)
        epoch_duration = time.time() - epoch_start

        scheduler.step(val_mean_auc)
        logger.info(
            f"Epoch [{epoch:02d}/{args.epochs:02d}] ({epoch_duration:.1f}s) "
            f"LR: {current_lr:.6f} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Mean AUC: {val_mean_auc:.4f}"
        )

        # Checkpoint Saving
        if val_loss < best_metric or best_metric < 0 or val_mean_auc > best_metric:
            best_metric = val_mean_auc
            checkpoint = {
                "epoch": epoch,
                "label_columns": label_cols,
                "model_state_dict": model.state_dict(),
                "val_loss": float(val_loss),
                "val_mean_auc": float(val_mean_auc),
                "per_class_auc": per_class_auc,
                "author": "Debashis Mohapatra",
                "version": "2.0.0"
            }
            torch.save(checkpoint, best_checkpoint_path)
            # Also save matching backend model name
            backend_target_path = os.path.join(args.checkpoint_dir, "nih-model.pth")
            torch.save(checkpoint, backend_target_path)
            logger.info(f">> High-water mark reached! Saved model to {best_checkpoint_path} (AUC: {best_metric:.4f})")

    logger.info("Training sequence concluded successfully.")
    return best_checkpoint_path


def load_config(config_path):
    """Reads configuration from YAML if supplied."""
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    return {}


if __name__ == "__main__":
    initial_parser = argparse.ArgumentParser(add_help=False)
    initial_parser.add_argument("--config", type=str, default="training/config.yaml", help="Path to YAML configuration")
    initial_args, remaining_argv = initial_parser.parse_known_args()

    cfg = load_config(initial_args.config)

    parser = argparse.ArgumentParser(description="Neuro CXR Multi-Label DenseNet Trainer")
    parser.add_argument("--config", type=str, default=initial_args.config)
    parser.add_argument("--train_csv", type=str, default=cfg.get("train_csv", "data/train.csv"))
    parser.add_argument("--val_csv", type=str, default=cfg.get("val_csv", "data/val.csv"))
    parser.add_argument("--images_dir", type=str, default=cfg.get("images_dir", "data/images"))
    parser.add_argument("--img_col", type=str, default=cfg.get("img_col", "image_path"))
    parser.add_argument("--finding_col", type=str, default=cfg.get("finding_col", None))
    parser.add_argument("--batch_size", type=int, default=cfg.get("batch_size", 16))
    parser.add_argument("--lr", type=float, default=cfg.get("lr", 1e-4))
    parser.add_argument("--weight_decay", type=float, default=cfg.get("weight_decay", 1e-4))
    parser.add_argument("--epochs", type=int, default=cfg.get("epochs", 3))
    parser.add_argument("--img_size", type=int, default=cfg.get("img_size", 224))
    parser.add_argument("--device", type=str, default=cfg.get("device", "cpu"))
    parser.add_argument("--checkpoint_dir", type=str, default=cfg.get("checkpoint_dir", "models"))
    parser.add_argument("--num_workers", type=int, default=cfg.get("num_workers", 0))
    parser.add_argument("--pretrained", action="store_true", default=cfg.get("pretrained", True))
    parser.add_argument("--fp16", action="store_true", default=cfg.get("fp16", False))
    parser.add_argument("--use_pos_weight", action="store_true", default=cfg.get("use_pos_weight", True))

    cli_args = parser.parse_args(remaining_argv)
    cli_args.label_cols = NIH_15_CLASSES
    train(cli_args)
