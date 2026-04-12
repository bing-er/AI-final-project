"""
engine_exp1.py — Experiment 1: SINetV2 on ACD1K
COMP 9130 Capstone: Military Camouflage Object Detection
Author: Yansong Jia (Experiment 1 Lead)

Training engine for SINetV2 on the ACD1K dataset only.
Responsibilities:
  - Build ACD1K train/val dataloaders at 352×352 resolution.
  - Train SINetV2 with AdamW and BCE + Dice loss.
  - Track mIoU, F1 (Dice), MAE on the validation split.
  - Apply early stopping with patience=10 based on val mIoU.
  - Save best checkpoint and history.json for downstream evaluation.

NOTE:
  This script intentionally does NOT perform any final hold-out evaluation
  or cross-experiment comparison. The unified 200-image test is handled
  separately by Sepehr's final evaluation notebook.
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

# added for reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

from dataset import (
    CamouflageDataset,
    DATASET_STATS,
    get_train_transforms,
    get_val_transforms,
    get_val_files,
)
from sinetv2 import build_sinetv2


# ──────────────────────────────────────────────────────────────────────────────
# Metrics
# ──────────────────────────────────────────────────────────────────────────────

def compute_metrics(preds: torch.Tensor, masks: torch.Tensor):
    """
    Compute mIoU, F1 (Dice), MAE for a batch.

    Args:
        preds: FloatTensor [B, 1, H, W] — probabilities after sigmoid.
        masks: FloatTensor [B, 1, H, W] — binary ground truth {0, 1}.
    """
    preds_bin = (preds > 0.5).float()

    inter = (preds_bin * masks).sum(dim=(1, 2, 3))
    union = preds_bin.sum(dim=(1, 2, 3)) + masks.sum(dim=(1, 2, 3)) - inter

    iou_fg = (inter + 1e-6) / (union + 1e-6)

    iou_bg = ((1 - preds_bin) * (1 - masks)).sum(dim=(1, 2, 3))
    union_bg = (1 - preds_bin).sum(dim=(1, 2, 3)) + (1 - masks).sum(dim=(1, 2, 3)) - iou_bg
    iou_bg = (iou_bg + 1e-6) / (union_bg + 1e-6)

    miou = ((iou_fg + iou_bg) / 2).mean().item()

    dice = (2 * inter + 1e-6) / (preds_bin.sum(dim=(1, 2, 3)) + masks.sum(dim=(1, 2, 3)) + 1e-6)
    f1 = dice.mean().item()

    mae = (preds - masks).abs().mean().item()

    return {"mIoU": miou, "F1": f1, "MAE": mae}


# ──────────────────────────────────────────────────────────────────────────────
# Dataloaders (ACD1K only, 352×352)
# ──────────────────────────────────────────────────────────────────────────────

def build_acd1k_loaders(
    data_root: str,
    splits_dir: str,
    batch_size: int = 8,
    num_workers: int = 2,
    seed: int = 42,
    input_size: int = 352,
):
    """
    Build ACD1K train/val dataloaders with 352×352 transforms.
    """
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    data_root = Path(data_root)
    splits_dir = Path(splits_dir)

    stats = DATASET_STATS["ACD1K"]
    train_tf = get_train_transforms(stats["mean"], stats["std"], input_size=input_size)
    val_tf = get_val_transforms(stats["mean"], stats["std"], input_size=input_size)

    train_ds = CamouflageDataset(
        image_dir=data_root / "dataset-splitM/Training/images",
        mask_dir=data_root / "dataset-splitM/Training/GT",
        transform=train_tf,
        dataset_name="ACD1K",
        file_list=None,  # all 748 train images
    )

    val_files = get_val_files(splits_dir, "acd1k")  # 230-image val split
    val_ds = CamouflageDataset(
        image_dir=data_root / "dataset-splitM/Testing/images",
        mask_dir=data_root / "dataset-splitM/Testing/GT",
        transform=val_tf,
        dataset_name="ACD1K",
        file_list=val_files,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    print(f"  [DataLoader] ACD1K train: {len(train_ds)} samples, {len(train_loader)} batches")
    print(f"  [DataLoader] ACD1K val  : {len(val_ds)} samples, {len(val_loader)} batches")

    return train_loader, val_loader


# ──────────────────────────────────────────────────────────────────────────────
# Model / loss
# ──────────────────────────────────────────────────────────────────────────────

def build_model():
    """
    Build SINetV2 for binary segmentation.
    """
    model = build_sinetv2(in_channels=3, base_channels=32)
    return model


def forward_pass(model, images, masks):
    """
    Forward pass for SINetV2.

    Returns:
        loss (BCE + Dice) and probabilities after sigmoid.
    """
    logits = model(images)  # [B, 1, H, W]
    probs = torch.sigmoid(logits)

    bce = F.binary_cross_entropy_with_logits(logits, masks)
    inter = (probs * masks).sum()
    dice_loss = 1 - (2 * inter + 1) / (probs.sum() + masks.sum() + 1)
    loss = bce + dice_loss

    return loss, probs


# ──────────────────────────────────────────────────────────────────────────────
# Epoch loops
# ──────────────────────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, device, accum_steps: int = 1):
    model.train()
    total_loss = 0.0
    all_miou, all_f1, all_mae = [], [], []

    for i, batch in enumerate(loader):
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)

        loss, probs = forward_pass(model, images, masks)

        loss = loss / accum_steps
        loss.backward()

        if (i + 1) % accum_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

        total_loss += loss.item()
        m = compute_metrics(probs.detach(), masks)
        all_miou.append(m["mIoU"])
        all_f1.append(m["F1"])
        all_mae.append(m["MAE"])

    n = len(loader)
    return {
        "loss": total_loss / n,
        "mIoU": float(np.mean(all_miou)),
        "F1": float(np.mean(all_f1)),
        "MAE": float(np.mean(all_mae)),
    }


def validate_one_epoch(model, loader, device):
    model.eval()
    total_loss = 0.0
    all_miou, all_f1, all_mae = [], [], []

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)

            loss, probs = forward_pass(model, images, masks)
            total_loss += loss.item()

            m = compute_metrics(probs, masks)
            all_miou.append(m["mIoU"])
            all_f1.append(m["F1"])
            all_mae.append(m["MAE"])

    n = len(loader)
    return {
        "loss": total_loss / n,
        "mIoU": float(np.mean(all_miou)),
        "F1": float(np.mean(all_f1)),
        "MAE": float(np.mean(all_mae)),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Main training routine
# ──────────────────────────────────────────────────────────────────────────────

def train(args):
    # Reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    output_dir = Path(args.output_dir)
    final_dir = output_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    config = vars(args)
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    print(f"Config saved → {output_dir}/config.json")
    print(json.dumps(config, indent=2))

    # Dataloaders
    print("\n[DataLoaders]")
    train_loader, val_loader = build_acd1k_loaders(
        data_root=args.data_root,
        splits_dir=args.splits_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        input_size=352,
    )

    # Model
    print("\n[Model] Building SINetV2...")
    model = build_model().to(device)

    # Optimizer & scheduler
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # Training loop
    print(
        f"\n[Training] {args.epochs} epochs, lr={args.lr}, "
        f"batch_size={args.batch_size}, patience={args.patience}"
    )
    print("=" * 65)

    best_val_miou = 0.0
    best_epoch = 0
    patience_count = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        train_m = train_one_epoch(
            model, train_loader, optimizer, device, accum_steps=args.accum_steps
        )
        val_m = validate_one_epoch(model, val_loader, device)
        scheduler.step()

        row = {
            "epoch": epoch,
            "train_loss": round(train_m["loss"], 4),
            "train_mIoU": round(train_m["mIoU"], 4),
            "train_F1": round(train_m["F1"], 4),
            "train_MAE": round(train_m["MAE"], 4),
            "val_loss": round(val_m["loss"], 4),
            "val_mIoU": round(val_m["mIoU"], 4),
            "val_F1": round(val_m["F1"], 4),
            "val_MAE": round(val_m["MAE"], 4),
            "lr": round(scheduler.get_last_lr()[0], 8),
        }
        history.append(row)

        improved = val_m["mIoU"] > best_val_miou
        marker = " ◀ best" if improved else ""
        print(
            f"Ep {epoch:3d}/{args.epochs} | "
            f"train loss={train_m['loss']:.4f} mIoU={train_m['mIoU']:.4f} | "
            f"val loss={val_m['loss']:.4f} mIoU={val_m['mIoU']:.4f} "
            f"F1={val_m['F1']:.4f} MAE={val_m['MAE']:.4f}{marker}"
        )

        if improved:
            best_val_miou = val_m["mIoU"]
            best_epoch = epoch
            patience_count = 0
            torch.save(
                {
                    "epoch": epoch,
                    "state_dict": model.state_dict(),
                    "val_mIoU": best_val_miou,
                    "config": config,
                },
                final_dir / "best_model.pth",
            )
        else:
            patience_count += 1
            if patience_count >= args.patience:
                print(
                    f"\n[Early stopping] No improvement for {args.patience} "
                    f"epochs. Best val mIoU={best_val_miou:.4f} at epoch {best_epoch}."
                )
                break

        # Persist history every epoch
        with open(final_dir / "history.json", "w") as f:
            json.dump(history, f, indent=2)

    print("=" * 65)
    print(
        f"Training complete. Best val mIoU={best_val_miou:.4f} "
        f"at epoch {best_epoch}."
    )
    print(f"Best model saved → {final_dir}/best_model.pth")

    return best_val_miou


# ──────────────────────────────────────────────────────────────────────────────
# Argument parser
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Train Experiment 1 — SINetV2 on ACD1K"
    )
    p.add_argument("--data_root", default="data/", help="Dataset root")
    p.add_argument("--splits_dir", default="splits/", help="Splits JSON dir")
    p.add_argument(
        "--output_dir", default="outputs/exp1/", help="Base directory for outputs"
    )

    # Hyperparameters (tuned for Exp1)
    p.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate (try: 1e-4, 6e-5, 1e-5)",
    )
    p.add_argument("--weight_decay", type=float, default=1e-4)

    # Training settings
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument(
        "--accum_steps",
        type=int,
        default=1,
        help="Gradient accumulation steps (effective batch = batch_size * accum_steps)",
    )
    p.add_argument(
        "--patience",
        type=int,
        default=10,
        help="Early stopping patience (epochs)",
    )
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)

