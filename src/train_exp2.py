"""
train_exp2.py — Experiment 2: Transfer Learning (COD10K → ACD1K)
COMP 9130 Capstone: Military Camouflage Object Detection
Author: Sepehr Mansouri (Experiment 2 + Evaluation Lead)

Two-stage SegFormer-B2 pipeline:
  Stage 1 — Pretrain on COD10K with differential LR
             (encoder: 1e-4, decode head: 1e-3)
  Stage 2 — Fine-tune on ACD1K with lower uniform LR (1e-5)
             Loads best Stage-1 checkpoint to start.

Usage (Colab):
    # Stage 1 — COD10K pretraining
    !PYTHONPATH=/content/AI-final-project/src \
     python src/train_exp2.py \
        --stage 1 \
        --encoder_lr 1e-4 --head_lr 1e-3 \
        --epochs 50 --batch_size 16 \
        --data_root data/ --splits_dir splits/ \
        --output_dir outputs/exp2/stage1

    # Stage 2 — ACD1K fine-tuning
    !PYTHONPATH=/content/AI-final-project/src \
     python src/train_exp2.py \
        --stage 2 \
        --lr 1e-5 --epochs 50 --batch_size 16 \
        --stage1_weights outputs/exp2/stage1/best_model.pth \
        --data_root data/ --splits_dir splits/ \
        --output_dir outputs/exp2/stage2
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import SegformerForSemanticSegmentation
import torch.nn.functional as F

# Allow running from repo root or src/ 
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))
from dataset import build_dataloader


# Metrics  (mIoU, F1, MAE)
def compute_metrics(preds, masks):
    """
    Compute mIoU, F1 (Dice), MAE for a batch.
    preds : FloatTensor [B, 1, H, W] — probabilities after sigmoid
    masks : FloatTensor [B, 1, H, W] — binary ground truth {0, 1}
    Returns dict of scalar averages over the batch.
    """
    preds_bin = (preds > 0.5).float()

    inter = (preds_bin * masks).sum(dim=(1, 2, 3))
    union = preds_bin.sum(dim=(1, 2, 3)) + masks.sum(dim=(1, 2, 3)) - inter

    iou_fg  = (inter + 1e-6) / (union + 1e-6)
    iou_bg  = ((1 - preds_bin) * (1 - masks)).sum(dim=(1, 2, 3))
    union_bg = (1 - preds_bin).sum(dim=(1, 2, 3)) + (1 - masks).sum(dim=(1, 2, 3)) - iou_bg
    iou_bg  = (iou_bg + 1e-6) / (union_bg + 1e-6)
    miou    = ((iou_fg + iou_bg) / 2).mean().item()

    dice = (2 * inter + 1e-6) / (preds_bin.sum(dim=(1, 2, 3)) +
                                   masks.sum(dim=(1, 2, 3)) + 1e-6)
    f1   = dice.mean().item()

    mae = (preds - masks).abs().mean().item()

    return {'mIoU': miou, 'F1': f1, 'MAE': mae}


def build_model(stage1_weights=None):
    """
    Load SegFormer-B2 pretrained on ADE20K, replace head with binary output.
    Optionally load Stage-1 weights for Stage-2 fine-tuning.
    """
    model = SegformerForSemanticSegmentation.from_pretrained(
        'nvidia/segformer-b2-finetuned-ade-512-512',
        num_labels=1,
        ignore_mismatched_sizes=True,
    )

    if stage1_weights is not None:
        print(f"[Model] Loading Stage-1 weights from {stage1_weights}")
        ckpt = torch.load(stage1_weights, map_location='cpu')
        state_dict = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
        model.load_state_dict(state_dict, strict=True)
        epoch_str = str(ckpt.get('epoch', '?'))
        val_miou = ckpt.get('val_mIoU', None)
        val_miou_str = f"{val_miou:.4f}" if val_miou is not None else "?"
        print(f"  ✅ Stage-1 weights loaded (epoch {epoch_str}, "
              f"val mIoU={val_miou_str})")

    return model


def forward_pass(model, images, masks, input_size=512):
    """
    Run forward pass; upsample logits to input_size; return loss + probs.
    Loss = BCE + Dice (same formulation as train_exp3.py).
    """
    outputs   = model(pixel_values=images, labels=None)
    logits    = outputs.logits                          # [B, 1, H/4, W/4]
    upsampled = F.interpolate(logits, size=(input_size, input_size),
                               mode='bilinear', align_corners=False)
    probs     = torch.sigmoid(upsampled)

    # Binary cross-entropy + Dice loss
    bce  = F.binary_cross_entropy_with_logits(upsampled, masks)
    inter = (probs * masks).sum()
    dice_loss = 1 - (2 * inter + 1) / (probs.sum() + masks.sum() + 1)
    loss = bce + dice_loss

    return loss, probs


# Train / validation loops

def run_epoch(model, loader, optimizer, device, train=True, accum_steps=1):
    model.train() if train else model.eval()
    total_loss = 0.0
    all_miou, all_f1, all_mae = [], [], []

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for i, batch in enumerate(loader):
            images = batch['image'].to(device)
            masks  = batch['mask'].to(device)

            loss, probs = forward_pass(model, images, masks)

            if train:
                loss = loss / accum_steps
                loss.backward()
                if (i + 1) % accum_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad()

            total_loss += loss.item()
            m = compute_metrics(probs.detach(), masks)
            all_miou.append(m['mIoU'])
            all_f1.append(m['F1'])
            all_mae.append(m['MAE'])

    n = len(loader)
    return {'loss': total_loss/n, 'mIoU': np.mean(all_miou),
            'F1': np.mean(all_f1), 'MAE': np.mean(all_mae)}


# Main training function
def train(args):
    # Reproducibility 
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save run config
    config = vars(args)
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)
    print(f'Config saved → {output_dir}/config.json')
    print(json.dumps(config, indent=2))

    # Dataloaders
    print('\n[DataLoaders]')

    if args.stage == 1:
        # Stage 1: train on COD10K, validate on COD10K val
        train_loader = build_dataloader(
            args.data_root, condition='cod10k', split='train',
            batch_size=args.batch_size, num_workers=args.num_workers,
            oversample_acd1k=False, seed=args.seed,
            splits_dir=args.splits_dir,
        )
        val_loader = build_dataloader(
            args.data_root, condition='cod10k', split='val',
            batch_size=args.batch_size, num_workers=args.num_workers,
            oversample_acd1k=False, seed=args.seed,
            splits_dir=args.splits_dir,
        )
    else:
        # Stage 2: train on ACD1K, validate on ACD1K val
        train_loader = build_dataloader(
            args.data_root, condition='acd1k', split='train',
            batch_size=args.batch_size, num_workers=args.num_workers,
            oversample_acd1k=False, seed=args.seed,
            splits_dir=args.splits_dir,
        )
        val_loader = build_dataloader(
            args.data_root, condition='acd1k', split='val',
            batch_size=args.batch_size, num_workers=args.num_workers,
            oversample_acd1k=False, seed=args.seed,
            splits_dir=args.splits_dir,
        )

    # Model
    print('\n[Model] Loading SegFormer-B2 (ADE20K pretrained)...')
    stage1_wt = args.stage1_weights if args.stage == 2 else None
    model = build_model(stage1_weights=stage1_wt).to(device)

    # Optimizer & Scheduler
    if args.stage == 1:
        # Differential LR: slower encoder, faster decode head
        encoder_params = [p for n, p in model.named_parameters()
                          if 'decode_head' not in n and p.requires_grad]
        head_params    = [p for n, p in model.named_parameters()
                          if 'decode_head' in n and p.requires_grad]
        optimizer = AdamW([
            {'params': encoder_params, 'lr': args.encoder_lr},
            {'params': head_params,    'lr': args.head_lr},
        ], weight_decay=args.weight_decay)
        lr_display = f"encoder_lr={args.encoder_lr}, head_lr={args.head_lr}"
    else:
        # Stage 2: uniform lower LR to prevent catastrophic forgetting
        optimizer = AdamW(
            model.parameters(), lr=args.lr,
            weight_decay=args.weight_decay,
        )
        lr_display = f"lr={args.lr}"

    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # Training loop
    stage_label = "Stage 1 (COD10K pretrain)" if args.stage == 1 \
                  else "Stage 2 (ACD1K fine-tune)"
    print(f'\n[Training] {stage_label}, {args.epochs} epochs, {lr_display}')
    print('=' * 65)

    best_val_miou  = 0.0
    best_epoch     = 0
    patience_count = 0
    history        = []

    for epoch in range(1, args.epochs + 1):
        train_m = run_epoch(model, train_loader, optimizer, device,
                            train=True, accum_steps=args.accum_steps)
        val_m = run_epoch(model, val_loader, optimizer, device,
                          train=False, accum_steps=1)
        scheduler.step()

        row = {
            'epoch':      epoch,
            'train_loss': round(train_m['loss'], 4),
            'train_mIoU': round(train_m['mIoU'], 4),
            'train_F1':   round(train_m['F1'],   4),
            'train_MAE':  round(train_m['MAE'],  4),
            'val_loss':   round(val_m['loss'],   4),
            'val_mIoU':   round(val_m['mIoU'],   4),
            'val_F1':     round(val_m['F1'],     4),
            'val_MAE':    round(val_m['MAE'],    4),
            'lr':         round(scheduler.get_last_lr()[0], 8),
        }
        history.append(row)

        improved = val_m['mIoU'] > best_val_miou
        marker   = ' ◀ best' if improved else ''
        print(
            f"Ep {epoch:3d}/{args.epochs} | "
            f"train loss={train_m['loss']:.4f} mIoU={train_m['mIoU']:.4f} | "
            f"val loss={val_m['loss']:.4f} mIoU={val_m['mIoU']:.4f} "
            f"F1={val_m['F1']:.4f} MAE={val_m['MAE']:.4f}{marker}"
        )

        if improved:
            best_val_miou  = val_m['mIoU']
            best_epoch     = epoch
            patience_count = 0
            torch.save({
                'epoch':      epoch,
                'state_dict': model.state_dict(),
                'val_mIoU':   best_val_miou,
                'config':     config,
            }, output_dir / 'best_model.pth')
        else:
            patience_count += 1
            if patience_count >= args.patience:
                print(f'\n[Early stopping] No improvement for {args.patience} '
                      f'epochs. Best val mIoU={best_val_miou:.4f} at '
                      f'epoch {best_epoch}.')
                break

        # Save history every epoch (safe for Colab disconnects)
        with open(output_dir / 'history.json', 'w') as f:
            json.dump(history, f, indent=2)

    print('=' * 65)
    print(f'Training complete. Best val mIoU={best_val_miou:.4f} '
          f'at epoch {best_epoch}.')
    print(f'Best model saved → {output_dir}/best_model.pth')

    return best_val_miou


# Argument parser

def parse_args():
    p = argparse.ArgumentParser(
        description='Train Experiment 2 — SegFormer-B2 Transfer Learning '
                    '(COD10K → ACD1K)'
    )
    p.add_argument('--stage', type=int, required=True, choices=[1, 2],
                   help='1 = COD10K pretrain, 2 = ACD1K fine-tune')
    p.add_argument('--data_root',   default='data/',     help='Dataset root')
    p.add_argument('--splits_dir',  default='splits/',   help='Splits JSON dir')
    p.add_argument('--output_dir',  required=True,       help='Save dir')

    # Stage 1 hyperparameters (differential LR)
    p.add_argument('--encoder_lr',  type=float, default=1e-4,
                   help='Stage 1: encoder learning rate (default: 1e-4)')
    p.add_argument('--head_lr',     type=float, default=1e-3,
                   help='Stage 1: decode-head learning rate (default: 1e-3)')

    # Stage 2 hyperparameters
    p.add_argument('--lr',          type=float, default=1e-5,
                   help='Stage 2: uniform learning rate (default: 1e-5)')
    p.add_argument('--stage1_weights', type=str, default=None,
                   help='Path to Stage-1 best_model.pth (required for Stage 2)')

    # Common training settings
    p.add_argument('--weight_decay',type=float, default=1e-4)
    p.add_argument('--epochs',      type=int,   default=50)
    p.add_argument('--batch_size',  type=int,   default=16)
    p.add_argument('--accum_steps', type=int,   default=1,
                   help='Gradient accumulation steps')
    p.add_argument('--patience',    type=int,   default=10,
                   help='Early stopping patience (epochs)')
    p.add_argument('--num_workers', type=int,   default=2)
    p.add_argument('--seed',        type=int,   default=42)

    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    if args.stage == 2 and args.stage1_weights is None:
        print("ERROR: --stage1_weights is required for Stage 2.")
        print("  Example: --stage1_weights outputs/exp2/stage1/best_model.pth")
        sys.exit(1)
    train(args)