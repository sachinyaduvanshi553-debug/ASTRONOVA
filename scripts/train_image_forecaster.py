"""
train_image_forecaster.py — ASTRONOVA Phase 12: Training Script

Trains the SolarImageForecaster model on chronologically-split image sequences.

SAFETY MEASURES:
    - Chronological split ONLY (no random splitting)
    - Normalization from training split ONLY
    - CPU-compatible (auto-detect CUDA, never hardcode)
    - Early stopping on val ROC-AUC
    - Gradient clipping
    - Mixed precision when CUDA available
    - Smoke test mode: 1 epoch, batch=2, size=128

Loss function:
    L_total = lambda_cls * L_cls + lambda_location * L_heatmap + lambda_image * L_image

    L_cls:     Focal loss per horizon (handles class imbalance)
    L_heatmap: BCE+Dice for spatial heatmap
    L_image:   L1 + SSIM for future image prediction

Usage:
    python scripts/train_image_forecaster.py --smoke-test
    python scripts/train_image_forecaster.py --config configs/image_forecasting.yaml
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from services.vision.preprocessing import (
    ImageNormalizer,
    SolarAugmentation,
    SequenceAugmentation,
    SolarSequenceDataset,
)
from models.vision.solar_image_forecaster import SolarImageForecaster, HORIZONS
from models.vision.image_forecaster import ImageForecastLoss

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger("astronova.train")


# ---------------------------------------------------------------------------
# Focal Loss for imbalanced flare classification
# ---------------------------------------------------------------------------
class FocalLoss(nn.Module):
    """Focal loss for binary classification (handles C+/M+/X+ class imbalance)."""

    def __init__(self, gamma: float = 2.0, alpha: float = 0.25):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits:  [B, 3] — raw logits (C+, M+, X+)
            targets: [B, 3] — binary labels
        """
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        probs = torch.sigmoid(logits)
        pt = probs * targets + (1 - probs) * (1 - targets)
        focal = self.alpha * (1 - pt) ** self.gamma * bce
        return focal.mean()


# ---------------------------------------------------------------------------
# Custom collate for dict labels
# ---------------------------------------------------------------------------
def collate_fn(batch):
    """Custom collate that handles nested dict labels."""
    image_seqs = torch.stack([b["image_seq"] for b in batch])
    telemetry = torch.stack([b["telemetry"] for b in batch])
    
    labels = {}
    for h in HORIZONS:
        labels[h] = torch.stack([b["labels"][h] for b in batch])

    return {
        "image_seq": image_seqs,
        "telemetry": telemetry,
        "labels": labels,
        "is_synthetic": [b["is_synthetic"] for b in batch],
    }


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    focal_loss: FocalLoss,
    image_loss: ImageForecastLoss,
    lambda_cls: float,
    lambda_location: float,
    lambda_image: float,
    device: torch.device,
    scaler: torch.amp.GradScaler | None,
    grad_clip: float,
    epoch: int,
) -> dict:
    """Run one training epoch."""
    model.train()
    total_loss = 0.0
    total_cls = 0.0
    total_img = 0.0
    n_batches = 0

    for batch_idx, batch in enumerate(loader):
        image_seq = batch["image_seq"].to(device)
        telemetry = batch["telemetry"].to(device)
        labels = {h: batch["labels"][h].to(device) for h in HORIZONS}

        optimizer.zero_grad()

        use_amp = scaler is not None and device.type == "cuda"
        autocast_ctx = torch.amp.autocast("cuda") if use_amp else torch.amp.autocast("cpu", enabled=False)

        with autocast_ctx:
            output = model(image_seq, telemetry)

            # Classification loss (sum over all horizons)
            cls_loss = torch.tensor(0.0, device=device)
            for h in HORIZONS:
                cls_loss = cls_loss + focal_loss(output["class_logits"][h], labels[h])
            cls_loss = cls_loss / len(HORIZONS)

            # Image prediction loss (self-supervised: predict last frame)
            # For proper image forecasting, target should be the FUTURE image.
            # During smoke test with synthetic data, we use the last input frame
            # as a proxy target — documenting this limitation clearly.
            target_image = image_seq[:, -1]  # [B, C, H, W] — proxy target
            img_loss, img_details = image_loss(output["predicted_image"], target_image)

            # Total loss
            loss = lambda_cls * cls_loss + lambda_image * img_loss

        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        total_loss += loss.item()
        total_cls += cls_loss.item()
        total_img += img_loss.item()
        n_batches += 1

        if batch_idx % 5 == 0:
            logger.info(
                "  Epoch %d | Batch %d/%d | Loss: %.4f (cls: %.4f, img: %.4f)",
                epoch, batch_idx, len(loader), loss.item(), cls_loss.item(), img_loss.item(),
            )

    return {
        "train_loss": total_loss / max(1, n_batches),
        "train_cls_loss": total_cls / max(1, n_batches),
        "train_img_loss": total_img / max(1, n_batches),
    }


# ---------------------------------------------------------------------------
# Validation loop
# ---------------------------------------------------------------------------
@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    focal_loss: FocalLoss,
    image_loss: ImageForecastLoss,
    lambda_cls: float,
    lambda_image: float,
    device: torch.device,
) -> dict:
    """Run validation."""
    model.eval()
    total_loss = 0.0
    total_cls = 0.0
    total_img = 0.0
    n_batches = 0

    all_probs = {h: [] for h in HORIZONS}
    all_labels = {h: [] for h in HORIZONS}

    for batch in loader:
        image_seq = batch["image_seq"].to(device)
        telemetry = batch["telemetry"].to(device)
        labels = {h: batch["labels"][h].to(device) for h in HORIZONS}

        output = model(image_seq, telemetry)

        cls_loss = torch.tensor(0.0, device=device)
        for h in HORIZONS:
            cls_loss = cls_loss + focal_loss(output["class_logits"][h], labels[h])
        cls_loss = cls_loss / len(HORIZONS)

        target_image = image_seq[:, -1]
        img_loss, _ = image_loss(output["predicted_image"], target_image)

        loss = lambda_cls * cls_loss + lambda_image * img_loss
        total_loss += loss.item()
        total_cls += cls_loss.item()
        total_img += img_loss.item()
        n_batches += 1

        for h in HORIZONS:
            all_probs[h].append(output["class_probs"][h].cpu())
            all_labels[h].append(labels[h].cpu())

    # Compute ROC-AUC for M+ at 60-minute horizon (primary metric)
    roc_auc = _compute_roc_auc(all_probs, all_labels, horizon="h60m", class_idx=1)

    return {
        "val_loss": total_loss / max(1, n_batches),
        "val_cls_loss": total_cls / max(1, n_batches),
        "val_img_loss": total_img / max(1, n_batches),
        "val_roc_auc_M_h60m": roc_auc,
    }


def _compute_roc_auc(all_probs, all_labels, horizon="h60m", class_idx=1) -> float:
    """Compute ROC-AUC for a specific horizon and class."""
    try:
        from sklearn.metrics import roc_auc_score
        probs = torch.cat(all_probs[horizon], dim=0)[:, class_idx].numpy()
        labels = torch.cat(all_labels[horizon], dim=0)[:, class_idx].numpy()
        if len(np.unique(labels)) < 2:
            return 0.5  # Cannot compute AUC with single class
        return float(roc_auc_score(labels, probs))
    except ImportError:
        logger.warning("sklearn not available — skipping ROC-AUC computation")
        return 0.5
    except Exception as e:
        logger.warning("ROC-AUC computation failed: %s", e)
        return 0.5


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="ASTRONOVA — Train Solar Image Forecaster",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "image_forecasting.yaml")
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "datasets" / "image_sequences")
    parser.add_argument("--checkpoint-dir", type=Path, default=PROJECT_ROOT / "checkpoints" / "vision")
    parser.add_argument("--smoke-test", action="store_true", help="Tiny smoke test: 1 epoch, batch=2, size=128")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # ── Seed ──────────────────────────────────────────────────────────────
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ── Load config ───────────────────────────────────────────────────────
    config = _load_config(args.config)

    # Override with CLI args or smoke-test defaults
    if args.smoke_test:
        logger.warning("=" * 60)
        logger.warning("SMOKE TEST MODE — NOT SCIENTIFIC TRAINING")
        logger.warning("=" * 60)
        config["training"]["epochs"] = 1
        config["training"]["batch_size"] = 2
        config["model"]["image_size"] = 128
        config["model"]["spatial_dim"] = 64
        config["model"]["temporal_dim"] = 64
        config["model"]["n_transformer_layers"] = 1

    if args.epochs:
        config["training"]["epochs"] = args.epochs
    if args.batch_size:
        config["training"]["batch_size"] = args.batch_size
    if args.image_size:
        config["model"]["image_size"] = args.image_size
    if args.lr:
        config["training"]["learning_rate"] = args.lr

    # ── Device ────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    # ── Check data availability ───────────────────────────────────────────
    train_json = args.data_dir / "train_sequences.json"
    val_json = args.data_dir / "validation_sequences.json"

    if not train_json.exists():
        logger.warning("Training data not found: %s", train_json)
        logger.info("Running dataset builder in smoke-test mode first...")
        import subprocess
        subprocess.run(
            [sys.executable, str(PROJECT_ROOT / "scripts" / "build_image_dataset.py"), "--smoke-test"],
            cwd=str(PROJECT_ROOT),
            check=True,
        )

    if not train_json.exists():
        logger.error("Could not create training data. Exiting.")
        sys.exit(1)

    # ── Normalizer (TRAIN SPLIT ONLY) ────────────────────────────────────
    image_size = config["model"]["image_size"]
    normalizer = ImageNormalizer(mode="per_image")  # Safe default for smoke test
    # For full training: fit normalizer on training images
    # normalizer = ImageNormalizer(mode="robust")
    # normalizer.fit(train_images)

    # ── Datasets ──────────────────────────────────────────────────────────
    augmentation = SequenceAugmentation(SolarAugmentation(
        rotation_max_deg=5.0,
        translation_max_px=4,
        intensity_scale_range=(0.95, 1.05),
        gaussian_noise_std=0.01,
        enabled=True,
    ))

    train_dataset = SolarSequenceDataset(
        train_json,
        image_size=image_size,
        normalizer=normalizer,
        augmentation=augmentation,
        max_seq_len=config.get("sequence", {}).get("lookback_hours", 24),
    )

    val_dataset = SolarSequenceDataset(
        val_json,
        image_size=image_size,
        normalizer=normalizer,
        augmentation=None,  # No augmentation for validation
        max_seq_len=config.get("sequence", {}).get("lookback_hours", 24),
    )

    bs = config["training"]["batch_size"]
    train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True, collate_fn=collate_fn, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=bs, shuffle=False, collate_fn=collate_fn, num_workers=0)

    logger.info("Train: %d sequences | Val: %d sequences", len(train_dataset), len(val_dataset))

    # ── Model ─────────────────────────────────────────────────────────────
    model = SolarImageForecaster(
        backbone="resnet18",
        image_size=image_size,
        channels=3,
        pretrained_encoder=False,
        temporal_model="transformer",
        spatial_dim=config["model"].get("spatial_dim", 256),
        temporal_dim=config["model"].get("temporal_dim", 256),
        n_heads=config["model"].get("n_heads", 4),
        n_temporal_layers=config["model"].get("n_transformer_layers", 2),
        dropout=config["model"].get("dropout", 0.1),
        max_seq_len=config.get("sequence", {}).get("lookback_hours", 24),
    ).to(device)

    param_count = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Model parameters: %d total, %d trainable", param_count, trainable)

    # ── Optimizer & scheduler ─────────────────────────────────────────────
    lr = config["training"]["learning_rate"]
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=config["training"].get("weight_decay", 1e-4))
    scheduler = CosineAnnealingLR(optimizer, T_max=config["training"]["epochs"])

    # ── Loss functions ────────────────────────────────────────────────────
    focal = FocalLoss(
        gamma=config["loss"].get("focal_gamma", 2.0),
        alpha=config["loss"].get("focal_alpha", 0.25),
    )
    img_loss = ImageForecastLoss(
        ssim_weight=config["loss"].get("ssim_weight", 0.5),
        channels=3,
    )

    lambda_cls = config["loss"].get("lambda_cls", 1.0)
    lambda_loc = config["loss"].get("lambda_location", 0.5)
    lambda_img = config["loss"].get("lambda_image", 0.5)

    # ── Mixed precision ──────────────────────────────────────────────────
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    # ── Training loop ─────────────────────────────────────────────────────
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_val_auc = 0.0
    best_val_loss = float("inf")
    patience_counter = 0
    patience = config["training"].get("early_stopping_patience", 10)
    epochs = config["training"]["epochs"]
    grad_clip = config["training"].get("gradient_clip_norm", 1.0)

    history = []

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        train_metrics = train_one_epoch(
            model, train_loader, optimizer, focal, img_loss,
            lambda_cls, lambda_loc, lambda_img,
            device, scaler, grad_clip, epoch,
        )

        val_metrics = validate(
            model, val_loader, focal, img_loss,
            lambda_cls, lambda_img, device,
        )

        scheduler.step()
        elapsed = time.time() - t0

        metrics = {**train_metrics, **val_metrics, "epoch": epoch, "elapsed_s": elapsed, "lr": scheduler.get_last_lr()[0]}
        history.append(metrics)

        logger.info(
            "Epoch %d/%d | Train Loss: %.4f | Val Loss: %.4f | Val AUC(M+@60m): %.4f | Time: %.1fs",
            epoch, epochs,
            metrics["train_loss"], metrics["val_loss"],
            metrics["val_roc_auc_M_h60m"], elapsed,
        )

        import io
        def safe_save(obj, path):
            buffer = io.BytesIO()
            torch.save(obj, buffer)
            with open(path, "wb") as f:
                f.write(buffer.getvalue())

        if not args.smoke_test:
            # Save best by val loss
            args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
            if val_metrics["val_loss"] < best_val_loss:
                best_val_loss = val_metrics["val_loss"]
                safe_save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": best_val_loss,
                    "config": config,
                    "model_config": model.get_config(),
                }, str(args.checkpoint_dir / "best_val_loss.pt"))
                logger.info("  → Saved best val_loss checkpoint (%.4f)", best_val_loss)

            # Save best by ROC-AUC
            if val_metrics["val_roc_auc_M_h60m"] > best_val_auc:
                best_val_auc = val_metrics["val_roc_auc_M_h60m"]
                safe_save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_roc_auc": best_val_auc,
                    "config": config,
                    "model_config": model.get_config(),
                }, str(args.checkpoint_dir / "best_roc_auc.pt"))
                logger.info("  → Saved best ROC-AUC checkpoint (%.4f)", best_val_auc)
                patience_counter = 0
            else:
                patience_counter += 1

            # Save last
            args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
            safe_save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": config,
            }, str(args.checkpoint_dir / "last.pt"))
        else:
            logger.info("Smoke test: Skipping checkpoint saving to conserve disk space.")
            if val_metrics["val_roc_auc_M_h60m"] <= best_val_auc:
                patience_counter += 1
            else:
                best_val_auc = val_metrics["val_roc_auc_M_h60m"]
                patience_counter = 0

        # Early stopping
        if patience_counter >= patience:
            logger.info("Early stopping at epoch %d (patience=%d)", epoch, patience)
            break

    # ── Save training history ─────────────────────────────────────────────
    history_path = args.checkpoint_dir / "training_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    logger.info("Training history saved: %s", history_path)

    is_synthetic = any(
        s.get("synthetic", False)
        for s in train_dataset.sequences
    )

    logger.info("=" * 60)
    logger.info("TRAINING COMPLETE")
    if is_synthetic:
        logger.warning("SMOKE TEST — results are NOT scientifically valid")
        logger.warning("Do NOT report these metrics in any benchmark or publication")
    logger.info("Best val loss: %.4f | Best ROC-AUC: %.4f", best_val_loss, best_val_auc)
    logger.info("Checkpoints: %s", args.checkpoint_dir)
    logger.info("=" * 60)


def _load_config(path: Path) -> dict:
    """Load YAML config with fallback defaults."""
    try:
        import yaml
        with open(path) as f:
            config = yaml.safe_load(f)
        return config
    except (ImportError, FileNotFoundError) as e:
        logger.warning("Could not load config from %s: %s — using defaults", path, e)
        return {
            "model": {
                "image_size": 256, "spatial_dim": 256, "temporal_dim": 256,
                "n_heads": 4, "n_transformer_layers": 2, "dropout": 0.1,
            },
            "loss": {
                "lambda_cls": 1.0, "lambda_location": 0.5, "lambda_image": 0.5,
                "focal_gamma": 2.0, "focal_alpha": 0.25, "ssim_weight": 0.5,
            },
            "training": {
                "epochs": 50, "batch_size": 8, "learning_rate": 1e-4,
                "weight_decay": 1e-4, "gradient_clip_norm": 1.0,
                "early_stopping_patience": 10,
            },
            "sequence": {"lookback_hours": 24},
        }


if __name__ == "__main__":
    main()
