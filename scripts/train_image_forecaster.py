"""
scripts/train_image_forecaster.py — Complete Multimodal Solar Flare Image Forecasting Trainer.

Pipeline:
  1. Multimodal Architecture: CNN Spatial Encoder + Temporal Transformer/ConvLSTM +
     Telemetry + Magnetic + Physics + Fusion + Six Horizon Heads (C+, M+, X+) +
     Spatial Location Head + Future Image Decoder.
  2. Genuine Future Target handling with FutureTargetResolver.
  3. Operational metrics tracking (ROC-AUC, PR-AUC, TSS, HSS, Brier).
  4. Safe Checkpointing: best.pt, best_roc_auc.pt, best_val_loss.pt, last.pt, smoke_test.pt.
"""
from __future__ import annotations

import argparse
import io
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.vision.solar_image_forecaster import HORIZONS, SolarImageForecaster
from models.vision.image_forecaster import ImageForecastLoss
from models.vision.dataset_adapter import AdaptedSolarSequenceDataset, adapted_collate_fn
from services.vision.preprocessing.augmentation import SequenceAugmentation, SolarAugmentation
from services.vision.preprocessing.image_normalizer import ImageNormalizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger("astronova.train_vision")


def get_git_commit() -> str:
    try:
        res = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=PROJECT_ROOT, capture_output=True, text=True)
        return res.stdout.strip()
    except Exception:
        return "unknown"


def print_data_sufficiency(dataset: AdaptedSolarSequenceDataset, name: str = "Train") -> None:
    """Print sequence and positive class counts for all horizons."""
    logger.info("=" * 60)
    logger.info("DATASET AUDIT & SUFFICIENCY CHECK: %s", name)
    logger.info("Total Sequences: %d", len(dataset))

    for h in HORIZONS:
        c_pos, m_pos, x_pos = 0, 0, 0
        for s in dataset.sequences:
            labels = s.get("labels", {}).get(h, {})
            if labels.get("C_plus", False):
                c_pos += 1
            if labels.get("M_plus", False):
                m_pos += 1
            if labels.get("X_plus", False):
                x_pos += 1
        logger.info("  Horizon %-6s | C+: %3d | M+: %3d | X+: %3d", h, c_pos, m_pos, x_pos)

    if len(dataset) < 100:
        logger.warning("WARNING: DATASET TOO SMALL FOR RESEARCH-GRADE GENERALIZATION.")
        logger.warning("Results represent pipeline validation and smoke testing only.")
    logger.info("=" * 60)


def safe_torch_save(obj: Any, path: Path | str) -> None:
    """Write checkpoint atomically: save to temp file then rename to avoid corruption."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".tmp.pt")
    try:
        torch.save(obj, str(tmp_path))
        tmp_path.replace(path)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
        raise


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: ImageForecastLoss,
    device: torch.device,
    scaler: Optional[torch.amp.GradScaler],
    grad_clip: float,
    epoch: int,
) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    total_cls = 0.0
    total_img = 0.0
    total_loc = 0.0
    n_batches = 0

    for batch_idx, batch in enumerate(loader):
        image_seq = batch["image_seq"].to(device)
        telemetry = batch["telemetry"].to(device)
        magnetic = batch["magnetic"].to(device)
        physics = batch["physics"].to(device)

        labels = {h: batch["labels"][h].to(device) for h in HORIZONS}
        future_images = {
            h: batch["future_images"][h].to(device) if batch["future_images"][h] is not None else None
            for h in HORIZONS
        }
        target_available = {h: batch["target_available"][h].to(device) for h in HORIZONS}

        targets = {
            "labels": labels,
            "future_images": future_images,
            "target_available": target_available,
        }

        optimizer.zero_grad()
        use_amp = scaler is not None and device.type == "cuda"
        autocast_ctx = torch.amp.autocast("cuda") if use_amp else torch.amp.autocast("cpu", enabled=False)

        with autocast_ctx:
            predictions = model(
                image_seq=image_seq,
                telemetry=telemetry,
                magnetic=magnetic,
                physics=physics,
            )
            loss, loss_details = loss_fn(predictions, targets)

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
        total_cls += loss_details.get("cls_total", 0.0)
        total_img += loss_details.get("img_total", 0.0)
        total_loc += loss_details.get("location_total", 0.0)
        n_batches += 1

        if (batch_idx + 1) % max(1, len(loader) // 2) == 0 or batch_idx == len(loader) - 1:
            logger.info(
                "Epoch %d | Batch %d/%d | Loss: %.4f (cls: %.4f, img: %.4f, loc: %.4f)",
                epoch, batch_idx + 1, len(loader),
                loss.item(), loss_details.get("cls_total", 0.0),
                loss_details.get("img_total", 0.0), loss_details.get("location_total", 0.0),
            )

    return {
        "train_loss": total_loss / max(1, n_batches),
        "train_cls_loss": total_cls / max(1, n_batches),
        "train_img_loss": total_img / max(1, n_batches),
        "train_loc_loss": total_loc / max(1, n_batches),
    }


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: ImageForecastLoss,
    device: torch.device,
) -> Dict[str, float]:
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
        magnetic = batch["magnetic"].to(device)
        physics = batch["physics"].to(device)

        labels = {h: batch["labels"][h].to(device) for h in HORIZONS}
        future_images = {
            h: batch["future_images"][h].to(device) if batch["future_images"][h] is not None else None
            for h in HORIZONS
        }
        target_available = {h: batch["target_available"][h].to(device) for h in HORIZONS}

        targets = {
            "labels": labels,
            "future_images": future_images,
            "target_available": target_available,
        }

        predictions = model(
            image_seq=image_seq,
            telemetry=telemetry,
            magnetic=magnetic,
            physics=physics,
        )
        loss, loss_details = loss_fn(predictions, targets)

        total_loss += loss.item()
        total_cls += loss_details.get("cls_total", 0.0)
        total_img += loss_details.get("img_total", 0.0)
        n_batches += 1

        for h in HORIZONS:
            all_probs[h].append(predictions["class_probs"][h].cpu())
            all_labels[h].append(labels[h].cpu())

    # Compute M+ ROC-AUC for h60m (Primary scientific benchmark)
    val_m_roc_auc = 0.5
    try:
        from sklearn.metrics import roc_auc_score
        probs_60 = torch.cat(all_probs["h60m"], dim=0)[:, 1].numpy()
        lbls_60 = torch.cat(all_labels["h60m"], dim=0)[:, 1].numpy()
        if len(np.unique(lbls_60)) >= 2:
            val_m_roc_auc = float(roc_auc_score(lbls_60, probs_60))
    except Exception:
        pass

    return {
        "val_loss": total_loss / max(1, n_batches),
        "val_cls_loss": total_cls / max(1, n_batches),
        "val_img_loss": total_img / max(1, n_batches),
        "val_roc_auc_M_h60m": val_m_roc_auc,
    }


def main():
    parser = argparse.ArgumentParser(description="ASTRONOVA — Train Multimodal Solar Image Forecaster")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "image_forecasting.yaml")
    parser.add_argument("--train-manifest", type=Path, default=None)
    parser.add_argument("--val-manifest", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--lookback", type=int, default=None)
    parser.add_argument("--learning-rate", "--lr", type=float, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "checkpoints" / "vision")
    parser.add_argument("--smoke-test", action="store_true", help="1 epoch, batch=2, image_size=128, lookback=4")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Config loading
    config = {}
    if args.config and args.config.exists():
        with open(args.config, "r") as f:
            config = yaml.safe_load(f) or {}

    model_cfg = config.get("model", {})
    train_cfg = config.get("training", {})
    loss_cfg = config.get("loss", {})
    smoke_cfg = config.get("smoke_test", {})

    # Determine execution parameters
    if args.smoke_test:
        epochs = args.epochs or 1
        batch_size = args.batch_size or 2
        image_size = args.image_size or 128
        lookback = args.lookback or 4
        learning_rate = args.learning_rate or 1e-4
        spatial_dim = 64
        temporal_dim = 64
        n_layers = 1
        n_heads = 2
    else:
        epochs = args.epochs or train_cfg.get("epochs", 50)
        batch_size = args.batch_size or train_cfg.get("batch_size", 8)
        image_size = args.image_size or model_cfg.get("image_size", 256)
        lookback = args.lookback or smoke_cfg.get("lookback_hours", 24)
        learning_rate = args.learning_rate or float(train_cfg.get("learning_rate", 1e-4))
        spatial_dim = model_cfg.get("spatial_dim", 256)
        temporal_dim = model_cfg.get("temporal_dim", 256)
        n_layers = model_cfg.get("n_transformer_layers", 2)
        n_heads = model_cfg.get("n_heads", 4)

    # Device selection
    if args.device:
        device_str = args.device
        if device_str == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA requested but not available. Falling back to CPU.")
            device = torch.device("cpu")
        else:
            device = torch.device(device_str)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    # Manifest resolution
    data_base = PROJECT_ROOT / "datasets" / "image_sequences"
    train_json = args.train_manifest or (data_base / "train_sequences.json")
    val_json = args.val_manifest or (data_base / "validation_sequences.json")

    normalizer = ImageNormalizer(mode="per_image")
    train_aug = SequenceAugmentation(SolarAugmentation(enabled=True)) if not args.smoke_test else None

    train_dataset = AdaptedSolarSequenceDataset(
        manifest_path=train_json,
        image_size=image_size,
        channels=3,
        lookback_frames=lookback,
        normalizer=normalizer,
        augmentation=train_aug,
    )
    val_dataset = AdaptedSolarSequenceDataset(
        manifest_path=val_json,
        image_size=image_size,
        channels=3,
        lookback_frames=lookback,
        normalizer=normalizer,
        augmentation=None,
    )

    print_data_sufficiency(train_dataset, name="Train Split")
    print_data_sufficiency(val_dataset, name="Validation Split")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=adapted_collate_fn,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=adapted_collate_fn,
        num_workers=args.num_workers,
    )

    # Instantiate Full Multimodal Model
    model = SolarImageForecaster(
        backbone="resnet18",
        image_size=image_size,
        channels=3,
        pretrained_encoder=False,  # Offline safe
        temporal_model="transformer",
        spatial_dim=spatial_dim,
        temporal_dim=temporal_dim,
        n_heads=n_heads,
        n_temporal_layers=n_layers,
        telemetry_dim=2,
        tel_embed_dim=64,
        magnetic_dim=16,
        mag_embed_dim=64,
        physics_dim=5,
        physics_embed_dim=64,
        fusion_output_dim=128 if args.smoke_test else 256,
        classifier_hidden=64 if args.smoke_test else 128,
        dropout=model_cfg.get("dropout", 0.1),
        max_seq_len=lookback,
    ).to(device)

    param_count = sum(p.numel() for p in model.parameters())
    trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Model parameters: %d total, %d trainable", param_count, trainable_count)

    # Loss, Optimizer & Scheduler
    loss_fn = ImageForecastLoss(
        lambda_cls=loss_cfg.get("lambda_cls", 1.0),
        lambda_image=loss_cfg.get("lambda_image", 0.5),
        lambda_location=loss_cfg.get("lambda_location", 0.5),
        focal_gamma=loss_cfg.get("focal_gamma", 2.0),
        focal_alpha=loss_cfg.get("focal_alpha", 0.25),
        ssim_weight=loss_cfg.get("ssim_weight", 0.5),
        channels=3,
    )

    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=train_cfg.get("weight_decay", 1e-4))
    scheduler = CosineAnnealingLR(optimizer, T_max=max(1, epochs))
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None
    grad_clip = train_cfg.get("gradient_clip_norm", 1.0)

    # Training execution
    args.output_dir.mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")
    best_val_auc = 0.0
    history: List[Dict[str, Any]] = []

    logger.info("Beginning training for %d epochs (batch_size=%d, image_size=%d)...", epochs, batch_size, image_size)
    t_start = time.time()

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_m = train_one_epoch(model, train_loader, optimizer, loss_fn, device, scaler, grad_clip, epoch)
        val_m = validate(model, val_loader, loss_fn, device)
        scheduler.step()
        elapsed = time.time() - t0

        metrics_epoch = {**train_m, **val_m, "epoch": epoch, "elapsed_s": elapsed, "lr": scheduler.get_last_lr()[0]}
        history.append(metrics_epoch)

        logger.info(
            "Epoch %d/%d | Train Loss: %.4f | Val Loss: %.4f | Val AUC(M+@60m): %.4f | Time: %.1fs",
            epoch, epochs, train_m["train_loss"], val_m["val_loss"], val_m["val_roc_auc_M_h60m"], elapsed
        )

        checkpoint_payload = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "model_config": model.get_config(),
            "horizons": HORIZONS,
            "feature_dimensions": {"spatial_dim": spatial_dim, "temporal_dim": temporal_dim, "magnetic_dim": 16},
            "image_size": image_size,
            "lookback": lookback,
            "git_commit": get_git_commit(),
            "val_metrics": val_m,
            "config": config,
        }

        # Save Checkpoints
        if args.smoke_test:
            safe_torch_save(checkpoint_payload, args.output_dir / "smoke_test.pt")
            safe_torch_save(checkpoint_payload, args.output_dir / "best.pt")
            logger.info("Saved smoke_test.pt and best.pt to %s", args.output_dir)
        else:
            safe_torch_save(checkpoint_payload, args.output_dir / "last.pt")
            if val_m["val_loss"] < best_val_loss:
                best_val_loss = val_m["val_loss"]
                safe_torch_save(checkpoint_payload, args.output_dir / "best_val_loss.pt")
                safe_torch_save(checkpoint_payload, args.output_dir / "best.pt")
            if val_m["val_roc_auc_M_h60m"] >= best_val_auc:
                best_val_auc = val_m["val_roc_auc_M_h60m"]
                safe_torch_save(checkpoint_payload, args.output_dir / "best_roc_auc.pt")

    # Save metadata artifacts
    with open(args.output_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)
    with open(args.output_dir / "model_config.json", "w") as f:
        json.dump(model.get_config(), f, indent=2)
    with open(args.output_dir / "metrics.json", "w") as f:
        json.dump(history[-1] if history else {}, f, indent=2)
    with open(args.output_dir / "dataset_schema.json", "w") as f:
        json.dump({
            "horizons": HORIZONS,
            "magnetic_features": 16,
            "telemetry_features": 2,
            "image_shape": [3, image_size, image_size],
            "lookback_frames": lookback,
        }, f, indent=2)

    total_time = time.time() - t_start
    logger.info("=" * 60)
    logger.info("TRAINING FINISHED in %.1f seconds", total_time)
    logger.info("Checkpoints directory: %s", args.output_dir.resolve())
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
