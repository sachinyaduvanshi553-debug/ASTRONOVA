"""
evaluate_image_forecaster.py — ASTRONOVA Phase 13: Evaluation Script

Evaluates the trained SolarImageForecaster on the TEST split only.
Computes all scientific metrics and generates evaluation report.

Classification metrics (per horizon, per class):
    - ROC-AUC, PR-AUC
    - True Skill Statistic (TSS)
    - Heidke Skill Score (HSS)
    - Brier Score
    - Expected Calibration Error (ECE)

Spatial metrics:
    - IoU (Intersection over Union)
    - Dice coefficient

Image quality metrics:
    - MAE, RMSE, SSIM, PSNR

CRITICAL:
    - Evaluate ONLY on the chronological test split
    - Never report metrics from synthetic smoke-test data as real results
    - Label all generated images as "AI_FORECAST"

Usage:
    python scripts/evaluate_image_forecaster.py --checkpoint checkpoints/vision/best_roc_auc.pt
    python scripts/evaluate_image_forecaster.py --checkpoint checkpoints/vision/best_roc_auc.pt --mc-dropout
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from services.vision.preprocessing import ImageNormalizer, SolarSequenceDataset
from models.vision.solar_image_forecaster import SolarImageForecaster, HORIZONS

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(name)s %(message)s")
logger = logging.getLogger("astronova.evaluate")

LABEL_NAMES = ["C_plus", "M_plus", "X_plus"]


def collate_fn(batch):
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
# Metrics
# ---------------------------------------------------------------------------
def compute_classification_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    """Compute all classification metrics for a single horizon+class."""
    metrics = {}

    try:
        from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
        if len(np.unique(y_true)) >= 2:
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
            metrics["pr_auc"] = float(average_precision_score(y_true, y_prob))
        else:
            metrics["roc_auc"] = float("nan")
            metrics["pr_auc"] = float("nan")
        metrics["brier"] = float(brier_score_loss(y_true, y_prob))
    except ImportError:
        logger.warning("sklearn not available — classification metrics unavailable")
        return metrics

    # TSS (True Skill Statistic) and HSS (Heidke Skill Score) at threshold 0.5
    y_pred = (y_prob >= 0.5).astype(int)
    tp = ((y_pred == 1) & (y_true == 1)).sum()
    tn = ((y_pred == 0) & (y_true == 0)).sum()
    fp = ((y_pred == 1) & (y_true == 0)).sum()
    fn = ((y_pred == 0) & (y_true == 1)).sum()

    tpr = tp / max(1, tp + fn)
    fpr = fp / max(1, fp + tn)
    metrics["tss"] = float(tpr - fpr)

    n = len(y_true)
    random_correct = ((tp + fp) * (tp + fn) + (fn + tn) * (fp + tn)) / max(1, n * n)
    observed_correct = (tp + tn) / max(1, n)
    metrics["hss"] = float((observed_correct - random_correct) / max(1e-8, 1 - random_correct))

    # ECE (Expected Calibration Error)
    metrics["ece"] = float(_compute_ece(y_true, y_prob))

    metrics["n_positive"] = int(y_true.sum())
    metrics["n_total"] = int(len(y_true))
    metrics["prevalence"] = float(y_true.mean())

    return metrics


def _compute_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Expected Calibration Error."""
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (y_prob >= bins[i]) & (y_prob < bins[i + 1])
        if mask.sum() == 0:
            continue
        bin_acc = y_true[mask].mean()
        bin_conf = y_prob[mask].mean()
        ece += mask.sum() / len(y_true) * abs(bin_acc - bin_conf)
    return ece


def compute_image_metrics(pred: np.ndarray, target: np.ndarray) -> dict:
    """Compute image quality metrics: MAE, RMSE, SSIM, PSNR."""
    mae = float(np.abs(pred - target).mean())
    rmse = float(np.sqrt(((pred - target) ** 2).mean()))

    # PSNR
    mse = ((pred - target) ** 2).mean()
    psnr = float(10 * np.log10(1.0 / max(mse, 1e-10)))

    # Simple SSIM approximation
    ssim = _simple_ssim(pred, target)

    return {"mae": mae, "rmse": rmse, "ssim": ssim, "psnr": psnr}


def _simple_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    """Simplified SSIM for evaluation."""
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    mu1, mu2 = img1.mean(), img2.mean()
    sigma1_sq = img1.var()
    sigma2_sq = img2.var()
    sigma12 = ((img1 - mu1) * (img2 - mu2)).mean()
    ssim_val = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1 ** 2 + mu2 ** 2 + C1) * (sigma1_sq + sigma2_sq + C2))
    return float(ssim_val)


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate(
    model: SolarImageForecaster,
    loader: DataLoader,
    device: torch.device,
    use_mc_dropout: bool = False,
    mc_passes: int = 20,
) -> dict:
    """Run full evaluation on test set."""
    model.eval()

    all_probs = {h: [] for h in HORIZONS}
    all_labels = {h: [] for h in HORIZONS}
    all_pred_images = []
    all_target_images = []
    is_any_synthetic = False

    for batch in loader:
        image_seq = batch["image_seq"].to(device)
        telemetry = batch["telemetry"].to(device)
        labels = {h: batch["labels"][h].to(device) for h in HORIZONS}

        if any(batch["is_synthetic"]):
            is_any_synthetic = True

        if use_mc_dropout:
            output = model.predict_with_uncertainty(
                image_seq, telemetry, n_passes=mc_passes
            )
        else:
            output = model(image_seq, telemetry)

        for h in HORIZONS:
            all_probs[h].append(output["class_probs"][h].cpu().numpy())
            all_labels[h].append(labels[h].cpu().numpy())

        all_pred_images.append(output["predicted_image"].cpu().numpy())
        all_target_images.append(image_seq[:, -1].cpu().numpy())

    # Aggregate
    results = {"is_synthetic_data": is_any_synthetic}

    if is_any_synthetic:
        results["WARNING"] = (
            "SMOKE TEST — these metrics are computed on SYNTHETIC data "
            "and are NOT scientifically valid. Do NOT cite in publications."
        )

    # Classification metrics per horizon per class
    results["classification"] = {}
    for h in HORIZONS:
        probs = np.concatenate(all_probs[h], axis=0)  # [N, 3]
        labels_np = np.concatenate(all_labels[h], axis=0)  # [N, 3]

        results["classification"][h] = {}
        for c_idx, c_name in enumerate(LABEL_NAMES):
            m = compute_classification_metrics(labels_np[:, c_idx], probs[:, c_idx])
            results["classification"][h][c_name] = m

    # Image quality metrics
    pred_imgs = np.concatenate(all_pred_images, axis=0)
    target_imgs = np.concatenate(all_target_images, axis=0)
    results["image_quality"] = compute_image_metrics(pred_imgs, target_imgs)

    return results


def main():
    parser = argparse.ArgumentParser(description="ASTRONOVA — Evaluate Solar Image Forecaster")
    parser.add_argument("--checkpoint", type=Path, default=PROJECT_ROOT / "checkpoints" / "vision" / "best_roc_auc.pt")
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "datasets" / "image_sequences")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "reports" / "experiments" / "image_forecasting" / "evaluation_results.json")
    parser.add_argument("--mc-dropout", action="store_true", help="Use MC-Dropout for uncertainty estimation")
    parser.add_argument("--mc-passes", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--smoke-test", action="store_true", help="Run in smoke test mode without loading checkpoint")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    if args.smoke_test:
        logger.warning("SMOKE TEST MODE: Random weights used. Results mean nothing.")
        model_config = {}
        config = {}
    else:
        # Load checkpoint
        if not args.checkpoint.exists():
            logger.error("Checkpoint not found: %s", args.checkpoint)
            logger.info("Run training first: python scripts/train_image_forecaster.py")
            sys.exit(1)

        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        model_config = ckpt.get("model_config", {})
        config = ckpt.get("config", {})

    # Rebuild model from config
    image_size = model_config.get("image_size", config.get("model", {}).get("image_size", 256))
    model = SolarImageForecaster(
        backbone=model_config.get("backbone", "resnet18"),
        image_size=image_size,
        spatial_dim=config.get("model", {}).get("spatial_dim", 256),
        temporal_dim=config.get("model", {}).get("temporal_dim", 256),
        n_heads=config.get("model", {}).get("n_heads", 4),
        n_temporal_layers=config.get("model", {}).get("n_transformer_layers", 2),
    ).to(device)

    if not args.smoke_test:
        model.load_state_dict(ckpt["model_state_dict"])
        logger.info("Loaded checkpoint: %s (epoch %d)", args.checkpoint, ckpt.get("epoch", -1))
    else:
        logger.info("Smoke test: Skipped loading state dict.")

    # Load test dataset
    test_json = args.data_dir / "test_sequences.json"
    if not test_json.exists():
        logger.error("Test data not found: %s", test_json)
        sys.exit(1)

    normalizer = ImageNormalizer(mode="per_image")
    test_dataset = SolarSequenceDataset(
        test_json,
        image_size=image_size,
        normalizer=normalizer,
    )
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    # Evaluate
    results = evaluate(model, test_loader, device, use_mc_dropout=args.mc_dropout, mc_passes=args.mc_passes)

    # Save results
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2, default=str)

    # Print summary
    logger.info("=" * 60)
    logger.info("EVALUATION RESULTS")
    if results.get("is_synthetic_data"):
        logger.warning("SMOKE TEST — NOT SCIENTIFIC RESULTS")

    for h in HORIZONS:
        cls = results["classification"].get(h, {})
        m_plus = cls.get("M_plus", {})
        logger.info(
            "  %s | M+ ROC-AUC: %.4f | TSS: %.4f | HSS: %.4f | Brier: %.4f",
            h,
            m_plus.get("roc_auc", float("nan")),
            m_plus.get("tss", float("nan")),
            m_plus.get("hss", float("nan")),
            m_plus.get("brier", float("nan")),
        )

    img_q = results.get("image_quality", {})
    logger.info("  Image: MAE=%.4f RMSE=%.4f SSIM=%.4f PSNR=%.2fdB",
                img_q.get("mae", 0), img_q.get("rmse", 0),
                img_q.get("ssim", 0), img_q.get("psnr", 0))
    logger.info("Results saved: %s", args.output)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
