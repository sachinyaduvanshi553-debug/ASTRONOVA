"""Standalone evaluation script for ASTRONOVA trained models.

Loads saved .pkl model artifacts and evaluates them against:
  - Test set from processed parquet
  - Full scientific metric suite (TSS, HSS, FAR, Brier, AUC)
  - Confusion matrix
  - ROC curve data
  - Feature importance (where supported)

Usage:
    python -m services.forecasting.training.evaluate
"""
from __future__ import annotations

import json
import logging
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from astronova_core.utils.scientific_metrics import (
    compute_tss, compute_hss, compute_far,
    compute_brier_score, compute_roc_curve,
    generate_confusion_matrix,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger("astronova.evaluate")

MODELS_DIR  = PROJECT_ROOT / "models" / "artifacts"
METRICS_DIR = PROJECT_ROOT / "models" / "metrics"
PROCESSED   = PROJECT_ROOT / "datasets" / "processed" / "goes_processed.parquet"
METRICS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAMES = ["random_forest", "xgboost", "lightgbm"]

_EXCLUDE = {"label_binary", "label_class", "quality_flag",
            "soft_xray_flux_log_scaled", "hard_xray_flux_log_scaled"}


def load_test_data() -> tuple[np.ndarray, np.ndarray, List[str]]:
    if not PROCESSED.exists():
        raise FileNotFoundError(f"Processed dataset not found: {PROCESSED}. Run build_dataset.py first.")
    df = pd.read_parquet(PROCESSED)
    feat_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in _EXCLUDE]
    split = int(len(df) * 0.80)
    X_test = df[feat_cols].iloc[split:].fillna(0.0).values.astype(np.float32)
    y_test = df["label_binary"].iloc[split:].fillna(0).values.astype(int)
    logger.info("Test set: %d rows | %d features | %d positives.", len(X_test), len(feat_cols), int(y_test.sum()))
    return X_test, y_test, feat_cols


def evaluate_saved_model(name: str, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
    pkl_path = MODELS_DIR / f"{name}.pkl"
    if not pkl_path.exists():
        logger.warning("Model not found: %s", pkl_path)
        return {}

    with open(pkl_path, "rb") as fh:
        model = pickle.load(fh)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred.astype(float)

    cm = generate_confusion_matrix(y_test, y_pred)
    roc = compute_roc_curve(y_test, y_prob)

    metrics: Dict[str, Any] = {
        "model":        name,
        "tss":          compute_tss(y_test, y_pred),
        "hss":          compute_hss(y_test, y_pred),
        "far":          compute_far(y_test, y_pred),
        "brier_score":  compute_brier_score(y_test, y_prob),
        "auc_roc":      roc["auc"],
        "accuracy":     float((y_pred == y_test).mean()),
        "confusion_matrix": cm,
        "roc_curve":    {"fpr": roc["fpr"][:10], "tpr": roc["tpr"][:10]},  # truncated for JSON
    }

    # Feature importance (RF / LGB / XGB)
    if hasattr(model, "feature_importances_"):
        imp = model.feature_importances_
        top_idx = np.argsort(imp)[-10:][::-1]
        metrics["top_feature_importances"] = {str(i): float(imp[i]) for i in top_idx}

    logger.info(
        "[%s] TSS=%+.3f  HSS=%+.3f  AUC=%.3f  FAR=%.3f  Brier=%.4f",
        name, metrics["tss"], metrics["hss"], metrics["auc_roc"],
        metrics["far"], metrics["brier_score"],
    )

    out_path = METRICS_DIR / f"{name}_eval.json"
    with open(out_path, "w") as fh:
        json.dump(metrics, fh, indent=2, default=str)
    logger.info("Evaluation saved → %s", out_path)
    return metrics


def run_evaluation() -> Dict[str, Any]:
    X_test, y_test, feat_cols = load_test_data()
    all_results = {}
    for name in MODEL_NAMES:
        all_results[name] = evaluate_saved_model(name, X_test, y_test)

    # Comparative summary
    logger.info("\n%s", "=" * 60)
    logger.info("EVALUATION SUMMARY")
    logger.info("%-14s  %6s  %6s  %6s  %6s", "Model", "TSS", "HSS", "AUC", "FAR")
    for name, m in all_results.items():
        if m:
            logger.info(
                "%-14s  %+.3f  %+.3f  %.3f  %.3f",
                name, m.get("tss", 0), m.get("hss", 0),
                m.get("auc_roc", 0), m.get("far", 0),
            )
    return all_results


if __name__ == "__main__":
    run_evaluation()
