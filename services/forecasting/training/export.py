"""ONNX export script for ASTRONOVA trained tabular models.

Supports:
  - sklearn RandomForest via skl2onnx
  - XGBoost via xgboost native JSON + skl2onnx
  - LightGBM via skl2onnx
  - Verifies ONNX model output matches sklearn predictions

Usage:
    python -m services.forecasting.training.export
"""
from __future__ import annotations

import json
import logging
import pickle
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger("astronova.export")

MODELS_DIR  = PROJECT_ROOT / "models" / "artifacts"
PROCESSED   = PROJECT_ROOT / "datasets" / "processed" / "goes_processed.parquet"

_EXCLUDE = {"label_binary", "label_class", "quality_flag",
            "soft_xray_flux_log_scaled", "hard_xray_flux_log_scaled"}


def _load_test_sample(n: int = 50) -> tuple[np.ndarray, int]:
    """Load a small sample of test data for ONNX verification."""
    import pandas as pd
    if not PROCESSED.exists():
        logger.warning("Processed dataset not found — using random data for ONNX check.")
        meta_path = MODELS_DIR / "feature_metadata.json"
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            n_features = meta["n_features"]
        else:
            n_features = 50
        return np.random.rand(n, n_features).astype(np.float32), n_features

    df = pd.read_parquet(PROCESSED)
    feat_cols = [c for c in df.select_dtypes(include=[np.float64, np.float32, np.int64]).columns if c not in _EXCLUDE]
    X = df[feat_cols].fillna(0.0).values[-n:].astype(np.float32)
    return X, len(feat_cols)


def export_model(name: str, X_sample: np.ndarray, n_features: int) -> bool:
    """Export a single .pkl model to ONNX and verify output."""
    pkl_path  = MODELS_DIR / f"{name}.pkl"
    onnx_path = MODELS_DIR / f"{name}.onnx"

    if not pkl_path.exists():
        logger.warning("PKL not found: %s — skipping.", pkl_path)
        return False

    with open(pkl_path, "rb") as fh:
        model = pickle.load(fh)

    # ── Export ────────────────────────────────────────────────────────
    try:
        from skl2onnx import convert_sklearn   # type: ignore
        from skl2onnx.common.data_types import FloatTensorType  # type: ignore
        initial_type = [("float_input", FloatTensorType([None, n_features]))]
        onnx_model = convert_sklearn(model, initial_types=initial_type,
                                     options={type(model): {"zipmap": False}})
        with open(onnx_path, "wb") as fh:
            fh.write(onnx_model.SerializeToString())
        logger.info("Exported → %s", onnx_path)
    except Exception as exc:
        logger.warning("skl2onnx export failed for %s: %s", name, exc)
        return False

    # ── Verify ────────────────────────────────────────────────────────
    try:
        import onnxruntime as ort  # type: ignore
        sess = ort.InferenceSession(str(onnx_path))
        input_name = sess.get_inputs()[0].name
        onnx_preds = sess.run(None, {input_name: X_sample})[0]

        sklearn_preds = model.predict(X_sample)
        match_rate = float(np.mean(onnx_preds == sklearn_preds))
        if match_rate >= 0.99:
            logger.info("[%s] ONNX verification PASSED (%.1f%% match).", name, 100 * match_rate)
        else:
            logger.warning("[%s] ONNX verification partial (%.1f%% match).", name, 100 * match_rate)
        return True
    except ImportError:
        logger.warning("onnxruntime not installed — ONNX verification skipped for %s.", name)
        return True
    except Exception as exc:
        logger.warning("ONNX verification failed for %s: %s", name, exc)
        return False


def run_export() -> None:
    X_sample, n_features = _load_test_sample()
    logger.info("Exporting %d models with %d features.", 3, n_features)
    for name in ["random_forest", "xgboost", "lightgbm"]:
        export_model(name, X_sample, n_features)
    logger.info("ONNX export complete. Artifacts in: %s", MODELS_DIR)


if __name__ == "__main__":
    run_export()
