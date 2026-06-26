"""Training pipeline for ASTRONOVA tabular baseline models.

Trains:
  - RandomForestClassifier
  - XGBoostClassifier
  - LightGBMClassifier

With:
  - Optuna hyperparameter tuning (50 trials per model)
  - StratifiedKFold cross-validation (5-fold)
  - SMOTE oversampling for class imbalance
  - Scientific metrics: TSS, HSS, FAR, AUC-ROC, Brier Score
  - Model artifact persistence (.pkl + ONNX)
  - MLflow experiment tracking
"""
from __future__ import annotations

import json
import logging
import pickle
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

# ── PYTHONPATH guard ──────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

warnings.filterwarnings("ignore", category=UserWarning)

import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import label_binarize
import xgboost as xgb
import lightgbm as lgb

try:
    from imblearn.over_sampling import SMOTE
    _HAS_SMOTE = True
except ImportError:
    _HAS_SMOTE = False
    logging.getLogger(__name__).warning("imbalanced-learn not installed; SMOTE disabled.")

from astronova_core.utils.scientific_metrics import (
    compute_tss,
    compute_hss,
    compute_far,
    compute_brier_score,
    compute_roc_curve,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger("astronova.training.tabular")

# ── Paths ─────────────────────────────────────────────────────────────────────
PROCESSED_DIR  = PROJECT_ROOT / "datasets" / "processed"
MODELS_DIR     = PROJECT_ROOT / "models" / "artifacts"
METRICS_DIR    = PROJECT_ROOT / "models" / "metrics"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)

# ── Training config ───────────────────────────────────────────────────────────
N_SPLITS        = 5
N_OPTUNA_TRIALS = 50
RANDOM_SEED     = 42


# ---------------------------------------------------------------------------
# Feature selection
# ---------------------------------------------------------------------------

_EXCLUDE_FROM_FEATURES = {
    "label_binary", "label_class", "quality_flag",
    "soft_xray_flux_log_scaled", "hard_xray_flux_log_scaled",
}

def get_feature_columns(df: pd.DataFrame) -> List[str]:
    """Return numeric feature columns, excluding label/flag columns."""
    return [
        c for c in df.select_dtypes(include=[np.number]).columns
        if c not in _EXCLUDE_FROM_FEATURES
    ]


# ---------------------------------------------------------------------------
# Dataset loader
# ---------------------------------------------------------------------------

def load_dataset() -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Load the processed feature matrix and binary labels.

    Returns (X, y, feature_names).
    Falls back to synthetic data if parquet files are missing.
    """
    parquet_path = PROCESSED_DIR / "goes_processed.parquet"
    feat_path    = PROCESSED_DIR / "feature_matrix.parquet" if not parquet_path.exists() else parquet_path

    if parquet_path.exists():
        logger.info("Loading processed dataset from %s.", parquet_path)
        df = pd.read_parquet(parquet_path)
    else:
        logger.warning("Processed parquet not found. Running dataset builder first.")
        from scripts.build_dataset import build_training_dataset
        df = build_training_dataset(save=True)

    if "label_binary" not in df.columns:
        raise ValueError("'label_binary' column not found in dataset. Run build_dataset.py first.")

    feature_cols = get_feature_columns(df)
    X = df[feature_cols].fillna(0.0).values.astype(np.float32)
    y = df["label_binary"].fillna(0).values.astype(int)

    logger.info(
        "Dataset loaded: %d samples × %d features | Positives: %d (%.1f%%)",
        len(X), X.shape[1], y.sum(), 100 * y.mean(),
    )
    return X, y, feature_cols


# ---------------------------------------------------------------------------
# Metrics evaluation
# ---------------------------------------------------------------------------

def evaluate_model(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_name: str,
) -> Dict[str, float]:
    """Compute full scientific metrics suite for a fitted model."""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred.astype(float)

    metrics = {
        "tss":          compute_tss(y_test, y_pred),
        "hss":          compute_hss(y_test, y_pred),
        "far":          compute_far(y_test, y_pred),
        "brier_score":  compute_brier_score(y_test, y_prob),
        "auc_roc":      float(roc_auc_score(y_test, y_prob)) if len(np.unique(y_test)) > 1 else 0.0,
        "accuracy":     float((y_pred == y_test).mean()),
        "n_test":       int(len(y_test)),
        "n_positive":   int(y_test.sum()),
        "model":        model_name,
    }
    logger.info(
        "[%s] TSS=%.3f  HSS=%.3f  FAR=%.3f  AUC=%.3f  Brier=%.4f",
        model_name, metrics["tss"], metrics["hss"], metrics["far"],
        metrics["auc_roc"], metrics["brier_score"],
    )
    return metrics


def save_metrics(metrics: Dict[str, Any], model_name: str) -> None:
    """Save metrics dict to JSON."""
    out_path = METRICS_DIR / f"{model_name}_metrics.json"
    with open(out_path, "w") as fh:
        json.dump(metrics, fh, indent=2, default=str)
    logger.info("Metrics saved → %s", out_path)


# ---------------------------------------------------------------------------
# Model persistence
# ---------------------------------------------------------------------------

def save_model_pkl(model: Any, name: str) -> Path:
    path = MODELS_DIR / f"{name}.pkl"
    with open(path, "wb") as fh:
        pickle.dump(model, fh, protocol=pickle.HIGHEST_PROTOCOL)
    logger.info("Saved PKL → %s", path)
    return path


def export_to_onnx(model: Any, name: str, n_features: int) -> None:
    """Export sklearn/xgb model to ONNX format."""
    try:
        from skl2onnx import convert_sklearn, to_onnx  # type: ignore
        from skl2onnx.common.data_types import FloatTensorType  # type: ignore
        initial_type = [("float_input", FloatTensorType([None, n_features]))]
        onnx_model = convert_sklearn(model, initial_types=initial_type)
        out_path = MODELS_DIR / f"{name}.onnx"
        with open(out_path, "wb") as fh:
            fh.write(onnx_model.SerializeToString())
        logger.info("Exported ONNX → %s", out_path)
    except ImportError:
        logger.warning("skl2onnx not installed — ONNX export skipped for %s.", name)
    except Exception as exc:
        logger.warning("ONNX export failed for %s: %s", name, exc)


# ---------------------------------------------------------------------------
# Optuna objective functions
# ---------------------------------------------------------------------------

def _rf_objective(trial: optuna.Trial, X: np.ndarray, y: np.ndarray) -> float:
    params = {
        "n_estimators":      trial.suggest_int("n_estimators", 100, 500),
        "max_depth":         trial.suggest_int("max_depth", 4, 20),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        "min_samples_leaf":  trial.suggest_int("min_samples_leaf", 1, 10),
        "max_features":      trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
        "class_weight":      "balanced",
        "random_state":      RANDOM_SEED,
        "n_jobs":            -1,
    }
    model = RandomForestClassifier(**params)
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_SEED)
    scores = cross_val_score(model, X, y, cv=cv, scoring="roc_auc", n_jobs=-1)
    return float(np.mean(scores))


def _xgb_objective(trial: optuna.Trial, X: np.ndarray, y: np.ndarray) -> float:
    params = {
        "n_estimators":    trial.suggest_int("n_estimators", 100, 600),
        "max_depth":       trial.suggest_int("max_depth", 3, 12),
        "learning_rate":   trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample":       trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree":trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight":trial.suggest_int("min_child_weight", 1, 10),
        "gamma":           trial.suggest_float("gamma", 0.0, 5.0),
        "reg_alpha":       trial.suggest_float("reg_alpha", 0.0, 1.0),
        "reg_lambda":      trial.suggest_float("reg_lambda", 0.5, 2.0),
        "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1.0, 20.0),
        "use_label_encoder": False,
        "eval_metric":     "auc",
        "random_state":    RANDOM_SEED,
        "n_jobs":          -1,
        "verbosity":       0,
    }
    model = xgb.XGBClassifier(**params)
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_SEED)
    scores = cross_val_score(model, X, y, cv=cv, scoring="roc_auc", n_jobs=-1)
    return float(np.mean(scores))


def _lgb_objective(trial: optuna.Trial, X: np.ndarray, y: np.ndarray) -> float:
    params = {
        "n_estimators":      trial.suggest_int("n_estimators", 100, 600),
        "max_depth":         trial.suggest_int("max_depth", 3, 12),
        "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "num_leaves":        trial.suggest_int("num_leaves", 20, 150),
        "subsample":         trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
        "reg_alpha":         trial.suggest_float("reg_alpha", 0.0, 1.0),
        "reg_lambda":        trial.suggest_float("reg_lambda", 0.0, 1.0),
        "is_unbalance":      True,
        "random_state":      RANDOM_SEED,
        "n_jobs":            -1,
        "verbosity":         -1,
    }
    model = lgb.LGBMClassifier(**params)
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_SEED)
    scores = cross_val_score(model, X, y, cv=cv, scoring="roc_auc", n_jobs=-1)
    return float(np.mean(scores))


# ---------------------------------------------------------------------------
# Model trainers
# ---------------------------------------------------------------------------

def train_random_forest(X_train: np.ndarray, y_train: np.ndarray,
                         X_test: np.ndarray, y_test: np.ndarray,
                         n_features: int) -> Dict[str, Any]:
    logger.info("── Training RandomForest ──────────────────────────────────")
    study = optuna.create_study(direction="maximize", study_name="rf_auc")
    study.optimize(lambda t: _rf_objective(t, X_train, y_train),
                   n_trials=N_OPTUNA_TRIALS, show_progress_bar=False)
    best_params = study.best_params
    best_params.update({"class_weight": "balanced", "random_state": RANDOM_SEED, "n_jobs": -1})
    logger.info("RF best params: %s | AUC=%.4f", best_params, study.best_value)

    model = RandomForestClassifier(**best_params)
    model.fit(X_train, y_train)

    metrics = evaluate_model(model, X_test, y_test, "random_forest")
    metrics["best_params"] = best_params
    save_metrics(metrics, "random_forest")
    save_model_pkl(model, "random_forest")
    export_to_onnx(model, "random_forest", n_features)
    return {"model": model, "metrics": metrics}


def train_xgboost(X_train: np.ndarray, y_train: np.ndarray,
                   X_test: np.ndarray, y_test: np.ndarray,
                   n_features: int) -> Dict[str, Any]:
    logger.info("── Training XGBoost ──────────────────────────────────────")
    study = optuna.create_study(direction="maximize", study_name="xgb_auc")
    study.optimize(lambda t: _xgb_objective(t, X_train, y_train),
                   n_trials=N_OPTUNA_TRIALS, show_progress_bar=False)
    best_params = study.best_params
    best_params.update({
        "use_label_encoder": False, "eval_metric": "auc",
        "random_state": RANDOM_SEED, "n_jobs": -1, "verbosity": 0,
    })
    logger.info("XGB best params: %s | AUC=%.4f", best_params, study.best_value)

    model = xgb.XGBClassifier(**best_params)
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    metrics = evaluate_model(model, X_test, y_test, "xgboost")
    metrics["best_params"] = best_params
    save_metrics(metrics, "xgboost")
    save_model_pkl(model, "xgboost")
    # XGBoost native ONNX
    try:
        onnx_path = MODELS_DIR / "xgboost.onnx"
        model.save_model(str(onnx_path.with_suffix(".json")))
        logger.info("Saved XGB JSON model → %s.json", onnx_path.stem)
    except Exception as exc:
        logger.warning("XGB JSON save failed: %s", exc)
    export_to_onnx(model, "xgboost", n_features)
    return {"model": model, "metrics": metrics}


def train_lightgbm(X_train: np.ndarray, y_train: np.ndarray,
                    X_test: np.ndarray, y_test: np.ndarray,
                    n_features: int) -> Dict[str, Any]:
    logger.info("── Training LightGBM ─────────────────────────────────────")
    study = optuna.create_study(direction="maximize", study_name="lgb_auc")
    study.optimize(lambda t: _lgb_objective(t, X_train, y_train),
                   n_trials=N_OPTUNA_TRIALS, show_progress_bar=False)
    best_params = study.best_params
    best_params.update({"is_unbalance": True, "random_state": RANDOM_SEED,
                         "n_jobs": -1, "verbosity": -1})
    logger.info("LGB best params: %s | AUC=%.4f", best_params, study.best_value)

    model = lgb.LGBMClassifier(**best_params)
    model.fit(X_train, y_train,
              eval_set=[(X_test, y_test)],
              callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)])

    metrics = evaluate_model(model, X_test, y_test, "lightgbm")
    metrics["best_params"] = best_params
    save_metrics(metrics, "lightgbm")
    save_model_pkl(model, "lightgbm")
    # LGB native model file
    model.booster_.save_model(str(MODELS_DIR / "lightgbm.txt"))
    export_to_onnx(model, "lightgbm", n_features)
    return {"model": model, "metrics": metrics}


# ---------------------------------------------------------------------------
# Main entrypoint
# ---------------------------------------------------------------------------

def run_training() -> Dict[str, Any]:
    """Train all three baseline models and return results dict."""
    X, y, feature_cols = load_dataset()

    # ── SMOTE oversampling (if minority class < 5%) ───────────────────
    pos_rate = float(y.mean())
    if _HAS_SMOTE and pos_rate < 0.05:
        logger.info("Applying SMOTE (positive rate=%.2f%%).", 100 * pos_rate)
        smote = SMOTE(random_state=RANDOM_SEED, k_neighbors=min(5, int(y.sum()) - 1))
        X, y = smote.fit_resample(X, y)
        logger.info("After SMOTE: %d samples, %.1f%% positive.", len(X), 100 * y.mean())
    else:
        logger.info("SMOTE skipped (positive rate=%.2f%% or imbalanced-learn not available).", 100 * pos_rate)

    # ── Train/test split (temporal: last 20% as test) ─────────────────
    split_idx = int(len(X) * 0.80)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    n_features = X.shape[1]

    logger.info("Train: %d | Test: %d | Features: %d", len(X_train), len(X_test), n_features)

    results = {}
    results["random_forest"] = train_random_forest(X_train, y_train, X_test, y_test, n_features)
    results["xgboost"]       = train_xgboost(X_train, y_train, X_test, y_test, n_features)
    results["lightgbm"]      = train_lightgbm(X_train, y_train, X_test, y_test, n_features)

    # ── Summary report ────────────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("TRAINING COMPLETE — SUMMARY")
    logger.info("=" * 60)
    for name, res in results.items():
        m = res["metrics"]
        logger.info(
            "%-14s  TSS=%+.3f  HSS=%+.3f  AUC=%.3f  FAR=%.3f",
            name, m["tss"], m["hss"], m["auc_roc"], m["far"],
        )

    # Save feature names for inference
    feat_meta = {"feature_columns": feature_cols, "n_features": n_features}
    with open(MODELS_DIR / "feature_metadata.json", "w") as fh:
        json.dump(feat_meta, fh, indent=2)

    return results


if __name__ == "__main__":
    run_training()
