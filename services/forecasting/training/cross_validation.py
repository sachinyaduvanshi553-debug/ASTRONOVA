"""TimeSeriesSplit Cross-Validation for ASTRONOVA Solar Flare Forecasters.

Performs 5-fold TimeSeriesSplit cross-validation to evaluate generalization
performance across all target forecasting horizons (+15m, +30m, +1h, +6h).
Saves results to reports/cross_validation_report.json.

Usage:
    python -m services.forecasting.training.cross_validation
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.model_selection import TimeSeriesSplit
from torch.utils.data import DataLoader, TensorDataset

# ── PYTHONPATH guard ──────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.data.dataset import RealGoesDataset
from ml.models.lightgbm_model import LightGBMForecaster

from ml.models.bilstm import BiLSTMForecaster
from ml.models.xgboost_model import XGBoostForecaster
from ml.training.metrics import calculate_classification_metrics, calculate_regression_metrics
from services.forecasting.training.calibration_metrics import expected_calibration_error_multiclass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger("astronova.cross_validation")

DATA_PATH = "data/sample/real_time_goes.csv"
REPORTS_DIR = PROJECT_ROOT / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def train_lstm_fold(
    X_train: np.ndarray, y_class_train: np.ndarray, y_reg_train: np.ndarray, epochs: int = 2
) -> BiLSTMForecaster:
    """Train a BiLSTM forecaster for a few epochs on CPU for validation."""
    device = torch.device("cpu")
    model = BiLSTMForecaster(input_size=15, num_classes=5, num_horizons=4).to(device)
    model.train()

    # Create DataLoader
    dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_class_train, dtype=torch.long),
        torch.tensor(y_reg_train, dtype=torch.float32),
    )
    loader = DataLoader(dataset, batch_size=64, shuffle=False)  # Keep order for TS

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=1e-4)
    ce_loss = torch.nn.CrossEntropyLoss()
    mse_loss = torch.nn.MSELoss()

    for _epoch in range(epochs):
        for bx, by_class, by_reg in loader:
            optimizer.zero_grad()
            class_preds, reg_preds = model(bx.to(device), return_tuple=True)

            loss = 0.0
            for h in range(4):
                loss += 0.7 * ce_loss(class_preds[:, h, :], by_class[:, h])
                loss += 0.3 * mse_loss(reg_preds[:, h, 0], by_reg[:, h, 0])

            loss.backward()
            optimizer.step()

    model.eval()
    return model


def make_multiclass_targets_robust(
    X_train: np.ndarray, y_class_train: np.ndarray, y_reg_train: np.ndarray, num_classes: int = 4
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Appends dummy rows to training data to ensure all classes in [0, num_classes-1] are present in every horizon."""
    n_horizons = y_class_train.shape[1]
    extra_X = []
    extra_yc = []
    extra_yr = []

    for h in range(n_horizons):
        unique = np.unique(y_class_train[:, h])
        missing = [c for c in range(num_classes) if c not in unique]
        for mc in missing:
            # Create a dummy row for this missing class
            dummy_x = np.zeros((1, *X_train.shape[1:]))
            dummy_yc = np.zeros((1, n_horizons), dtype=y_class_train.dtype)
            # Set this horizon's class to the missing class
            dummy_yc[0, :] = mc
            dummy_yr = np.zeros((1, n_horizons, 1), dtype=y_reg_train.dtype)

            extra_X.append(dummy_x)
            extra_yc.append(dummy_yc)
            extra_yr.append(dummy_yr)

    if extra_X:
        X_train = np.concatenate([X_train, *extra_X], axis=0)
        y_class_train = np.concatenate([y_class_train, *extra_yc], axis=0)
        y_reg_train = np.concatenate([y_reg_train, *extra_yr], axis=0)

    return X_train, y_class_train, y_reg_train


def run_cross_validation() -> dict[str, Any]:
    logger.info("Loading dataset from %s...", DATA_PATH)
    dataset = RealGoesDataset(DATA_PATH)
    X = dataset.X.numpy()
    y_class = dataset.y_class.numpy()
    y_reg = dataset.y_reg.numpy()

    logger.info("Dataset shape: X=%s, y_class=%s, y_reg=%s", X.shape, y_class.shape, y_reg.shape)

    n_splits = 5
    tscv = TimeSeriesSplit(n_splits=n_splits)

    horizons = ["15m", "30m", "1h", "6h"]
    models = ["xgboost", "lightgbm", "lstm"]

    # Store results per model, fold, and horizon
    results: dict[str, Any] = {m: {h: [] for h in horizons} for m in models}

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X), 1):
        logger.info("--- Processing Fold %d/%d ---", fold, n_splits)

        X_train, X_test = X[train_idx], X[test_idx]
        y_class_train, y_class_test = y_class[train_idx], y_class[test_idx]
        y_reg_train, y_reg_test = y_reg[train_idx], y_reg[test_idx]

        logger.info("Fold %d sizes - Train: %d, Test: %d", fold, len(train_idx), len(test_idx))

        # Ensure all classes are present in every training split to avoid XGB/LGB fit errors
        X_train_fit, y_class_train_fit, y_reg_train_fit = make_multiclass_targets_robust(
            X_train, y_class_train, y_reg_train, num_classes=4
        )

        # 1. Train XGBoost
        logger.info("Training XGBoost...")
        xgb_model = XGBoostForecaster(input_size=15, seq_len=10, num_classes=5, num_horizons=4)
        xgb_model.fit(X_train_fit, y_class_train_fit, y_reg_train_fit)
        xgb_probs, xgb_regs = xgb_model.predict(X_test)

        # 2. Train LightGBM
        logger.info("Training LightGBM...")
        lgb_model = LightGBMForecaster(input_size=15, seq_len=10, num_classes=5, num_horizons=4)
        lgb_model.fit(X_train_fit, y_class_train_fit, y_reg_train_fit)
        lgb_probs, lgb_regs = lgb_model.predict(X_test)

        # 3. Train LSTM
        logger.info("Training LSTM...")
        lstm_model = train_lstm_fold(X_train_fit, y_class_train_fit, y_reg_train_fit, epochs=2)
        with torch.no_grad():
            x_t = torch.tensor(X_test, dtype=torch.float32)
            lstm_probs, lstm_regs = lstm_model(x_t, return_tuple=True)
            lstm_probs = lstm_probs.numpy()
            lstm_regs = lstm_regs.numpy()

        predictions = {
            "xgboost": (xgb_probs, xgb_regs),
            "lightgbm": (lgb_probs, lgb_regs),
            "lstm": (lstm_probs, lstm_regs),
        }

        # Evaluate each model on each horizon
        for model_name, (probs, regs) in predictions.items():
            for h_idx, h_name in enumerate(horizons):
                # Target ground truth for this horizon
                yc_true = y_class_test[:, h_idx]
                yr_true = y_reg_test[:, h_idx, 0]

                # Predictions for this horizon
                yc_prob = probs[:, h_idx, :]

                # Slice to class count present in yc_true (e.g. keep classes 0..max_c)
                max_c = int(np.max(yc_true))
                yc_prob = yc_prob[:, : max_c + 1]
                # Renormalize
                row_sums = yc_prob.sum(axis=1, keepdims=True)
                yc_prob = np.divide(yc_prob, row_sums, out=np.zeros_like(yc_prob), where=row_sums > 0)

                yc_pred = np.argmax(yc_prob, axis=1)
                yr_pred = regs[:, h_idx, 0]

                # Compute metrics
                clf_metrics = calculate_classification_metrics(yc_true, yc_pred, yc_prob)
                reg_metrics = calculate_regression_metrics(yr_true, yr_pred)
                ece = expected_calibration_error_multiclass(yc_true, yc_prob)

                # Add stable MAPE calculation on log10 target
                mape = np.mean(np.abs((yr_true - yr_pred) / (yr_true + 1e-8))) * 100

                fold_metrics = {
                    "fold": fold,
                    "accuracy": float(clf_metrics.get("accuracy", 0.0)),
                    "f1": float(clf_metrics.get("f1", 0.0)),
                    "roc_auc": float(clf_metrics.get("roc_auc", 0.0))
                    if not np.isnan(clf_metrics.get("roc_auc", 0.0))
                    else 0.0,
                    "ece": float(ece),
                    "rmse": float(reg_metrics.get("rmse", 0.0)),
                    "mae": float(reg_metrics.get("mae", 0.0)),
                    "r2": float(reg_metrics.get("r2", 0.0)),
                    "mape": float(mape),
                }

                results[model_name][h_name].append(fold_metrics)

    # Compute averages across folds
    cv_summary: dict[str, Any] = {}
    for model_name in models:
        cv_summary[model_name] = {}
        for h_name in horizons:
            folds_data = results[model_name][h_name]
            metrics_keys = ["accuracy", "f1", "roc_auc", "ece", "rmse", "mae", "r2", "mape"]

            avg_metrics = {}
            for k in metrics_keys:
                vals = [f[k] for f in folds_data if not np.isnan(f[k])]
                avg_metrics[k] = float(np.mean(vals)) if vals else 0.0

            cv_summary[model_name][h_name] = {"mean": avg_metrics, "folds": folds_data}

    # Print results summary table
    print("\n" + "=" * 80)
    print("TIME-SERIES CROSS-VALIDATION SUMMARY (5 FOLDS AVERAGE)")
    print("=" * 80)
    print(f"{'Model':<12} | {'Horizon':<8} | {'ROC-AUC':<8} | {'F1':<8} | {'ECE':<8} | {'RMSE':<8} | {'MAPE':<8}")
    print("-" * 80)
    for model_name in models:
        for h_name in horizons:
            m = cv_summary[model_name][h_name]["mean"]
            print(
                f"{model_name:<12} | {h_name:<8} | {m['roc_auc']:.4f} | {m['f1']:.4f} | {m['ece']:.4f} | {m['rmse']:.4f} | {m['mape']:.2f}%"
            )
        print("-" * 80)

    # Save report
    report_path = REPORTS_DIR / "cross_validation_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(cv_summary, f, indent=2)
    logger.info("Cross-validation report saved → %s", report_path)

    return cv_summary


if __name__ == "__main__":
    run_cross_validation()
