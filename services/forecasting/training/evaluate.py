"""Scientific Evaluation Suite for ASTRONOVA Forecasters.

Loads trained XGBoost, LightGBM, BiLSTM, and Ensemble models.
Evaluates them on the out-of-time test split (last 20%) of the sequence dataset.
Computes comprehensive metrics per horizon (+15m, +30m, +1h, +6h).
Generates 7 publication-quality plots and an evaluation report in reports/.

Usage:
    python -m services.forecasting.training.evaluate
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import (
    auc,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import label_binarize

# ── PYTHONPATH guard ──────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.data.dataset import RealGoesDataset
from ml.models.lightgbm_model import LightGBMForecaster

from ml.models.bilstm import BiLSTMForecaster
from ml.models.xgboost_model import XGBoostForecaster
from ml.training.metrics import calculate_classification_metrics, calculate_regression_metrics
from services.forecasting.training.calibration_metrics import (
    compute_calibration_curve_data,
    expected_calibration_error_multiclass,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger("astronova.evaluate")

# Setup Matplotlib styles
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update(
    {
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "figure.titlesize": 14,
        "figure.dpi": 150,
    }
)

DATA_PATH = "data/sample/real_time_goes.csv"
REPORTS_DIR = PROJECT_ROOT / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def load_test_split() -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Loads dataset and splits into out-of-time test split (last 20%)."""
    logger.info("Loading dataset from %s...", DATA_PATH)
    dataset = RealGoesDataset(DATA_PATH)
    X = dataset.X.numpy()
    y_class = dataset.y_class.numpy()
    y_reg = dataset.y_reg.numpy()

    split_idx = int(len(X) * 0.80)
    X_test = X[split_idx:]
    y_class_test = y_class[split_idx:]
    y_reg_test = y_reg[split_idx:]

    logger.info("Test split size: %d samples", len(X_test))
    return X_test, y_class_test, y_reg_test, dataset.feature_cols


def load_all_models() -> tuple[XGBoostForecaster, LightGBMForecaster, BiLSTMForecaster]:
    """Loads pre-trained forecaster models."""
    logger.info("Loading pre-trained XGBoost model...")
    xgb = XGBoostForecaster.load("models/xgboost/model.pkl")

    logger.info("Loading pre-trained LightGBM model...")
    lgb = LightGBMForecaster.load("models/lightgbm/model.pkl")

    logger.info("Loading pre-trained BiLSTM model...")
    lstm = BiLSTMForecaster(input_size=15, num_horizons=4)
    lstm.load_state_dict(torch.load("models/lstm/best.pt", map_location="cpu"))
    lstm.eval()

    return xgb, lgb, lstm


def robust_roc_auc_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Computes multiclass OVR ROC-AUC score, handling missing classes in y_true."""
    unique_classes = np.unique(y_true)
    if len(unique_classes) < 2:
        return float("nan")

    # If it's a binary case (e.g. only classes [2, 3] are present)
    if len(unique_classes) == 2:
        y_true_binary = (y_true == unique_classes[1]).astype(int)
        col0 = int(unique_classes[0])
        col1 = int(unique_classes[1])
        if col0 < y_prob.shape[1] and col1 < y_prob.shape[1]:
            prob_pos = y_prob[:, col1] / (y_prob[:, col0] + y_prob[:, col1] + 1e-9)
        else:
            prob_pos = y_prob[:, 1]
        try:
            return float(roc_auc_score(y_true_binary, prob_pos))
        except Exception as e:
            logger.error("Exception in binary roc_auc_score: %s", e)
            return float("nan")

    # Multiclass OVR
    try:
        valid_cols = [int(c) for c in unique_classes if int(c) < y_prob.shape[1]]
        if len(valid_cols) < 2:
            return float("nan")

        y_prob_filtered = y_prob[:, valid_cols]
        row_sums = y_prob_filtered.sum(axis=1, keepdims=True)
        y_prob_filtered = np.divide(y_prob_filtered, row_sums, out=np.zeros_like(y_prob_filtered), where=row_sums > 0)

        class_map = {c: idx for idx, c in enumerate(valid_cols)}
        mask = np.isin(y_true, valid_cols)
        if not np.any(mask):
            return float("nan")
        y_true_filtered = y_true[mask]
        y_prob_filtered = y_prob_filtered[mask]

        y_true_mapped = np.vectorize(class_map.get)(y_true_filtered)
        return float(roc_auc_score(y_true_mapped, y_prob_filtered, multi_class="ovr", average="weighted"))
    except Exception as e:
        logger.error("Exception in multiclass roc_auc_score: %s", e)
        return float("nan")


def evaluate_predictions(
    y_class_true: np.ndarray,
    y_reg_true: np.ndarray,
    probs: np.ndarray,
    regs: np.ndarray,
    horizons: list[str],
) -> dict[str, dict[str, float]]:
    """Computes comprehensive evaluation metrics for a model across all horizons."""
    metrics_by_horizon = {}

    for h_idx, h_name in enumerate(horizons):
        yc_true = y_class_true[:, h_idx]
        yr_true = y_reg_true[:, h_idx, 0]

        yc_prob = probs[:, h_idx, :]
        yc_pred = np.argmax(yc_prob, axis=1)
        yr_pred = regs[:, h_idx, 0]

        # Classification
        clf_metrics = calculate_classification_metrics(yc_true, yc_pred, yc_prob)
        ece = expected_calibration_error_multiclass(yc_true, yc_prob)
        roc_auc_val = robust_roc_auc_score(yc_true, yc_prob)

        # Multiclass Brier Score
        n_classes = yc_prob.shape[1]
        yc_true_bin = label_binarize(yc_true, classes=range(n_classes))
        if yc_true_bin.shape[1] != n_classes:
            temp = np.zeros((len(yc_true), n_classes))
            for c in range(yc_true_bin.shape[1]):
                temp[:, c] = yc_true_bin[:, c]
            yc_true_bin = temp
        brier = float(np.mean(np.sum((yc_prob - yc_true_bin) ** 2, axis=1)))

        # PR-AUC (OVR average)
        pr_aucs = []
        for c in range(n_classes):
            c_true = (yc_true == c).astype(int)
            c_prob = yc_prob[:, c]
            if np.sum(c_true) > 0:
                p, r, _ = precision_recall_curve(c_true, c_prob)
                pr_aucs.append(auc(r, p))
        mean_pr_auc = float(np.mean(pr_aucs)) if pr_aucs else 0.0

        # Regression
        reg_metrics = calculate_regression_metrics(yr_true, yr_pred)
        mape = float(np.mean(np.abs((yr_true - yr_pred) / (yr_true + 1e-8))) * 100)

        metrics_by_horizon[h_name] = {
            "accuracy": float(clf_metrics.get("accuracy", 0.0)),
            "precision": float(clf_metrics.get("precision", 0.0)),
            "recall": float(clf_metrics.get("recall", 0.0)),
            "f1": float(clf_metrics.get("f1", 0.0)),
            "roc_auc": roc_auc_val,
            "pr_auc": mean_pr_auc,
            "brier_score": brier,
            "ece": ece,
            "rmse": float(reg_metrics.get("rmse", 0.0)),
            "mae": float(reg_metrics.get("mae", 0.0)),
            "r2": float(reg_metrics.get("r2", 0.0)),
            "mape": mape,
            # Space Weather Metrics
            "tss": float(clf_metrics.get("tss", 0.0)),
            "hss": float(clf_metrics.get("hss", 0.0)),
            "far": float(clf_metrics.get("far", 0.0)),
        }

    return metrics_by_horizon


# ── Plot Generation Helpers ───────────────────────────────────────────────────


def plot_confusion_matrices(y_true: np.ndarray, probs: np.ndarray, classes: list[str], horizons: list[str]):
    _fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.ravel()

    for idx, (h_name, ax) in enumerate(zip(horizons, axes, strict=False)):
        yc_true = y_true[:, idx]
        yc_pred = np.argmax(probs[:, idx, :], axis=1)

        cm = confusion_matrix(yc_true, yc_pred, labels=range(len(classes)))

        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            ax=ax,
            xticklabels=classes,
            yticklabels=classes,
            cbar=False,
        )
        ax.set_title(f"Horizon: {h_name}")
        ax.set_xlabel("Predicted Class")
        ax.set_ylabel("True Class")

    plt.suptitle("Confusion Matrices (Ensemble Forecaster)", fontsize=16)
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "confusion_matrix.png", dpi=300)
    plt.close()
    logger.info("Saved reports/confusion_matrix.png")


def plot_roc_curves(y_true: np.ndarray, probs: np.ndarray, classes: list[str], horizon_idx: int = 0):
    yc_true = y_true[:, horizon_idx]
    yc_prob = probs[:, horizon_idx, :]

    plt.figure(figsize=(8, 6))
    for c_idx, c_name in enumerate(classes):
        c_true = (yc_true == c_idx).astype(int)
        c_prob = yc_prob[:, c_idx]

        if np.sum(c_true) > 0:
            fpr, tpr, _ = roc_curve(c_true, c_prob)
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, lw=2, label=f"Class {c_name} (AUC = {roc_auc:.3f})")

    plt.plot([0, 1], [0, 1], "k--", lw=1.5, label="Random Guess")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("One-vs-Rest ROC Curves (+15m Horizon)")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "roc_curve.png", dpi=300)
    plt.close()
    logger.info("Saved reports/roc_curve.png")


def plot_pr_curves(y_true: np.ndarray, probs: np.ndarray, classes: list[str], horizon_idx: int = 0):
    yc_true = y_true[:, horizon_idx]
    yc_prob = probs[:, horizon_idx, :]

    plt.figure(figsize=(8, 6))
    for c_idx, c_name in enumerate(classes):
        c_true = (yc_true == c_idx).astype(int)
        c_prob = yc_prob[:, c_idx]

        if np.sum(c_true) > 0:
            precision, recall, _ = precision_recall_curve(c_true, c_prob)
            pr_auc = auc(recall, precision)
            plt.plot(recall, precision, lw=2, label=f"Class {c_name} (AUC = {pr_auc:.3f})")

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("One-vs-Rest Precision-Recall Curves (+15m Horizon)")
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "pr_curve.png", dpi=300)
    plt.close()
    logger.info("Saved reports/pr_curve.png")


def plot_feature_importance(xgb_model: XGBoostForecaster, lgb_model: LightGBMForecaster, feature_names: list[str]):
    # Get importances from 15m horizon classifiers
    xgb_imp = xgb_model.classifiers[0].feature_importances_
    lgb_imp = lgb_model.classifiers[0].feature_importances_

    # Standardize and average
    xgb_imp = xgb_imp / np.sum(xgb_imp)
    lgb_imp = lgb_imp / np.sum(lgb_imp)
    avg_imp = (xgb_imp + lgb_imp) / 2.0

    # Feature names are flattened [seq_len, features].
    # Let's map flattened indices back to temporal names e.g., magnetic_complexity (t-0)
    seq_len = 10
    len(feature_names)
    expanded_names = []
    for t in range(seq_len):
        lag = seq_len - 1 - t
        for f in feature_names:
            expanded_names.append(f"{f} (t-{lag})")

    # Sort top 15 features
    top_indices = np.argsort(avg_imp)[-15:]
    top_names = [expanded_names[i] for i in top_indices]
    top_scores = avg_imp[top_indices]

    plt.figure(figsize=(10, 6))
    plt.barh(top_names, top_scores, color="steelblue", edgecolor="black", height=0.6)
    plt.xlabel("Average Normalized Importance (XGBoost + LightGBM)")
    plt.title("Top 15 Temporal Feature Importances (+15m Classifier)")
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "feature_importance.png", dpi=300)
    plt.close()
    logger.info("Saved reports/feature_importance.png")


def plot_residuals(y_true: np.ndarray, regs: np.ndarray, horizon_idx: int = 0):
    yr_true = y_true[:, horizon_idx, 0]
    yr_pred = regs[:, horizon_idx, 0]
    residuals = yr_true - yr_pred

    _fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Scatter plot
    ax1.scatter(yr_pred, yr_true, alpha=0.4, color="darkorange", edgecolor="none")
    min_val = min(yr_true.min(), yr_pred.min())
    max_val = max(yr_true.max(), yr_pred.max())
    ax1.plot([min_val, max_val], [min_val, max_val], "k--", lw=1.5, label="Perfect Forecast")
    ax1.set_xlabel("Predicted Log Soft X-ray Flux")
    ax1.set_ylabel("True Log Soft X-ray Flux")
    ax1.set_title("Actual vs. Predicted")
    ax1.legend()

    # Histogram of residuals
    sns.histplot(residuals, kde=True, ax=ax2, color="teal", edgecolor="black", bins=30)
    ax2.axvline(0, color="red", linestyle="--", lw=1.5)
    ax2.set_xlabel("Residuals (True - Predicted)")
    ax2.set_ylabel("Count")
    ax2.set_title("Residuals Distribution")

    plt.suptitle("Residuals Analysis (+15m Regression Head)", fontsize=15)
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "residual_analysis.png", dpi=300)
    plt.close()
    logger.info("Saved reports/residual_analysis.png")


def plot_calibration_curves(y_true: np.ndarray, probs: np.ndarray, horizon_idx: int = 0):
    # Use binary target: significant flare (class M/X, i.e., class >= 2)
    yc_true_bin = (y_true[:, horizon_idx] >= 2).astype(int)
    # Ensemble probability of flare
    yc_prob_flare = np.sum(probs[:, horizon_idx, 2:], axis=1)

    plt.figure(figsize=(8, 6))

    # Plot perfect calibration
    plt.plot([0, 1], [0, 1], "k--", lw=1.5, label="Perfectly Calibrated")

    # Compute and plot calibration curve for ensemble
    if np.sum(yc_true_bin) > 0:
        prob_true, prob_pred = compute_calibration_curve_data(yc_true_bin, yc_prob_flare, n_bins=10)
        plt.plot(prob_pred, prob_true, marker="o", lw=2, label="Ensemble")

    plt.xlabel("Mean Predicted Probability (Significant Flare)")
    plt.ylabel("Fraction of Positives (True Frequency)")
    plt.title("Reliability Diagram (+15m Horizon)")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "calibration_curve.png", dpi=300)
    plt.close()
    logger.info("Saved reports/calibration_curve.png")


def plot_horizon_comparison(all_metrics: dict[str, dict[str, dict[str, float]]], horizons: list[str]):
    _fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.ravel()

    metrics_to_plot = ["roc_auc", "f1", "rmse", "ece"]
    titles = [
        "ROC-AUC (Higher is Better)",
        "F1 Score (Higher is Better)",
        "RMSE (Lower is Better)",
        "ECE (Lower is Better)",
    ]
    models = list(all_metrics.keys())

    for idx, (metric, title) in enumerate(zip(metrics_to_plot, titles, strict=False)):
        ax = axes[idx]
        for model in models:
            y_vals = [all_metrics[model][h][metric] for h in horizons]
            ax.plot(horizons, y_vals, marker="s", lw=2, label=model.upper())
        ax.set_title(title)
        ax.set_xlabel("Forecast Horizon")
        ax.set_ylabel("Value")
        if idx == 0:
            ax.legend(loc="lower left")

    plt.suptitle("Model Degradation Across Forecast Horizons", fontsize=16)
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "horizon_comparison.png", dpi=300)
    plt.close()
    logger.info("Saved reports/horizon_comparison.png")


# ── Main Run ─────────────────────────────────────────────────────────────────


def main():
    X_test, y_class_test, y_reg_test, feature_names = load_test_split()
    xgb_model, lgb_model, lstm_model = load_all_models()

    horizons = ["15m", "30m", "1h", "6h"]
    classes = ["A/B", "C", "M", "X"]

    logger.info("Running inference on test set (XGBoost)...")
    t0 = time.time()
    xgb_probs, xgb_regs = xgb_model.predict(X_test)
    xgb_lat = (time.time() - t0) / len(X_test) * 1000

    logger.info("Running inference on test set (LightGBM)...")
    t0 = time.time()
    lgb_probs, lgb_regs = lgb_model.predict(X_test)
    lgb_lat = (time.time() - t0) / len(X_test) * 1000

    logger.info("Running inference on test set (BiLSTM)...")
    t0 = time.time()
    with torch.no_grad():
        x_t = torch.tensor(X_test, dtype=torch.float32)
        lstm_probs, lstm_regs = lstm_model(x_t, return_tuple=True)
        lstm_probs = lstm_probs.numpy()
        lstm_regs = lstm_regs.numpy()
    lstm_lat = (time.time() - t0) / len(X_test) * 1000

    # Ensemble (0.3 LSTM + 0.4 XGB + 0.3 LGB)
    logger.info("Computing Ensemble predictions...")
    ens_probs = 0.3 * lstm_probs + 0.4 * xgb_probs + 0.3 * lgb_probs
    ens_regs = 0.3 * lstm_regs + 0.4 * xgb_regs + 0.3 * lgb_regs
    ens_lat = xgb_lat + lgb_lat + lstm_lat  # sum of individual inference calls if run sequentially

    all_predictions = {
        "xgboost": (xgb_probs, xgb_regs, xgb_lat),
        "lightgbm": (lgb_probs, lgb_regs, lgb_lat),
        "lstm": (lstm_probs, lstm_regs, lstm_lat),
        "ensemble": (ens_probs, ens_regs, ens_lat),
    }

    # Compute all metrics
    all_metrics = {}
    for model_name, (probs, regs, _lat) in all_predictions.items():
        logger.info("Computing evaluation metrics for %s...", model_name)
        all_metrics[model_name] = evaluate_predictions(y_class_test, y_reg_test, probs, regs, horizons)

    # Generate Plots (using Ensemble)
    logger.info("Generating plots...")
    plot_confusion_matrices(y_class_test, ens_probs, classes, horizons)
    plot_roc_curves(y_class_test, ens_probs, classes, horizon_idx=0)
    plot_pr_curves(y_class_test, ens_probs, classes, horizon_idx=0)
    plot_feature_importance(xgb_model, lgb_model, feature_names)
    plot_residuals(y_reg_test, ens_regs, horizon_idx=0)
    plot_calibration_curves(y_class_test, ens_probs, horizon_idx=0)
    plot_horizon_comparison(all_metrics, horizons)

    # Print metric tables to console
    for h_name in horizons:
        print("\n" + "=" * 80)
        print(f"EVALUATION RESULTS FOR HORIZON: {h_name}")
        print("=" * 80)
        print(
            f"{'Model':<12} | {'ROC-AUC':<8} | {'PR-AUC':<8} | {'F1':<8} | {'ECE':<8} | {'Brier':<8} | {'RMSE':<8} | {'MAPE':<8}"
        )
        print("-" * 80)
        for model in all_predictions:
            m = all_metrics[model][h_name]
            print(
                f"{model:<12} | {m['roc_auc']:.4f} | {m['pr_auc']:.4f} | {m['f1']:.4f} | {m['ece']:.4f} | {m['brier_score']:.4f} | {m['rmse']:.4f} | {m['mape']:.2f}%"
            )
        print("-" * 80)

    # Save raw metrics to JSON
    with open(REPORTS_DIR / "evaluation_metrics.json", "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2)

    # Generate markdown report
    # Let's check target criteria status for ensemble at +15m
    ens_15m = all_metrics["ensemble"]["15m"]

    roc_status = "✅" if ens_15m["roc_auc"] > 0.85 else "❌"
    f1_status = "✅" if ens_15m["f1"] > 0.80 else "❌"
    ece_status = "✅" if ens_15m["ece"] < 0.05 else "❌"
    inf_status = "✅" if ens_lat < 100.0 else "❌"

    report_md = f"""# AstroNova Model Evaluation Report

This report presents a thorough, publication-grade evaluation of the AstroNova solar flare forecasting models: **XGBoost**, **LightGBM**, **BiLSTM**, and their weighted **Ensemble** (`0.3*BiLSTM + 0.4*XGBoost + 0.3*LightGBM`).

Testing was performed on an out-of-time test split (last 20% of the timeline) containing **{len(X_test)}** prediction sequences.

## 🚀 Success Criteria Audit (+15 min Horizon)

| Metric | Target | Ensemble Score | Status |
| :--- | :--- | :--- | :---: |
| **ROC-AUC** | > 0.85 | **{ens_15m["roc_auc"]:.4f}** | {roc_status} |
| **F1-Score (Weighted)** | > 0.80 | **{ens_15m["f1"]:.4f}** | {f1_status} |
| **Calibration Error (ECE)** | < 0.05 | **{ens_15m["ece"]:.4f}** | {ece_status} |
| **Inference Latency** | < 100ms | **{ens_lat:.2f}ms** | {inf_status} |

## 📊 Comprehensive Performance Across Horizons

### classification Metrics

| Horizon | Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC | ECE | Brier |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
"""
    for h_name in horizons:
        for model in all_predictions:
            m = all_metrics[model][h_name]
            report_md += f"| {h_name} | {model.upper()} | {m['accuracy']:.4f} | {m['precision']:.4f} | {m['recall']:.4f} | {m['f1']:.4f} | {m['roc_auc']:.4f} | {m['pr_auc']:.4f} | {m['ece']:.4f} | {m['brier_score']:.4f} |\n"

    report_md += """
### Regression Metrics (Log Soft X-Ray Flux)

| Horizon | Model | RMSE | MAE | R² | MAPE (Log target) |
| :--- | :--- | :---: | :---: | :---: | :---: |
"""
    for h_name in horizons:
        for model in all_predictions:
            m = all_metrics[model][h_name]
            report_md += f"| {h_name} | {model.upper()} | {m['rmse']:.4f} | {m['mae']:.4f} | {m['r2']:.4f} | {m['mape']:.2f}% |\n"

    report_md += """
### Space Weather Operational Metrics (Binary Event Detection: class >= M)

| Horizon | Model | TSS (True Skill Stat) | HSS (Heidke Skill Score) | FAR (False Alarm Ratio) |
| :--- | :--- | :---: | :---: | :---: |
"""
    for h_name in horizons:
        for model in all_predictions:
            m = all_metrics[model][h_name]
            report_md += f"| {h_name} | {model.upper()} | {m['tss']:.4f} | {m['hss']:.4f} | {m['far']:.4f} |\n"

    report_md += """
## 🎨 Visualization Artifacts

- **Confusion Matrices**: Shows multiclass accuracy and classification errors across all 4 horizons.
  ![Confusion Matrix](confusion_matrix.png)

- **Receiver Operating Characteristic (ROC)**: One-vs-rest ROC curves for A/B, C, M, and X flare classes.
  ![ROC Curve](roc_curve.png)

- **Precision-Recall Curve (PR)**: Displays the precision-recall trade-offs, which are highly critical under heavy class imbalance (flares are rare).
  ![PR Curve](pr_curve.png)

- **Temporal Feature Importance**: Indicates the most critical input features and their historical lags.
  ![Feature Importance](feature_importance.png)

- **Residual Analysis**: Evaluates the regression predictions for log flux. The right panel validates that residual errors are normally distributed.
  ![Residual Analysis](residual_analysis.png)

- **Calibration Curves**: Shows how well predicted probabilities map to actual observations (reliability diagram).
  ![Calibration Curve](calibration_curve.png)

- **Horizon Comparison**: Illustrates the performance degradation of all models across longer forecast timelines.
  ![Horizon Comparison](horizon_comparison.png)
"""

    # Save report
    report_path = REPORTS_DIR / "evaluation_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_md)

    logger.info("Saved reports/evaluation_report.md successfully.")
    print("Scientific evaluation complete! See reports/ directory.")


def run_evaluation():
    """Wrapper function for backward compatibility."""
    main()


if __name__ == "__main__":
    main()
