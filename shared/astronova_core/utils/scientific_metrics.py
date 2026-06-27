import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import auc, brier_score_loss, precision_recall_curve, roc_curve

logger = logging.getLogger("astronova.scientific_metrics")

def _get_confusion_matrix_elements(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[int, int, int, int]:
    """Helper to compute TP, FP, FN, TN for binary classification."""
    y_true_bool = np.array(y_true, dtype=bool)
    y_pred_bool = np.array(y_pred, dtype=bool)

    tp = int(np.sum(y_true_bool & y_pred_bool))
    fp = int(np.sum(~y_true_bool & y_pred_bool))
    fn = int(np.sum(y_true_bool & ~y_pred_bool))
    tn = int(np.sum(~y_true_bool & ~y_pred_bool))

    return tp, fp, fn, tn

def compute_tss(y_true: list[int] | np.ndarray, y_pred: list[int] | np.ndarray) -> float:
    """
    True Skill Statistic (TSS) / Hanssen-Kuipers Discriminant.
    TSS = TPR - FPR = TP / (TP + FN) - FP / (FP + TN)
    Range: [-1, 1], where 1 is perfect forecast, 0 is random/constant forecast.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    tp, fp, fn, tn = _get_confusion_matrix_elements(y_true, y_pred)

    sensitivity_denom = tp + fn
    specificity_denom = fp + tn

    tpr = tp / sensitivity_denom if sensitivity_denom > 0 else 0.0
    fpr = fp / specificity_denom if specificity_denom > 0 else 0.0

    tss = tpr - fpr
    return float(tss)

def compute_hss(y_true: list[int] | np.ndarray, y_pred: list[int] | np.ndarray) -> float:
    """
    Heidke Skill Score (HSS).
    Measures the fractional improvement of the forecast over standard random forecasts.
    Range: [-inf, 1], where 1 is perfect, 0 is no skill.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    tp, fp, fn, tn = _get_confusion_matrix_elements(y_true, y_pred)

    total = tp + fp + fn + tn
    if total == 0:
        return 0.0

    expected_correct = ((tp + fn) * (tp + fp) + (tn + fn) * (tn + fp)) / total
    observed_correct = tp + tn

    denom = total - expected_correct
    if denom == 0:
        return 0.0

    hss = (observed_correct - expected_correct) / denom
    return float(hss)

def compute_far(y_true: list[int] | np.ndarray, y_pred: list[int] | np.ndarray, return_ratio: bool = True) -> float:
    """
    False Alarm Ratio (FAR) or False Alarm Rate (FPR).
    If return_ratio is True (default for space weather): FAR = FP / (TP + FP)
    If return_ratio is False: FPR = FP / (FP + TN)
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    tp, fp, _fn, tn = _get_confusion_matrix_elements(y_true, y_pred)

    if return_ratio:
        denom = tp + fp
        return float(fp / denom) if denom > 0 else 0.0
    else:
        denom = fp + tn
        return float(fp / denom) if denom > 0 else 0.0

def compute_brier_score(y_true: list[int] | np.ndarray, y_prob: list[float] | np.ndarray) -> float:
    """
    Brier Score (BS). Mean squared error of probability forecasts.
    Range: [0, 1], lower is better.
    """
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    if len(y_true) == 0:
        return 0.0
    return float(brier_score_loss(y_true, y_prob))

def compute_calibration_error(y_true: list[int] | np.ndarray, y_prob: list[float] | np.ndarray, n_bins: int = 10) -> float:
    """
    Expected Calibration Error (ECE).
    Weighted average of difference between confidence and accuracy in each bin.
    """
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)

    if len(y_true) == 0:
        return 0.0

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    total_samples = len(y_true)

    for i in range(n_bins):
        bin_lower = bin_edges[i]
        bin_upper = bin_edges[i + 1]

        # Find indices of samples falling into current bin
        in_bin = (y_prob >= bin_lower) & (y_prob < bin_upper) if i < n_bins - 1 else (y_prob >= bin_lower) & (y_prob <= bin_upper)
        bin_count = np.sum(in_bin)

        if bin_count > 0:
            bin_acc = np.mean(y_true[in_bin])
            bin_conf = np.mean(y_prob[in_bin])
            ece += (bin_count / total_samples) * np.abs(bin_acc - bin_conf)

    return float(ece)

def compute_reliability_curve(y_true: list[int] | np.ndarray, y_prob: list[float] | np.ndarray, n_bins: int = 10) -> dict[str, list[float]]:
    """
    Computes points for a reliability diagram (calibration curve).
    """
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    true_probabilities = []
    pred_probabilities = []
    bin_counts = []

    for i in range(n_bins):
        bin_lower = bin_edges[i]
        bin_upper = bin_edges[i + 1]

        in_bin = (y_prob >= bin_lower) & (y_prob < bin_upper) if i < n_bins - 1 else (y_prob >= bin_lower) & (y_prob <= bin_upper)
        bin_count = np.sum(in_bin)

        if bin_count > 0:
            true_probabilities.append(float(np.mean(y_true[in_bin])))
            pred_probabilities.append(float(np.mean(y_prob[in_bin])))
            bin_counts.append(int(bin_count))

    return {
        "true_probabilities": true_probabilities,
        "pred_probabilities": pred_probabilities,
        "bin_counts": bin_counts
    }

def compute_roc_curve(y_true: list[int] | np.ndarray, y_prob: list[float] | np.ndarray) -> dict[str, Any]:
    """
    Compute Receiver Operating Characteristic (ROC) curve and Area Under Curve (AUC).
    """
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)

    if len(np.unique(y_true)) < 2:
        return {"fpr": [], "tpr": [], "auc": 0.0}

    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    return {
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist(),
        "thresholds": thresholds.tolist(),
        "auc": float(roc_auc)
    }

def compute_pr_curve(y_true: list[int] | np.ndarray, y_prob: list[float] | np.ndarray) -> dict[str, Any]:
    """
    Compute Precision-Recall curve and Area Under Curve (AUC).
    """
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)

    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    pr_auc = auc(recall, precision)

    return {
        "precision": precision.tolist(),
        "recall": recall.tolist(),
        "thresholds": thresholds.tolist(),
        "auc": float(pr_auc)
    }

def generate_confusion_matrix(y_true: list[int] | np.ndarray, y_pred: list[int] | np.ndarray) -> dict[str, int]:
    """
    Generates standard binary confusion matrix counts.
    """
    tp, fp, fn, tn = _get_confusion_matrix_elements(np.array(y_true), np.array(y_pred))
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn
    }

def compute_lead_time_stats(event_catalog: pd.DataFrame, predictions: list[dict[str, Any]]) -> dict[str, float]:
    """
    Computes lead time statistics for successfully forecasted flares.
    - lead_time = event_start_time - time_of_prediction_above_threshold
    """
    lead_times = []

    # Simple matching logic: find events and see when the forecast first crossed 50% probability
    # within a 1-hour window prior to the event.
    for _, event in event_catalog.iterrows():
        event_start = pd.to_datetime(event['start_time'])
        # Look for predictions prior to event_start (up to 2 hours before)
        event_preds = [
            p for p in predictions
            if pd.to_datetime(p['timestamp']) < event_start
            and pd.to_datetime(p['timestamp']) >= event_start - pd.Timedelta(hours=2)
            and p.get('probability', 0.0) >= 0.5
        ]

        if event_preds:
            # Sort by timestamp ascending to find the earliest detection
            event_preds.sort(key=lambda x: pd.to_datetime(x['timestamp']))
            earliest_detection = pd.to_datetime(event_preds[0]['timestamp'])
            lead_time_min = (event_start - earliest_detection).total_seconds() / 60.0
            lead_times.append(lead_time_min)

    if not lead_times:
        return {"mean_lead_time": 0.0, "median_lead_time": 0.0, "max_lead_time": 0.0}

    return {
        "mean_lead_time": float(np.mean(lead_times)),
        "median_lead_time": float(np.median(lead_times)),
        "max_lead_time": float(np.max(lead_times))
    }
