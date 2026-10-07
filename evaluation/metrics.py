"""
Comprehensive Solar Physics & ML Evaluation Metrics.
Computes:
- Accuracy, Precision, Recall, F1
- ROC-AUC, PR-AUC
- TSS (True Skill Statistic / Hanssen & Kuipers discriminant)
- HSS (Heidke Skill Score)
- FAR (False Alarm Ratio), CSI (Critical Success Index / Threat Score)
- Brier Score, Confusion Matrix
"""

import logging
from typing import Dict, Optional, Tuple, Union
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    auc,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

logger = logging.getLogger(__name__)


def compute_all_metrics(
    y_true: Union[np.ndarray, list],
    y_prob: Union[np.ndarray, list],
    threshold: float = 0.5,
) -> Dict[str, Union[float, int, list]]:
    """
    Computes all standard solar flare forecasting and classification metrics.
    """
    y_true = np.asarray(y_true, dtype=int).ravel()
    y_prob = np.asarray(y_prob, dtype=float).ravel()
    y_pred = (y_prob >= threshold).astype(int)

    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    tn, fp, fn, tp = int(tn), int(fp), int(fn), int(tp)

    # Standard metrics
    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    # ROC-AUC & PR-AUC
    try:
        roc_auc = float(roc_auc_score(y_true, y_prob))
    except Exception:
        roc_auc = 0.5

    try:
        p_curve, r_curve, _ = precision_recall_curve(y_true, y_prob)
        pr_auc = float(auc(r_curve, p_curve))
    except Exception:
        pr_auc = 0.0

    # Solar physics skill scores:
    # TSS = Recall - False Positive Rate = (TP / (TP + FN)) - (FP / (FP + TN))
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    tss = float(tpr - fpr)

    # HSS = 2*(TP*TN - FP*FN) / ((TP+FN)*(FN+TN) + (TP+FP)*(FP+TN))
    denominator = ((tp + fn) * (fn + tn)) + ((tp + fp) * (fp + tn))
    if denominator > 0:
        hss = float(2.0 * (tp * tn - fp * fn) / denominator)
    else:
        hss = 0.0

    # FAR = FP / (TP + FP)
    far = float(fp / (tp + fp)) if (tp + fp) > 0 else 0.0

    # CSI = TP / (TP + FP + FN)
    csi = float(tp / (tp + fp + fn)) if (tp + fp + fn) > 0 else 0.0

    # Brier Score
    brier = float(brier_score_loss(y_true, y_prob))

    return {
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "roc_auc": round(roc_auc, 4),
        "pr_auc": round(pr_auc, 4),
        "tss": round(tss, 4),
        "hss": round(hss, 4),
        "far": round(far, 4),
        "csi": round(csi, 4),
        "brier_score": round(brier, 4),
        "threshold": float(threshold),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "total_samples": len(y_true),
        "positive_samples": int((y_true == 1).sum()),
        "negative_samples": int((y_true == 0).sum()),
    }
