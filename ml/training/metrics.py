import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)


def calculate_classification_metrics(y_true, y_pred, y_prob=None):
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }

    if y_prob is not None:
        try:
            metrics["roc_auc"] = roc_auc_score(y_true, y_prob, multi_class="ovr")
        except ValueError:
            metrics["roc_auc"] = float("nan")

    # Custom Space Weather Metrics
    # True Skill Statistic (TSS) = Recall + Specificity - 1
    # For binary simplification (flare vs no flare):
    is_flare_true = (y_true >= 2).astype(int)  # M and X class
    is_flare_pred = (y_pred >= 2).astype(int)

    tn, fp, fn, tp = confusion_matrix(is_flare_true, is_flare_pred, labels=[0, 1]).ravel()

    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

    metrics["tss"] = tpr - fpr
    metrics["far"] = fp / (tp + fp) if (tp + fp) > 0 else 0
    metrics["pod"] = tpr

    # Heidke Skill Score
    expected_correct = (
        ((tp + fn) * (tp + fp) + (tn + fn) * (tn + fp)) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    )
    metrics["hss"] = (
        (tp + tn - expected_correct) / (tp + tn + fp + fn - expected_correct)
        if (tp + tn + fp + fn - expected_correct) > 0
        else 0
    )

    return metrics


def calculate_regression_metrics(y_true, y_pred):
    return {
        "mse": mean_squared_error(y_true, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
        "mae": mean_absolute_error(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
    }
