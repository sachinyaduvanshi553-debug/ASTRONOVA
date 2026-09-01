import numpy as np
from sklearn.metrics import brier_score_loss


def expected_calibration_error(y_true, y_prob, n_bins=10):
    """
    Computes Expected Calibration Error (ECE).
    y_true: True binary labels
    y_prob: Predicted probabilities
    """
    bins = np.linspace(0, 1, n_bins + 1)
    binids = np.digitize(y_prob, bins) - 1

    ece = 0.0
    for i in range(n_bins):
        bin_idx = binids == i
        if np.sum(bin_idx) > 0:
            bin_true = y_true[bin_idx]
            bin_prob = y_prob[bin_idx]

            acc = np.mean(bin_true)
            conf = np.mean(bin_prob)

            ece += (np.sum(bin_idx) / len(y_true)) * np.abs(acc - conf)

    return ece


def brier_score(y_true, y_prob):
    """
    Computes the Brier Score.
    """
    return brier_score_loss(y_true, y_prob)


def expected_calibration_error_multiclass(y_true, y_prob, n_bins=10):
    """
    Computes ECE for multi-class classification.
    y_prob: [N, C] array of probabilities
    """
    preds = np.argmax(y_prob, axis=1)
    confs = np.max(y_prob, axis=1)
    accs = (preds == y_true).astype(float)

    return expected_calibration_error(accs, confs, n_bins)


def compute_calibration_curve_data(y_true, y_prob, n_bins=10):
    """
    Computes calibration curve (reliability diagram) data for binary classification.
    Returns (prob_true, prob_pred)
    """
    from sklearn.calibration import calibration_curve

    return calibration_curve(y_true, y_prob, n_bins=n_bins, strategy="uniform")
