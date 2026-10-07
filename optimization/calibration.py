"""
Probability Calibration Module for Solar Flare Forecasting.
Evaluates:
1. Raw Model Probabilities
2. Platt Scaling (Sigmoidal calibration via LogisticRegression)
3. Isotonic Regression Calibration
Calculates Expected Calibration Error (ECE), Brier Score, and reliability curve coordinates.
Calibration models are fitted strictly using validation data.
"""

import json
import logging
import os
import pickle
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss

logger = logging.getLogger(__name__)


def compute_expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """
    Computes Expected Calibration Error (ECE) across n_bins.
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    total_samples = len(y_true)

    for i in range(n_bins):
        bin_lower, bin_upper = bin_boundaries[i], bin_boundaries[i + 1]
        in_bin = (y_prob >= bin_lower) & (y_prob < bin_upper) if i < n_bins - 1 else (y_prob >= bin_lower) & (y_prob <= bin_upper)
        bin_count = int(in_bin.sum())
        if bin_count > 0:
            bin_acc = float(np.mean(y_true[in_bin]))
            bin_conf = float(np.mean(y_prob[in_bin]))
            ece += (bin_count / total_samples) * abs(bin_acc - bin_conf)

    return float(ece)


class SolarProbabilityCalibrator:
    """
    Fits and applies probability calibration models.
    """

    def __init__(self, method: str = "platt"):
        self.method = method  # "platt" or "isotonic"
        self.calibrator = None
        self.is_fitted = False

    def fit(self, y_val_prob: np.ndarray, y_val_true: np.ndarray) -> "SolarProbabilityCalibrator":
        """
        Fits calibration strictly on validation predictions.
        """
        y_prob = np.asarray(y_val_prob, dtype=np.float64).reshape(-1, 1)
        y_true = np.asarray(y_val_true, dtype=int).ravel()

        if self.method == "platt":
            # Platt scaling: LogisticRegression on logits or probabilities
            self.calibrator = LogisticRegression(C=1.0, solver="lbfgs")
            self.calibrator.fit(y_prob, y_true)
        elif self.method == "isotonic":
            self.calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
            self.calibrator.fit(y_prob.ravel(), y_true)
        else:
            raise ValueError(f"Unknown calibration method: {self.method}")

        self.is_fitted = True
        logger.info("Fitted %s calibration on %d validation samples.", self.method, len(y_true))
        return self

    def transform(self, y_prob: np.ndarray) -> np.ndarray:
        """
        Transforms raw probabilities into calibrated probabilities.
        """
        if not self.is_fitted:
            return np.asarray(y_prob, dtype=np.float32)

        arr = np.asarray(y_prob, dtype=np.float64)
        if self.method == "platt":
            cal_prob = self.calibrator.predict_proba(arr.reshape(-1, 1))[:, 1]
        elif self.method == "isotonic":
            cal_prob = self.calibrator.predict(arr.ravel())
        else:
            cal_prob = arr

        return np.clip(cal_prob, 0.0, 1.0).astype(np.float32)

    def evaluate_calibration(self, y_prob: np.ndarray, y_true: np.ndarray) -> Dict[str, float]:
        """
        Calculates Brier score and ECE before and after calibration.
        """
        raw_brier = float(brier_score_loss(y_true, y_prob))
        raw_ece = compute_expected_calibration_error(y_true, y_prob)

        cal_prob = self.transform(y_prob) if self.is_fitted else y_prob
        cal_brier = float(brier_score_loss(y_true, cal_prob))
        cal_ece = compute_expected_calibration_error(y_true, cal_prob)

        return {
            "method": self.method,
            "raw_brier_score": round(raw_brier, 4),
            "calibrated_brier_score": round(cal_brier, 4),
            "raw_ece": round(raw_ece, 4),
            "calibrated_ece": round(cal_ece, 4),
        }

    def save(self, filepath: str = "checkpoints/calibrator.pkl") -> None:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filepath: str = "checkpoints/calibrator.pkl") -> "SolarProbabilityCalibrator":
        with open(filepath, "rb") as f:
            return pickle.load(f)
