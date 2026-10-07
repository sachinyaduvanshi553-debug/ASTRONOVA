"""
Error Analysis Module for Solar Flare Forecasting.
Conducts deep diagnostic analysis of:
1. False Positives (Actual = 0, Predicted = 1) -> active regions with strong fields that did not erupt within 24h
2. False Negatives (Actual = 1, Predicted = 0) -> eruptions occurring in lower-activity or emerging regions
"""

import json
import logging
import os
from typing import Dict, List, Optional
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class SolarErrorAnalyzer:
    """
    Performs scientific post-mortem on classification errors.
    """

    def __init__(self, feature_names: List[str]):
        self.feature_names = feature_names

    def analyze_errors(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        y_pred_prob: np.ndarray,
        threshold: float = 0.5,
        top_k: int = 10,
    ) -> Dict[str, any]:
        """
        Analyzes false positive and false negative samples.
        """
        y_true = np.asarray(y_test, dtype=int).ravel()
        y_prob = np.asarray(y_pred_prob, dtype=float).ravel()
        y_pred = (y_prob >= threshold).astype(int)

        fp_indices = np.where((y_true == 0) & (y_pred == 1))[0]
        fn_indices = np.where((y_true == 1) & (y_pred == 0))[0]
        tp_indices = np.where((y_true == 1) & (y_pred == 1))[0]
        tn_indices = np.where((y_true == 0) & (y_pred == 0))[0]

        logger.info(
            "Error breakdown: TP=%d, TN=%d, FP=%d, FN=%d (Total errors: %d)",
            len(tp_indices), len(tn_indices), len(fp_indices), len(fn_indices), len(fp_indices) + len(fn_indices)
        )

        def summarize_group(indices: np.ndarray, group_name: str) -> Dict[str, any]:
            if len(indices) == 0:
                return {"count": 0, "mean_confidence": 0.0, "top_feature_means": {}}

            group_probs = y_prob[indices]
            group_X = X_test[indices]  # (K, 24, 16)
            # Average feature values across all time steps
            feat_means = np.mean(group_X, axis=(0, 1))

            feature_summary = {
                self.feature_names[i]: float(feat_means[i])
                for i in range(len(self.feature_names))
            }

            # Top confidence samples
            sorted_by_conf = np.argsort(-group_probs) if "Positive" in group_name else np.argsort(group_probs)
            representative_indices = indices[sorted_by_conf[:top_k]].tolist()

            return {
                "count": len(indices),
                "mean_confidence": float(np.mean(group_probs)),
                "std_confidence": float(np.std(group_probs)),
                "feature_means": feature_summary,
                "representative_sample_indices": representative_indices,
            }

        fp_analysis = summarize_group(fp_indices, "False Positives")
        fn_analysis = summarize_group(fn_indices, "False Negatives")

        return {
            "total_samples": len(y_true),
            "threshold_used": float(threshold),
            "false_positive_analysis": fp_analysis,
            "false_negative_analysis": fn_analysis,
        }

    def save_analysis(self, report: Dict[str, any], output_path: str = "reports/error_analysis.json") -> None:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)
        logger.info("Saved error analysis to %s", output_path)
