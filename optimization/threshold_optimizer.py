"""
Threshold Optimization Module for Solar Flare Forecasting.
Evaluates classification thresholds strictly on validation set predictions across [0.05, 0.95].
Finds threshold that optimizes True Skill Statistic (TSS) or F1 score.
Locks threshold for test-set evaluation.
"""

import json
import logging
import os
from typing import Dict, List, Tuple, Union
import numpy as np
import pandas as pd
from evaluation.metrics import compute_all_metrics

logger = logging.getLogger(__name__)


class ThresholdOptimizer:
    """
    Optimizes operating decision thresholds using validation set metrics only.
    """

    def __init__(
        self,
        metric_to_optimize: str = "tss",
        threshold_range: Tuple[float, float, int] = (0.05, 0.95, 91),
    ):
        self.metric_to_optimize = metric_to_optimize
        self.thresholds = np.linspace(threshold_range[0], threshold_range[1], threshold_range[2])
        self.optimal_threshold: float = 0.5
        self.best_metric_value: float = -1.0
        self.sweep_results_df: Optional[pd.DataFrame] = None

    def fit(self, y_val: np.ndarray, y_val_prob: np.ndarray) -> "ThresholdOptimizer":
        """
        Sweeps thresholds across validation data.
        """
        records = []
        best_val = -float("inf")
        best_thresh = 0.5

        for th in self.thresholds:
            metrics = compute_all_metrics(y_val, y_val_prob, threshold=float(th))
            val = metrics.get(self.metric_to_optimize, 0.0)
            metrics["threshold"] = round(float(th), 4)
            records.append(metrics)

            if val > best_val:
                best_val = val
                best_thresh = float(th)

        self.sweep_results_df = pd.DataFrame(records)
        self.optimal_threshold = round(best_thresh, 4)
        self.best_metric_value = round(best_val, 4)

        logger.info(
            "Optimal threshold determined on validation set: %.4f (Validation %s = %.4f)",
            self.optimal_threshold,
            self.metric_to_optimize.upper(),
            self.best_metric_value,
        )
        return self

    def save(self, filepath: str = "checkpoints/threshold.json") -> None:
        """
        Persists the locked threshold to disk.
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        payload = {
            "optimal_threshold": self.optimal_threshold,
            "metric_optimized": self.metric_to_optimize,
            "validation_score": self.best_metric_value,
        }
        with open(filepath, "w") as f:
            json.dump(payload, f, indent=2)
        logger.info("Saved locked threshold to %s", filepath)

    @classmethod
    def load(cls, filepath: str = "checkpoints/threshold.json") -> float:
        """
        Loads locked threshold.
        """
        with open(filepath, "r") as f:
            data = json.load(f)
        return float(data.get("optimal_threshold", 0.5))
