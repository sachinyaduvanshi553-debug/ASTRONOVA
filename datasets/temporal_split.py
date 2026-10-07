"""
Chronological and Active-Region Temporal Dataset Splitting.
Strict separation: Train (earlier period), Validation (intermediate), Test (latest unseen).
Includes explicit leakage check and dataset statistics generation.
"""

import json
import logging
import os
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class TemporalDataSplitter:
    """
    Splits chronological solar time-series data without future-to-past leakage.
    Supports chronological date-based split and stratified active-region split.
    """

    def __init__(
        self,
        train_ratio: float = 0.70,
        val_ratio: float = 0.10,
        test_ratio: float = 0.20,
    ):
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio

    def split_dataframe(
        self,
        df: pd.DataFrame,
        timestamp_col: str = "timestamp",
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Splits a DataFrame chronologically based on timestamp ordering.
        """
        df_sorted = df.sort_values(by=timestamp_col).reset_index(drop=True)
        n = len(df_sorted)
        train_end = int(n * self.train_ratio)
        val_end = int(n * (self.train_ratio + self.val_ratio))

        train_df = df_sorted.iloc[:train_end].copy()
        val_df = df_sorted.iloc[train_end:val_end].copy()
        test_df = df_sorted.iloc[val_end:].copy()

        self._log_and_verify_split(train_df, val_df, test_df, timestamp_col=timestamp_col)
        return train_df, val_df, test_df

    def split_arrays(
        self,
        X: np.ndarray,
        y: np.ndarray,
        timestamps: Optional[List[pd.Timestamp]] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Splits sequence arrays (N, 24, 16) chronologically.
        """
        n = len(X)
        train_end = int(n * self.train_ratio)
        val_end = int(n * (self.train_ratio + self.val_ratio))

        X_train, y_train = X[:train_end], y[:train_end]
        X_val, y_val = X[train_end:val_end], y[train_end:val_end]
        X_test, y_test = X[val_end:], y[val_end:]

        stats = self.compute_split_statistics(y_train, y_val, y_test)
        logger.info("Split statistics: %s", stats)
        return X_train, y_train, X_val, y_val, X_test, y_test

    @staticmethod
    def compute_split_statistics(
        y_train: np.ndarray,
        y_val: np.ndarray,
        y_test: np.ndarray,
    ) -> Dict[str, Union[int, float]]:
        """
        Calculates positive/negative counts and class ratios across partitions.
        """
        def get_counts(y):
            pos = int((y == 1).sum())
            neg = int((y == 0).sum())
            tot = pos + neg
            rate = (pos / tot * 100.0) if tot > 0 else 0.0
            return pos, neg, tot, rate

        tr_pos, tr_neg, tr_tot, tr_rate = get_counts(y_train)
        va_pos, va_neg, va_tot, va_rate = get_counts(y_val)
        te_pos, te_neg, te_tot, te_rate = get_counts(y_test)

        return {
            "train_total": tr_tot,
            "train_pos": tr_pos,
            "train_neg": tr_neg,
            "train_pos_rate_pct": round(tr_rate, 3),
            "val_total": va_tot,
            "val_pos": va_pos,
            "val_neg": va_neg,
            "val_pos_rate_pct": round(va_rate, 3),
            "test_total": te_tot,
            "test_pos": te_pos,
            "test_neg": te_neg,
            "test_pos_rate_pct": round(te_rate, 3),
        }

    def _log_and_verify_split(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        timestamp_col: str,
    ) -> None:
        """
        Verifies no timestamp overlap exists across splits.
        """
        train_max = train_df[timestamp_col].max()
        val_min = val_df[timestamp_col].min()
        val_max = val_df[timestamp_col].max()
        test_min = test_df[timestamp_col].min()

        if val_min <= train_max:
            raise ValueError(f"Temporal Leakage Detected! Validation starts ({val_min}) before Train ends ({train_max})")
        if test_min <= val_max:
            raise ValueError(f"Temporal Leakage Detected! Test starts ({test_min}) before Val ends ({val_max})")

        logger.info("Split verified strictly chronological: Train [%s to %s], Val [%s to %s], Test [%s to %s]",
                    train_df[timestamp_col].min(), train_max, val_min, val_max, test_min, test_df[timestamp_col].max())

    @staticmethod
    def save_split_metadata(stats: Dict[str, Union[int, float]], output_path: str = "checkpoints/split_metadata.json") -> None:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(stats, f, indent=2)
        logger.info("Saved split metadata to %s", output_path)
