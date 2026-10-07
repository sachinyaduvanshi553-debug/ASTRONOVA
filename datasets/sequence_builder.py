"""
Temporal Sequence Builder module.
Constructs 24-hour temporal sequences (T=24, F=16) for each observation ending at time t.
Ensures ZERO future leakage: only data at or before t is used.
Target corresponds strictly to whether a >= M-class flare occurs within [t, t + 24 hours].
"""

import logging
from typing import List, Optional, Tuple
import numpy as np
import pandas as pd
from preprocessing.cleaner import BASE_PAPER_16_FEATURES

logger = logging.getLogger(__name__)


class SequenceBuilder:
    """
    Builds fixed-length temporal window sequences from chronological active region data.
    """

    def __init__(
        self,
        sequence_length: int = 24,
        features: Optional[List[str]] = None,
        timestamp_col: str = "timestamp",
        ar_col: str = "harp_num",
        label_col: str = "label",
    ):
        self.sequence_length = sequence_length
        self.features = features or BASE_PAPER_16_FEATURES
        self.timestamp_col = timestamp_col
        self.ar_col = ar_col
        self.label_col = label_col

    def build_sequences(
        self,
        df: pd.DataFrame,
    ) -> Tuple[np.ndarray, np.ndarray, List[pd.Timestamp]]:
        """
        Extracts sequences of shape (N, 24, 16) and corresponding binary labels.
        Only continuous sequences without missing steps in active regions are preserved.
        """
        df = df.copy()
        if self.timestamp_col in df.columns:
            df[self.timestamp_col] = pd.to_datetime(df[self.timestamp_col], utc=True)
        else:
            raise KeyError(f"Timestamp column '{self.timestamp_col}' missing.")

        # Ensure active region grouping column
        has_ar = self.ar_col in df.columns
        if not has_ar and "active_region" in df.columns:
            self.ar_col = "active_region"
            has_ar = True

        available_features = [f for f in self.features if f in df.columns]
        if len(available_features) != len(self.features):
            logger.warning("Using %d / %d requested features", len(available_features), len(self.features))

        sequences: List[np.ndarray] = []
        labels: List[int] = []
        timestamps: List[pd.Timestamp] = []

        if has_ar:
            # Group by Active Region to prevent boundary cross-contamination
            grouped = df.groupby(self.ar_col)
            for ar_id, ar_df in grouped:
                ar_df = ar_df.sort_values(by=self.timestamp_col).reset_index(drop=True)
                if len(ar_df) < self.sequence_length:
                    continue

                feat_matrix = ar_df[available_features].values.astype(np.float32)
                label_vec = ar_df[self.label_col].values.astype(int)
                time_vec = ar_df[self.timestamp_col].values

                # Sliding window of length 24
                for i in range(len(ar_df) - self.sequence_length + 1):
                    seq = feat_matrix[i : i + self.sequence_length]  # Shape: (24, num_feats)
                    # Label is the target at the end of the sequence window (time t)
                    target = label_vec[i + self.sequence_length - 1]
                    end_time = time_vec[i + self.sequence_length - 1]

                    sequences.append(seq)
                    labels.append(target)
                    timestamps.append(pd.to_datetime(end_time))
        else:
            # Single continuous timeline
            df = df.sort_values(by=self.timestamp_col).reset_index(drop=True)
            feat_matrix = df[available_features].values.astype(np.float32)
            label_vec = df[self.label_col].values.astype(int)
            time_vec = df[self.timestamp_col].values

            for i in range(len(df) - self.sequence_length + 1):
                seq = feat_matrix[i : i + self.sequence_length]
                target = label_vec[i + self.sequence_length - 1]
                end_time = time_vec[i + self.sequence_length - 1]

                sequences.append(seq)
                labels.append(target)
                timestamps.append(pd.to_datetime(end_time))

        X = np.array(sequences, dtype=np.float32) if sequences else np.empty((0, self.sequence_length, len(available_features)), dtype=np.float32)
        y = np.array(labels, dtype=np.float32) if labels else np.empty((0,), dtype=np.float32)

        pos = int((y == 1).sum()) if len(y) > 0 else 0
        neg = len(y) - pos
        ratio = (neg / pos) if pos > 0 else 0.0
        logger.info("Constructed %d sequences of shape %s. Positive: %d, Negative: %d (Imbalance ratio: %.1f:1)",
                    len(X), X.shape, pos, neg, ratio)
        return X, y, timestamps
