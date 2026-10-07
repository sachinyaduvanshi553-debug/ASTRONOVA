"""
ANOVA Feature Selection module for Solar Flare Forecasting.
Calculates ANOVA F-value across classes on training data only.
Applies Min-Max normalization and filters features meeting the threshold (>= 0.1).
"""

import logging
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from sklearn.feature_selection import f_classif
from sklearn.preprocessing import MinMaxScaler

logger = logging.getLogger(__name__)


class ANOVASelector:
    """
    Selects photospheric magnetic features using normalized ANOVA F-scores.
    """

    def __init__(self, threshold: float = 0.1):
        self.threshold = threshold
        self.f_scores_: Dict[str, float] = {}
        self.normalized_scores_: Dict[str, float] = {}
        self.selected_features_: List[str] = []

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        feature_names: List[str],
    ) -> "ANOVASelector":
        """
        Fits ANOVA on training data strictly.
        """
        X_mat = X[feature_names].values
        y_vec = y.values.ravel()

        # Handle any NaN/Inf
        X_mat = np.nan_to_num(X_mat, nan=0.0, posinf=0.0, neginf=0.0)

        f_vals, p_vals = f_classif(X_mat, y_vec)
        f_vals = np.nan_to_num(f_vals, nan=0.0)

        # Min-Max scale F-scores to [0, 1]
        scaler = MinMaxScaler()
        f_norm = scaler.fit_transform(f_vals.reshape(-1, 1)).flatten()

        self.f_scores_ = {feat: float(f_vals[i]) for i, feat in enumerate(feature_names)}
        self.normalized_scores_ = {feat: float(f_norm[i]) for i, feat in enumerate(feature_names)}

        self.selected_features_ = [
            feat for feat, score in self.normalized_scores_.items() if score >= self.threshold
        ]

        logger.info(
            "ANOVA Selected %d / %d features with normalized threshold >= %.2f: %s",
            len(self.selected_features_),
            len(feature_names),
            self.threshold,
            self.selected_features_,
        )
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Filters dataframe to selected features.
        """
        return X[self.selected_features_]

    def get_scores_dataframe(self) -> pd.DataFrame:
        """
        Returns a DataFrame of all feature F-scores and selection status.
        """
        df = pd.DataFrame(
            [
                {
                    "feature": feat,
                    "raw_f_score": self.f_scores_.get(feat, 0.0),
                    "normalized_f_score": self.normalized_scores_.get(feat, 0.0),
                    "selected_by_anova": feat in self.selected_features_,
                }
                for feat in self.f_scores_
            ]
        ).sort_values(by="normalized_f_score", ascending=False).reset_index(drop=True)
        return df
