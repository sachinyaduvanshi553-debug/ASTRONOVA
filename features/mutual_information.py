"""
Mutual Information Feature Selection module for Solar Flare Forecasting.
Computes non-linear mutual information shared between photospheric magnetic features and flare labels.
Applies Min-Max normalization and filters features meeting the threshold (>= 0.2).
"""

import logging
from typing import Dict, List
import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import MinMaxScaler

logger = logging.getLogger(__name__)


class MutualInformationSelector:
    """
    Selects photospheric magnetic features using normalized Mutual Information scores.
    """

    def __init__(self, threshold: float = 0.2, random_state: int = 42, n_neighbors: int = 5):
        self.threshold = threshold
        self.random_state = random_state
        self.n_neighbors = n_neighbors
        self.mi_scores_: Dict[str, float] = {}
        self.normalized_scores_: Dict[str, float] = {}
        self.selected_features_: List[str] = []

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        feature_names: List[str],
    ) -> "MutualInformationSelector":
        """
        Computes MI strictly on training data.
        """
        X_mat = X[feature_names].values
        y_vec = y.values.ravel()

        X_mat = np.nan_to_num(X_mat, nan=0.0, posinf=0.0, neginf=0.0)

        # Compute mutual information
        mi_vals = mutual_info_classif(
            X_mat,
            y_vec,
            n_neighbors=self.n_neighbors,
            random_state=self.random_state,
        )
        mi_vals = np.nan_to_num(mi_vals, nan=0.0)

        # Min-Max scale MI scores to [0, 1]
        scaler = MinMaxScaler()
        mi_norm = scaler.fit_transform(mi_vals.reshape(-1, 1)).flatten()

        self.mi_scores_ = {feat: float(mi_vals[i]) for i, feat in enumerate(feature_names)}
        self.normalized_scores_ = {feat: float(mi_norm[i]) for i, feat in enumerate(feature_names)}

        self.selected_features_ = [
            feat for feat, score in self.normalized_scores_.items() if score >= self.threshold
        ]

        logger.info(
            "MI Selected %d / %d features with normalized threshold >= %.2f: %s",
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
        Returns a DataFrame of all feature MI scores and selection status.
        """
        df = pd.DataFrame(
            [
                {
                    "feature": feat,
                    "raw_mi_score": self.mi_scores_.get(feat, 0.0),
                    "normalized_mi_score": self.normalized_scores_.get(feat, 0.0),
                    "selected_by_mi": feat in self.selected_features_,
                }
                for feat in self.mi_scores_
            ]
        ).sort_values(by="normalized_mi_score", ascending=False).reset_index(drop=True)
        return df
