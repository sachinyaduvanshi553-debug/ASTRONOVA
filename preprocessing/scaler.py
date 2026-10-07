"""
Scientific Feature Scaler module.
Ensures zero data leakage: fitted strictly on training data, persisted to disk,
and used to transform validation, test, and live inference samples.
"""

import os
import pickle
import logging
from typing import List, Optional, Union
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler, StandardScaler

logger = logging.getLogger(__name__)


class SolarFeatureScaler:
    """
    Wraps standard or robust scaling with strict train-only fitting safeguards.
    """

    def __init__(
        self,
        feature_cols: List[str],
        scaler_type: str = "robust",
        clip_range: Optional[tuple[float, float]] = (-10.0, 10.0),
    ):
        self.feature_cols = feature_cols
        self.scaler_type = scaler_type
        self.clip_range = clip_range
        self.scaler = RobustScaler() if scaler_type == "robust" else StandardScaler()
        self.is_fitted = False

    def fit(self, df_or_arr: Union[pd.DataFrame, np.ndarray]) -> "SolarFeatureScaler":
        """
        Fits the scaler on training data.
        """
        if isinstance(df_or_arr, pd.DataFrame):
            X = df_or_arr[self.feature_cols].values
        else:
            X = df_or_arr
        
        # Handle 3D sequence arrays (N, T, F)
        if X.ndim == 3:
            N, T, F = X.shape
            X = X.reshape(-1, F)

        self.scaler.fit(X)
        self.is_fitted = True
        logger.info("Fitted %s scaler on %d training records across %d features.", 
                    self.scaler_type, X.shape[0], len(self.feature_cols))
        return self

    def transform(self, df_or_arr: Union[pd.DataFrame, np.ndarray]) -> Union[pd.DataFrame, np.ndarray]:
        """
        Transforms data using the previously fitted scaler.
        """
        if not self.is_fitted:
            raise RuntimeError("Cannot transform data: SolarFeatureScaler is not fitted yet!")

        if isinstance(df_or_arr, pd.DataFrame):
            df = df_or_arr.copy()
            X = df[self.feature_cols].values
            X_scaled = self.scaler.transform(X)
            if self.clip_range:
                X_scaled = np.clip(X_scaled, self.clip_range[0], self.clip_range[1])
            df[self.feature_cols] = X_scaled
            return df
        else:
            arr = np.array(df_or_arr, dtype=np.float32)
            orig_shape = arr.shape
            if arr.ndim == 3:
                N, T, F = arr.shape
                flat_arr = arr.reshape(-1, F)
                scaled = self.scaler.transform(flat_arr)
                if self.clip_range:
                    scaled = np.clip(scaled, self.clip_range[0], self.clip_range[1])
                return scaled.reshape(N, T, F).astype(np.float32)
            elif arr.ndim == 2:
                scaled = self.scaler.transform(arr)
                if self.clip_range:
                    scaled = np.clip(scaled, self.clip_range[0], self.clip_range[1])
                return scaled.astype(np.float32)
            else:
                raise ValueError(f"Unsupported array dimension: {arr.ndim}")

    def fit_transform(self, df_or_arr: Union[pd.DataFrame, np.ndarray]) -> Union[pd.DataFrame, np.ndarray]:
        """
        Fit on training data and transform.
        """
        return self.fit(df_or_arr).transform(df_or_arr)

    def save(self, filepath: str) -> None:
        """
        Persists the fitted scaler to a pickle file.
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "wb") as f:
            pickle.dump(self, f)
        logger.info("Saved fitted scaler to %s", filepath)

    @classmethod
    def load(cls, filepath: str) -> "SolarFeatureScaler":
        """
        Loads a persisted scaler.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Scaler file not found: {filepath}")
        with open(filepath, "rb") as f:
            scaler = pickle.load(f)
        logger.info("Loaded scaler from %s", filepath)
        return scaler
