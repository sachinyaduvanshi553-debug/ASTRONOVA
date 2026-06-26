"""Normalization pipeline for ASTRONOVA solar flare flux data.

Implements:
- Log10 transform for heavy-tailed flux distributions
- RobustScaler (median/IQR) to resist residual outliers
- MinMax scaling (optional, for neural network inputs)
- Inverse transform utilities
"""
from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler, MinMaxScaler

from services.processing.pipelines.base import BasePipeline

logger = logging.getLogger("astronova.processing.normalization")

# Columns to normalize (exclude flags and labels)
_FLUX_COLS = ["soft_xray_flux", "hard_xray_flux"]
_LOG_TRANSFORM_FLOOR = 1e-9   # W/m² – same as cleaning floor


class NormalizationPipeline(BasePipeline):
    """Log10 + RobustScaler normalization for GOES XRS flux columns.

    fit():      computes per-column log10 min/IQR from training data.
    transform():applies log10 transform followed by RobustScaler.
    inverse():  reverses scaling back to physical flux units (W/m²).

    Persists scaler state via save() / load() for reproducibility.
    """

    def __init__(
        self,
        flux_columns: List[str] = None,
        use_log: bool = True,
        use_minmax: bool = False,
    ) -> None:
        self.flux_columns = flux_columns or _FLUX_COLS
        self.use_log = use_log
        self.use_minmax = use_minmax
        self._scaler: Optional[RobustScaler | MinMaxScaler] = None
        self._fitted = False

    # ------------------------------------------------------------------
    def fit(self, df: pd.DataFrame) -> "NormalizationPipeline":
        """Fit scaler on ``df`` flux columns (after log transform if enabled)."""
        X = self._extract_and_log(df)
        if self.use_minmax:
            self._scaler = MinMaxScaler(feature_range=(0, 1))
        else:
            self._scaler = RobustScaler()
        self._scaler.fit(X)
        self._fitted = True
        logger.info("NormalizationPipeline fitted on %d rows.", len(df))
        return self

    # ------------------------------------------------------------------
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply normalization to ``df``."""
        if not self._fitted:
            logger.warning("NormalizationPipeline not fitted — running fit_transform.")
            self.fit(df)

        df = df.copy()
        X = self._extract_and_log(df)
        X_scaled = self._scaler.transform(X)

        suffix = "_log_scaled" if self.use_log else "_scaled"
        for i, col in enumerate(self.flux_columns):
            if col in df.columns:
                df[f"{col}{suffix}"] = X_scaled[:, i]

        logger.info("NormalizationPipeline transformed %d rows.", len(df))
        return df

    # ------------------------------------------------------------------
    def inverse_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Reverse scaling on log-scaled columns back to W/m² units."""
        if not self._fitted or self._scaler is None:
            raise RuntimeError("Pipeline must be fitted before inverse_transform.")

        df = df.copy()
        suffix = "_log_scaled" if self.use_log else "_scaled"
        scaled_cols = [f"{col}{suffix}" for col in self.flux_columns if f"{col}{suffix}" in df.columns]

        if not scaled_cols:
            logger.warning("No normalized columns found for inverse transform.")
            return df

        X_scaled = df[scaled_cols].values
        X_unscaled = self._scaler.inverse_transform(X_scaled)

        for i, col in enumerate(self.flux_columns):
            scaled_col = f"{col}{suffix}"
            if scaled_col in df.columns:
                raw = X_unscaled[:, i]
                if self.use_log:
                    raw = np.power(10.0, raw)   # undo log10
                df[f"{col}_physical"] = np.clip(raw, _LOG_TRANSFORM_FLOOR, 1e-3)

        return df

    # ------------------------------------------------------------------
    def save(self, path: str | Path) -> None:
        """Persist scaler to disk using pickle."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump({"scaler": self._scaler, "config": {
                "flux_columns": self.flux_columns,
                "use_log": self.use_log,
                "use_minmax": self.use_minmax,
            }}, fh)
        logger.info("NormalizationPipeline saved to %s.", path)

    @classmethod
    def load(cls, path: str | Path) -> "NormalizationPipeline":
        """Load a persisted NormalizationPipeline from disk."""
        with open(path, "rb") as fh:
            data = pickle.load(fh)
        config = data["config"]
        pipeline = cls(**config)
        pipeline._scaler = data["scaler"]
        pipeline._fitted = True
        logger.info("NormalizationPipeline loaded from %s.", path)
        return pipeline

    # ------------------------------------------------------------------
    def _extract_and_log(self, df: pd.DataFrame) -> np.ndarray:
        """Extract flux columns from df and apply log10 if enabled."""
        cols_present = [c for c in self.flux_columns if c in df.columns]
        if not cols_present:
            raise ValueError(f"None of the expected flux columns found: {self.flux_columns}")

        X = df[cols_present].values.astype(float)
        X = np.clip(X, _LOG_TRANSFORM_FLOOR, None)   # avoid log(0)
        if self.use_log:
            X = np.log10(X)
        return X
