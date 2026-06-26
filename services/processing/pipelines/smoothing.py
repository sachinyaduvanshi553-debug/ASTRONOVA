"""Production smoothing pipeline for ASTRONOVA solar flare time series.

Implements:
- Savitzky-Golay filter (polynomial regression smoothing)
- Gaussian kernel smoothing (configurable sigma)
- Exponential Weighted Moving Average (EWMA)
- Adaptive mode selector based on data length
"""
from __future__ import annotations

import logging
from typing import Literal

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter1d

from services.processing.pipelines.base import BasePipeline

logger = logging.getLogger("astronova.processing.smoothing")

_FLUX_COLS = ["soft_xray_flux", "hard_xray_flux"]


class SmoothingPipeline(BasePipeline):
    """Multi-mode smoothing pipeline for GOES XRS flux data.

    mode='savgol'   – Savitzky-Golay polynomial regression filter.
                      Preserves flare peak shapes well.
    mode='gaussian' – Gaussian kernel smoothing (sigma in data points).
    mode='ewma'     – Exponential Weighted Moving Average (span in minutes).
    mode='auto'     – Selects savgol if len >= window_length, else ewma.

    Parameters
    ----------
    mode:           Smoothing algorithm.
    window_length:  Number of data points in SavGol window (must be odd).
    polyorder:      Polynomial order for SavGol (must be < window_length).
    sigma:          Standard deviation for Gaussian filter (data points).
    ewma_span:      Span parameter for EWMA (roughly minutes at 1-min cadence).
    flux_columns:   Columns to smooth.
    """

    def __init__(
        self,
        mode: Literal["savgol", "gaussian", "ewma", "auto"] = "auto",
        window_length: int = 11,
        polyorder: int = 3,
        sigma: float = 2.0,
        ewma_span: int = 5,
        flux_columns: list = None,
    ) -> None:
        self.mode = mode
        self.window_length = window_length if window_length % 2 == 1 else window_length + 1
        self.polyorder = min(polyorder, self.window_length - 1)
        self.sigma = sigma
        self.ewma_span = ewma_span
        self.flux_columns = flux_columns or _FLUX_COLS

    def fit(self, df: pd.DataFrame) -> "SmoothingPipeline":
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            return df

        df = df.copy()
        n = len(df)
        mode = self.mode

        if mode == "auto":
            mode = "savgol" if n >= self.window_length else "ewma"

        for col in self.flux_columns:
            if col not in df.columns:
                continue

            values = df[col].values.astype(float)

            if mode == "savgol":
                if n >= self.window_length:
                    values = savgol_filter(values, self.window_length, self.polyorder)
                else:
                    logger.debug("SavGol skipped for '%s': n=%d < window=%d.", col, n, self.window_length)

            elif mode == "gaussian":
                values = gaussian_filter1d(values, sigma=self.sigma)

            elif mode == "ewma":
                values = df[col].ewm(span=self.ewma_span, adjust=False).mean().values

            # Clamp to physical floor after smoothing (filter can go slightly negative)
            values = np.clip(values, 1e-9, 1e-3)
            df[col] = values

        logger.debug("SmoothingPipeline (%s) applied to %d rows.", mode, n)
        return df
