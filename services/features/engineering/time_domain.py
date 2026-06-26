"""Time-domain feature engineering for ASTRONOVA solar flare forecasting.

Implements:
- Rolling statistics (mean, std, min, max, skew, kurtosis)
- Flux rate-of-change and acceleration
- Exponential Weighted Moving Average (EWMA) trend features
- Temporal context features (hour of day, day of year)
- Lag features for sequence modelling
- Shannon entropy of local flux windows
"""
from __future__ import annotations

import logging
from typing import List

import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis

logger = logging.getLogger("astronova.features.time_domain")

# Rolling windows in minutes (1-min cadence → direct window sizes)
_WINDOWS = [5, 15, 30, 60]


class TimeDomainFeatures:
    """Computes time-domain rolling and lag features.

    Features computed per window W in _WINDOWS:
        soft_flux_roll_mean_W
        soft_flux_roll_std_W
        soft_flux_roll_min_W
        soft_flux_roll_max_W
        soft_flux_roll_skew_W
        soft_flux_roll_kurt_W
        hard_flux_roll_mean_W
        hard_flux_roll_std_W

    Additional features:
        soft_flux_gradient      (1-min diff)
        hard_flux_gradient      (1-min diff)
        soft_flux_accel         (2nd diff)
        ewma_soft_5             (5-min EWMA)
        ewma_soft_15            (15-min EWMA)
        soft_rolling_entropy_30 (Shannon entropy over 30-min window)
        hour_of_day             (0–23, cyclically encoded: sin + cos)
        day_of_year             (1–365, cyclically encoded: sin + cos)
        lag_1, lag_5, lag_15    (lag features for soft_xray_flux)
    """

    def __init__(self, windows: List[int] = None, lag_steps: List[int] = None) -> None:
        self.windows   = windows or _WINDOWS
        self.lag_steps = lag_steps or [1, 5, 15, 30]

    # ------------------------------------------------------------------
    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """Append all time-domain features to ``df``."""
        df = df.copy()

        soft = df["soft_xray_flux"]
        hard = df["hard_xray_flux"]

        # ── Rolling statistics ────────────────────────────────────────
        for w in self.windows:
            roll_soft = soft.rolling(window=w, min_periods=1)
            roll_hard = hard.rolling(window=w, min_periods=1)

            df[f"soft_flux_roll_mean_{w}"] = roll_soft.mean()
            df[f"soft_flux_roll_std_{w}"]  = roll_soft.std().fillna(0.0)
            df[f"soft_flux_roll_min_{w}"]  = roll_soft.min()
            df[f"soft_flux_roll_max_{w}"]  = roll_soft.max()

            # Skew and kurtosis (need at least 3 points)
            df[f"soft_flux_roll_skew_{w}"] = roll_soft.apply(
                lambda x: float(skew(x)) if len(x) >= 3 else 0.0, raw=True
            ).fillna(0.0)
            df[f"soft_flux_roll_kurt_{w}"] = roll_soft.apply(
                lambda x: float(kurtosis(x)) if len(x) >= 4 else 0.0, raw=True
            ).fillna(0.0)

            df[f"hard_flux_roll_mean_{w}"] = roll_hard.mean()
            df[f"hard_flux_roll_std_{w}"]  = roll_hard.std().fillna(0.0)

        # ── Gradient and acceleration ─────────────────────────────────
        df["soft_flux_gradient"] = soft.diff().fillna(0.0)
        df["hard_flux_gradient"] = hard.diff().fillna(0.0)
        df["soft_flux_accel"]    = df["soft_flux_gradient"].diff().fillna(0.0)

        # ── EWMA trend features ───────────────────────────────────────
        df["ewma_soft_5"]  = soft.ewm(span=5,  adjust=False).mean()
        df["ewma_soft_15"] = soft.ewm(span=15, adjust=False).mean()

        # ── Shannon entropy over 30-min rolling window ────────────────
        df["soft_rolling_entropy_30"] = soft.rolling(window=30, min_periods=5).apply(
            self._shannon_entropy, raw=True
        ).fillna(0.0)

        # ── Temporal context features (cyclical encoding) ─────────────
        if isinstance(df.index, pd.DatetimeIndex):
            hour    = df.index.hour
            doy     = df.index.dayofyear
            df["hour_sin"]    = np.sin(2 * np.pi * hour / 24.0)
            df["hour_cos"]    = np.cos(2 * np.pi * hour / 24.0)
            df["doy_sin"]     = np.sin(2 * np.pi * doy / 365.0)
            df["doy_cos"]     = np.cos(2 * np.pi * doy / 365.0)

        # ── Lag features ──────────────────────────────────────────────
        for lag in self.lag_steps:
            df[f"soft_lag_{lag}"] = soft.shift(lag).bfill()

        logger.debug(
            "TimeDomainFeatures: computed features → %d columns total.",
            len(df.columns),
        )
        return df

    # ------------------------------------------------------------------
    @staticmethod
    def _shannon_entropy(arr: np.ndarray) -> float:
        """Shannon entropy of a normalized array (measures flux complexity)."""
        arr = np.clip(arr, 1e-30, None)
        arr_norm = arr / arr.sum()
        return float(-np.sum(arr_norm * np.log2(arr_norm + 1e-30)))
