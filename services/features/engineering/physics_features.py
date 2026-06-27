"""Physics-based feature engineering for ASTRONOVA solar flare forecasting.

Implements research-grade features following:
- GOES XRS X-ray spectral physics (NIST solar X-ray classifications)
- Solar flare lifecycle phase model (Quiescent → Pre-flare → Rise → Peak → Decay)
- ISRO/SWPC Solar Hazard Index (SHI) formulation
- Thermal/non-thermal radiation proxies
- Spectral hardness ratio (XRSA/XRSB)
"""
from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger("astronova.features.physics")

# GOES flux thresholds  (W/m²)
_CLASS_THRESHOLDS = {
    "A": (1e-8, 1e-7),
    "B": (1e-7, 1e-6),
    "C": (1e-6, 1e-5),
    "M": (1e-5, 1e-4),
    "X": (1e-4, np.inf),
}


def classify_flux(flux: float) -> str:
    """Classify a GOES XRSB flux value to a GOES class string (e.g., 'M4.2')."""
    for cls, (low, high) in _CLASS_THRESHOLDS.items():
        if low <= flux < high:
            multiplier = flux / low
            return f"{cls}{multiplier:.1f}"
    return "A0.0" if flux < 1e-8 else "X100+"


def flux_to_class_numeric(flux: float) -> float:
    """Convert GOES flux to a continuous numeric severity score (0–5 scale).

    A=0–1, B=1–2, C=2–3, M=3–4, X=4–5
    """
    levels = [1e-8, 1e-7, 1e-6, 1e-5, 1e-4]
    base = 0.0
    for i, threshold in enumerate(levels):
        if flux >= threshold:
            base = float(i)
    # Sub-class decimal
    for cls, (low, high) in _CLASS_THRESHOLDS.items():
        if low <= flux < min(high, 1e-3):
            sub = np.log10(flux / low)
            return base + sub
    return base



class PhysicsFeatures:
    """Computes physics-motivated features for each row in a GOES flux DataFrame.

    All computed features are appended as new columns.

    Features:
    ---------
    xray_ratio            : XRSB / XRSA  (spectral hardness proxy)
    spectral_hardness     : log10(XRSB / XRSA)  — harder = more energetic
    flux_severity         : Continuous class score (0–5)
    log_soft_flux         : log10(soft_xray_flux)
    log_hard_flux         : log10(hard_xray_flux)
    log_flux_diff         : log_soft – log_hard  (proxy for non-thermal component)
    soft_xray_gradient    : 1st diff of soft flux (W/m²/min)
    hard_xray_gradient    : 1st diff of hard flux
    flux_acceleration     : 2nd diff of soft flux (curvature)
    energy_proxy          : Cumulative trapezoidal integral (running sum)
    precursor_score       : Weighted combination of gradient × ratio
    lifecycle_phase       : Encoded flare phase (0=Quiet, 1=Pre-flare, 2=Rise, 3=Peak, 4=Decay)
    """

    PHASE_MAP = {
        "Quiescent": 0,
        "Pre-flare": 1,
        "Rise":      2,
        "Peak":      3,
        "Decay":     4,
    }

    def __init__(self, window: int = 10) -> None:
        """
        Parameters
        ----------
        window: rolling window (minutes) for lifecycle detection.
        """
        self.window = window

    # ------------------------------------------------------------------
    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """Append all physics features to ``df``."""
        df = df.copy()

        soft = df["soft_xray_flux"].values.astype(float)
        hard = df["hard_xray_flux"].values.astype(float)

        # Guard against zeros
        soft = np.clip(soft, 1e-9, None)
        hard = np.clip(hard, 1e-9, None)

        # ── Basic spectral features ───────────────────────────────────
        df["xray_ratio"]       = soft / hard
        df["spectral_hardness"]= np.log10(soft / hard)
        df["log_soft_flux"]    = np.log10(soft)
        df["log_hard_flux"]    = np.log10(hard)
        df["log_flux_diff"]    = df["log_soft_flux"] - df["log_hard_flux"]
        df["flux_severity"]    = [flux_to_class_numeric(f) for f in soft]

        # ── Gradient features ─────────────────────────────────────────
        df["soft_xray_gradient"] = np.gradient(soft)
        df["hard_xray_gradient"] = np.gradient(hard)
        df["flux_acceleration"]  = np.gradient(np.gradient(soft))

        # ── Cumulative energy proxy (trapezoidal running integral) ────
        energy = np.zeros(len(soft))
        for i in range(1, len(soft)):
            energy[i] = energy[i - 1] + 0.5 * (soft[i] + soft[i - 1])
        df["energy_proxy"] = energy

        # ── Precursor activity score ──────────────────────────────────
        grad_pos = np.maximum(df["soft_xray_gradient"].values, 0.0)
        df["precursor_score"] = (grad_pos * 1e5) + (soft / hard * 0.05)

        # ── Lifecycle phase (rolling window approach) ─────────────────
        phases = []
        w = self.window
        for i in range(len(soft)):
            window_slice = soft[max(0, i - w + 1): i + 1]
            phases.append(self._classify_phase(window_slice))
        df["lifecycle_phase"] = [self.PHASE_MAP[p] for p in phases]

        logger.debug("PhysicsFeatures: computed %d features on %d rows.", 14, len(df))
        return df

    # ------------------------------------------------------------------
    @staticmethod
    def _classify_phase(flux_window: np.ndarray) -> str:
        if len(flux_window) < 3:
            return "Quiescent"
        grad = np.gradient(flux_window)
        mean_flux = np.mean(flux_window)
        last_grad = grad[-1]
        if last_grad > 1e-5:
            return "Rise"
        elif last_grad > 1e-7:
            return "Pre-flare"
        elif last_grad < -1e-6:
            return "Decay"
        elif mean_flux > 1e-5:
            return "Peak"
        return "Quiescent"
