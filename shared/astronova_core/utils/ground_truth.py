import logging
from typing import Any

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

logger = logging.getLogger("astronova.ground_truth")

class FlareGroundTruthBuilder:
    """
    Builds a ground truth solar flare catalog from continuous flux telemetry.
    Segments events based on physics-based peak detection and gradient analysis.
    """

    def __init__(self, prom_threshold: float = 1e-6, min_distance_minutes: int = 15):
        """
        Args:
            prom_threshold: Minimum peak prominence in Watts/m^2.
            min_distance_minutes: Minimum separation between distinct flare events.
        """
        self.prom_threshold = prom_threshold
        self.min_distance_minutes = min_distance_minutes

    def classify_flux_goes(self, peak_flux: float) -> str:
        """Converts raw peak flux in W/m^2 to GOES class string (e.g., M1.5, X2.0)."""
        if peak_flux <= 0:
            return "A0.0"

        log_flux = np.log10(peak_flux)

        if log_flux < -8:
            val = peak_flux / 1e-9
            return f"A0.{max(1, int(val))}"
        elif log_flux < -7:
            val = peak_flux / 1e-8
            return f"A{val:.1f}"
        elif log_flux < -6:
            val = peak_flux / 1e-7
            return f"B{val:.1f}"
        elif log_flux < -5:
            val = peak_flux / 1e-6
            return f"C{val:.1f}"
        elif log_flux < -4:
            val = peak_flux / 1e-5
            return f"M{val:.1f}"
        else:
            val = peak_flux / 1e-4
            return f"X{val:.1f}"

    def segment_flare(
        self,
        df: pd.DataFrame,
        peak_idx: int,
        flux_col: str = "soft_xray_flux",
        time_col: str = "time"
    ) -> dict[str, Any] | None:
        """
        Segments a single flare given its peak index.
        Determines the start and end times by analyzing gradients and decay.
        """
        n = len(df)
        if peak_idx <= 0 or peak_idx >= n - 1:
            return None

        times = df[time_col].values
        fluxes = df[flux_col].values

        peak_time = pd.to_datetime(times[peak_idx])
        peak_flux = fluxes[peak_idx]

        # 1. Estimate background level before the flare
        # Scan backward up to 60 minutes to find the local minimum
        lookback = min(60, peak_idx)
        pre_flare_window = fluxes[peak_idx - lookback : peak_idx]

        if len(pre_flare_window) == 0:
            return None

        bg_flux = np.min(pre_flare_window)
        bg_idx = peak_idx - lookback + np.argmin(pre_flare_window)

        # Flare start time: when flux begins rising significantly above background
        # Or simply the local minimum before the peak
        start_idx = bg_idx
        # Refine start index: look for when the gradient becomes consistently positive
        for idx in range(bg_idx, peak_idx):
            if idx < n - 2:
                # 3-point rolling average gradient
                grad = fluxes[idx + 1] - fluxes[idx]
                if grad > 1e-8:
                    start_idx = idx
                    break

        start_time = pd.to_datetime(times[start_idx])

        # 2. Estimate flare end time (decay phase)
        # Scan forward up to 120 minutes
        lookforward = min(120, n - 1 - peak_idx)
        post_flare_window = fluxes[peak_idx : peak_idx + lookforward]

        if len(post_flare_window) == 0:
            return None

        # Target decay level: background + 1/e (approx 36.8%) of the flare peak amplitude above background
        flare_amplitude = peak_flux - bg_flux
        target_decay_flux = bg_flux + (1.0 / np.e) * flare_amplitude

        end_idx = peak_idx + lookforward - 1
        for idx in range(peak_idx, peak_idx + lookforward):
            if fluxes[idx] <= target_decay_flux:
                end_idx = idx
                break

        end_time = pd.to_datetime(times[end_idx])

        duration_minutes = (end_time - start_time).total_seconds() / 60.0
        rise_time_minutes = (peak_time - start_time).total_seconds() / 60.0
        decay_time_minutes = (end_time - peak_time).total_seconds() / 60.0

        return {
            "start_time": start_time,
            "peak_time": peak_time,
            "end_time": end_time,
            "peak_flux": float(peak_flux),
            "background_flux": float(bg_flux),
            "goes_class": self.classify_flux_goes(peak_flux),
            "duration_minutes": float(duration_minutes),
            "rise_time_minutes": float(rise_time_minutes),
            "decay_time_minutes": float(decay_time_minutes)
        }

    def build_catalog(
        self,
        df: pd.DataFrame,
        flux_col: str = "soft_xray_flux",
        time_col: str = "time"
    ) -> pd.DataFrame:
        """
        Scans a continuous flux DataFrame and detects all flare events.
        Returns a DataFrame representing the ground truth flare catalog.
        """
        if df.empty or len(df) < 5:
            return pd.DataFrame(columns=[
                "start_time", "peak_time", "end_time", "peak_flux",
                "background_flux", "goes_class", "duration_minutes",
                "rise_time_minutes", "decay_time_minutes"
            ])

        df_sorted = df.sort_values(by=time_col).reset_index(drop=True)
        fluxes = df_sorted[flux_col].fillna(1e-9).values

        # Detect peaks using scipy.signal.find_peaks
        # Distance constraint translates time distance to number of samples (assuming 1-minute cadence)
        peaks_indices, _ = find_peaks(
            fluxes,
            prominence=self.prom_threshold,
            distance=self.min_distance_minutes
        )

        flares = []
        for idx in peaks_indices:
            flare_data = self.segment_flare(df_sorted, idx, flux_col, time_col)
            if flare_data:
                flares.append(flare_data)

        catalog_df = pd.DataFrame(flares)
        if not catalog_df.empty:
            catalog_df = catalog_df.sort_values(by="start_time").reset_index(drop=True)

        logger.info(f"Built ground truth catalog with {len(catalog_df)} detected flare events.")
        return catalog_df
