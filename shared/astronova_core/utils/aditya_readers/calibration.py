import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger("astronova.calibration")

class AdityaCalibrator:
    """
    Calibrator for Aditya-L1 instruments (SoLEXS and HEL1OS).
    Applies temperature correction, gain drift compensation, and cross-calibration with GOES.
    """

    def __init__(self, temp_coefficient: float = -0.002, reference_temp: float = 20.0):
        self.temp_coefficient = temp_coefficient
        self.reference_temp = reference_temp

    def calibrate_solexs(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calibrates SoLEXS flux using detector temperature and gain drift models.
        """
        df = df.copy()

        # 1. Temperature correction: flux_corr = flux * (1 + temp_coeff * (temp - ref_temp))
        if 'detector_temp' in df.columns and 'soft_xray_flux' in df.columns:
            # Prevent unphysical temp inputs
            temps = df['detector_temp'].clip(-50, 100)
            correction_factor = 1.0 + self.temp_coefficient * (temps - self.reference_temp)
            df['soft_xray_flux'] = df['soft_xray_flux'] * correction_factor

        # 2. Gain drift compensation: instruments experience small sensitivity loss over time
        # We model a linear drift of 0.01% per day since a reference date (e.g., launch on Sep 2, 2023)
        if 'time' in df.columns and 'soft_xray_flux' in df.columns:
            launch_date = pd.to_datetime("2023-09-02")
            times = pd.to_datetime(df['time'])
            days_since_launch = (times - launch_date).dt.total_seconds() / (24 * 3600)
            drift_factor = 1.0 - (0.0001 * days_since_launch)
            # Clip drift factor to prevent division by zero or negative values
            drift_factor = np.clip(drift_factor, 0.5, 1.0)
            df['soft_xray_flux'] = df['soft_xray_flux'] / drift_factor

        return df

    def calibrate_helios(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calibrates HEL1OS count rate data.
        Applies dead-time correction if raw_counts exist and not already corrected.
        """
        df = df.copy()
        if 'raw_counts' in df.columns and 'counts_per_sec' in df.columns:
            # Dead time correction: R_true = R_obs / (1 - R_obs * tau)
            tau = 1e-5 # 10 microsecond dead time
            obs_rate = df['raw_counts']
            denom = 1.0 - obs_rate * tau
            denom = np.clip(denom, 0.01, 1.0)
            df['counts_per_sec'] = obs_rate / denom

            # Count to flux conversion (standard conversion: 1 count/sec = 1e-11 W/m^2)
            df['hard_xray_flux'] = df['counts_per_sec'] * 1e-11

        return df

    def cross_calibrate(self, aditya_df: pd.DataFrame, goes_df: pd.DataFrame) -> dict[str, Any]:
        """
        Performs cross-calibration validation of Aditya-L1 SoLEXS against NOAA GOES.
        Finds overlapping time intervals and fits a linear relationship: GOES = scale * SoLEXS + offset.
        """
        # Ensure time columns are datetime and set as index
        a_df = aditya_df[['time', 'soft_xray_flux']].copy().rename(columns={'soft_xray_flux': 'solexs_flux'})
        g_df = goes_df[['time', 'soft_xray_flux']].copy().rename(columns={'soft_xray_flux': 'goes_flux'})

        a_df['time'] = pd.to_datetime(a_df['time']).dt.round('1Min')
        g_df['time'] = pd.to_datetime(g_df['time']).dt.round('1Min')

        # Merge on time
        merged = pd.merge(a_df, g_df, on='time').dropna()

        if len(merged) < 10:
            logger.warning("Insufficient overlapping data points for cross-calibration.")
            return {"scale_factor": 1.0, "offset": 0.0, "r_squared": 0.0, "overlapping_points": len(merged)}

        # Fit linear regression: y = mx + c
        x = merged['solexs_flux'].values
        y = merged['goes_flux'].values

        # Linear regression using numpy polyfit
        slope, intercept = np.polyfit(x, y, 1)

        # Compute R-squared correlation
        correlation_matrix = np.corrcoef(x, y)
        r_squared = correlation_matrix[0, 1] ** 2 if correlation_matrix.shape == (2, 2) else 0.0

        logger.info(f"Cross-calibration fit: GOES = {slope:.3f} * SoLEXS + {intercept:.3e} (R^2 = {r_squared:.4f})")

        return {
            "scale_factor": float(slope),
            "offset": float(intercept),
            "r_squared": float(r_squared),
            "overlapping_points": len(merged)
        }
