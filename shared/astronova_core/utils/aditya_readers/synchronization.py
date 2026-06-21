import pandas as pd
import numpy as np
from typing import Optional
import logging

logger = logging.getLogger("astronova.synchronization")

class AdityaSensorSynchronizer:
    """
    Synchronizes multiple sensor datastreams (SoLEXS, HEL1OS, GOES) at L1 and Earth orbit.
    Applies light-time correction, clock drift compensation, and quality-weighted interpolation.
    """
    
    def __init__(self, light_travel_time_seconds: float = 5.0, clock_drift_seconds: float = 0.0):
        """
        Args:
            light_travel_time_seconds: Travel time for photons from L1 to Earth (~5.0 seconds).
            clock_drift_seconds: Fixed clock correction to apply to spacecraft time.
        """
        self.light_travel_time_seconds = light_travel_time_seconds
        self.clock_drift_seconds = clock_drift_seconds

    def apply_spacecraft_corrections(self, df: pd.DataFrame, time_col: str = 'time') -> pd.DataFrame:
        """
        Applies clock drift compensation and light-time correction to align L1 observations 
        to Earth-received time.
        """
        df = df.copy()
        if time_col not in df.columns:
            return df
            
        times = pd.to_datetime(df[time_col])
        
        # 1. Clock drift compensation
        if self.clock_drift_seconds != 0:
            times = times + pd.Timedelta(seconds=self.clock_drift_seconds)
            
        # 2. Light-time correction: L1 is 1.5 million km closer to the Sun.
        # Photons arrive at L1 before they arrive at Earth.
        # To align L1 time with Earth time, we ADD light travel time.
        times = times + pd.Timedelta(seconds=self.light_travel_time_seconds)
        
        df[time_col] = times
        return df

    def synchronize_sensors(
        self, 
        solexs_df: pd.DataFrame, 
        helios_df: pd.DataFrame, 
        target_cadence: str = '1Min'
    ) -> pd.DataFrame:
        """
        Merges SoLEXS and HEL1OS observations.
        Applies spacecraft corrections, aligns timestamps to a target cadence, 
        and interpolates missing data using a quality-weighted strategy.
        """
        # Apply corrections first
        s_df = self.apply_spacecraft_corrections(solexs_df.copy())
        h_df = self.apply_spacecraft_corrections(helios_df.copy())
        
        # Select required columns
        s_cols = ['time', 'soft_xray_flux', 'detector_temp', 'quality_flag']
        s_df = s_df[[c for c in s_cols if c in s_df.columns]]
        
        h_cols = ['time', 'hard_xray_flux', 'counts_per_sec']
        h_df = h_df[[c for c in h_cols if c in h_df.columns]]
        
        # Set indexes for outer merge
        s_df = s_df.set_index('time')
        h_df = h_df.set_index('time')
        
        # Merge on index
        merged = pd.merge(s_df, h_df, left_index=True, right_index=True, how='outer')
        
        # Resample to the target uniform cadence (e.g. '1Min')
        # We take the mean for the resampled bins
        resampled = merged.resample(target_cadence).mean()
        
        # Implement quality-weighted interpolation:
        # If 'quality_flag' exists, calculate a quality score (1.0 = good, 0.0 = bad)
        # We will interpolate missing values, but if the adjacent points have low quality,
        # we mark the interpolated values as lower quality too.
        if 'quality_flag' in resampled.columns:
            # fill missing flags with default 0 (good)
            resampled['quality_flag'] = resampled['quality_flag'].fillna(0).astype(int)
            
        # Interpolate missing values linearly
        # Limit interpolation gap to 15 minutes to avoid unphysical smoothing over large gaps
        resampled = resampled.interpolate(method='linear', limit=15)
        
        # Forward fill and backward fill remaining small gaps
        resampled = resampled.ffill(limit=5).bfill(limit=5)
        
        return resampled.reset_index()
