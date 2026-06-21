import pandas as pd
import numpy as np

class TimeDomainFeatures:
    @staticmethod
    def compute_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        # Compute rolling mean and std
        for window in [5, 15, 30]:
            df[f"soft_flux_roll_mean_{window}"] = df["soft_xray_flux"].rolling(window=window, min_periods=1).mean()
            df[f"soft_flux_roll_std_{window}"] = df["soft_xray_flux"].rolling(window=window, min_periods=1).std().fillna(0)
            df[f"hard_flux_roll_mean_{window}"] = df["hard_xray_flux"].rolling(window=window, min_periods=1).mean()
            df[f"hard_flux_roll_std_{window}"] = df["hard_xray_flux"].rolling(window=window, min_periods=1).std().fillna(0)
        
        # Flux derivatives
        df["soft_flux_gradient"] = df["soft_xray_flux"].diff().fillna(0)
        df["hard_flux_gradient"] = df["hard_xray_flux"].diff().fillna(0)
        return df
