import pandas as pd
from scipy.signal import savgol_filter
from services.processing.pipelines.base import BasePipeline

class SmoothingPipeline(BasePipeline):
    def __init__(self, window_length: int = 11, polyorder: int = 3):
        self.window_length = window_length
        self.polyorder = polyorder

    def fit(self, df: pd.DataFrame) -> 'SmoothingPipeline':
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        # Apply Savitzky-Golay filter to smooth fluxes
        if len(df) >= self.window_length:
            df["soft_xray_flux"] = savgol_filter(df["soft_xray_flux"], self.window_length, self.polyorder)
            df["hard_xray_flux"] = savgol_filter(df["hard_xray_flux"], self.window_length, self.polyorder)
        return df
