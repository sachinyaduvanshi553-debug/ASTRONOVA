import pandas as pd
import numpy as np
from services.processing.pipelines.base import BasePipeline

class DataCleaningPipeline(BasePipeline):
    def fit(self, df: pd.DataFrame) -> 'DataCleaningPipeline':
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        # Drop duplicate records
        df = df.drop_duplicates(subset=["time"])
        # Fill missing values
        df["soft_xray_flux"] = df["soft_xray_flux"].interpolate(method="linear").ffill().bfill()
        df["hard_xray_flux"] = df["hard_xray_flux"].interpolate(method="linear").ffill().bfill()
        # Handle outliers using IQR
        for col in ["soft_xray_flux", "hard_xray_flux"]:
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            df[col] = np.clip(df[col], lower_bound, upper_bound)
        return df
