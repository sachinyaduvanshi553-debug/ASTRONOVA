import numpy as np
import pandas as pd


def compute_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Computes derivatives, seasonality, and temporal features."""

    # Gradients (1st derivative)
    df["soft_gradient"] = df["soft_xray_flux"].diff().fillna(0)
    df["hard_gradient"] = df["hard_xray_flux"].diff().fillna(0)

    # Acceleration (2nd derivative)
    df["flux_acceleration"] = df["soft_gradient"].diff().fillna(0)

    # Time seasonality
    if not pd.api.types.is_datetime64_any_dtype(df["time"]):
        df["time"] = pd.to_datetime(df["time"])

    df["hour_sin"] = np.sin(2 * np.pi * df["time"].dt.hour / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * df["time"].dt.hour / 24.0)

    df["doy_sin"] = np.sin(2 * np.pi * df["time"].dt.dayofyear / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * df["time"].dt.dayofyear / 365.25)

    return df
