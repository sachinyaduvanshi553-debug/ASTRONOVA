import pandas as pd


def compute_statistical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Computes rolling window statistical features."""

    # 15-minute rolling
    df["soft_rolling_mean_15"] = df["soft_xray_flux"].rolling(window=15, min_periods=1).mean()
    df["soft_rolling_std_15"] = df["soft_xray_flux"].rolling(window=15, min_periods=1).std().fillna(0)

    # 30-minute rolling
    df["soft_rolling_max_30"] = df["soft_xray_flux"].rolling(window=30, min_periods=1).max()
    df["soft_rolling_min_30"] = df["soft_xray_flux"].rolling(window=30, min_periods=1).min()

    return df
