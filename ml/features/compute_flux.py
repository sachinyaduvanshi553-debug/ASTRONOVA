import numpy as np
import pandas as pd


def compute_flux_features(df: pd.DataFrame) -> pd.DataFrame:
    """Computes basic flux features: log transformations and ratios."""
    # Ensure minimum flux to avoid log(0)
    soft_flux = df["soft_xray_flux"].clip(lower=1e-9)
    hard_flux = df["hard_xray_flux"].clip(lower=1e-10)

    df["log_soft_flux"] = np.log10(soft_flux)
    df["log_hard_flux"] = np.log10(hard_flux)
    df["xray_ratio"] = soft_flux / hard_flux

    return df
