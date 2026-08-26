import numpy as np
import pandas as pd


def compute_physics_features(df: pd.DataFrame) -> pd.DataFrame:
    """Computes physics-informed features (simulated if external data unavailable)."""

    # Time since previous flare (assume flare is flux > 1e-6)
    is_flare = df["soft_xray_flux"] > 1e-6
    df.index[is_flare].tolist()

    time_since_flare = []
    last_flare_idx = -10000  # Large negative number

    for idx in df.index:
        if is_flare[idx]:
            last_flare_idx = idx
            time_since_flare.append(0)
        else:
            time_since_flare.append(idx - last_flare_idx)

    df["time_since_prev_flare"] = time_since_flare

    # Simulated NOAA Active Region Count (derived loosely from background flux levels to mock correlation)
    background_proxy = df["soft_xray_flux"].rolling(1440, min_periods=1).mean()
    df["noaa_ar_count"] = (np.log10(background_proxy.clip(1e-9)) + 9) * 2
    df["noaa_ar_count"] = df["noaa_ar_count"].clip(0, 15).astype(int)

    # Simulated Magnetic Complexity Index
    df["magnetic_complexity"] = df["xray_ratio"] * df["noaa_ar_count"] * 0.1
    df["magnetic_complexity"] = df["magnetic_complexity"].clip(0, 5)

    return df
