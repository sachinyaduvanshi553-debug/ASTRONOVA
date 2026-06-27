import pandas as pd

from astronova_core.utils.physics import compute_xray_ratio


class PhysicsFeatures:
    @staticmethod
    def compute_physics_features(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        # Compute soft/hard x-ray ratio
        df["xray_ratio"] = df.apply(
            lambda row: compute_xray_ratio(row["soft_xray_flux"], row["hard_xray_flux"]), axis=1
        )
        return df
