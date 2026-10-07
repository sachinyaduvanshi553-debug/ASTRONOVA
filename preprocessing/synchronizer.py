"""
Data Synchronizer module.
Merges SDO/HMI SHARPs and cgem.Lorentz series on (timestamp, active_region),
aligns with GOES flare catalog, assigns the binary target label (>= M-class flare in next 24 hours),
and outputs a clean, synchronized multi-feature tabular dataset.
"""

import os
import logging
from typing import Optional, List
import numpy as np
import pandas as pd
from preprocessing.cleaner import SolarDataCleaner, BASE_PAPER_25_FEATURES

logger = logging.getLogger(__name__)


class SolarDataSynchronizer:
    """
    Synchronizes SHARP parameters, Lorentz forces, and flare catalog labels.
    """

    def __init__(
        self,
        prediction_horizon_hours: int = 24,
        features: Optional[List[str]] = None,
    ):
        self.prediction_horizon = pd.Timedelta(hours=prediction_horizon_hours)
        self.features = features or BASE_PAPER_25_FEATURES
        self.cleaner = SolarDataCleaner(features=self.features)

    def synchronize(
        self,
        hmi_df: pd.DataFrame,
        lorentz_df: Optional[pd.DataFrame] = None,
        flares_df: Optional[pd.DataFrame] = None,
        output_path: Optional[str] = "data/processed/solar_flare_features_2010_2018.parquet",
    ) -> pd.DataFrame:
        """
        Merges datasets, creates 24h lookahead binary flare label, and validates completeness.
        """
        logger.info("Starting synchronization: HMI shape %s", hmi_df.shape)

        # 1. Merge HMI and Lorentz data if available
        if lorentz_df is not None and not lorentz_df.empty:
            # Merge on timestamp and harp_num / active_region
            merge_keys = ["timestamp"]
            for key in ["harp_num", "active_region"]:
                if key in hmi_df.columns and key in lorentz_df.columns:
                    merge_keys.append(key)
                    break

            merged_df = pd.merge(hmi_df, lorentz_df, on=merge_keys, how="inner", suffixes=("", "_lorentz"))
        else:
            merged_df = hmi_df.copy()

        # 2. Clean and handle missing/invalid values
        cleaned_df = self.cleaner.clean_dataframe(merged_df, timestamp_col="timestamp")

        # 3. Compute flare labels if flares_df is provided
        if flares_df is not None and not flares_df.empty:
            cleaned_df = self._assign_flare_labels(cleaned_df, flares_df)
        elif "label" not in cleaned_df.columns and "target" not in cleaned_df.columns:
            # If label not present, initialize target column
            cleaned_df["label"] = 0

        # Ensure label column is named 'label' and is integer {0, 1}
        if "target" in cleaned_df.columns and "label" not in cleaned_df.columns:
            cleaned_df["label"] = cleaned_df["target"]

        cleaned_df["label"] = cleaned_df["label"].astype(int)

        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            cleaned_df.to_parquet(output_path, index=False)
            logger.info("Saved synchronized dataset to %s with shape %s", output_path, cleaned_df.shape)

        return cleaned_df

    def _assign_flare_labels(self, data_df: pd.DataFrame, flares_df: pd.DataFrame) -> pd.DataFrame:
        """
        Assigns label=1 if an M- or X-class flare occurs within [t, t + 24 hours], else 0.
        """
        df = data_df.copy()
        df["label"] = 0

        flare_time_col = "start_time" if "start_time" in flares_df.columns else "peak_time"
        major_flares = flares_df[flares_df.get("is_major_flare", True)].copy()

        if major_flares.empty:
            logger.warning("No major flares found in catalog; all labels set to 0.")
            return df

        flare_times = pd.to_datetime(major_flares[flare_time_col], utc=True).sort_values().values
        flare_ars = major_flares["active_region"].values if "active_region" in major_flares.columns else None

        # Check for each observation whether a flare occurs in [timestamp, timestamp + 24h]
        labels = np.zeros(len(df), dtype=int)
        timestamps = df["timestamp"].values

        for i, t in enumerate(timestamps):
            t_end = t + np.timedelta64(24, "h")
            # Flare occurs if any flare_time is in [t, t_end]
            in_window = (flare_times >= t) & (flare_times <= t_end)
            if np.any(in_window):
                labels[i] = 1

        df["label"] = labels
        pos_count = int(labels.sum())
        logger.info("Assigned flare labels: %d positive (%.2f%%), %d negative",
                    pos_count, 100.0 * pos_count / len(labels), len(labels) - pos_count)
        return df
