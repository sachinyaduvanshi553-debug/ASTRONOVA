"""
Data cleaning and quality assurance for SDO/HMI SHARPs and cgem.Lorentz parameters.
Implements missing value handling, invalid value clipping, and duplicate removal.
"""

import logging
from typing import List, Optional
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Base paper's 25 photospheric magnetic field parameters
BASE_PAPER_25_FEATURES: List[str] = [
    "ABSNJZH",
    "AREA_ACR",
    "MEANALP",
    "MEANGBH",
    "MEANGBT",
    "MEANGBZ",
    "MEANJZD",
    "MEANJZH",
    "MEANPOT",
    "MEANSHR",
    "SHRGT45",
    "TOTBSQ",
    "TOTFX",
    "TOTFY",
    "TOTFZ",
    "TOTPOT",
    "TOTUSJH",
    "TOTUSJZ",
    "USFLUX",
    "SAVNCPP",
    "R_VALUE",
    "EPSX",
    "EPSY",
    "EPSZ",
    "MEANF",
]

# The 16 features selected by ANOVA union Mutual Information in the base paper
BASE_PAPER_16_FEATURES: List[str] = [
    "ABSNJZH",
    "AREA_ACR",
    "MEANALP",
    "MEANJZH",
    "MEANPOT",
    "MEANSHR",
    "R_VALUE",
    "SAVNCPP",
    "TOTBSQ",
    "TOTFX",
    "TOTFY",
    "TOTFZ",
    "TOTPOT",
    "TOTUSJH",
    "TOTUSJZ",
    "USFLUX",
]


class SolarDataCleaner:
    """
    Cleans raw SDO/HMI and Lorentz dataframes ensuring numerical validity,
    no corrupt values (-9999, NaN, Inf), and consistent chronological ordering.
    """

    def __init__(
        self,
        features: Optional[List[str]] = None,
        fill_strategy: str = "interpolate",
        max_consecutive_nans: int = 3,
    ):
        self.features = features or BASE_PAPER_25_FEATURES
        self.fill_strategy = fill_strategy
        self.max_consecutive_nans = max_consecutive_nans

    def clean_dataframe(
        self,
        df: pd.DataFrame,
        timestamp_col: str = "timestamp",
        ar_col: str = "harp_num",
    ) -> pd.DataFrame:
        """
        Cleans the input DataFrame:
        1. Ensures datetime timestamp and sorts chronologically per AR.
        2. Replaces invalid sentinel values (-9999, -inf, inf) with NaN.
        3. Interpolates short gaps per active region.
        4. Drops remaining unrecoverable NaNs.
        5. Removes duplicate timestamps per AR.
        """
        df = df.copy()

        # 1. Normalize timestamp and active region identifier
        if timestamp_col in df.columns:
            df[timestamp_col] = pd.to_datetime(df[timestamp_col], utc=True)
        else:
            raise KeyError(f"Timestamp column '{timestamp_col}' not found in dataframe.")

        if ar_col not in df.columns and "active_region" in df.columns:
            df[ar_col] = df["active_region"]

        # Normalize column names if needed (e.g. AREA-ACR -> AREA_ACR, R-VALUE -> R_VALUE)
        renames = {"AREA-ACR": "AREA_ACR", "R-VALUE": "R_VALUE"}
        for old_col, new_col in renames.items():
            if old_col in df.columns and new_col not in df.columns:
                df[new_col] = df[old_col]

        # Ensure all required features exist
        available_features = [f for f in self.features if f in df.columns]
        missing_feats = set(self.features) - set(available_features)
        if missing_feats:
            logger.warning("Features missing from dataset: %s", missing_feats)

        # 2. Replace sentinel/corrupted values
        for feat in available_features:
            df[feat] = pd.to_numeric(df[feat], errors="coerce")
            df[feat] = df[feat].replace([-9999.0, -99999.0, np.inf, -np.inf], np.nan)

        # 3. Sort chronologically per active region
        sort_cols = [ar_col, timestamp_col] if ar_col in df.columns else [timestamp_col]
        df = df.sort_values(by=sort_cols).reset_index(drop=True)

        # 4. Remove duplicate entries
        df = df.drop_duplicates(subset=sort_cols, keep="last").reset_index(drop=True)

        # 5. Missing value handling per AR group
        if ar_col in df.columns:
            cleaned_groups = []
            for _, group in df.groupby(ar_col):
                group = group.copy()
                if self.fill_strategy == "interpolate":
                    group[available_features] = group[available_features].interpolate(
                        method="linear", limit=self.max_consecutive_nans, limit_direction="both"
                    )
                group[available_features] = group[available_features].ffill().bfill()
                cleaned_groups.append(group)
            df = pd.concat(cleaned_groups, ignore_index=True)
        else:
            if self.fill_strategy == "interpolate":
                df[available_features] = df[available_features].interpolate(
                    method="linear", limit=self.max_consecutive_nans, limit_direction="both"
                )
            df[available_features] = df[available_features].ffill().bfill()

        # Drop any records still having NaNs in essential features
        df = df.dropna(subset=available_features).reset_index(drop=True)

        # Ensure final chronological sort
        df = df.sort_values(by=sort_cols).reset_index(drop=True)
        logger.info("Cleaned dataframe shape: %s", df.shape)
        return df
