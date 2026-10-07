"""
SDO/HMI SHARPs Data Loader.
Loads Space-weather HMI Active Region Patches (SHARPs) series parameters.
Strictly adheres to May 2010 - May 2018 study window.
"""

import os
import glob
import logging
from typing import List, Optional
import pandas as pd

logger = logging.getLogger(__name__)

HMI_SHARP_FEATURES = [
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
    "TOTPOT",
    "TOTUSJH",
    "TOTUSJZ",
    "USFLUX",
    "SAVNCPP",
    "R_VALUE",
]


class HMISharpLoader:
    """
    Loads and indexes SDO/HMI SHARP data series.
    """

    def __init__(self, data_dir: str = "data/raw/sdo_hmi"):
        self.data_dir = data_dir

    def load_series(
        self,
        start_date: str = "2010-05-01",
        end_date: str = "2018-05-31",
        file_pattern: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Loads HMI SHARPs data matching the specified date window.
        """
        os.makedirs(self.data_dir, exist_ok=True)
        files = glob.glob(os.path.join(self.data_dir, file_pattern or "*.parquet"))
        if not files:
            files = glob.glob(os.path.join(self.data_dir, "*.csv"))

        if not files:
            logger.warning("No HMI SHARP files found in %s", self.data_dir)
            return pd.DataFrame()

        dfs = []
        for f in sorted(files):
            try:
                if f.endswith(".parquet"):
                    df = pd.read_parquet(f)
                else:
                    df = pd.read_csv(f)
                dfs.append(df)
            except Exception as e:
                logger.error("Error reading HMI file %s: %s", f, e)

        if not dfs:
            return pd.DataFrame()

        combined_df = pd.concat(dfs, ignore_index=True)
        if "timestamp" in combined_df.columns:
            combined_df["timestamp"] = pd.to_datetime(combined_df["timestamp"], utc=True)
            start_ts = pd.to_datetime(start_date, utc=True)
            end_ts = pd.to_datetime(end_date, utc=True)
            combined_df = combined_df[(combined_df["timestamp"] >= start_ts) & (combined_df["timestamp"] <= end_ts)]

        return combined_df
