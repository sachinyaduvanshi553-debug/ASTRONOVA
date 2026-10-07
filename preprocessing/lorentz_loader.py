"""
cgem.Lorentz Data Loader.
Loads integrated Lorentz force series estimates from JSOC.
"""

import os
import glob
import logging
from typing import List, Optional
import pandas as pd

logger = logging.getLogger(__name__)

LORENTZ_FEATURES = [
    "TOTBSQ",
    "TOTFX",
    "TOTFY",
    "TOTFZ",
    "EPSX",
    "EPSY",
    "EPSZ",
    "MEANF",
]


class LorentzLoader:
    """
    Loads and parses cgem.Lorentz integrated Lorentz force series.
    """

    def __init__(self, data_dir: str = "data/raw/cgem_lorentz"):
        self.data_dir = data_dir

    def load_series(
        self,
        start_date: str = "2010-05-01",
        end_date: str = "2018-05-31",
        file_pattern: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Loads Lorentz force parameters within the target study window.
        """
        os.makedirs(self.data_dir, exist_ok=True)
        files = glob.glob(os.path.join(self.data_dir, file_pattern or "*.parquet"))
        if not files:
            files = glob.glob(os.path.join(self.data_dir, "*.csv"))

        if not files:
            logger.warning("No Lorentz files found in %s", self.data_dir)
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
                logger.error("Error reading Lorentz file %s: %s", f, e)

        if not dfs:
            return pd.DataFrame()

        combined_df = pd.concat(dfs, ignore_index=True)
        if "timestamp" in combined_df.columns:
            combined_df["timestamp"] = pd.to_datetime(combined_df["timestamp"], utc=True)
            start_ts = pd.to_datetime(start_date, utc=True)
            end_ts = pd.to_datetime(end_date, utc=True)
            combined_df = combined_df[(combined_df["timestamp"] >= start_ts) & (combined_df["timestamp"] <= end_ts)]

        return combined_df
