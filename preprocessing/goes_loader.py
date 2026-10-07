"""
GOES X-ray Flare Catalog Loader.
Parses NCEI/NOAA GOES flare event catalogs, extracts M- and X-class flare events,
and aligns them with active regions and peak/start times.
"""

import os
import glob
import logging
from typing import List, Optional
import pandas as pd

logger = logging.getLogger(__name__)


class GOESFlareLoader:
    """
    Loads and standardizes GOES X-ray flare events for binary labeling.
    Target definition: >= M-class flare in next 24 hours.
    """

    def __init__(self, data_dir: str = "data/raw/goes_flares"):
        self.data_dir = data_dir

    def load_flares(
        self,
        start_date: str = "2010-05-01",
        end_date: str = "2018-05-31",
        min_class: str = "M",
    ) -> pd.DataFrame:
        """
        Loads flare events and filters for >= min_class (M, X).
        """
        os.makedirs(self.data_dir, exist_ok=True)
        files = glob.glob(os.path.join(self.data_dir, "*.csv")) + glob.glob(os.path.join(self.data_dir, "*.parquet"))

        if not files:
            logger.warning("No GOES flare files found in %s", self.data_dir)
            return pd.DataFrame()

        dfs = []
        for f in files:
            try:
                if f.endswith(".parquet"):
                    df = pd.read_parquet(f)
                else:
                    df = pd.read_csv(f)
                dfs.append(df)
            except Exception as e:
                logger.error("Error loading flare file %s: %s", f, e)

        if not dfs:
            return pd.DataFrame()

        combined_df = pd.concat(dfs, ignore_index=True)

        # Standardize columns: start_time, peak_time, end_time, flare_class, active_region
        if "start_time" in combined_df.columns:
            combined_df["start_time"] = pd.to_datetime(combined_df["start_time"], utc=True)
        if "peak_time" in combined_df.columns:
            combined_df["peak_time"] = pd.to_datetime(combined_df["peak_time"], utc=True)

        if "flare_class" in combined_df.columns:
            # Filter for >= M class (M or X)
            combined_df["is_major_flare"] = combined_df["flare_class"].str.startswith(("M", "X"))
        else:
            combined_df["is_major_flare"] = False

        start_ts = pd.to_datetime(start_date, utc=True)
        end_ts = pd.to_datetime(end_date, utc=True)
        time_col = "peak_time" if "peak_time" in combined_df.columns else "start_time"
        if time_col in combined_df.columns:
            combined_df = combined_df[(combined_df[time_col] >= start_ts) & (combined_df[time_col] <= end_ts)]

        return combined_df
