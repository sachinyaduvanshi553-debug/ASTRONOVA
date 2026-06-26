"""Temporal alignment pipeline for ASTRONOVA multi-source solar data.

Merges GOES XRS flux with NOAA event labels and other ancillary streams
onto a common 1-minute UTC time grid using forward-fill joining.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from services.processing.pipelines.base import BasePipeline

logger = logging.getLogger("astronova.processing.alignment")

# Standard cadence for ASTRONOVA unified dataset
_RESAMPLE_FREQ = "1min"


class AlignmentPipeline(BasePipeline):
    """Resamples a DataFrame to a fixed 1-minute UTC grid.

    Steps:
    1. Ensure DatetimeIndex is UTC-aware and sorted.
    2. Resample to _RESAMPLE_FREQ using mean aggregation (flux channels).
    3. Forward-fill categorical / flag columns (quality_flag, flare_class).
    4. Back-fill any leading NaN values introduced by resampling.
    """

    def __init__(
        self,
        freq: str = _RESAMPLE_FREQ,
        fill_limit: int = 10,
    ) -> None:
        self.freq = freq
        self.fill_limit = fill_limit

    def fit(self, df: pd.DataFrame) -> "AlignmentPipeline":
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            logger.warning("AlignmentPipeline: empty input.")
            return df

        df = df.copy()

        # ── Ensure UTC DatetimeIndex ──────────────────────────────────
        if not isinstance(df.index, pd.DatetimeIndex):
            if "time" in df.columns:
                df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
                df = df.dropna(subset=["time"]).set_index("time")
            else:
                raise ValueError("AlignmentPipeline requires a DatetimeIndex or 'time' column.")

        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        else:
            df.index = df.index.tz_convert("UTC")

        df = df.sort_index()

        # ── Identify numeric vs categorical columns ───────────────────
        num_cols  = df.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols  = [c for c in df.columns if c not in num_cols]

        # ── Resample numeric columns: mean ────────────────────────────
        df_num = df[num_cols].resample(self.freq).mean() if num_cols else pd.DataFrame(index=pd.date_range(df.index[0], df.index[-1], freq=self.freq, tz="UTC"))

        # ── Resample categorical / flag columns: forward-fill ─────────
        if cat_cols:
            df_cat = df[cat_cols].resample(self.freq).ffill()
            df_resampled = df_num.join(df_cat, how="outer")
        else:
            df_resampled = df_num

        # ── Fill gaps ─────────────────────────────────────────────────
        for col in num_cols:
            if col in df_resampled.columns:
                df_resampled[col] = df_resampled[col].interpolate(
                    method="time", limit=self.fill_limit
                ).ffill(limit=self.fill_limit).bfill(limit=self.fill_limit)

        logger.info(
            "AlignmentPipeline: %d → %d rows at %s cadence.",
            len(df), len(df_resampled), self.freq,
        )
        return df_resampled


def merge_goes_and_events(
    goes_df: pd.DataFrame,
    events_df: pd.DataFrame,
    window_minutes: int = 60,
) -> pd.DataFrame:
    """Merge GOES XRS flux with NOAA event labels using vectorized approach.

    For each minute in ``goes_df``, assigns a positive label (1) if an M or X
    class flare *starts* within the next ``window_minutes`` minutes.

    Parameters
    ----------
    goes_df:        Cleaned GOES flux DataFrame (DatetimeIndex, UTC).
    events_df:      Parsed NOAA events DataFrame with 'start_time', 'flare_class'.
    window_minutes: Label look-forward window in minutes.

    Returns
    -------
    goes_df with extra columns:
        label_class  – GOES flare class string  (e.g. 'X1.5', '' = no event)
        label_binary – 1 if M or X class within window, else 0
    """
    merged = goes_df.copy()
    merged["label_class"]  = ""
    merged["label_binary"] = 0

    if events_df is None or events_df.empty:
        logger.warning("merge_goes_and_events: events_df is empty; labels will be zero.")
        return merged

    # ── Ensure UTC-aware timestamps on events ─────────────────────────
    events_df = events_df.copy()
    for tcol in ["start_time", "peak_time", "end_time"]:
        if tcol in events_df.columns:
            col = pd.to_datetime(events_df[tcol], errors="coerce")
            if col.dt.tz is None:
                col = col.dt.tz_localize("UTC")
            else:
                col = col.dt.tz_convert("UTC")
            events_df[tcol] = col

    # Keep only M/X class flares for binary labelling
    flare_events = events_df[
        events_df["flare_class"].str.startswith(("M", "X"), na=False)
    ].copy()

    if flare_events.empty:
        logger.warning("No M/X class events in events_df — all labels will be 0.")
        return merged

    # Ensure GOES index is UTC-aware
    if merged.index.tz is None:
        merged.index = merged.index.tz_localize("UTC")

    window = pd.Timedelta(minutes=window_minutes)
    goes_times = merged.index

    # Vectorized: for each event mark [start, start - window] range as 1
    # (look-BACK: a label is set at times where the flare will happen in the future)
    for _, event_row in flare_events.iterrows():
        evt_start = event_row["start_time"]
        cls       = str(event_row.get("flare_class", ""))
        # All GOES timestamps in [evt_start - window, evt_start]
        mask = (goes_times >= evt_start - window) & (goes_times <= evt_start)
        merged.loc[mask, "label_binary"] = 1
        merged.loc[mask, "label_class"]  = cls

    positive = int(merged["label_binary"].sum())
    logger.info(
        "merge_goes_and_events: %d total rows, %d M/X positive labels (%.1f%%).",
        len(merged), positive, 100 * positive / max(len(merged), 1),
    )
    return merged

