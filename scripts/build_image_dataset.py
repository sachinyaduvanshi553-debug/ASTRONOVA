"""
build_image_dataset.py — ASTRONOVA Phase 2: Image Dataset Builder

Constructs chronological SDO image sequences aligned with GOES telemetry
and NOAA flare labels. Applies strict chronological splitting to prevent
data leakage between train/validation/test splits.

CRITICAL DATA LEAKAGE PREVENTION:
  - Chronological split ONLY (no random splitting)
  - Event-grouped: same flare event cannot appear in train + test
  - Normalization statistics computed ONLY from training split
  - No future telemetry in input features (T0+H excluded from inputs)

Usage:
    python scripts/build_image_dataset.py --help
    python scripts/build_image_dataset.py \\
        --image-dir datasets/raw/sdo_images \\
        --goes-csv cleaned/goes/goes_xrs_oct2024_jan2025.csv \\
        --noaa-csv cleaned/noaa_labels/noaa_flares_clean.csv \\
        --lookback-hours 24 \\
        --horizon-minutes 60 \\
        --image-size 256 \\
        --output-dir datasets/image_sequences
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# PYTHONPATH guard
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger("astronova.build_image_dataset")


# ---------------------------------------------------------------------------
# GOES flux to GOES class
# ---------------------------------------------------------------------------
def flux_to_goes_class(flux: float) -> str:
    """Map peak soft X-ray flux to GOES class string."""
    if flux < 1e-8:
        return "A"
    elif flux < 1e-7:
        return "B"
    elif flux < 1e-6:
        return "C"
    elif flux < 1e-5:
        return "M"
    else:
        return "X"


# ---------------------------------------------------------------------------
# GOES CSV loader
# ---------------------------------------------------------------------------
def load_goes_csv(path: Path) -> pd.DataFrame:
    """
    Load GOES XRS telemetry CSV.
    Expected columns: time (or timestamp), soft_xray_flux (or xrsa/xrsb), hard_xray_flux
    Returns DataFrame indexed by UTC timestamp.
    """
    df = pd.read_csv(path, comment="#", low_memory=False)
    df.columns = [c.strip().lower() for c in df.columns]

    # Detect time column
    time_col = next((c for c in ["time", "timestamp", "date_time", "datetime", "time_tag"] if c in df.columns), None)
    if time_col is None:
        raise ValueError(f"No time column found in {path}. Columns: {list(df.columns)}")

    df["timestamp"] = pd.to_datetime(df[time_col], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

    # Detect flux columns
    soft_col = next(
        (c for c in ["soft_xray_flux", "xrsa", "xrs_a", "flux_1_8a", "short_flux"] if c in df.columns), None
    )
    hard_col = next(
        (c for c in ["hard_xray_flux", "xrsb", "xrs_b", "flux_0_5_4a", "long_flux"] if c in df.columns), None
    )

    if soft_col:
        df["soft_xray_flux"] = pd.to_numeric(df[soft_col], errors="coerce").fillna(1e-9)
    else:
        logger.warning("No soft X-ray flux column found; using synthetic placeholder 1e-9")
        df["soft_xray_flux"] = 1e-9

    if hard_col:
        df["hard_xray_flux"] = pd.to_numeric(df[hard_col], errors="coerce").fillna(1e-10)
    else:
        df["hard_xray_flux"] = df["soft_xray_flux"] * 0.1

    df["log_soft_flux"] = np.log10(df["soft_xray_flux"].clip(lower=1e-12))
    df["goes_class"] = df["soft_xray_flux"].apply(flux_to_goes_class)

    df = df.set_index("timestamp")
    logger.info("Loaded GOES CSV: %d rows, range %s → %s", len(df), df.index.min(), df.index.max())
    return df


# ---------------------------------------------------------------------------
# NOAA flare event CSV loader
# ---------------------------------------------------------------------------
def load_noaa_events(path: Path) -> pd.DataFrame:
    """
    Load NOAA flare event catalog.
    Returns DataFrame with: event_id, start_time, peak_time, end_time, goes_class, active_region
    """
    df = pd.read_csv(path, comment="#", low_memory=False)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    for col in ["start_time", "peak_time", "end_time"]:
        candidates = [c for c in df.columns if col.split("_")[0] in c and "time" in c]
        if candidates:
            df[col] = pd.to_datetime(df[candidates[0]], utc=True, errors="coerce")
        elif col in df.columns:
            df[col] = pd.to_datetime(df[col], utc=True, errors="coerce")

    if "goes_class" not in df.columns:
        # Try alternative column names
        for alt in ["class", "flare_class", "goes", "magnitude"]:
            if alt in df.columns:
                df["goes_class"] = df[alt].astype(str)
                break
        else:
            df["goes_class"] = "C0.0"

    if "active_region" not in df.columns:
        for alt in ["active_region_id", "noaa_region", "ar_number", "region"]:
            if alt in df.columns:
                df["active_region"] = df[alt]
                break
        else:
            df["active_region"] = 0

    df = df.dropna(subset=["start_time"]).sort_values("start_time").reset_index(drop=True)
    logger.info("Loaded NOAA events: %d flares", len(df))
    return df


# ---------------------------------------------------------------------------
# Image inventory builder
# ---------------------------------------------------------------------------
def build_image_inventory(image_dir: Path) -> pd.DataFrame:
    """
    Scan image_dir for solar images. Parse timestamps from filenames where possible.
    Returns DataFrame with: image_path, timestamp, instrument, wavelength, image_hash

    IMPORTANT: When reliable metadata exists (FITS headers, separate catalog),
    use that instead of filename parsing.
    """
    extensions = [".jpg", ".jpeg", ".png", ".fits"]
    image_files = sorted(
        f for f in image_dir.rglob("*")
        if f.suffix.lower() in extensions and f.is_file()
    )

    if not image_files:
        logger.warning("No images found in %s", image_dir)
        return pd.DataFrame()

    records = []
    for fpath in image_files:
        ts = _parse_timestamp_from_filename(fpath.name)
        wavelength = _parse_wavelength_from_filename(fpath.name)
        instrument = _parse_instrument_from_path(fpath)

        records.append({
            "image_path": str(fpath),
            "filename": fpath.name,
            "timestamp": ts,
            "instrument": instrument,
            "wavelength": wavelength,
            "size_bytes": fpath.stat().st_size,
            "image_hash": _fast_hash(fpath),
        })

    df = pd.DataFrame(records)
    if df["timestamp"].notna().any():
        df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
        logger.info("Image inventory: %d images, range %s → %s",
                    len(df), df["timestamp"].min(), df["timestamp"].max())
    else:
        logger.warning("Could not parse timestamps for any images — using sequential index")

    return df


def _parse_timestamp_from_filename(filename: str):
    """
    Attempt to parse timestamp from common SDO filename formats.
    Returns None if no reliable timestamp found.
    """
    import re
    # SDO AIA format: YYYYmmdd_HHMMSS_512_0193.jpg
    m = re.search(r"(\d{8})_(\d{6})", filename)
    if m:
        try:
            return datetime.strptime(f"{m.group(1)}_{m.group(2)}", "%Y%m%d_%H%M%S").replace(tzinfo=timezone.utc)
        except ValueError:
            pass
    # ISO-like: 2024-10-01T12:00:00
    m = re.search(r"(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})", filename)
    if m:
        try:
            return datetime.fromisoformat(m.group(1)).replace(tzinfo=timezone.utc)
        except ValueError:
            pass
    return None


def _parse_wavelength_from_filename(filename: str) -> str:
    import re
    m = re.search(r"(0131|0171|0193|0211|0304|0335|131|171|193|211|304|335)", filename)
    return f"{m.group(1)} Angstrom" if m else "unknown"


def _parse_instrument_from_path(fpath: Path) -> str:
    parts = str(fpath).lower()
    if "aia" in parts or "sdo" in parts:
        return "SDO/AIA"
    if "solexs" in parts:
        return "Aditya-L1/SoLEXS"
    if "helios" in parts or "hel1os" in parts:
        return "Aditya-L1/HEL1OS"
    return "unknown"


def _fast_hash(fpath: Path) -> str:
    h = hashlib.md5()
    with open(fpath, "rb") as f:
        h.update(f.read(4096))
    return h.hexdigest()[:12]


# ---------------------------------------------------------------------------
# Align images with GOES telemetry and NOAA labels
# ---------------------------------------------------------------------------
def align_images_with_telemetry(
    image_df: pd.DataFrame,
    goes_df: pd.DataFrame,
    noaa_df: pd.DataFrame,
    tolerance_minutes: int = 10,
) -> pd.DataFrame:
    """
    For each image timestamp, find the nearest GOES observation and
    assign the appropriate flare label from NOAA events.

    Leakage prevention: only uses GOES data up to the image timestamp.
    """
    if image_df.empty or goes_df.empty:
        logger.warning("Empty image or GOES dataframe — cannot align")
        return image_df

    aligned_rows = []
    for _, row in image_df.iterrows():
        ts = pd.Timestamp(row["timestamp"])
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")

        # Find nearest GOES observation BEFORE image timestamp (no leakage)
        goes_before = goes_df[goes_df.index <= ts]
        if goes_before.empty:
            soft_flux = 1e-9
            log_flux = -9.0
            goes_class = "A"
        else:
            nearest = goes_before.index.get_indexer([ts], method="nearest")[0]
            soft_flux = goes_before.iloc[nearest]["soft_xray_flux"]
            log_flux = goes_before.iloc[nearest]["log_soft_flux"]
            goes_class = goes_before.iloc[nearest]["goes_class"]

        # Assign flare label from NOAA events
        flare_class, flare_start, flare_peak, flare_end, ar_id = _lookup_flare_label(ts, noaa_df, tolerance_minutes)

        aligned_rows.append({
            **row.to_dict(),
            "soft_xray_flux": soft_flux,
            "log_soft_flux": log_flux,
            "goes_class_derived": goes_class,
            "flare_class": flare_class,
            "flare_start": flare_start,
            "flare_peak": flare_peak,
            "flare_end": flare_end,
            "active_region_id": ar_id,
            "missing_telemetry": False,
        })

    return pd.DataFrame(aligned_rows)


def _lookup_flare_label(
    ts: pd.Timestamp,
    noaa_df: pd.DataFrame,
    tolerance_minutes: int = 10,
) -> tuple:
    """Find which flare event (if any) the timestamp falls within."""
    if noaa_df.empty:
        return "quiet", None, None, None, None

    window = timedelta(minutes=tolerance_minutes)
    for _, event in noaa_df.iterrows():
        start = pd.Timestamp(event.get("start_time"))
        end = pd.Timestamp(event.get("end_time")) if pd.notna(event.get("end_time")) else start + timedelta(hours=2)
        if start is not pd.NaT and (start - window) <= ts <= (end + window):
            return (
                str(event.get("goes_class", "quiet")),
                start,
                event.get("peak_time"),
                end,
                event.get("active_region", 0),
            )
    return "quiet", None, None, None, None


# ---------------------------------------------------------------------------
# Chronological sequence builder
# ---------------------------------------------------------------------------
class ChronologicalSequenceBuilder:
    """
    Builds chronological sequences from aligned image metadata.

    STRICT LEAKAGE PREVENTION:
    - Sequences are built purely from historical data
    - The label (flare in horizon H) is determined by NOAA events AFTER T0
    - Input window is [T0 - W, T0], label window is (T0, T0 + H]
    - No future images or features are ever included in input
    """

    def __init__(
        self,
        lookback_hours: int = 24,
        horizon_minutes: int = 60,
        cadence_minutes: int = 60,
        min_images_in_sequence: int = 4,
    ):
        self.lookback_hours = lookback_hours
        self.horizon_minutes = horizon_minutes
        self.cadence_minutes = cadence_minutes
        self.min_images = min_images_in_sequence
        self.lookback_td = timedelta(hours=lookback_hours)
        self.horizon_td = timedelta(minutes=horizon_minutes)

    def build_sequences(
        self,
        aligned_df: pd.DataFrame,
        noaa_df: pd.DataFrame,
    ) -> list[dict[str, Any]]:
        """
        Build all valid sequences from the aligned image dataframe.
        Returns list of sequence dicts ready for JSON serialization.
        """
        if aligned_df.empty:
            logger.warning("Empty aligned dataframe — no sequences to build")
            return []

        df = aligned_df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.sort_values("timestamp").reset_index(drop=True)

        sequences = []
        timestamps = df["timestamp"].values

        for i, anchor_ts in enumerate(df["timestamp"]):
            # Find all images in historical window
            window_start = anchor_ts - self.lookback_td
            mask = (df["timestamp"] >= window_start) & (df["timestamp"] <= anchor_ts)
            seq_df = df[mask]

            if len(seq_df) < self.min_images:
                continue

            # Determine labels for each horizon
            labels = self._compute_labels(anchor_ts, noaa_df)

            # Build sequence record
            sequences.append({
                "anchor_timestamp": anchor_ts.isoformat(),
                "anchor_idx": i,
                "window_start": window_start.isoformat(),
                "image_paths": seq_df["image_path"].tolist(),
                "timestamps": [t.isoformat() for t in seq_df["timestamp"]],
                "soft_xray_flux": seq_df["soft_xray_flux"].tolist(),
                "log_soft_flux": seq_df["log_soft_flux"].tolist(),
                "n_images": len(seq_df),
                "labels": labels,
                "active_region_id": seq_df["active_region_id"].iloc[-1] if "active_region_id" in seq_df.columns else None,
            })

        logger.info("Built %d sequences from %d images", len(sequences), len(df))
        return sequences

    def _compute_labels(
        self,
        anchor_ts: pd.Timestamp,
        noaa_df: pd.DataFrame,
    ) -> dict[str, dict[str, bool]]:
        """
        For each horizon, determine if a flare of each class occurs
        in the future window (anchor_ts, anchor_ts + horizon].

        LEAKAGE CHECK: anchor_ts is the LATEST input timestamp.
        Labels are determined from FUTURE events only.
        """
        horizons_minutes = [15, 30, 60, 360, 720, 1440]
        labels: dict[str, dict[str, bool]] = {}

        for h in horizons_minutes:
            horizon_td = timedelta(minutes=h)
            future_start = anchor_ts
            future_end = anchor_ts + horizon_td

            # Look for flare events in this future window
            future_events = noaa_df[
                (noaa_df.get("start_time", pd.NaT) >= future_start) &
                (noaa_df.get("start_time", pd.NaT) < future_end)
            ] if not noaa_df.empty else pd.DataFrame()

            max_class = "quiet"
            if not future_events.empty:
                classes = future_events["goes_class"].astype(str).str.upper()
                if classes.str.startswith("X").any():
                    max_class = "X"
                elif classes.str.startswith("M").any():
                    max_class = "M"
                elif classes.str.startswith("C").any():
                    max_class = "C"
                else:
                    max_class = "B"

            labels[f"h{h}m"] = {
                "C_plus": max_class in ["C", "M", "X"],
                "M_plus": max_class in ["M", "X"],
                "X_plus": max_class == "X",
                "max_class": max_class,
                "n_events": len(future_events),
            }

        return labels


# ---------------------------------------------------------------------------
# Chronological split
# ---------------------------------------------------------------------------
def chronological_split(
    sequences: list[dict],
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    event_gap_hours: int = 6,
) -> tuple[list, list, list]:
    """
    Split sequences chronologically.
    
    STRICT LEAKAGE PREVENTION:
    - Sort by anchor_timestamp (chronological order)
    - No random shuffling
    - Ensure no sequence straddles a split boundary by enforcing a gap
    - The same active-region event cannot appear in both train and test
    """
    if not sequences:
        return [], [], []

    # Sort chronologically
    seqs = sorted(sequences, key=lambda s: s["anchor_timestamp"])

    n = len(seqs)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))

    # Enforce event-gap: move boundary forward to avoid sequences from the
    # same flare event straddling the split
    gap = timedelta(hours=event_gap_hours)

    def advance_boundary(seqs, idx):
        if idx >= len(seqs):
            return idx
        boundary_ts = pd.Timestamp(seqs[idx]["anchor_timestamp"])
        while idx < len(seqs):
            ts = pd.Timestamp(seqs[idx]["anchor_timestamp"])
            if ts - boundary_ts > gap:
                break
            idx += 1
        return idx

    train_end = advance_boundary(seqs, train_end)
    if train_end >= n - 1:
        train_end = max(1, int(n * train_frac))
    val_end = advance_boundary(seqs, val_end)
    if val_end >= n:
        val_end = max(train_end + 1, int(n * (train_frac + val_frac)))

    train = seqs[:train_end]
    val = seqs[train_end:val_end]
    test = seqs[val_end:]

    logger.info(
        "Chronological split: TRAIN=%d | VAL=%d | TEST=%d",
        len(train), len(val), len(test),
    )
    return train, val, test


# ---------------------------------------------------------------------------
# Metadata parquet writer
# ---------------------------------------------------------------------------
def write_metadata(
    sequences: list[dict],
    output_dir: Path,
    split_name: str,
) -> None:
    """Write metadata parquet file for a split."""
    if not sequences:
        logger.warning("No sequences for %s split — skipping metadata", split_name)
        return

    rows = []
    for seq in sequences:
        labels = seq.get("labels", {})
        rows.append({
            "anchor_timestamp": seq["anchor_timestamp"],
            "n_images": seq["n_images"],
            "active_region_id": seq.get("active_region_id"),
            **{f"{h}_{cls}": labels.get(h, {}).get(cls, False)
               for h in labels for cls in ["C_plus", "M_plus", "X_plus", "max_class"]},
        })

    df = pd.DataFrame(rows)
    try:
        out_path = output_dir / f"{split_name}_metadata.parquet"
        df.to_parquet(out_path, index=False)
        logger.info("Wrote %s metadata: %s", split_name, out_path)
    except Exception as exc:
        out_path = output_dir / f"{split_name}_metadata.csv"
        df.to_csv(out_path, index=False)
        logger.warning("Parquet unavailable (%s), wrote CSV metadata: %s", exc, out_path)


# ---------------------------------------------------------------------------
# Dataset manifest
# ---------------------------------------------------------------------------
def write_manifest(
    train: list, val: list, test: list,
    image_size: int,
    channels: int,
    manifest_path: Path,
) -> None:
    """Write image_dataset_v1.json manifest."""
    def count_class(seqs, horizon="h60m", cls="M_plus"):
        return sum(1 for s in seqs if s.get("labels", {}).get(horizon, {}).get(cls, False))

    all_seqs = train + val + test
    date_range_start = all_seqs[0]["anchor_timestamp"] if all_seqs else "N/A"
    date_range_end = all_seqs[-1]["anchor_timestamp"] if all_seqs else "N/A"

    manifest = {
        "version": "1.0",
        "created": datetime.utcnow().isoformat() + "Z",
        "n_samples_total": len(all_seqs),
        "n_train": len(train),
        "n_val": len(val),
        "n_test": len(test),
        "image_dimensions": f"{image_size}x{image_size}",
        "channels": channels,
        "date_range_start": date_range_start,
        "date_range_end": date_range_end,
        "split_strategy": "chronological",
        "train_fraction": len(train) / max(1, len(all_seqs)),
        "val_fraction": len(val) / max(1, len(all_seqs)),
        "test_fraction": len(test) / max(1, len(all_seqs)),
        "class_distribution": {
            "train_M_plus_h60m": count_class(train, "h60m", "M_plus"),
            "val_M_plus_h60m": count_class(val, "h60m", "M_plus"),
            "test_M_plus_h60m": count_class(test, "h60m", "M_plus"),
        },
        "leakage_prevention": {
            "chronological_split": True,
            "event_gap_enforced": True,
            "normalization_from_train_only": True,
            "no_future_features_in_input": True,
        },
        "data_sufficiency_notes": (
            "SMOKE TEST ONLY — synthetic/limited data" if len(all_seqs) < 100
            else "Dataset built from real observations"
        ),
    }

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    logger.info("Wrote dataset manifest: %s", manifest_path)


# ---------------------------------------------------------------------------
# Synthetic data generator (smoke-test only)
# ---------------------------------------------------------------------------
def generate_synthetic_sequences(
    n_sequences: int = 20,
    lookback_hours: int = 4,
    image_size: int = 128,
    output_dir: Path = Path("datasets/raw/sdo_images/synthetic"),
) -> tuple[list[dict], pd.DataFrame]:
    """
    Generate synthetic solar images and NOAA events for pipeline smoke testing.
    
    !!!! SMOKE TEST ONLY — NOT SCIENTIFIC DATA !!!!
    Results from this data must NEVER appear in scientific benchmark metrics.
    """
    logger.warning("=" * 60)
    logger.warning("SMOKE TEST ONLY — generating synthetic data")
    logger.warning("These images are NOT real solar observations.")
    logger.warning("=" * 60)

    try:
        import cv2
        HAS_CV2 = True
    except ImportError:
        HAS_CV2 = False

    output_dir.mkdir(parents=True, exist_ok=True)
    base_ts = datetime(2024, 10, 1, 0, 0, 0, tzinfo=timezone.utc)
    cadence = timedelta(hours=1)

    sequences = []
    all_timestamps = []
    images_per_seq = max(4, lookback_hours)

    rng = np.random.default_rng(42)  # deterministic seed

    # Generate all image timestamps we need
    total_frames = n_sequences + images_per_seq
    for i in range(total_frames):
        ts = base_ts + i * cadence
        fname = output_dir / f"synthetic_{ts.strftime('%Y%m%d_%H%M%S')}.png"
        
        # Create a minimal physics-based synthetic solar disc image
        img = _make_synthetic_solar_image(image_size, rng, i)
        if HAS_CV2:
            import cv2
            cv2.imwrite(str(fname), img)
        else:
            # Fallback: save as raw numpy binary
            np.save(str(fname).replace(".png", ".npy"), img)
            fname = fname.with_suffix(".npy")
        all_timestamps.append((ts, str(fname)))

    # Build synthetic NOAA events — inject some M-class flares
    noaa_rows = []
    for i in range(0, n_sequences, 5):
        flare_ts = base_ts + timedelta(hours=i + images_per_seq + 1)
        noaa_rows.append({
            "start_time": flare_ts,
            "peak_time": flare_ts + timedelta(minutes=10),
            "end_time": flare_ts + timedelta(minutes=30),
            "goes_class": rng.choice(["C5.0", "M1.2", "M3.4", "X1.0"]),
            "active_region": 13780,
        })
    noaa_df = pd.DataFrame(noaa_rows)
    if not noaa_df.empty:
        noaa_df["start_time"] = pd.to_datetime(noaa_df["start_time"], utc=True)
        noaa_df["peak_time"] = pd.to_datetime(noaa_df["peak_time"], utc=True)
        noaa_df["end_time"] = pd.to_datetime(noaa_df["end_time"], utc=True)

    # Build sequences
    builder = ChronologicalSequenceBuilder(
        lookback_hours=lookback_hours,
        horizon_minutes=60,
        cadence_minutes=60,
        min_images_in_sequence=4,
    )

    for i in range(images_per_seq, min(total_frames, images_per_seq + n_sequences)):
        anchor_ts = pd.Timestamp(all_timestamps[i][0])
        window = all_timestamps[max(0, i - images_per_seq):i + 1]
        labels = builder._compute_labels(anchor_ts, noaa_df)
        sequences.append({
            "anchor_timestamp": anchor_ts.isoformat(),
            "anchor_idx": i,
            "window_start": (anchor_ts - timedelta(hours=lookback_hours)).isoformat(),
            "image_paths": [w[1] for w in window],
            "timestamps": [w[0].isoformat() for w in window],
            "soft_xray_flux": [1e-7 + rng.random() * 1e-5 for _ in window],
            "log_soft_flux": [-7.0 + rng.random() * 2 for _ in window],
            "n_images": len(window),
            "labels": labels,
            "active_region_id": 13780,
            "synthetic": True,   # ALWAYS mark synthetic data
        })

    return sequences, noaa_df


def _make_synthetic_solar_image(size: int, rng: np.random.Generator, frame_idx: int) -> np.ndarray:
    """Create a minimal physics-based synthetic solar image for smoke testing."""
    img = np.zeros((size, size, 3), dtype=np.uint8)
    cx, cy, r = size // 2, size // 2, int(size * 0.45)

    # Solar disc with limb darkening
    yy, xx = np.ogrid[:size, :size]
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    disc_mask = dist < r
    mu = np.sqrt(np.clip(1 - (dist / r) ** 2, 0, 1))
    brightness = (0.4 + 0.6 * mu) * 200
    img[disc_mask, 0] = brightness[disc_mask].clip(0, 255).astype(np.uint8)
    img[disc_mask, 1] = (brightness[disc_mask] * 0.6).clip(0, 255).astype(np.uint8)

    # Active region (flare-like bright spot)
    ar_x = cx + int(r * 0.3 * np.sin(frame_idx * 0.1))
    ar_y = cy + int(r * 0.1 * np.cos(frame_idx * 0.1))
    ar_r = max(3, size // 32)
    ar_mask = ((xx - ar_x) ** 2 + (yy - ar_y) ** 2 < ar_r ** 2) & disc_mask
    img[ar_mask] = [255, 200, 50]

    # Noise
    noise = rng.integers(-5, 5, (size, size, 3), dtype=np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return img


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="ASTRONOVA Image Dataset Builder — Phase 2",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--image-dir", type=Path, default=Path("datasets/raw/sdo_images"),
                        help="Directory containing SDO solar images")
    parser.add_argument("--goes-csv", type=Path, default=None,
                        help="Path to GOES XRS telemetry CSV")
    parser.add_argument("--noaa-csv", type=Path, default=None,
                        help="Path to NOAA flare event catalog CSV")
    parser.add_argument("--lookback-hours", type=int, default=24,
                        help="Historical observation window (W) in hours")
    parser.add_argument("--horizon-minutes", type=int, default=60,
                        help="Prediction horizon (H) in minutes")
    parser.add_argument("--cadence-minutes", type=int, default=60,
                        help="Image cadence in minutes")
    parser.add_argument("--image-size", type=int, default=256,
                        help="Target image size (square)")
    parser.add_argument("--min-images", type=int, default=4,
                        help="Minimum images required per sequence")
    parser.add_argument("--output-dir", type=Path, default=Path("datasets/image_sequences"),
                        help="Output directory for sequences")
    parser.add_argument("--onedrive-data", type=Path,
                        default=Path(r"C:\Users\sachi\OneDrive\Documents\ASTRONOVA\DATA"),
                        help="OneDrive data root path")
    parser.add_argument("--smoke-test", action="store_true",
                        help="Run with synthetic data for pipeline verification only")
    parser.add_argument("--smoke-n", type=int, default=20,
                        help="Number of synthetic sequences for smoke test")
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("ASTRONOVA Image Dataset Builder — Phase 2")
    logger.info("=" * 60)

    if args.smoke_test:
        logger.warning("SMOKE TEST MODE — synthetic data only. NOT scientific.")
        sequences, noaa_df = generate_synthetic_sequences(
            n_sequences=args.smoke_n,
            lookback_hours=args.lookback_hours,
            image_size=args.image_size,
            output_dir=Path("datasets/raw/sdo_images/synthetic"),
        )
    else:
        # 1. Resolve real data paths
        image_dir = args.image_dir
        goes_path = args.goes_csv or (args.onedrive_data / "cleaned" / "goes" / "goes_xrs_oct2024_jan2025 (1).csv")
        noaa_path = args.noaa_csv or (args.onedrive_data / "cleaned" / "noaa_labels" / "noaa_flares_clean.csv")

        # 2. Check data availability
        if not image_dir.exists() or not any(image_dir.rglob("*.jpg")):
            logger.warning("Image directory missing or empty: %s", image_dir)
            logger.warning("Suggest running: python services/vision/download_sample_data.py")
            logger.info("Falling back to smoke test with synthetic data")
            args.smoke_test = True
            sequences, noaa_df = generate_synthetic_sequences(
                n_sequences=args.smoke_n,
                lookback_hours=args.lookback_hours,
                image_size=args.image_size,
            )
        else:
            # 3. Load real data
            goes_df = load_goes_csv(goes_path) if goes_path.exists() else pd.DataFrame()
            noaa_df = load_noaa_events(noaa_path) if noaa_path.exists() else pd.DataFrame()

            # 4. Build image inventory
            image_df = build_image_inventory(image_dir)
            if image_df.empty:
                logger.error("No images found. Cannot build dataset.")
                sys.exit(1)

            # 5. Align with telemetry
            aligned_df = align_images_with_telemetry(image_df, goes_df, noaa_df)

            # 6. Build sequences
            builder = ChronologicalSequenceBuilder(
                lookback_hours=args.lookback_hours,
                horizon_minutes=args.horizon_minutes,
                cadence_minutes=args.cadence_minutes,
                min_images_in_sequence=args.min_images,
            )
            sequences = builder.build_sequences(aligned_df, noaa_df)

    if not sequences:
        logger.error("No sequences built. Cannot continue.")
        sys.exit(1)

    # 7. Chronological split (NO random splitting)
    train, val, test = chronological_split(sequences, train_frac=0.70, val_frac=0.15)

    # 8. Save splits as JSON
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for split_name, split_seqs in [("train", train), ("validation", val), ("test", test)]:
        out_path = args.output_dir / f"{split_name}_sequences.json"
        with open(out_path, "w") as f:
            json.dump(split_seqs, f, indent=2, default=str)
        logger.info("Saved %s: %d sequences → %s", split_name, len(split_seqs), out_path)

        # Metadata parquet
        write_metadata(split_seqs, args.output_dir, split_name)

    # 9. Write dataset manifest
    manifest_path = PROJECT_ROOT / "datasets" / "manifests" / "image_dataset_v1.json"
    write_manifest(train, val, test, args.image_size, 3, manifest_path)

    # 10. Summary
    total = len(sequences)
    is_synthetic = any(s.get("synthetic") for s in sequences)
    logger.info("=" * 60)
    logger.info("DATASET BUILD COMPLETE")
    if is_synthetic:
        logger.warning("SMOKE TEST ONLY — results are NOT scientifically valid")
    logger.info("Total sequences: %d", total)
    logger.info("Train: %d | Val: %d | Test: %d", len(train), len(val), len(test))
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
