"""Dataset builder for ASTRONOVA — Phase 1C training data.

Orchestrates the full pipeline:
  GOES NC file → Clean → Align → Smooth → Normalize → PhysicsFeatures → TimeDomainFeatures
  → merge NOAA labels → validate → save to datasets/processed/

Usage:
    python -m scripts.build_dataset
    or: from scripts.build_dataset import build_training_dataset
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ── PYTHONPATH guard ──────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from services.processing.pipelines import (
    DataCleaningPipeline,
    AlignmentPipeline,
    NormalizationPipeline,
    SmoothingPipeline,
    ValidationPipeline,
    read_goes_nc,
    parse_noaa_events,
    merge_goes_and_events,
)
from services.features.engineering.physics_features import PhysicsFeatures
from services.features.engineering.time_domain import TimeDomainFeatures

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger("astronova.build_dataset")

RAW_DIR       = PROJECT_ROOT / "datasets" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "datasets" / "processed"
FEATURES_DIR  = PROJECT_ROOT / "datasets" / "features"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
FEATURES_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Fallback synthetic dataset (physics-compliant) when real data is absent
# ---------------------------------------------------------------------------

def _generate_synthetic_goes(n_minutes: int = 1440) -> pd.DataFrame:
    """Generate a physics-compliant synthetic GOES XRS 1-day dataset.

    Uses realistic A/B/C/M/X class background + injected flare pulses.
    """
    rng = np.random.default_rng(seed=42)
    t = pd.date_range("2026-06-21 00:00:00", periods=n_minutes, freq="1min", tz="UTC")

    # Background: log-normal around B2 level
    background = 2e-7 * np.ones(n_minutes)
    background *= np.exp(rng.normal(0, 0.15, n_minutes))   # ±15% noise

    # Inject 4 synthetic flares
    for start_min, peak_mult, duration in [
        (120, 500, 20),   # C5 flare
        (360, 4000, 35),  # M4 flare
        (600, 15000, 60), # X1.5 flare
        (900, 1200, 25),  # M1 flare
    ]:
        if start_min + duration >= n_minutes:
            continue
        rise  = duration // 3
        decay = duration - rise
        envelope = np.concatenate([
            np.linspace(1, peak_mult, rise),
            np.linspace(peak_mult, 1, decay),
        ])
        end_min = start_min + len(envelope)
        background[start_min:end_min] *= envelope

    soft_flux = np.clip(background, 1e-9, 1e-3)
    # Hard flux ≈ 1/5 of soft (XRSA/XRSB ratio)
    hard_flux = soft_flux * rng.uniform(0.15, 0.25, n_minutes)

    df = pd.DataFrame({
        "soft_xray_flux": soft_flux,
        "hard_xray_flux": hard_flux,
        "quality_flag":   np.zeros(n_minutes, dtype=int),
    }, index=t)
    logger.info("Generated %d-row synthetic GOES dataset.", n_minutes)
    return df


def _generate_synthetic_events() -> pd.DataFrame:
    """Synthetic NOAA event list matching the synthetic GOES data."""
    events = [
        {"event_id": 1, "start_time": pd.Timestamp("2026-06-21 02:00", tz="UTC"),
         "peak_time": pd.Timestamp("2026-06-21 02:07", tz="UTC"),
         "end_time":  pd.Timestamp("2026-06-21 02:20", tz="UTC"),
         "flare_class": "C5.0", "location": "S15W20", "region": 3420},
        {"event_id": 2, "start_time": pd.Timestamp("2026-06-21 06:00", tz="UTC"),
         "peak_time": pd.Timestamp("2026-06-21 06:15", tz="UTC"),
         "end_time":  pd.Timestamp("2026-06-21 07:00", tz="UTC"),
         "flare_class": "M4.2", "location": "N10E45", "region": 3421},
        {"event_id": 3, "start_time": pd.Timestamp("2026-06-21 10:00", tz="UTC"),
         "peak_time": pd.Timestamp("2026-06-21 10:20", tz="UTC"),
         "end_time":  pd.Timestamp("2026-06-21 11:00", tz="UTC"),
         "flare_class": "X1.5", "location": "S20W10", "region": 3422},
        {"event_id": 4, "start_time": pd.Timestamp("2026-06-21 15:00", tz="UTC"),
         "peak_time": pd.Timestamp("2026-06-21 15:10", tz="UTC"),
         "end_time":  pd.Timestamp("2026-06-21 15:30", tz="UTC"),
         "flare_class": "M1.0", "location": "N10E45", "region": 3421},
    ]
    return pd.DataFrame(events)


# ---------------------------------------------------------------------------
# Main pipeline runner
# ---------------------------------------------------------------------------

def build_training_dataset(
    goes_nc_path: Path | None = None,
    noaa_events_path: Path | None = None,
    save: bool = True,
) -> pd.DataFrame:
    """Execute the full GOES + NOAA → features pipeline.

    Returns the final feature-enriched DataFrame.
    """
    logger.info("=" * 60)
    logger.info("ASTRONOVA Dataset Builder — Phase 1B")
    logger.info("=" * 60)

    # ── Step 1: Load raw data ─────────────────────────────────────────
    goes_nc_path = goes_nc_path or next(
        (p for p in (RAW_DIR / "goes_xray").glob("*.nc")), None
    )
    noaa_path = noaa_events_path or next(
        (p for p in (RAW_DIR / "noaa_events").glob("*.txt")), None
    )

    if goes_nc_path and goes_nc_path.exists():
        logger.info("Loading GOES NC: %s", goes_nc_path)
        raw_goes = read_goes_nc(str(goes_nc_path))
    else:
        logger.warning("GOES NC not found — using synthetic dataset.")
        raw_goes = _generate_synthetic_goes()

    if noaa_path and noaa_path.exists():
        logger.info("Loading NOAA events: %s", noaa_path)
        events_df = parse_noaa_events(str(noaa_path))
    else:
        logger.warning("NOAA events file not found — using synthetic events.")
        events_df = _generate_synthetic_events()

    logger.info("Raw GOES shape: %s", raw_goes.shape)

    # ── Step 2: Clean ─────────────────────────────────────────────────
    cleaner = DataCleaningPipeline(spike_zscore_threshold=4.0)
    df = cleaner.fit_transform(raw_goes)
    logger.info("After cleaning: %s", df.shape)

    # ── Step 3: Align to 1-minute UTC grid ───────────────────────────
    aligner = AlignmentPipeline(freq="1min")
    df = aligner.fit_transform(df)
    logger.info("After alignment: %s", df.shape)

    # ── Step 4: Smooth ────────────────────────────────────────────────
    smoother = SmoothingPipeline(mode="auto")
    df = smoother.fit_transform(df)
    logger.info("After smoothing: %s", df.shape)

    # ── Step 5: Normalize ─────────────────────────────────────────────
    normalizer = NormalizationPipeline(use_log=True)
    df = normalizer.fit_transform(df)
    normalizer.save(PROCESSED_DIR / "normalizer.pkl")
    logger.info("After normalization: %s", df.shape)

    # ── Step 6: Physics features ──────────────────────────────────────
    physics = PhysicsFeatures(window=10)
    df = physics.compute(df)
    logger.info("After physics features: %s", df.shape)

    # ── Step 7: Time-domain features ──────────────────────────────────
    time_feats = TimeDomainFeatures()
    df = time_feats.compute(df)
    logger.info("After time-domain features: %s", df.shape)

    # ── Step 8: Merge labels ──────────────────────────────────────────
    df = merge_goes_and_events(df, events_df, window_minutes=60)
    logger.info("After label merge: %s", df.shape)

    # ── Step 9: Validate ──────────────────────────────────────────────
    validator = ValidationPipeline()
    report = validator.validate(df)
    logger.info(report.summary())
    if not report.passed:
        logger.error("Validation FAILED. Review errors above before training.")

    # ── Step 10: Save ─────────────────────────────────────────────────
    if save:
        out_path = PROCESSED_DIR / "goes_processed.parquet"
        df.to_parquet(out_path, engine="pyarrow")
        logger.info("Saved processed dataset → %s  (%d rows × %d cols)", out_path, *df.shape)

        # Save feature matrix (drop label columns)
        feature_cols = [c for c in df.columns if c not in ("label_class",)]
        feat_path = FEATURES_DIR / "feature_matrix.parquet"
        df[feature_cols].to_parquet(feat_path, engine="pyarrow")
        logger.info("Saved feature matrix → %s", feat_path)

    return df


if __name__ == "__main__":
    df = build_training_dataset(save=True)
    logger.info("Dataset build complete. Final shape: %s", df.shape)
    logger.info("Columns: %s", list(df.columns))
    logger.info("Label distribution:\n%s", df["label_binary"].value_counts().to_string())
