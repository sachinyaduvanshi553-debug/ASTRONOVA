"""Validation pipeline for ASTRONOVA solar flare datasets.

Checks:
- Schema completeness (required columns present)
- Temporal continuity (gap detection)
- Physical range bounds (GOES class thresholds)
- Class balance report
- Data quality summary report
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger("astronova.processing.validation")

_REQUIRED_COLS = ["soft_xray_flux", "hard_xray_flux"]
_FLUX_FLOOR    = 1e-9   # W/m²
_FLUX_CEIL     = 1e-3   # W/m²
_MAX_GAP_MIN   = 5      # Alert if gap > 5 minutes in 1-min cadence data


@dataclass
class ValidationReport:
    """Container for dataset validation results."""
    n_rows:            int = 0
    n_cols:            int = 0
    missing_columns:   List[str] = field(default_factory=list)
    nan_counts:        Dict[str, int] = field(default_factory=dict)
    out_of_range:      Dict[str, int] = field(default_factory=dict)
    time_gaps:         List[str] = field(default_factory=list)
    n_time_gaps:       int = 0
    class_balance:     Dict[str, int] = field(default_factory=dict)
    quality_flag_dist: Dict[str, int] = field(default_factory=dict)
    passed:            bool = True
    warnings:          List[str] = field(default_factory=list)
    errors:            List[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"Validation {'PASSED' if self.passed else 'FAILED'}",
            f"  Rows: {self.n_rows:,}  |  Cols: {self.n_cols}",
            f"  Missing columns: {self.missing_columns or 'none'}",
            f"  NaN counts: {self.nan_counts}",
            f"  Out-of-range counts: {self.out_of_range}",
            f"  Time gaps > {_MAX_GAP_MIN}min: {self.n_time_gaps}",
            f"  Class balance: {self.class_balance}",
            f"  Quality flags: {self.quality_flag_dist}",
        ]
        if self.warnings:
            lines.append(f"  WARNINGS ({len(self.warnings)}):")
            lines += [f"    - {w}" for w in self.warnings]
        if self.errors:
            lines.append(f"  ERRORS ({len(self.errors)}):")
            lines += [f"    - {e}" for e in self.errors]
        return "\n".join(lines)


class ValidationPipeline:
    """Non-transforming validator that inspects a cleaned + aligned DataFrame.

    Usage:
        report = ValidationPipeline().validate(df)
        print(report.summary())
    """

    def __init__(
        self,
        required_columns: Optional[List[str]] = None,
        flux_columns: Optional[List[str]] = None,
        max_gap_minutes: int = _MAX_GAP_MIN,
    ) -> None:
        self.required_columns = required_columns or _REQUIRED_COLS
        self.flux_columns     = flux_columns or ["soft_xray_flux", "hard_xray_flux"]
        self.max_gap_minutes  = max_gap_minutes

    # ------------------------------------------------------------------
    def validate(self, df: pd.DataFrame) -> ValidationReport:
        report = ValidationReport()

        if df is None or df.empty:
            report.passed = False
            report.errors.append("DataFrame is None or empty.")
            return report

        report.n_rows = len(df)
        report.n_cols = len(df.columns)

        # ── 1. Schema check ───────────────────────────────────────────
        report.missing_columns = [c for c in self.required_columns if c not in df.columns]
        if report.missing_columns:
            report.errors.append(f"Missing required columns: {report.missing_columns}")
            report.passed = False

        # ── 2. NaN counts ─────────────────────────────────────────────
        for col in df.columns:
            n_nan = int(df[col].isna().sum())
            if n_nan > 0:
                report.nan_counts[col] = n_nan
                pct = 100 * n_nan / len(df)
                msg = f"Column '{col}' has {n_nan} NaN values ({pct:.1f}%)."
                if pct > 10.0:
                    report.errors.append(msg)
                    report.passed = False
                else:
                    report.warnings.append(msg)

        # ── 3. Physical range check ───────────────────────────────────
        for col in self.flux_columns:
            if col not in df.columns:
                continue
            n_below = int((df[col] < _FLUX_FLOOR).sum())
            n_above = int((df[col] > _FLUX_CEIL).sum())
            if n_below + n_above > 0:
                report.out_of_range[col] = n_below + n_above
                report.warnings.append(
                    f"Column '{col}': {n_below} below floor, {n_above} above ceiling."
                )

        # ── 4. Temporal continuity ────────────────────────────────────
        if isinstance(df.index, pd.DatetimeIndex) and len(df) > 1:
            deltas = df.index.to_series().diff().dropna()
            gap_thresh = pd.Timedelta(minutes=self.max_gap_minutes)
            large_gaps = deltas[deltas > gap_thresh]
            report.n_time_gaps = len(large_gaps)
            for ts, gap in large_gaps.items():
                report.time_gaps.append(f"{ts}: gap = {gap}")
            if report.n_time_gaps > 0:
                report.warnings.append(
                    f"{report.n_time_gaps} temporal gaps > {self.max_gap_minutes}min detected."
                )

        # ── 5. Class balance ──────────────────────────────────────────
        if "label_binary" in df.columns:
            vc = df["label_binary"].value_counts().to_dict()
            report.class_balance = {str(k): int(v) for k, v in vc.items()}
            pos = report.class_balance.get("1", 0)
            total = sum(report.class_balance.values())
            if total > 0 and pos / total < 0.01:
                report.warnings.append(
                    f"Severe class imbalance: only {pos}/{total} positive M/X labels ({100*pos/total:.2f}%)."
                )

        if "flare_class" in df.columns:
            vc = df["flare_class"].value_counts().to_dict()
            report.class_balance.update({f"class_{k}": int(v) for k, v in vc.items()})

        # ── 6. Quality flag distribution ──────────────────────────────
        if "quality_flag" in df.columns:
            vc = df["quality_flag"].value_counts().to_dict()
            flag_labels = {0: "clean", 1: "imputed", 2: "spike_removed"}
            report.quality_flag_dist = {
                flag_labels.get(int(k), str(k)): int(v) for k, v in vc.items()
            }

        logger.info("ValidationPipeline:\n%s", report.summary())
        return report
