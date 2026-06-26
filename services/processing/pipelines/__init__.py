"""Processing pipelines package for ASTRONOVA.

Exposes:
    BasePipeline         – Abstract base class
    DataCleaningPipeline – Cleaning + spike removal + IQR
    AlignmentPipeline    – Temporal resampling to 1-min UTC grid
    NormalizationPipeline– Log10 + RobustScaler normalization
    SmoothingPipeline    – SavGol / Gaussian / EWMA smoothing
    ValidationPipeline   – Schema + continuity + class balance checks
    ValidationReport     – Structured report dataclass
    read_goes_nc         – GOES XRS NetCDF reader
    parse_noaa_events    – NOAA SWPC event list parser
    merge_goes_and_events– Label merger
"""
from services.processing.pipelines.base import BasePipeline
from services.processing.pipelines.cleaning import (
    DataCleaningPipeline,
    read_goes_nc,
    parse_noaa_events,
)
from services.processing.pipelines.alignment import (
    AlignmentPipeline,
    merge_goes_and_events,
)
from services.processing.pipelines.normalization import NormalizationPipeline
from services.processing.pipelines.smoothing import SmoothingPipeline
from services.processing.pipelines.validation import ValidationPipeline, ValidationReport

__all__ = [
    "BasePipeline",
    "DataCleaningPipeline",
    "AlignmentPipeline",
    "NormalizationPipeline",
    "SmoothingPipeline",
    "ValidationPipeline",
    "ValidationReport",
    "read_goes_nc",
    "parse_noaa_events",
    "merge_goes_and_events",
]
