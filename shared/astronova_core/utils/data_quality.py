import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, List, Union
from datetime import datetime
import logging

logger = logging.getLogger("astronova.data_quality")

def compute_quality_score(row: Union[pd.Series, Dict[str, Any]]) -> float:
    """
    Computes a data quality score between 0.0 and 1.0 for a single observation point.
    Decrements are applied based on flags, abnormal temperatures, or out-of-bounds flux values.
    """
    score = 1.0
    
    # 1. Quality flag check
    quality_flag = row.get("quality_flag", 0)
    if quality_flag != 0:
        # If flag indicates corrupted data (e.g. flag > 1), drop score significantly
        if quality_flag > 1:
            score -= 0.5
        else:
            score -= 0.2
            
    # 2. Detector temperature check
    # Optimal temp for SoLEXS/HEL1OS is typically around 20-25C. Severe deviations affect calibration.
    temp = row.get("detector_temp", 25.0)
    if not pd.isna(temp):
        if temp < -10.0 or temp > 50.0:
            score -= 0.4
        elif temp < 10.0 or temp > 35.0:
            score -= 0.15
            
    # 3. Flux checking
    soft_flux = row.get("soft_xray_flux", 0.0)
    hard_flux = row.get("hard_xray_flux", 0.0)
    
    # Flux cannot be negative physically
    if soft_flux < 0:
        score -= 0.3
    if hard_flux < 0:
        score -= 0.3
        
    # Standard GOES background check (quiescent flux shouldn't drop below 1e-9)
    if 0.0 < soft_flux < 1e-10:
        score -= 0.1
        
    return float(max(0.0, min(1.0, score)))

def analyze_gaps(df: pd.DataFrame, time_col: str = "time", expected_cadence_minutes: float = 1.0) -> Dict[str, Any]:
    """
    Analyzes gaps in the timeseries data.
    Returns gap count, max gap duration, and percentage of missing records.
    """
    if df.empty or len(df) < 2:
        return {
            "total_gaps": 0,
            "max_gap_minutes": 0.0,
            "missing_percentage": 0.0,
            "gap_intervals": []
        }
        
    times = pd.to_datetime(df[time_col]).sort_values()
    diffs = times.diff().dropna()
    
    expected_delta = pd.Timedelta(minutes=expected_cadence_minutes)
    
    # Gaps are where the difference is greater than 1.5 * expected cadence
    gaps = diffs[diffs > 1.5 * expected_delta]
    
    total_gaps = len(gaps)
    max_gap = float(gaps.max().total_seconds() / 60.0) if total_gaps > 0 else 0.0
    
    # Estimate missing records
    total_span_seconds = (times.max() - times.min()).total_seconds()
    expected_records = (total_span_seconds / (expected_cadence_minutes * 60.0)) + 1
    actual_records = len(df)
    
    missing_percentage = max(0.0, (1 - (actual_records / expected_records)) * 100) if expected_records > 0 else 0.0
    
    return {
        "total_gaps": total_gaps,
        "max_gap_minutes": max_gap,
        "missing_percentage": float(missing_percentage)
    }

def detect_sensor_drift(df: pd.DataFrame, column: str, window: int = 120, threshold_std_factor: float = 3.0) -> bool:
    """
    Detects potential sensor drift or calibration degradation by comparing short-term rolling mean 
    against a longer-term baseline mean.
    """
    if df.empty or len(df) < window * 2 or column not in df.columns:
        return False
        
    series = df[column].dropna()
    if len(series) < window * 2:
        return False
        
    # Compute rolling mean and std
    rolling_mean = series.rolling(window=window).mean()
    overall_mean = series.mean()
    overall_std = series.std()
    
    if pd.isna(overall_std) or overall_std == 0:
        return False
        
    # Check if the recent rolling mean deviates from overall mean by threshold
    recent_rolling_mean = rolling_mean.iloc[-1]
    deviation = abs(recent_rolling_mean - overall_mean)
    
    if deviation > threshold_std_factor * overall_std:
        logger.warning(f"Sensor drift detected on column '{column}'. Deviation: {deviation:.2e}, limit: {threshold_std_factor * overall_std:.2e}")
        return True
        
    return False

def validate_temporal_consistency(df: pd.DataFrame, time_col: str = "time") -> Dict[str, Any]:
    """
    Validates temporal consistency:
    - Monotonically increasing timestamps
    - No duplicate timestamps
    - No future timestamps
    """
    if df.empty:
        return {"is_valid": True, "issues": ["Empty DataFrame"]}
        
    times = pd.to_datetime(df[time_col])
    
    is_monotonic = times.is_monotonic_increasing
    has_duplicates = times.duplicated().any()
    
    now = datetime.now()
    has_future_times = (times > now).any()
    
    issues = []
    if not is_monotonic:
        issues.append("Timestamps are not monotonically increasing")
    if has_duplicates:
        issues.append("Duplicate timestamps detected")
    if has_future_times:
        issues.append("Timestamps in the future detected")
        
    return {
        "is_valid": len(issues) == 0,
        "is_monotonic": is_monotonic,
        "has_duplicates": has_duplicates,
        "has_future_times": has_future_times,
        "issues": issues
    }

def detect_outliers_modified_zscore(df: pd.DataFrame, column: str, threshold: float = 3.5) -> pd.Series:
    """
    Identifies outliers using the robust Modified Z-Score.
    Modified Z-score = 0.6745 * (x - median) / MAD
    Returns a boolean Series (True for outlier).
    """
    if df.empty or column not in df.columns:
        return pd.Series(dtype=bool)
        
    x = df[column]
    median = x.median()
    mad = np.median(np.abs(x - median))
    
    if mad == 0:
        # Fall back to standard standard deviation if MAD is 0
        std = x.std()
        if pd.isna(std) or std == 0:
            return pd.Series(False, index=df.index)
        z_scores = (x - x.mean()) / std
        return np.abs(z_scores) > threshold
        
    modified_z_scores = 0.6745 * (x - median) / mad
    return np.abs(modified_z_scores) > threshold
