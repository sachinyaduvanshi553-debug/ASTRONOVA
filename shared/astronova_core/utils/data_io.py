import pandas as pd
import json
import os
from typing import Dict, Any, List, Union
import logging

logger = logging.getLogger("astronova.data_io")

# Delay imports of readers to avoid circular dependencies
def get_solexs_reader():
    from astronova_core.utils.aditya_readers.solexs_reader import SolexsFitsReader
    return SolexsFitsReader()

def get_helios_reader():
    from astronova_core.utils.aditya_readers.helios_reader import HeliosCdfReader
    return HeliosCdfReader()

def read_fits(filepath: str) -> pd.DataFrame:
    """Reads a Level-1 FITS file using SolexsFitsReader."""
    return get_solexs_reader().read_level1_fits(filepath)

def read_cdf(filepath: str) -> pd.DataFrame:
    """Reads a Level-1 CDF file using HeliosCdfReader."""
    return get_helios_reader().read_level1_cdf(filepath)

def read_csv_solexs(filepath: str) -> pd.DataFrame:
    """Reads a SoLEXS observation from CSV file."""
    # Try using SolexsFitsReader's CSV reader first as it standardizes columns
    try:
        return get_solexs_reader().read_csv(filepath)
    except Exception:
        df = pd.read_csv(filepath)
        df['time'] = pd.to_datetime(df['time'])
        return df

def read_json_data(filepath: str) -> pd.DataFrame:
    """Reads observation data from JSON file."""
    try:
        return get_solexs_reader().read_json(filepath)
    except Exception:
        with open(filepath, 'r') as f:
            data = json.load(f)
        if isinstance(data, list):
            df = pd.DataFrame(data)
        else:
            df = pd.DataFrame([data])
        df['time'] = pd.to_datetime(df['time'])
        return df

def detect_file_format(filepath: str) -> str:
    """
    Auto-detects the format of a data file by looking at its extension or header.
    Returns: 'fits', 'cdf', 'csv', 'json', or 'unknown'
    """
    _, ext = os.path.splitext(filepath.lower())
    if ext in ['.fits', '.fit', '.fts']:
        return 'fits'
    elif ext in ['.cdf']:
        return 'cdf'
    elif ext in ['.csv']:
        return 'csv'
    elif ext in ['.json']:
        return 'json'
    
    # Try reading the first few bytes for signatures if extension is ambiguous
    try:
        with open(filepath, 'rb') as f:
            header_bytes = f.read(4)
            if header_bytes == b'SIMPLE': # FITS signature
                return 'fits'
            elif header_bytes == b'\xcdf\x00\x00' or header_bytes == b'\x00\x00\xff\xff': # CDF signature
                return 'cdf'
    except Exception:
        pass
        
    return 'unknown'

def validate_schema(df: pd.DataFrame, required_columns: List[str]) -> bool:
    """
    Validates that a DataFrame contains the required columns and is not empty.
    """
    if df is None or df.empty:
        logger.warning("DataFrame is empty or None")
        return False
        
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        logger.warning(f"Schema validation failed. Missing columns: {missing_cols}")
        return False
        
    return True
