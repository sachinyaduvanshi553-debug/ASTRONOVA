import argparse
import glob
import os
import sys

import pandas as pd

# Ensure shared directory is on the path so we can import astronova_core
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'shared'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import logging

from astronova_core.utils.aditya_readers.calibration import AdityaCalibrator
from astronova_core.utils.aditya_readers.helios_reader import HeliosCdfReader
from astronova_core.utils.data_quality import validate_temporal_consistency

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("helios_dataset_builder")

def build_helios_dataset(input_dir: str, output_file: str):
    logger.info(f"Building clean HEL1OS dataset from {input_dir}...")

    reader = HeliosCdfReader()
    calibrator = AdityaCalibrator()

    # Find all CDF and CSV files in input_dir
    cdf_files = glob.glob(os.path.join(input_dir, "*.cdf"))
    csv_files = glob.glob(os.path.join(input_dir, "*helios*.csv"))

    all_dfs = []

    # Parse CDF files
    for fp in cdf_files:
        try:
            df = reader.read_level1_cdf(fp)
            logger.info(f"Successfully read CDF file: {fp} ({len(df)} rows)")
            all_dfs.append(df)
        except Exception as e:
            logger.error(f"Error reading CDF file {fp}: {e}")

    # Parse CSV files
    for fp in csv_files:
        try:
            df = reader.read_csv(fp)
            logger.info(f"Successfully read CSV file: {fp} ({len(df)} rows)")
            all_dfs.append(df)
        except Exception as e:
            logger.error(f"Error reading CSV file {fp}: {e}")

    if not all_dfs:
        logger.error(f"No HEL1OS files found in {input_dir}")
        return

    # Concat and sort by time
    combined_df = pd.concat(all_dfs, ignore_index=True)
    combined_df = combined_df.drop_duplicates(subset=['time']).sort_values(by='time').reset_index(drop=True)

    # Validate temporal consistency
    temp_validity = validate_temporal_consistency(combined_df)
    logger.info(f"Temporal validity report: {temp_validity}")

    # Calibrate data (dead-time correction and counts to flux conversion)
    logger.info("Applying dead-time correction and flux calibration to HEL1OS counts...")
    calibrated_df = calibrator.calibrate_helios(combined_df)

    # Save clean dataset
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    calibrated_df.to_parquet(output_file, index=False)
    logger.info(f"Saved clean HEL1OS dataset to {output_file} ({len(calibrated_df)} records)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process raw HEL1OS telemetry into clean Parquet dataset.")
    parser.add_argument("--input-dir", type=str, default="data/sample", help="Directory containing raw HEL1OS files.")
    parser.add_argument("--output-file", type=str, default="data/sample/clean_helios.parquet", help="Path to save clean Parquet file.")
    args = parser.parse_args()

    build_helios_dataset(args.input_dir, args.output_file)
