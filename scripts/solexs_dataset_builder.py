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
from astronova_core.utils.aditya_readers.solexs_reader import SolexsFitsReader
from astronova_core.utils.data_quality import compute_quality_score, validate_temporal_consistency

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("solexs_dataset_builder")

def build_solexs_dataset(input_dir: str, output_file: str):
    logger.info(f"Building clean SoLEXS dataset from {input_dir}...")

    reader = SolexsFitsReader()
    calibrator = AdityaCalibrator()

    # Find all FITS, CSV and JSON files in input_dir
    fits_files = glob.glob(os.path.join(input_dir, "*.fits")) + glob.glob(os.path.join(input_dir, "*.fit"))
    csv_files = glob.glob(os.path.join(input_dir, "*solexs*.csv"))
    json_files = glob.glob(os.path.join(input_dir, "*solexs*.json"))

    all_dfs = []

    # Parse FITS files
    for fp in fits_files:
        try:
            df = reader.read_level1_fits(fp)
            logger.info(f"Successfully read FITS file: {fp} ({len(df)} rows)")
            all_dfs.append(df)
        except Exception as e:
            logger.error(f"Error reading FITS file {fp}: {e}")

    # Parse CSV files
    for fp in csv_files:
        try:
            df = reader.read_csv(fp)
            logger.info(f"Successfully read CSV file: {fp} ({len(df)} rows)")
            all_dfs.append(df)
        except Exception as e:
            logger.error(f"Error reading CSV file {fp}: {e}")

    # Parse JSON files
    for fp in json_files:
        try:
            df = reader.read_json(fp)
            logger.info(f"Successfully read JSON file: {fp} ({len(df)} rows)")
            all_dfs.append(df)
        except Exception as e:
            logger.error(f"Error reading JSON file {fp}: {e}")

    if not all_dfs:
        logger.error(f"No SoLEXS files found in {input_dir}")
        return

    # Concat and sort by time
    combined_df = pd.concat(all_dfs, ignore_index=True)
    combined_df = combined_df.drop_duplicates(subset=['time']).sort_values(by='time').reset_index(drop=True)

    # Validate temporal consistency
    temp_validity = validate_temporal_consistency(combined_df)
    logger.info(f"Temporal validity report: {temp_validity}")

    # Calibrate data
    logger.info("Applying temperature and gain-drift calibration to SoLEXS flux...")
    calibrated_df = calibrator.calibrate_solexs(combined_df)

    # Compute data quality scores
    logger.info("Computing observation quality scores...")
    calibrated_df['quality_score'] = calibrated_df.apply(compute_quality_score, axis=1)

    # Save clean dataset
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    calibrated_df.to_parquet(output_file, index=False)
    logger.info(f"Saved clean SoLEXS dataset to {output_file} ({len(calibrated_df)} records)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process raw SoLEXS telemetry into clean Parquet dataset.")
    parser.add_argument("--input-dir", type=str, default="data/sample", help="Directory containing raw SoLEXS files.")
    parser.add_argument("--output-file", type=str, default="data/sample/clean_solexs.parquet", help="Path to save clean Parquet file.")
    args = parser.parse_args()

    build_solexs_dataset(args.input_dir, args.output_file)
