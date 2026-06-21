import pandas as pd
import numpy as np
import os
import argparse
import sys

# Ensure shared directory is on the path so we can import astronova_core
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'shared'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from astronova_core.utils.aditya_readers.synchronization import AdityaSensorSynchronizer
from astronova_core.utils.physics import compute_xray_ratio
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("data_fusion")

def fuse_datasets(solexs_parquet: str, helios_parquet: str, output_parquet: str):
    logger.info(f"Fusing SoLEXS ({solexs_parquet}) and HEL1OS ({helios_parquet}) datasets...")
    
    if not os.path.exists(solexs_parquet):
        logger.error(f"SoLEXS parquet not found: {solexs_parquet}")
        return
    if not os.path.exists(helios_parquet):
        logger.error(f"HEL1OS parquet not found: {helios_parquet}")
        return
        
    solexs_df = pd.read_parquet(solexs_parquet)
    helios_df = pd.read_parquet(helios_parquet)
    
    # Synchronize datasets using AdityaSensorSynchronizer (1-minute cadence)
    synchronizer = AdityaSensorSynchronizer()
    fused_df = synchronizer.synchronize_sensors(solexs_df, helios_df, target_cadence='1Min')
    
    # Compute derived fields
    logger.info("Computing derived physics-based fields (gradients, ratios, and quality metrics)...")
    
    # 1. soft_xray_flux and hard_xray_flux must be clean
    fused_df['soft_xray_flux'] = fused_df['soft_xray_flux'].fillna(1e-9)
    fused_df['hard_xray_flux'] = fused_df['hard_xray_flux'].fillna(1e-11)
    
    # 2. X-ray ratio (soft / hard)
    soft = fused_df['soft_xray_flux'].values
    hard = fused_df['hard_xray_flux'].values
    ratio = np.zeros_like(soft)
    mask = (soft > 0) & (hard > 0)
    ratio[mask] = soft[mask] / hard[mask]
    fused_df['xray_ratio'] = ratio
    
    # 3. Flux gradient dF/dt (W/m^2/min)
    fused_df['flux_gradient'] = np.gradient(fused_df['soft_xray_flux'].values)
    
    # 4. Flux acceleration d2F/dt2 (W/m^2/min^2)
    fused_df['flux_acceleration'] = np.gradient(fused_df['flux_gradient'].values)
    
    # 5. Cumulative energy (integral of flux)
    fused_df['cumulative_energy'] = np.cumsum(fused_df['soft_xray_flux'].values)
    
    # 6. Default observation quality if quality_score missing
    if 'quality_score' not in fused_df.columns:
        fused_df['quality_score'] = 1.0
    else:
        fused_df['quality_score'] = fused_df['quality_score'].fillna(1.0)
        
    # Save fused dataset
    os.makedirs(os.path.dirname(output_parquet), exist_ok=True)
    fused_df.to_parquet(output_parquet, index=False)
    
    # Also save as CSV for inspection/backup
    csv_path = output_parquet.replace('.parquet', '.csv')
    fused_df.to_csv(csv_path, index=False)
    
    logger.info(f"Fuesd dataset created successfully at {output_parquet} ({len(fused_df)} records)")
    logger.info(f"CSV mirror saved to {csv_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fuse calibrated SoLEXS and HEL1OS Parquet files.")
    parser.add_argument("--solexs-file", type=str, default="data/sample/clean_solexs.parquet", help="Path to clean SoLEXS Parquet.")
    parser.add_argument("--helios-file", type=str, default="data/sample/clean_helios.parquet", help="Path to clean HEL1OS Parquet.")
    parser.add_argument("--output-file", type=str, default="data/sample/aditya_fusion_dataset.parquet", help="Path to save fused Parquet file.")
    args = parser.parse_args()
    
    fuse_datasets(args.solexs_file, args.helios_file, args.output_file)
