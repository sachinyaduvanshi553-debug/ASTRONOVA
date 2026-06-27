import logging
import os
from datetime import datetime

import cdflib
import numpy as np
import pandas as pd
from astropy.io import fits

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("astronova.generate_realistic_data")

def generate_flare_profile(
    times: pd.DatetimeIndex,
    start_time: datetime,
    peak_time: datetime,
    decay_tau_minutes: float,
    peak_flux: float,
    bg_flux: float
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generates a physically realistic soft X-ray flux profile and corresponding hard X-ray count rate
    profile based on the Neupert effect (Hard X-rays correlate with the derivative of soft X-rays).
    """
    n = len(times)
    soft_flux = np.full(n, bg_flux)
    hard_counts = np.zeros(n)

    t_start = pd.to_datetime(start_time)
    t_peak = pd.to_datetime(peak_time)

    rise_duration = (t_peak - t_start).total_seconds() / 60.0
    amplitude = peak_flux - bg_flux

    for i, t in enumerate(times):
        if t < t_start:
            continue
        elif t_start <= t <= t_peak:
            # Rise phase: Sinusoidal rise
            frac = (t - t_start).total_seconds() / 60.0 / rise_duration
            soft_flux[i] = bg_flux + amplitude * (np.sin(np.pi / 2 * frac) ** 2)

            # Neupert Effect: Hard X-ray count rate is proportional to dF_soft/dt
            # dF/dt = amplitude * 2 * sin(pi/2 * frac) * cos(pi/2 * frac) * (pi/2 / rise_duration)
            grad = amplitude * np.sin(np.pi * frac) * (np.pi / (2 * rise_duration))
            # Convert gradient to hard X-ray counts (arbitrary scale)
            hard_counts[i] = max(0.0, grad * 1.5e8)
        else:
            # Decay phase: Exponential decay
            t_decay = (t - t_peak).total_seconds() / 60.0
            soft_flux[i] = bg_flux + amplitude * np.exp(-t_decay / decay_tau_minutes)
            # Hard X-ray counts decay very rapidly after peak
            hard_counts[i] = max(0.0, hard_counts[i-1] * np.exp(-t_decay / 2.0) if i > 0 else 0.0)

    return soft_flux, hard_counts

def main():
    logger.info("Generating physically realistic Aditya-L1 datasets...")

    # 1. Setup timestamps (1-minute cadence for 24 hours)
    start_epoch = datetime(2026, 6, 20, 0, 0, 0)
    times = pd.date_range(start=start_epoch, periods=1440, freq="1Min")

    bg_flux = 1.5e-8 # B-class quiescent background

    # Initialize flux arrays
    soft_flux = np.full(1440, bg_flux)
    hard_counts = np.full(1440, 10.0) # Quiescent background count rate

    # Inject multiple flares (Cycle 25 statistics)
    # Flare 1: C-class flare around 04:00 (duration ~40 min)
    f1_soft, f1_hard = generate_flare_profile(
        times,
        datetime(2026, 6, 20, 3, 50),
        datetime(2026, 6, 20, 4, 10),
        15.0,
        4.5e-6, # C4.5
        bg_flux
    )

    # Flare 2: M-class flare around 10:00 (duration ~60 min)
    f2_soft, f2_hard = generate_flare_profile(
        times,
        datetime(2026, 6, 20, 9, 30),
        datetime(2026, 6, 20, 10, 0),
        20.0,
        2.5e-5, # M2.5
        bg_flux
    )

    # Flare 3: Extreme X-class flare around 18:00 (duration ~100 min)
    f3_soft, f3_hard = generate_flare_profile(
        times,
        datetime(2026, 6, 20, 17, 40),
        datetime(2026, 6, 20, 18, 00),
        30.0,
        1.2e-4, # X1.2
        bg_flux
    )

    # Combine signals
    soft_flux = soft_flux + (f1_soft - bg_flux) + (f2_soft - bg_flux) + (f3_soft - bg_flux)
    hard_counts = hard_counts + f1_hard + f2_hard + f3_hard

    # Add detector noise and temperature variations
    np.random.seed(42)
    soft_flux += np.random.normal(0, 1e-9, 1440)
    soft_flux = np.clip(soft_flux, 1e-10, None)

    hard_counts += np.random.normal(0, 1.5, 1440)
    hard_counts = np.clip(hard_counts, 0.0, None)

    # Temperature drift (diurnal cycle based on orbit)
    detector_temp = 22.0 + 3.0 * np.sin(2 * np.pi * np.arange(1440) / 1440) + np.random.normal(0, 0.05, 1440)

    # 2. Write SoLEXS FITS File
    out_dir = "data/sample"
    os.makedirs(out_dir, exist_ok=True)

    solexs_fits_path = os.path.join(out_dir, "solexs_raw_sample.fits")

    # Create Primary HDU
    primary_hdu = fits.PrimaryHDU()
    primary_hdu.header['DATE-OBS'] = start_epoch.isoformat()
    primary_hdu.header['EXPTIME'] = 60.0
    primary_hdu.header['VERSION'] = '1.0.0'
    primary_hdu.header['SC_POS_X'] = 1498000.0 # L1 coordinates approx
    primary_hdu.header['SC_POS_Y'] = 12000.0
    primary_hdu.header['SC_POS_Z'] = -34000.0

    # Times relative to start epoch in seconds
    times_sec = (times - times[0]).total_seconds().values

    # Create Binary Table HDU
    col1 = fits.Column(name='TIME', format='D', array=times_sec)
    col2 = fits.Column(name='FLUX_SOFT', format='E', array=soft_flux)
    col3 = fits.Column(name='DETECTOR_TEMP', format='E', array=detector_temp)
    col4 = fits.Column(name='QUALITY_FLAG', format='I', array=np.zeros(1440, dtype=int))

    tb_hdu = fits.BinTableHDU.from_columns(fits.ColDefs([col1, col2, col3, col4]))

    hdul = fits.HDUList([primary_hdu, tb_hdu])
    hdul.writeto(solexs_fits_path, overwrite=True)
    logger.info(f"Wrote SoLEXS FITS data to {solexs_fits_path}")

    # Write HEL1OS CDF File
    helios_cdf_path = os.path.join(out_dir, "helios_raw_sample.cdf")
    if os.path.exists(helios_cdf_path):
        os.remove(helios_cdf_path)

    cdf = cdflib.cdfwrite.CDF(helios_cdf_path)

    # Convert times to CDF Epoch formats
    cdf_times = []
    for t in times:
        cdf_times.append([t.year, t.month, t.day, t.hour, t.minute, t.second, 0])
    epochs = cdflib.cdfepoch.compute_epoch(cdf_times)

    spec_epoch = {
        'Variable': 'Epoch',
        'Data_Type': 31, # CDF_EPOCH
        'Num_Elements': 1,
        'Rec_Vary': True,
        'Dim_Sizes': []
    }
    spec_counts = {
        'Variable': 'HARD_XRAY_COUNTS',
        'Data_Type': 45, # CDF_DOUBLE
        'Num_Elements': 1,
        'Rec_Vary': True,
        'Dim_Sizes': []
    }

    cdf.write_var(spec_epoch, var_data=epochs)
    cdf.write_var(spec_counts, var_data=hard_counts)
    cdf.close()
    logger.info(f"Wrote HEL1OS CDF data to {helios_cdf_path}")

    # Also write CSV copies for easy preview and verification
    solexs_df = pd.DataFrame({
        "time": times,
        "soft_xray_flux": soft_flux,
        "detector_temp": detector_temp,
        "quality_flag": 0
    })
    solexs_df.to_csv(os.path.join(out_dir, "solexs_raw_sample.csv"), index=False)

    helios_df = pd.DataFrame({
        "time": times,
        "hard_xray_flux": hard_counts * 1e-11,
        "counts_per_sec": hard_counts
    })
    helios_df.to_csv(os.path.join(out_dir, "helios_raw_sample.csv"), index=False)

    logger.info("Generation complete! All synthetic products saved.")

if __name__ == "__main__":
    main()
