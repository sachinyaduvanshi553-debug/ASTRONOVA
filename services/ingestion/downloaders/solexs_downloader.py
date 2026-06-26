import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from services.ingestion.downloaders.base_downloader import BaseDownloader

class SoLEXSDownloader(BaseDownloader):
    def generate_fallback_data(self, dest_path: str):
        """Generate simulated SoLEXS (Soft X-ray spectrometer) CSV observations."""
        start_time = datetime(2026, 6, 21)
        times = [start_time + timedelta(minutes=i) for i in range(1440)]
        
        np.random.seed(202)
        # SoLEXS is focused on soft X-rays (1.0 to 22.0 keV)
        soft_xray = 1e-7 + np.abs(np.random.normal(0, 1.2e-8, 1440))
        hard_xray = 1e-8 + np.abs(np.random.normal(0, 1.2e-9, 1440))
        
        # Simulate a M-class flare in SoLEXS
        for t in range(580, 640):
            if t < 600:
                factor = (t - 580) / 20
                soft_xray[t] += 4.2e-5 * (factor ** 2)
                hard_xray[t] += 4.2e-6 * (factor ** 2)
            else:
                factor = np.exp(-(t - 600) / 45)
                soft_xray[t] += 4.2e-5 * factor
                hard_xray[t] += 4.2e-6 * factor

        df = pd.DataFrame({
            "time": [t.strftime("%Y-%m-%dT%H:%M:%S") for t in times],
            "soft_xray_flux": soft_xray,
            "hard_xray_flux": hard_xray,
            "energy_band_lo": [1.0] * 1440,
            "energy_band_hi": [22.0] * 1440,
            "quality_flag": [0] * 1440
        })
        df.to_csv(dest_path, index=False)
