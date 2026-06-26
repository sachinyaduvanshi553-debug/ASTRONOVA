import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from services.ingestion.downloaders.base_downloader import BaseDownloader

class AdityaDownloader(BaseDownloader):
    def generate_fallback_data(self, dest_path: str):
        """Generate simulated Aditya-L1 CSV observations."""
        start_time = datetime(2026, 6, 21)
        times = [start_time + timedelta(minutes=i) for i in range(1440)]
        
        np.random.seed(101)
        soft_xray = 1e-7 + np.abs(np.random.normal(0, 1.5e-8, 1440))
        hard_xray = 1e-8 + np.abs(np.random.normal(0, 1.5e-9, 1440))
        
        # Simulate an M-class flare in Aditya data
        for t in range(580, 640):
            if t < 600:
                factor = (t - 580) / 20
                soft_xray[t] += 5e-5 * (factor ** 2)
                hard_xray[t] += 5e-6 * (factor ** 2)
            else:
                factor = np.exp(-(t - 600) / 30)
                soft_xray[t] += 5e-5 * factor
                hard_xray[t] += 5e-6 * factor
                
        df = pd.DataFrame({
            "time": [t.strftime("%Y-%m-%dT%H:%M:%S") for t in times],
            "soft_xray_flux": soft_xray,
            "hard_xray_flux": hard_xray,
            "energy_band_lo": [1.0] * 1440,
            "energy_band_hi": [10.0] * 1440,
            "quality_flag": [0] * 1440
        })
        df.to_csv(dest_path, index=False)
