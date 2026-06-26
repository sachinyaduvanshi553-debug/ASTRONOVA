import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from services.ingestion.downloaders.base_downloader import BaseDownloader

class GOESDownloader(BaseDownloader):
    def generate_fallback_data(self, dest_path: str):
        """Generate physics-compliant simulated GOES X-ray flux daily data."""
        # Generate 1-minute resolution data for 1 day
        start_time = datetime(2026, 6, 21)
        times = [start_time + timedelta(minutes=i) for i in range(1440)]
        
        # Physics model: solar X-ray background + flares
        # Background: 1.0e-7 W/m^2 (C-class background)
        np.random.seed(42)
        base_soft = 1e-7 + np.abs(np.random.normal(0, 2e-8, 1440))
        base_hard = 1e-8 + np.abs(np.random.normal(0, 2e-9, 1440))
        
        soft_flux = base_soft.copy()
        hard_flux = base_hard.copy()
        
        # Add 3 solar flares of various sizes (C, M, X class)
        # Flare 1: C8.5 flare at minute 200
        # Flare 2: M4.2 flare at minute 600
        # Flare 3: X1.5 flare at minute 1000
        flares = [
            {"start": 200, "peak": 220, "amp_soft": 8.5e-6, "amp_hard": 8.5e-7, "decay": 40},
            {"start": 600, "peak": 615, "amp_soft": 4.2e-5, "amp_hard": 8.4e-6, "decay": 60},
            {"start": 1000, "peak": 1010, "amp_soft": 1.5e-4, "amp_hard": 3.0e-5, "decay": 120}
        ]
        
        for flare in flares:
            st = flare["start"]
            pk = flare["peak"]
            decay = flare["decay"]
            
            # Rise phase (exponential rise)
            for t in range(st, pk):
                factor = (t - st) / (pk - st)
                soft_flux[t] += flare["amp_soft"] * (factor ** 2)
                hard_flux[t] += flare["amp_hard"] * (factor ** 2)
                
            # Decay phase (exponential decay)
            for t in range(pk, 1440):
                factor = np.exp(-(t - pk) / decay)
                soft_flux[t] += flare["amp_soft"] * factor
                hard_flux[t] += flare["amp_hard"] * factor
        
        # Save as CSV
        df = pd.DataFrame({
            "time": [t.strftime("%Y-%m-%dT%H:%M:%S") for t in times],
            "soft_xray_flux": soft_flux,
            "hard_xray_flux": hard_flux
        })
        df.to_csv(dest_path, index=False)
        logger = self.config.get("logger")
        if logger:
            logger.info(f"Generated realistic GOES fallback data file at {dest_path}")
