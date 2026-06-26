import os
import numpy as np
from services.ingestion.downloaders.base_downloader import BaseDownloader

class SDODownloader(BaseDownloader):
    def generate_fallback_data(self, dest_path: str):
        """Generate a lightweight simulated SDO AIA observation file."""
        # Save a mock numpy array of shape (6, 64, 64) representing 6 wavelength channels
        # of a 64x64 solar image.
        np.random.seed(404)
        mock_image = np.random.randint(0, 255, size=(6, 64, 64), dtype=np.uint8)
        
        # Save as a compressed numpy file or binary file
        np.save(dest_path, mock_image)
