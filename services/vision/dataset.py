import os
import glob
from typing import Optional, Any, Dict
import numpy as np
import torch
from torch.utils.data import Dataset

from .data.alignment import MultiSourceAligner
from .data.sequence_builder import SequenceBuilder
from .data.augmentations import SolarAugmentations
from .data.loader import InnerSolarDataset


class SolarImageDataset(Dataset):
    """
    Authentic dataset that loads real SDO images, parses their timestamps,
    and temporally aligns them with real GOES X-ray telemetry data and physics catalogs.
    
    This is a wrapper around the new modular data pipeline for backward compatibility.
    """
    def __init__(
        self, 
        image_dir: str, 
        goes_csv_path: Optional[str] = None,
        helios_csv_path: Optional[str] = None,
        solexs_csv_path: Optional[str] = None,
        noaa_catalog_path: Optional[str] = None,
        cme_catalog_path: Optional[str] = None,
        preprocessor=None, 
        sequence_length: int = 4, 
        prediction_horizon: int = 60,
        is_training: bool = False
    ):
        self.image_dir = image_dir
        self.goes_csv_path = goes_csv_path
        self.is_training = is_training
        
        # Check if basic files exist
        if not os.path.exists(image_dir):
            raise FileNotFoundError(f"Image directory not found: {image_dir}")

        if goes_csv_path and os.path.exists(goes_csv_path):
            # Align data
            aligner = MultiSourceAligner(
                image_dir=image_dir,
                goes_csv=goes_csv_path,
                helios_csv=helios_csv_path,
                solexs_csv=solexs_csv_path,
                noaa_csv=noaa_catalog_path,
                cme_csv=cme_catalog_path
            )
            df_aligned = aligner.align()
            
            # Build sequences
            builder = SequenceBuilder(seq_len=sequence_length, prediction_horizon_minutes=prediction_horizon)
            sequences = builder.build_sequences(df_aligned)
        else:
            # Standalone image sequences (for testing or image-only inference)
            img_files = sorted(
                glob.glob(os.path.join(image_dir, "*.jpg")) +
                glob.glob(os.path.join(image_dir, "*.png")) +
                glob.glob(os.path.join(image_dir, "*.jpeg"))
            )
            sequences = []
            if len(img_files) >= sequence_length + 1:
                for i in range(len(img_files) - sequence_length):
                    seq_imgs = img_files[i:i + sequence_length]
                    tgt_img = img_files[i + sequence_length]
                    sequences.append({
                        "image_paths": seq_imgs,
                        "target_image_path": tgt_img,
                        "telemetry": np.zeros(10, dtype=np.float32),
                        "physics": np.zeros(5, dtype=np.float32),
                        "flare_class": 0,
                        "log_flux": -7.0,
                        "timestamp": "2026-01-01T00:00:00Z"
                    })
            else:
                sequences = []
        
        # Use inner dataset to handle image loading and augmentations
        self.inner_dataset = InnerSolarDataset(sequences, is_training=is_training)

    def __len__(self) -> int:
        return len(self.inner_dataset)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.inner_dataset[idx]
