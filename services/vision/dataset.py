import os
from typing import Optional, Any, Dict
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
        goes_csv_path: str,
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
        if not os.path.exists(goes_csv_path):
            raise FileNotFoundError(f"GOES CSV not found: {goes_csv_path}")

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
        
        # Use inner dataset to handle image loading and augmentations
        self.inner_dataset = InnerSolarDataset(sequences, is_training=is_training)

    def __len__(self) -> int:
        return len(self.inner_dataset)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.inner_dataset[idx]
