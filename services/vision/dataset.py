import os
import glob
import torch
from torch.utils.data import Dataset
import numpy as np
import cv2
import pandas as pd
from datetime import datetime

class SolarImageDataset(Dataset):
    """
    Authentic dataset that loads real SDO images, parses their timestamps,
    and temporally aligns them with real GOES X-ray telemetry data.
    """
    def __init__(self, image_dir, goes_csv_path, preprocessor=None, sequence_length=1, is_training=False):
        self.image_dir = image_dir
        self.goes_csv_path = goes_csv_path
        self.preprocessor = preprocessor
        self.sequence_length = sequence_length
        self.is_training = is_training
        
        # 1. Load telemetry
        if os.path.exists(goes_csv_path):
            self.df_goes = pd.read_csv(goes_csv_path)
            self.df_goes['time'] = pd.to_datetime(self.df_goes['time'], utc=True)
            self.df_goes = self.df_goes.sort_values('time').reset_index(drop=True)
            self.has_telemetry = True
        else:
            self.has_telemetry = False
            print(f"Warning: GOES CSV not found at {goes_csv_path}. Using zero tensors.")
            
        # 2. Load images and extract timestamps
        # Files are named like YYYYMMDD_HHMMSS_RES_WAVELENGTH.jpg
        self.image_records = []
        if os.path.exists(image_dir):
            paths = sorted(glob.glob(os.path.join(image_dir, "*.jpg")))
            for p in paths:
                filename = os.path.basename(p)
                try:
                    # Parse YYYYMMDD_HHMMSS
                    dt_str = filename[:15]
                    dt = datetime.strptime(dt_str, "%Y%m%d_%H%M%S")
                    dt = pd.to_datetime(dt, utc=True)
                    self.image_records.append({'path': p, 'time': dt})
                except Exception:
                    # Ignore files that don't match pattern
                    continue
        
        # Sort image records by time
        self.image_records = sorted(self.image_records, key=lambda x: x['time'])

    def __len__(self):
        total_required = self.sequence_length + 1
        return max(0, len(self.image_records) - total_required + 1)

    def _get_telemetry_for_time(self, target_time):
        if not self.has_telemetry:
            return torch.zeros(10)
            
        # Find closest telemetry row before or at target_time
        # In a real heavy dataset, merge_asof is done once. For simplicity here:
        mask = self.df_goes['time'] <= target_time
        filtered = self.df_goes[mask]
        if len(filtered) == 0:
            return torch.zeros(10)
            
        latest_row = filtered.iloc[-1]
        
        # Extract features (xrsa, xrsb). Pad to 10 for model compatibility.
        # Log-scale the X-ray fluxes because they are tiny (e.g., 1e-8)
        xrsa = latest_row.get('xrsa_flux', 1e-9)
        xrsb = latest_row.get('xrsb_flux', 1e-9)
        
        # Handle zeros/NaNs before log10
        xrsa = np.log10(max(xrsa, 1e-10))
        xrsb = np.log10(max(xrsb, 1e-10))
        
        # The model expects 10 features, we fill the rest with time-derived features
        hour = target_time.hour / 24.0
        minute = target_time.minute / 60.0
        
        telemetry = torch.tensor([xrsa, xrsb, hour, minute] + [0.0] * 6, dtype=torch.float32)
        return telemetry

    def __getitem__(self, idx):
        # Input sequence
        input_records = self.image_records[idx : idx + self.sequence_length]
        target_record = self.image_records[idx + self.sequence_length]
        
        input_images = []
        for record in input_records:
            img = cv2.imread(record['path'])
            if img is None:
                img = np.zeros((512, 512, 3), dtype=np.uint8)
            else:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            input_images.append(img)
            
        target_img = cv2.imread(target_record['path'])
        if target_img is None:
            target_img = np.zeros((512, 512, 3), dtype=np.uint8)
        else:
            target_img = cv2.cvtColor(target_img, cv2.COLOR_BGR2RGB)
            
        if self.preprocessor:
            input_tensors = self.preprocessor.process_sequence(input_images, is_training=self.is_training)
            target_tensor = self.preprocessor.process_sequence([target_img], is_training=False)[0]
        else:
            # Fallback scaling
            input_tensors = [torch.from_numpy(i).permute(2,0,1).float() / 255.0 for i in input_images]
            target_tensor = torch.from_numpy(target_img).permute(2,0,1).float() / 255.0
            
        input_tensor_stack = torch.stack(input_tensors)
        
        # Get telemetry for the LAST frame in the sequence
        last_frame_time = input_records[-1]['time']
        telemetry = self._get_telemetry_for_time(last_frame_time)
        
        # Dummy physics features (since NOAA flares are sparse, we zero-pad for now to keep it realistic without breaking)
        physics = torch.zeros(5, dtype=torch.float32)
        
        return {
            "image": input_tensor_stack,      # Shape: (T, C, H, W)
            "telemetry": telemetry,           # Shape: (10,)
            "physics": physics,               # Shape: (5,)
            "target": target_tensor           # Shape: (C, H, W)
        }
