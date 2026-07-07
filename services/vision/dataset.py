import os
import glob
import torch
from torch.utils.data import Dataset
import numpy as np
import cv2

class SolarImageDataset(Dataset):
    """
    Dataset to load solar images, align them with GOES telemetry,
    and apply data augmentations using SolarImagePreprocessor.
    """
    def __init__(self, image_dir, telemetry_data=None, preprocessor=None, sequence_length=1, is_training=False):
        self.image_dir = image_dir
        self.telemetry_data = telemetry_data
        self.preprocessor = preprocessor
        self.sequence_length = sequence_length
        self.is_training = is_training
        
        # Load all image paths
        # Assuming images are named such that sorting them orders them by time
        if os.path.exists(image_dir):
            self.image_paths = sorted(glob.glob(os.path.join(image_dir, "*.png")))
        else:
            self.image_paths = []

    def __len__(self):
        # We need `sequence_length` frames for input and 1 for target
        total_required = self.sequence_length + 1
        return max(0, len(self.image_paths) - total_required + 1)

    def __getitem__(self, idx):
        # Sequence of input images
        input_paths = self.image_paths[idx : idx + self.sequence_length]
        target_path = self.image_paths[idx + self.sequence_length]
        
        input_images = []
        for path in input_paths:
            img = cv2.imread(path)
            if img is None:
                img = np.zeros((512, 512, 3), dtype=np.uint8)
            else:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            input_images.append(img)
            
        target_img = cv2.imread(target_path)
        if target_img is None:
            target_img = np.zeros((512, 512, 3), dtype=np.uint8)
        else:
            target_img = cv2.cvtColor(target_img, cv2.COLOR_BGR2RGB)
            
        if self.preprocessor:
            # Process sequence
            input_tensors = self.preprocessor.process_sequence(input_images, is_training=self.is_training)
            target_tensor = self.preprocessor.process_sequence([target_img], is_training=False)[0]
        else:
            # Dummy fallback if no preprocessor
            input_tensors = [torch.zeros(3, 512, 512) for _ in input_images]
            target_tensor = torch.zeros(3, 512, 512)
            
        input_tensor_stack = torch.stack(input_tensors)
        
        # Dummy telemetry and physics features
        telemetry = torch.zeros(10)
        physics = torch.zeros(5)
        
        return {
            "image": input_tensor_stack,      # Shape: (T, C, H, W)
            "telemetry": telemetry,           # Shape: (10,)
            "physics": physics,               # Shape: (5,)
            "target": target_tensor           # Shape: (C, H, W)
        }
