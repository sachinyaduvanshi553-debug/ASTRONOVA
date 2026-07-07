import numpy as np
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from typing import List, Tuple, Dict, Any, Optional
from datetime import datetime

class SolarImagePreprocessor:
    """
    Handles preprocessing of raw solar imagery from SDO, SOHO, Aditya-L1.
    """
    def __init__(self, target_size: int = 512, augment: bool = False):
        self.target_size = target_size
        self.augment = augment

        # Base transforms applied to all images (validation/inference)
        self.base_transforms = A.Compose([
            A.Resize(height=target_size, width=target_size),
            A.Normalize(
                mean=[0.485, 0.456, 0.406], # standard ImageNet mean, ideally replaced with solar mean
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2()
        ])

        # Augmentation transforms for training
        self.train_transforms = A.Compose([
            A.Resize(height=target_size, width=target_size),
            A.RandomRotate90(p=0.5),
            A.Flip(p=0.5),
            A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.2),
            A.RandomBrightnessContrast(p=0.2),
            A.OneOf([
                A.GaussNoise(p=1.0),
                A.MultiplicativeNoise(p=1.0),
            ], p=0.2),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2()
        ])

    def preprocess_image(self, image: np.ndarray, is_training: bool = False) -> Any:
        """
        Apply resizing, normalization, and optionally augmentations.
        Returns a torch Tensor.
        """
        # Ensure image is in RGB format if it's single channel (or handle channels differently depending on dataset)
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        if is_training and self.augment:
            augmented = self.train_transforms(image=image)
        else:
            augmented = self.base_transforms(image=image)
            
        return augmented["image"]

    @staticmethod
    def equalize_histogram(image: np.ndarray) -> np.ndarray:
        """Apply CLAHE for better contrast in solar features."""
        if len(image.shape) == 3:
            # Convert to LAB for CLAHE
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            cl = clahe.apply(l)
            limg = cv2.merge((cl,a,b))
            return cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
        else:
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            return clahe.apply(image)

    @staticmethod
    def align_solar_disk(image: np.ndarray) -> np.ndarray:
        """
        Aligns the solar disk to the center. 
        (Mock implementation - normally uses sunpy.map)
        """
        # In a full implementation, we'd find the center of mass or use FITS header info
        # to shift the image so the solar disk is perfectly centered.
        return image

    @staticmethod
    def remove_noise(image: np.ndarray) -> np.ndarray:
        """Basic noise removal."""
        return cv2.GaussianBlur(image, (5, 5), 0)

    def process_sequence(self, images: List[np.ndarray], is_training: bool = False) -> List[Any]:
        """Process a sequence of historical images."""
        processed_tensors = []
        for img in images:
            img = self.remove_noise(img)
            img = self.equalize_histogram(img)
            img = self.align_solar_disk(img)
            tensor = self.preprocess_image(img, is_training=is_training)
            processed_tensors.append(tensor)
        return processed_tensors

def synchronize_data(images: List[Dict], telemetry: List[Dict], tolerance_seconds: int = 300) -> List[Dict]:
    """
    Aligns images with nearest GOES telemetry within a tolerance.
    """
    aligned = []
    # Simplified O(N^2) for mock, in reality use sorted arrays or pandas merge_asof
    for img_record in images:
        img_time = img_record['timestamp']
        # Find closest telemetry
        closest_tel = None
        min_diff = float('inf')
        for tel_record in telemetry:
            diff = abs((img_time - tel_record['timestamp']).total_seconds())
            if diff < min_diff and diff <= tolerance_seconds:
                min_diff = diff
                closest_tel = tel_record
                
        if closest_tel:
            aligned.append({
                "image": img_record['image'],
                "timestamp": img_time,
                "telemetry": closest_tel
            })
    return aligned
