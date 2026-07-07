import numpy as np
import cv2
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from typing import List, Dict, Any
import pandas as pd

class SolarImagePreprocessor:
    """
    Handles preprocessing of raw solar imagery from SDO, SOHO, Aditya-L1.
    Includes authentic solar disk alignment, CLAHE enhancement, and noise removal.
    """
    def __init__(self, target_size: int = 512, augment: bool = False):
        self.target_size = target_size
        self.augment = augment

        # Base transforms applied to all images (validation/inference)
        self.base_transforms = A.Compose([
            A.Resize(height=target_size, width=target_size),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2()
        ])

        # Augmentation transforms for training
        self.train_transforms = A.Compose([
            A.Resize(height=target_size, width=target_size),
            A.RandomRotate90(p=0.5),
            A.Flip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.3),
            A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.3),
            A.OneOf([
                A.GaussNoise(p=1.0),
                A.GaussianBlur(blur_limit=3, p=1.0),
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
        # Ensure image is in RGB format if it's single channel
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)

        if is_training and self.augment:
            augmented = self.train_transforms(image=image)
        else:
            augmented = self.base_transforms(image=image)
            
        return augmented["image"]

    @staticmethod
    def equalize_histogram(image: np.ndarray) -> np.ndarray:
        """Apply CLAHE for better contrast in solar features like active regions and flares."""
        if len(image.shape) == 3:
            # Convert to LAB for CLAHE
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            cl = clahe.apply(l)
            limg = cv2.merge((cl, a, b))
            return cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
        else:
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            return clahe.apply(image)

    @staticmethod
    def align_solar_disk(image: np.ndarray) -> np.ndarray:
        """
        Authentic solar disk alignment using Hough circle detection.
        Detects the solar disk boundary and centers it in the frame.
        """
        if image is None or image.size == 0:
            return image

        h, w = image.shape[:2]
        center_x, center_y = w // 2, h // 2

        # Convert to grayscale for circle detection
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()

        # Apply Gaussian blur to reduce noise for Hough detection
        blurred = cv2.GaussianBlur(gray, (9, 9), 2)

        # Detect circles (solar disk) using Hough Circle Transform
        # Solar disk typically occupies 80-95% of the image in SDO data
        min_radius = int(min(h, w) * 0.3)
        max_radius = int(min(h, w) * 0.5)

        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=min(h, w) // 2,
            param1=50,
            param2=30,
            minRadius=min_radius,
            maxRadius=max_radius
        )

        if circles is not None:
            # Take the most prominent circle
            circles = np.round(circles[0, :]).astype(int)
            best_circle = circles[0]
            cx, cy, r = best_circle

            # Calculate shift needed to center the disk
            shift_x = center_x - cx
            shift_y = center_y - cy

            # Only shift if the offset is significant (> 2% of image size)
            if abs(shift_x) > w * 0.02 or abs(shift_y) > h * 0.02:
                M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
                image = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=0)

        return image

    @staticmethod
    def remove_noise(image: np.ndarray) -> np.ndarray:
        """
        Removes sensor noise while preserving solar features.
        Uses bilateral filter which preserves edges better than Gaussian blur.
        """
        if image is None or image.size == 0:
            return image
        return cv2.bilateralFilter(image, d=5, sigmaColor=50, sigmaSpace=50)

    @staticmethod
    def mask_off_limb(image: np.ndarray, margin: float = 0.02) -> np.ndarray:
        """
        Masks off-limb regions (outside the solar disk) to black.
        This removes stray light and cosmic ray artifacts outside the disk.
        """
        h, w = image.shape[:2]
        center_x, center_y = w // 2, h // 2
        
        # Approximate solar radius as ~45% of image dimension
        radius = int(min(h, w) * (0.45 + margin))
        
        # Create circular mask
        Y, X = np.ogrid[:h, :w]
        dist_from_center = np.sqrt((X - center_x) ** 2 + (Y - center_y) ** 2)
        mask = dist_from_center <= radius
        
        if len(image.shape) == 3:
            mask = mask[:, :, np.newaxis]
            
        return (image * mask).astype(image.dtype)

    def process_sequence(self, images: List[np.ndarray], is_training: bool = False) -> List[Any]:
        """Process a sequence of historical images through the full preprocessing pipeline."""
        processed_tensors = []
        for img in images:
            img = self.remove_noise(img)
            img = self.equalize_histogram(img)
            img = self.align_solar_disk(img)
            img = self.mask_off_limb(img)
            tensor = self.preprocess_image(img, is_training=is_training)
            processed_tensors.append(tensor)
        return processed_tensors


def synchronize_data(
    image_dir: str,
    goes_csv_path: str,
    tolerance_seconds: int = 300
) -> pd.DataFrame:
    """
    Aligns images with nearest GOES telemetry within a tolerance using pandas merge_asof.
    This is the production-grade version using vectorized operations.
    """
    import glob
    import os
    from datetime import datetime

    # Load GOES data
    df_goes = pd.read_csv(goes_csv_path)
    df_goes['time'] = pd.to_datetime(df_goes['time'], utc=True)
    df_goes = df_goes.sort_values('time').reset_index(drop=True)

    # Parse image timestamps from filenames
    image_records = []
    for path in sorted(glob.glob(os.path.join(image_dir, "*.jpg"))):
        filename = os.path.basename(path)
        try:
            dt_str = filename[:15]
            dt = datetime.strptime(dt_str, "%Y%m%d_%H%M%S")
            dt = pd.to_datetime(dt, utc=True)
            image_records.append({'path': path, 'time': dt})
        except Exception:
            continue

    if not image_records:
        return pd.DataFrame()

    df_images = pd.DataFrame(image_records).sort_values('time').reset_index(drop=True)

    # Merge using merge_asof for efficient nearest-neighbor temporal alignment
    aligned = pd.merge_asof(
        df_images,
        df_goes,
        on='time',
        direction='nearest',
        tolerance=pd.Timedelta(seconds=tolerance_seconds)
    )

    # Drop rows where telemetry wasn't found within tolerance
    aligned = aligned.dropna(subset=['xrsa_flux', 'xrsb_flux'])

    return aligned
