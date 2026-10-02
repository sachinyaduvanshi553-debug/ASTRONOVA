import numpy as np
import cv2
import torch
import torchvision.transforms as T
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

        self.normalize = T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )

    def preprocess_image(self, image: np.ndarray, is_training: bool = False) -> torch.Tensor:
        """
        Apply resizing, normalization, and optionally augmentations.
        Returns a torch Tensor [C, H, W] in float32.
        """
        # Ensure image is in RGB format if it's single channel or RGBA
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif len(image.shape) == 3 and image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
        elif len(image.shape) == 3 and image.shape[2] == 1:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        # Resize
        if image.shape[0] != self.target_size or image.shape[1] != self.target_size:
            image = cv2.resize(image, (self.target_size, self.target_size), interpolation=cv2.INTER_AREA)

        # Convert to float tensor [0, 1]
        tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

        if is_training and self.augment:
            # Random slight rotation
            if np.random.rand() > 0.5:
                angle = float(np.random.uniform(-5.0, 5.0))
                tensor = T.functional.rotate(tensor, angle)
            # Random slight brightness/contrast
            if np.random.rand() > 0.5:
                brightness = float(np.random.uniform(0.9, 1.1))
                tensor = (tensor * brightness).clamp(0.0, 1.0)

        # Apply normalization
        tensor = self.normalize(tensor)
        return tensor

    @staticmethod
    def equalize_histogram(image: np.ndarray) -> np.ndarray:
        """Apply CLAHE for better contrast in solar features like active regions and flares."""
        if len(image.shape) == 3:
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

        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()

        blurred = cv2.GaussianBlur(gray, (9, 9), 2)
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
            circles = np.round(circles[0, :]).astype(int)
            best_circle = circles[0]
            cx, cy, r = best_circle
            shift_x = center_x - cx
            shift_y = center_y - cy

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
        """
        h, w = image.shape[:2]
        center_x, center_y = w // 2, h // 2
        radius = int(min(h, w) * (0.45 + margin))
        
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
    """
    import glob
    import os
    from datetime import datetime

    df_goes = pd.read_csv(goes_csv_path)
    df_goes['time'] = pd.to_datetime(df_goes['time'], utc=True)
    df_goes = df_goes.sort_values('time').reset_index(drop=True)

    image_records = []
    for path in sorted(glob.glob(os.path.join(image_dir, "*.jpg")) + glob.glob(os.path.join(image_dir, "*.png"))):
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

    aligned = pd.merge_asof(
        df_images,
        df_goes,
        on='time',
        direction='nearest',
        tolerance=pd.Timedelta(seconds=tolerance_seconds)
    )

    aligned = aligned.dropna(subset=['xrsa_flux', 'xrsb_flux'])
    return aligned
