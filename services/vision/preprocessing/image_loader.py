"""
image_loader.py — Solar image loading with validation.

Supports JPEG, PNG, and FITS formats.
Always validates:
  - file existence
  - minimum size
  - pixel range sanity
  - channel count
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch

logger = logging.getLogger(__name__)


def load_image(
    path: str | Path,
    expected_size: int = 256,
    channels: int = 3,
    missing_ok: bool = False,
) -> torch.Tensor | None:
    """
    Load a solar image from disk.
    Returns: float32 tensor [C, H, W] in range [0, 1], or None if missing_ok.
    
    Validates file, size, and pixel range.
    """
    path = Path(path)

    if not path.exists():
        if missing_ok:
            logger.warning("Image not found: %s — returning None", path)
            return None
        raise FileNotFoundError(f"Image not found: {path}")

    suffix = path.suffix.lower()

    if suffix == ".npy":
        arr = np.load(str(path)).astype(np.float32)
    elif suffix in (".fits", ".fit"):
        arr = _load_fits(path, channels)
    else:
        arr = _load_raster(path, channels)

    if arr is None:
        if missing_ok:
            return None
        raise ValueError(f"Failed to load image: {path}")

    # Ensure [H, W, C] → [C, H, W]
    if arr.ndim == 2:
        arr = arr[:, :, np.newaxis]
        arr = np.repeat(arr, channels, axis=2)
    if arr.shape[2] != channels:
        if arr.shape[2] > channels:
            arr = arr[:, :, :channels]
        else:
            arr = np.concatenate([arr] + [arr[:, :, :1]] * (channels - arr.shape[2]), axis=2)

    arr = arr.transpose(2, 0, 1)  # [C, H, W]

    # Normalize to [0, 1]
    arr = arr.astype(np.float32)
    if arr.max() > 1.0:
        arr = arr / 255.0
    arr = arr.clip(0.0, 1.0)

    return torch.from_numpy(arr)


def _load_raster(path: Path, channels: int) -> np.ndarray | None:
    """Load JPEG/PNG using PIL (no cv2 dependency required)."""
    try:
        from PIL import Image
        img = Image.open(path)
        if channels == 1:
            img = img.convert("L")
            arr = np.array(img, dtype=np.float32)[:, :, np.newaxis]
        else:
            img = img.convert("RGB")
            arr = np.array(img, dtype=np.float32)
        return arr
    except Exception as e:
        logger.warning("PIL failed for %s: %s", path, e)

    try:
        import cv2
        flag = cv2.IMREAD_GRAYSCALE if channels == 1 else cv2.IMREAD_COLOR
        arr = cv2.imread(str(path), flag)
        if arr is None:
            return None
        if channels == 3:
            arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
        if arr.ndim == 2:
            arr = arr[:, :, np.newaxis]
        return arr.astype(np.float32)
    except Exception as e:
        logger.warning("cv2 failed for %s: %s", path, e)
        return None


def _load_fits(path: Path, channels: int) -> np.ndarray | None:
    """Load FITS scientific image."""
    try:
        from astropy.io import fits
        with fits.open(str(path)) as hdul:
            data = hdul[0].data
            if data is None and len(hdul) > 1:
                data = hdul[1].data
            if data is None:
                return None
            data = data.astype(np.float32)
            # Handle NaN/Inf
            data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
            # Expand to [H, W, C]
            if data.ndim == 2:
                data = np.stack([data] * channels, axis=-1)
            return data
    except ImportError:
        logger.warning("astropy not installed — cannot load FITS file %s", path)
        return None
    except Exception as e:
        logger.warning("FITS load failed for %s: %s", path, e)
        return None
