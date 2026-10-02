"""
image_normalizer.py — Solar image normalization.

CRITICAL: Normalization statistics (mean, std, min, max, percentiles) must
ONLY be computed from the TRAINING split. Never from validation or test data.

Supported modes:
  1. minmax    — [0, 1] scaling using train min/max
  2. zscore    — zero mean, unit variance using train mean/std
  3. robust    — IQR-based robust scaling (immune to solar flare outliers)
  4. per_image — normalize each image independently (no train statistics needed)
  5. dataset_level — full dataset statistics (ONLY compute from train)
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import torch

logger = logging.getLogger(__name__)


class ImageNormalizer:
    """
    Solar image normalizer with leakage-safe statistics.
    
    Usage:
        normalizer = ImageNormalizer(mode="robust")
        normalizer.fit(train_images)           # Compute stats from TRAIN only
        normalizer.save("checkpoints/vision/normalization.json")
        
        # For inference:
        normalizer = ImageNormalizer.load("checkpoints/vision/normalization.json")
        normalized = normalizer.transform(image_tensor)
    """

    SUPPORTED_MODES = {"minmax", "zscore", "robust", "per_image", "dataset_level"}

    def __init__(self, mode: str = "robust"):
        if mode not in self.SUPPORTED_MODES:
            raise ValueError(f"mode must be one of {self.SUPPORTED_MODES}, got: {mode}")
        self.mode = mode
        self._stats: dict = {}
        self._fitted = False

    def fit(self, images: list[torch.Tensor] | np.ndarray) -> "ImageNormalizer":
        """
        Compute normalization statistics from TRAINING data only.
        
        Args:
            images: List of float32 tensors [C, H, W] or numpy arrays.
        """
        if self.mode == "per_image":
            logger.info("per_image mode: no global statistics needed")
            self._fitted = True
            return self

        logger.info("Computing normalization statistics from %d training images (mode=%s)",
                    len(images), self.mode)

        all_pixels = []
        for img in images:
            if isinstance(img, torch.Tensor):
                arr = img.numpy()
            else:
                arr = np.asarray(img)
            all_pixels.append(arr.flatten())

        flat = np.concatenate(all_pixels)

        if self.mode == "minmax" or self.mode == "dataset_level":
            self._stats = {
                "min": float(flat.min()),
                "max": float(flat.max()),
                "mode": self.mode,
            }
        elif self.mode == "zscore":
            self._stats = {
                "mean": float(flat.mean()),
                "std": float(flat.std() + 1e-8),
                "mode": self.mode,
            }
        elif self.mode == "robust":
            self._stats = {
                "p2": float(np.percentile(flat, 2)),
                "p98": float(np.percentile(flat, 98)),
                "median": float(np.median(flat)),
                "iqr": float(np.percentile(flat, 75) - np.percentile(flat, 25) + 1e-8),
                "mode": self.mode,
            }

        self._fitted = True
        logger.info("Normalization stats: %s", self._stats)
        return self

    def transform(self, image: torch.Tensor) -> torch.Tensor:
        """
        Normalize a single image tensor [C, H, W].
        """
        if self.mode == "per_image":
            vmin, vmax = image.min(), image.max()
            return (image - vmin) / (vmax - vmin + 1e-8)

        if not self._fitted:
            raise RuntimeError("Normalizer not fitted. Call fit() with training data first.")

        if self.mode in ("minmax", "dataset_level"):
            return (image - self._stats["min"]) / (self._stats["max"] - self._stats["min"] + 1e-8)
        elif self.mode == "zscore":
            return (image - self._stats["mean"]) / self._stats["std"]
        elif self.mode == "robust":
            return (image - self._stats["p2"]) / (self._stats["p98"] - self._stats["p2"] + 1e-8)
        return image

    def inverse_transform(self, image: torch.Tensor) -> torch.Tensor:
        """Reverse normalization for visualization."""
        if self.mode == "per_image":
            return image  # Cannot reverse per-image normalization globally

        if not self._fitted:
            raise RuntimeError("Normalizer not fitted.")

        if self.mode in ("minmax", "dataset_level"):
            return image * (self._stats["max"] - self._stats["min"]) + self._stats["min"]
        elif self.mode == "zscore":
            return image * self._stats["std"] + self._stats["mean"]
        elif self.mode == "robust":
            return image * (self._stats["p98"] - self._stats["p2"]) + self._stats["p2"]
        return image

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"mode": self.mode, "fitted": self._fitted, "stats": self._stats}
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)
        logger.info("Normalization stats saved: %s", path)

    @classmethod
    def load(cls, path: str | Path) -> "ImageNormalizer":
        with open(path) as f:
            payload = json.load(f)
        obj = cls(mode=payload["mode"])
        obj._stats = payload.get("stats", {})
        obj._fitted = payload.get("fitted", False)
        logger.info("Loaded normalizer: mode=%s, fitted=%s", obj.mode, obj._fitted)
        return obj
