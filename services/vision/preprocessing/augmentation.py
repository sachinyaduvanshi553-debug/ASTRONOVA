"""
augmentation.py — Physically reasonable solar image augmentations.

RULES for solar images:
  - Small rotations ONLY (solar disc orientation has physical meaning)
  - NO horizontal/vertical flips (heliographic coordinates matter)
  - NO elastic distortions (would corrupt magnetic morphology)
  - Intensity scaling within narrow bounds (detector sensitivity variation)
  - Gaussian noise (detector read noise simulation)
  - Small translations (pointing jitter)
  - Contrast adjustment (atmospheric/detector variation)

All augmentations are applied ONLY to training data.
"""
from __future__ import annotations

import random

import numpy as np
import torch
import torch.nn.functional as F


class SolarAugmentation:
    """
    Physically reasonable augmentations for solar images.
    All parameters are configurable.
    """

    def __init__(
        self,
        rotation_max_deg: float = 5.0,
        translation_max_px: int = 4,
        intensity_scale_range: tuple[float, float] = (0.95, 1.05),
        gaussian_noise_std: float = 0.01,
        contrast_range: tuple[float, float] = (0.9, 1.1),
        enabled: bool = True,
    ):
        self.rotation_max_deg = rotation_max_deg
        self.translation_max_px = translation_max_px
        self.intensity_scale_range = intensity_scale_range
        self.gaussian_noise_std = gaussian_noise_std
        self.contrast_range = contrast_range
        self.enabled = enabled

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        """
        Apply augmentations to a single image tensor [C, H, W].
        Returns augmented tensor [C, H, W].
        """
        if not self.enabled:
            return image

        # Small rotation
        if self.rotation_max_deg > 0 and random.random() < 0.5:
            angle = random.uniform(-self.rotation_max_deg, self.rotation_max_deg)
            image = self._rotate(image, angle)

        # Small translation
        if self.translation_max_px > 0 and random.random() < 0.5:
            tx = random.randint(-self.translation_max_px, self.translation_max_px)
            ty = random.randint(-self.translation_max_px, self.translation_max_px)
            image = self._translate(image, tx, ty)

        # Intensity scaling
        if random.random() < 0.5:
            scale = random.uniform(*self.intensity_scale_range)
            image = (image * scale).clamp(0.0, 1.0)

        # Gaussian noise
        if self.gaussian_noise_std > 0 and random.random() < 0.5:
            noise = torch.randn_like(image) * self.gaussian_noise_std
            image = (image + noise).clamp(0.0, 1.0)

        # Contrast
        if random.random() < 0.3:
            factor = random.uniform(*self.contrast_range)
            mean = image.mean()
            image = ((image - mean) * factor + mean).clamp(0.0, 1.0)

        return image

    def _rotate(self, image: torch.Tensor, angle_deg: float) -> torch.Tensor:
        """Rotate image by angle_deg using affine transform."""
        angle_rad = angle_deg * np.pi / 180.0
        cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
        theta = torch.tensor(
            [[cos_a, -sin_a, 0], [sin_a, cos_a, 0]],
            dtype=torch.float32,
        ).unsqueeze(0)
        grid = F.affine_grid(theta, image.unsqueeze(0).shape, align_corners=False)
        rotated = F.grid_sample(image.unsqueeze(0), grid, align_corners=False, padding_mode="border")
        return rotated.squeeze(0)

    def _translate(self, image: torch.Tensor, tx: int, ty: int) -> torch.Tensor:
        """Translate image by (tx, ty) pixels."""
        _, H, W = image.shape
        tx_norm = tx / (W / 2)
        ty_norm = ty / (H / 2)
        theta = torch.tensor(
            [[1, 0, tx_norm], [0, 1, ty_norm]],
            dtype=torch.float32,
        ).unsqueeze(0)
        grid = F.affine_grid(theta, image.unsqueeze(0).shape, align_corners=False)
        translated = F.grid_sample(image.unsqueeze(0), grid, align_corners=False, padding_mode="border")
        return translated.squeeze(0)


class SequenceAugmentation:
    """Apply the same spatial augmentation to all frames in a sequence."""

    def __init__(self, image_aug: SolarAugmentation):
        self.image_aug = image_aug

    def __call__(self, sequence: torch.Tensor) -> torch.Tensor:
        """
        Args:
            sequence: [T, C, H, W]
        Returns:
            augmented: [T, C, H, W] — same spatial transform for all frames
        """
        T, C, H, W = sequence.shape

        # Sample a single augmentation configuration and apply consistently
        angle = random.uniform(-self.image_aug.rotation_max_deg, self.image_aug.rotation_max_deg)
        tx = random.randint(-self.image_aug.translation_max_px, self.image_aug.translation_max_px)
        ty = random.randint(-self.image_aug.translation_max_px, self.image_aug.translation_max_px)
        intensity_scale = random.uniform(*self.image_aug.intensity_scale_range)
        noise_std = self.image_aug.gaussian_noise_std

        augmented = []
        for t in range(T):
            frame = sequence[t]
            if self.image_aug.enabled:
                if angle != 0:
                    frame = self.image_aug._rotate(frame, angle)
                if tx != 0 or ty != 0:
                    frame = self.image_aug._translate(frame, tx, ty)
                frame = (frame * intensity_scale).clamp(0.0, 1.0)
                if noise_std > 0:
                    frame = (frame + torch.randn_like(frame) * noise_std).clamp(0.0, 1.0)
            augmented.append(frame)

        return torch.stack(augmented, dim=0)
