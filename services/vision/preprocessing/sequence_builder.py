"""
sequence_builder.py — Chronological image sequence dataset for PyTorch.

CRITICAL LEAKAGE PREVENTION:
  - Sequences are loaded from JSON built by build_image_dataset.py
  - Chronological splitting is enforced BEFORE this class is instantiated
  - Normalization uses statistics passed in from the training split
  - No future images, telemetry, or labels appear in input tensors
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from .image_loader import load_image
from .image_normalizer import ImageNormalizer
from .augmentation import SequenceAugmentation

logger = logging.getLogger(__name__)

# 6 standard forecast horizons
HORIZONS = ["h15m", "h30m", "h60m", "h360m", "h720m", "h1440m"]
LABEL_TYPES = ["C_plus", "M_plus", "X_plus"]


class SolarSequenceDataset(Dataset):
    """
    PyTorch Dataset for chronological solar image sequences.
    
    Loads pre-built sequences from JSON files (output of build_image_dataset.py).
    Each sample returns:
        image_seq: [T, C, H, W] — historical image sequence
        telemetry: [T, F_tel]   — GOES flux features per timestep
        labels: dict of {horizon: {class: float}} — binary flare labels
        anchor_timestamp: str
        is_synthetic: bool      — True if smoke-test data
    
    Leakage guarantee:
        - Input window covers [T0-W, T0]
        - Labels cover (T0, T0+H] — purely future
        - No future features in input
    """

    def __init__(
        self,
        sequences_json: str | Path,
        image_size: int = 256,
        channels: int = 3,
        normalizer: ImageNormalizer | None = None,
        augmentation: SequenceAugmentation | None = None,
        max_seq_len: int = 24,
        missing_image_policy: str = "zero",  # "zero" | "skip"
    ):
        self.image_size = image_size
        self.channels = channels
        self.normalizer = normalizer
        self.augmentation = augmentation
        self.max_seq_len = max_seq_len
        self.missing_image_policy = missing_image_policy

        sequences_json = Path(sequences_json)
        if not sequences_json.exists():
            raise FileNotFoundError(f"Sequences JSON not found: {sequences_json}")

        with open(sequences_json) as f:
            self.sequences = json.load(f)

        n_synthetic = sum(1 for s in self.sequences if s.get("synthetic", False))
        if n_synthetic > 0:
            logger.warning(
                "Dataset contains %d synthetic sequences — NOT suitable for scientific evaluation",
                n_synthetic,
            )
        logger.info("Loaded %d sequences from %s", len(self.sequences), sequences_json)

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        seq = self.sequences[idx]

        # 1. Load image sequence [T, C, H, W]
        image_paths = seq.get("image_paths", [])
        image_seq = self._load_image_sequence(image_paths)

        # 2. Pad/truncate to max_seq_len
        T, C, H, W = image_seq.shape
        if T > self.max_seq_len:
            image_seq = image_seq[-self.max_seq_len:]  # keep most recent
        elif T < self.max_seq_len:
            pad = torch.zeros(self.max_seq_len - T, C, H, W)
            image_seq = torch.cat([pad, image_seq], dim=0)

        # 3. Augmentation (training only — caller sets this)
        if self.augmentation is not None:
            image_seq = self.augmentation(image_seq)

        # 4. Telemetry features [T_actual, 2] → padded to [max_seq_len, 2]
        soft_flux = seq.get("soft_xray_flux", [])
        log_flux = seq.get("log_soft_flux", [])
        telemetry = self._build_telemetry(soft_flux, log_flux)

        # 5. Labels — per horizon, per class
        raw_labels = seq.get("labels", {})
        labels = self._parse_labels(raw_labels)

        return {
            "image_seq": image_seq,          # [T, C, H, W]
            "telemetry": telemetry,           # [T, 2]
            "labels": labels,                 # dict[str, Tensor shape [3]]
            "anchor_timestamp": seq.get("anchor_timestamp", ""),
            "active_region_id": seq.get("active_region_id", 0) or 0,
            "is_synthetic": bool(seq.get("synthetic", False)),
            "n_images": seq.get("n_images", T),
        }

    def _load_image_sequence(self, paths: list[str]) -> torch.Tensor:
        """Load all images in sequence, handling missing files gracefully."""
        frames = []
        for p in paths:
            img = load_image(p, expected_size=self.image_size, channels=self.channels, missing_ok=True)
            if img is None:
                if self.missing_image_policy == "zero":
                    img = torch.zeros(self.channels, self.image_size, self.image_size)
                else:
                    continue
            else:
                # Resize if needed
                if img.shape[-1] != self.image_size or img.shape[-2] != self.image_size:
                    img = torch.nn.functional.interpolate(
                        img.unsqueeze(0),
                        size=(self.image_size, self.image_size),
                        mode="bilinear",
                        align_corners=False,
                    ).squeeze(0)
                # Normalize
                if self.normalizer is not None:
                    img = self.normalizer.transform(img)

            frames.append(img)

        if not frames:
            # Fallback: return zero sequence
            return torch.zeros(1, self.channels, self.image_size, self.image_size)

        return torch.stack(frames, dim=0)  # [T, C, H, W]

    def _build_telemetry(self, soft_flux: list, log_flux: list) -> torch.Tensor:
        """Build telemetry tensor [max_seq_len, 2]: (soft_flux, log_soft_flux)."""
        n = len(soft_flux)
        tel = np.zeros((n, 2), dtype=np.float32)
        for i, (sf, lf) in enumerate(zip(soft_flux, log_flux)):
            tel[i, 0] = float(sf) if sf is not None else 1e-9
            tel[i, 1] = float(lf) if lf is not None else -9.0

        tel_tensor = torch.from_numpy(tel)
        T = tel_tensor.shape[0]
        if T > self.max_seq_len:
            tel_tensor = tel_tensor[-self.max_seq_len:]
        elif T < self.max_seq_len:
            pad = torch.zeros(self.max_seq_len - T, 2)
            tel_tensor = torch.cat([pad, tel_tensor], dim=0)

        return tel_tensor

    def _parse_labels(self, raw_labels: dict) -> dict[str, torch.Tensor]:
        """Parse label dict into per-horizon float tensors [3] = [P(C+), P(M+), P(X+)]."""
        labels = {}
        for h in HORIZONS:
            h_labels = raw_labels.get(h, {})
            labels[h] = torch.tensor([
                float(h_labels.get("C_plus", False)),
                float(h_labels.get("M_plus", False)),
                float(h_labels.get("X_plus", False)),
            ], dtype=torch.float32)
        return labels

    def get_label_statistics(self) -> dict[str, dict[str, float]]:
        """Compute class prevalence for all horizons (for loss weighting)."""
        stats: dict[str, dict[str, int]] = {
            h: {"C_plus": 0, "M_plus": 0, "X_plus": 0, "total": 0}
            for h in HORIZONS
        }
        for seq in self.sequences:
            raw = seq.get("labels", {})
            for h in HORIZONS:
                hl = raw.get(h, {})
                stats[h]["total"] += 1
                for cls in LABEL_TYPES:
                    if hl.get(cls, False):
                        stats[h][cls] += 1

        prevalence = {}
        for h, cnts in stats.items():
            total = max(1, cnts["total"])
            prevalence[h] = {
                cls: cnts[cls] / total for cls in LABEL_TYPES
            }
        return prevalence
