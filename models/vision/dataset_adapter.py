"""
models/vision/dataset_adapter.py — Dataset Adapter for the Multimodal Vision Pipeline.

Integrates:
  - Chronological sequence manifests (train_sequences.json, validation_sequences.json, test_sequences.json)
  - FutureTargetResolver for authentic, timestamp-matched future images without leakage
  - Support for Telemetry, Photospheric Magnetic (16 HMI parameters), and Physics features
  - Standardized dictionary sample structure conforming to pipeline specifications.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from torch.utils.data import Dataset

from services.vision.preprocessing.augmentation import SequenceAugmentation, SolarAugmentation
from services.vision.preprocessing.image_loader import load_image
from services.vision.preprocessing.image_normalizer import ImageNormalizer
from .future_target_resolver import FutureTargetResolver, HORIZONS

logger = logging.getLogger("astronova.vision.dataset_adapter")


class AdaptedSolarSequenceDataset(Dataset):
    """
    Adapter dataset wrapping chronological sequences and resolving future target images.
    """

    def __init__(
        self,
        manifest_path: str | Path,
        image_size: int = 128,
        channels: int = 3,
        lookback_frames: int = 4,
        normalizer: Optional[ImageNormalizer] = None,
        augmentation: Optional[SequenceAugmentation] = None,
        target_resolver: Optional[FutureTargetResolver] = None,
        image_root: Optional[str | Path] = None,
    ):
        self.manifest_path = Path(manifest_path)
        self.image_size = image_size
        self.channels = channels
        self.lookback_frames = lookback_frames
        self.normalizer = normalizer or ImageNormalizer(mode="per_image")
        self.augmentation = augmentation
        self.image_root = Path(image_root) if image_root else None

        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Sequence manifest not found at: {self.manifest_path}")

        with open(self.manifest_path, "r", encoding="utf-8") as f:
            self.sequences: List[Dict[str, Any]] = json.load(f)

        # Set up Target Resolver
        if target_resolver is not None:
            self.target_resolver = target_resolver
        else:
            self.target_resolver = FutureTargetResolver(
                image_size=image_size,
                channels=channels,
                normalizer=self.normalizer,
            )
            # Index all available images from dataset directory if possible
            if self.image_root and self.image_root.exists():
                self.target_resolver.build_index_from_directory(self.image_root)
            else:
                # Default search in datasets/raw/sdo_images
                default_img_dir = self.manifest_path.parents[1] / "raw" / "sdo_images"
                if default_img_dir.exists():
                    self.target_resolver.build_index_from_directory(default_img_dir)

        logger.info(
            "Adapted dataset initialized with %d sequences from %s (image_size=%d, lookback=%d)",
            len(self.sequences),
            self.manifest_path.name,
            self.image_size,
            self.lookback_frames,
        )

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        seq = self.sequences[idx]
        anchor_ts = seq.get("anchor_timestamp", "")
        raw_paths = seq.get("image_paths", [])

        # 1. Load historical input sequence [T, C, H, W]
        image_tensors = []
        for p in raw_paths:
            img_tensor = load_image(p, expected_size=self.image_size, channels=self.channels, missing_ok=True)
            if img_tensor is None:
                img_tensor = torch.zeros(self.channels, self.image_size, self.image_size)
            else:
                if img_tensor.shape[-1] != self.image_size or img_tensor.shape[-2] != self.image_size:
                    img_tensor = torch.nn.functional.interpolate(
                        img_tensor.unsqueeze(0),
                        size=(self.image_size, self.image_size),
                        mode="bilinear",
                        align_corners=False,
                    ).squeeze(0)
                if self.normalizer is not None:
                    img_tensor = self.normalizer.transform(img_tensor)
            image_tensors.append(img_tensor)

        if not image_tensors:
            image_tensors = [torch.zeros(self.channels, self.image_size, self.image_size)]

        image_seq = torch.stack(image_tensors, dim=0)  # [T, C, H, W]

        # Truncate / pad to lookback_frames
        T = image_seq.shape[0]
        if T > self.lookback_frames:
            image_seq = image_seq[-self.lookback_frames:]
        elif T < self.lookback_frames:
            pad = torch.zeros(self.lookback_frames - T, self.channels, self.image_size, self.image_size)
            image_seq = torch.cat([pad, image_seq], dim=0)

        # 2. Augment (training only)
        if self.augmentation is not None:
            image_seq = self.augmentation(image_seq)

        # 3. Telemetry tensor [T, 2]
        soft_flux = seq.get("soft_xray_flux", [])
        log_flux = seq.get("log_soft_flux", [])
        n_tel = min(len(soft_flux), len(log_flux))
        tel_list = [
            [float(soft_flux[i]) if soft_flux[i] is not None else 1e-9,
             float(log_flux[i]) if log_flux[i] is not None else -9.0]
            for i in range(n_tel)
        ]
        if not tel_list:
            telemetry = torch.zeros(self.lookback_frames, 2, dtype=torch.float32)
        else:
            telemetry = torch.tensor(tel_list, dtype=torch.float32)
            if telemetry.shape[0] > self.lookback_frames:
                telemetry = telemetry[-self.lookback_frames:]
            elif telemetry.shape[0] < self.lookback_frames:
                pad_tel = torch.zeros(self.lookback_frames - telemetry.shape[0], 2)
                telemetry = torch.cat([pad_tel, telemetry], dim=0)

        # 4. Magnetic features [16] & Physics [5] (default zero-filled)
        magnetic = torch.tensor(seq.get("magnetic_features", [0.0] * 16), dtype=torch.float32)
        if magnetic.numel() != 16:
            magnetic = torch.zeros(16, dtype=torch.float32)

        physics = torch.tensor(seq.get("physics_features", [0.0] * 5), dtype=torch.float32)
        if physics.numel() != 5:
            physics = torch.zeros(5, dtype=torch.float32)

        # 5. Horizon Labels: dict[h -> Tensor[3]] (C+, M+, X+)
        raw_labels = seq.get("labels", {})
        labels: Dict[str, torch.Tensor] = {}
        for h in HORIZONS:
            hl = raw_labels.get(h, {})
            labels[h] = torch.tensor([
                float(hl.get("C_plus", False)),
                float(hl.get("M_plus", False)),
                float(hl.get("X_plus", False)),
            ], dtype=torch.float32)

        # 6. Resolve genuine future targets
        future_images, target_avail = self.target_resolver.resolve_targets(
            anchor_timestamp=anchor_ts,
            input_image_paths=raw_paths,
        )

        return {
            "image_seq": image_seq,
            "telemetry": telemetry,
            "magnetic": magnetic,
            "physics": physics,
            "labels": labels,
            "future_images": future_images,
            "target_available": target_avail,
            "anchor_timestamp": anchor_ts,
            "timestamp": anchor_ts,
            "is_synthetic": bool(seq.get("synthetic", False)),
            "active_region_id": seq.get("active_region_id", 0) or 0,
        }


def adapted_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Collate function for AdaptedSolarSequenceDataset."""
    image_seqs = torch.stack([b["image_seq"] for b in batch])
    telemetry = torch.stack([b["telemetry"] for b in batch])
    magnetic = torch.stack([b["magnetic"] for b in batch])
    physics = torch.stack([b["physics"] for b in batch])

    labels: Dict[str, torch.Tensor] = {}
    for h in HORIZONS:
        labels[h] = torch.stack([b["labels"][h] for b in batch])

    future_images: Dict[str, Optional[torch.Tensor]] = {}
    target_available: Dict[str, torch.Tensor] = {}

    for h in HORIZONS:
        all_present = all(b["future_images"].get(h) is not None for b in batch)
        if all_present:
            future_images[h] = torch.stack([b["future_images"][h] for b in batch])
        else:
            future_images[h] = None
        target_available[h] = torch.tensor([b["target_available"].get(h, False) for b in batch], dtype=torch.bool)

    return {
        "image_seq": image_seqs,
        "telemetry": telemetry,
        "magnetic": magnetic,
        "physics": physics,
        "labels": labels,
        "future_images": future_images,
        "target_available": target_available,
        "anchor_timestamps": [b["anchor_timestamp"] for b in batch],
        "is_synthetic": [b["is_synthetic"] for b in batch],
    }
