"""
models/vision/future_target_resolver.py — Resolves genuine observed future image targets.

CRITICAL RESEARCH RULE:
    - Never substitute the last input frame as the future target.
    - Only return true observed images matching the exact forecasting horizon.
    - If no future frame exists within tolerance, return None and mark target_available[h] = False.
    - Strictly reject timestamps <= input end time (T_anchor).
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import torch

from services.vision.preprocessing.image_loader import load_image
from services.vision.preprocessing.image_normalizer import ImageNormalizer

logger = logging.getLogger("astronova.vision.future_target_resolver")

HORIZONS = ["h15m", "h30m", "h60m", "h360m", "h720m", "h1440m"]
HORIZON_MINUTES = {
    "h15m": 15,
    "h30m": 30,
    "h60m": 60,
    "h360m": 360,
    "h720m": 720,
    "h1440m": 1440,
}
# Tolerance window in minutes for matching future observation frames
HORIZON_TOLERANCE_MINUTES = {
    "h15m": 10,
    "h30m": 15,
    "h60m": 20,
    "h360m": 45,
    "h720m": 60,
    "h1440m": 90,
}


def parse_iso_timestamp(ts_str: str) -> Optional[datetime]:
    """Safely parse an ISO 8601 timestamp string into a timezone-aware UTC datetime."""
    if not ts_str:
        return None
    try:
        dt = pd.to_datetime(ts_str, utc=True)
        return dt.to_pydatetime()
    except Exception:
        return None


class FutureTargetResolver:
    """
    Finds and loads genuine future observation images corresponding to
    each forecasting horizon (h15m .. h1440m) given a sequence anchor timestamp.
    """

    def __init__(
        self,
        image_index: Optional[List[Dict[str, Any]]] = None,
        image_size: int = 256,
        channels: int = 3,
        normalizer: Optional[ImageNormalizer] = None,
    ):
        """
        Args:
            image_index: List of {"path": str, "timestamp": datetime} available in the repository / dataset.
            image_size: Target image resolution (e.g. 128 or 256).
            channels: Number of color channels (3).
            normalizer: Optional fitted ImageNormalizer instance.
        """
        self.image_size = image_size
        self.channels = channels
        self.normalizer = normalizer
        self.image_index: List[Dict[str, Any]] = image_index or []

    def build_index_from_directory(self, image_dir: str | Path) -> None:
        """Indexes all images in a directory with parsed timestamps."""
        image_dir = Path(image_dir)
        if not image_dir.exists():
            logger.warning("Image directory does not exist: %s", image_dir)
            return

        index = []
        for ext in ("*.jpg", "*.png", "*.jpeg"):
            for p in image_dir.rglob(ext):
                ts = self._extract_timestamp_from_path(p)
                if ts is not None:
                    index.append({"path": str(p), "timestamp": ts})

        index.sort(key=lambda x: x["timestamp"])
        self.image_index = index
        logger.info("Indexed %d images from %s", len(self.image_index), image_dir)

    def _extract_timestamp_from_path(self, path: Path) -> Optional[datetime]:
        """Extract timestamp from standard SDO/synthetic filename."""
        name = path.stem
        # Format example: synthetic_20241001_120000 or 20241001_120000
        parts = name.split("_")
        for i in range(len(parts) - 1):
            date_str, time_str = parts[i], parts[i + 1]
            if len(date_str) == 8 and len(time_str) >= 6 and date_str.isdigit() and time_str[:6].isdigit():
                try:
                    dt = datetime.strptime(f"{date_str}_{time_str[:6]}", "%Y%m%d_%H%M%S")
                    return dt.replace(tzinfo=timezone.utc)
                except Exception:
                    continue
        return None

    def resolve_targets(
        self,
        anchor_timestamp: str,
        input_image_paths: Optional[List[str]] = None,
    ) -> Tuple[Dict[str, Optional[torch.Tensor]], Dict[str, bool]]:
        """
        Locates future observed images strictly after anchor_timestamp.

        Returns:
            future_images: dict[horizon -> Tensor[C, H, W] or None]
            target_available: dict[horizon -> bool]
        """
        anchor_dt = parse_iso_timestamp(anchor_timestamp)
        input_paths_set = set(input_image_paths or [])

        future_images: Dict[str, Optional[torch.Tensor]] = {h: None for h in HORIZONS}
        target_available: Dict[str, bool] = {h: False for h in HORIZONS}

        if anchor_dt is None or not self.image_index:
            return future_images, target_available

        for h in HORIZONS:
            target_dt = anchor_dt + timedelta(minutes=HORIZON_MINUTES[h])
            tol = timedelta(minutes=HORIZON_TOLERANCE_MINUTES[h])

            best_match = None
            best_diff = None

            for entry in self.image_index:
                img_dt = entry["timestamp"]
                img_path = entry["path"]

                # STRICT RULE: Must be strictly after anchor timestamp
                if img_dt <= anchor_dt:
                    continue
                # STRICT RULE: Must not be in historical input sequence
                if img_path in input_paths_set:
                    continue

                diff = abs((img_dt - target_dt).total_seconds())
                if diff <= tol.total_seconds():
                    if best_diff is None or diff < best_diff:
                        best_diff = diff
                        best_match = img_path

            if best_match is not None:
                tensor = load_image(
                    best_match,
                    expected_size=self.image_size,
                    channels=self.channels,
                    missing_ok=True,
                )
                if tensor is not None:
                    if tensor.shape[-1] != self.image_size or tensor.shape[-2] != self.image_size:
                        tensor = torch.nn.functional.interpolate(
                            tensor.unsqueeze(0),
                            size=(self.image_size, self.image_size),
                            mode="bilinear",
                            align_corners=False,
                        ).squeeze(0)
                    if self.normalizer is not None:
                        tensor = self.normalizer.transform(tensor)

                    future_images[h] = tensor
                    target_available[h] = True

        return future_images, target_available
