import base64
import io
import traceback
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
import torch
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from PIL import Image

from .inference import VisionInferencePipeline

router = APIRouter(prefix="/vision", tags=["vision"])

# ---------------------------------------------------------------------------
# Model singleton
# ---------------------------------------------------------------------------
_pipeline: Optional[VisionInferencePipeline] = None
_load_error: Optional[str] = None

try:
    _pipeline = VisionInferencePipeline()
except Exception as exc:
    _load_error = f"{type(exc).__name__}: {exc}"
    print(f"[vision] Failed to load VisionInferencePipeline – {_load_error}")


def _get_pipeline() -> VisionInferencePipeline:
    if _pipeline is None:
        raise HTTPException(
            status_code=503,
            detail=f"Vision model is not loaded. {_load_error or ''}",
        )
    return _pipeline


# ---------------------------------------------------------------------------
# Pydantic request / response schemas
# ---------------------------------------------------------------------------

class PredictRequest(BaseModel):
    """Request body for ``/vision/predict``.

    *image_paths* – list of absolute file paths to solar images on disk
    (at least 1).  They are loaded, resized to 512×512, normalised,
    and stacked into a ``(1, T, 3, 512, 512)`` tensor.

    *telemetry_data* – 1-D list of floats (padded / truncated to 10).
    *physics_data*   – 1-D list of floats (padded / truncated to 5).
    """
    image_paths: List[str]
    telemetry_data: List[float] = []
    physics_data: List[float] = []


class ExplainRequest(BaseModel):
    """Request body for ``/vision/explain``."""
    image_paths: List[str]
    telemetry_data: List[float] = []
    physics_data: List[float] = []


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

TARGET_SIZE = 512


def _load_images(paths: List[str]) -> List[np.ndarray]:
    """Load images from disk using OpenCV.  Returns a list of BGR uint8
    numpy arrays."""
    images: List[np.ndarray] = []
    for p in paths:
        path = Path(p)
        if not path.is_file():
            raise HTTPException(
                status_code=400,
                detail=f"Image file not found: {p}",
            )
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            raise HTTPException(
                status_code=400,
                detail=f"Could not decode image: {p}",
            )
        images.append(img)
    return images


def _ndarray_to_base64_png(arr: np.ndarray) -> str:
    """Encode a (C, H, W) or (H, W, C) float32 array as a base64 PNG
    string via PIL."""
    if arr.ndim == 4:
        arr = arr[0]                       # drop batch dim
    if arr.shape[0] in (1, 3):             # CHW → HWC
        arr = np.transpose(arr, (1, 2, 0))
    # Clip and convert to uint8
    arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    if arr.shape[2] == 1:
        arr = arr.squeeze(2)
    pil_img = Image.fromarray(arr)
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _ndarray_map_to_base64(arr: np.ndarray) -> str:
    """Encode a 2-D [0,1] float map as a grayscale base64 PNG."""
    arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    pil_img = Image.fromarray(arr, mode="L")
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.get("/health")
async def health():
    """Return model / pipeline health status."""
    if _pipeline is not None:
        return {
            "status": "ok",
            "device": str(_pipeline.device),
            "model_loaded": True,
        }
    return {
        "status": "unavailable",
        "model_loaded": False,
        "error": _load_error,
    }


@router.post("/predict")
async def predict_future_image(request: PredictRequest):
    """Generate a predicted solar image from a sequence of input frames,
    telemetry, and physics data.  Returns the prediction as a base64-
    encoded PNG together with confidence and flare probability."""
    pipeline = _get_pipeline()

    try:
        # 1. Load raw images from disk -----------------------------------
        raw_images = _load_images(request.image_paths)

        # 2. Run the inference pipeline (handles preprocessing internally)
        result = pipeline.predict(
            image_sequence=raw_images,
            telemetry=request.telemetry_data,
            physics=request.physics_data,
        )

        # 3. Encode predicted image to base64 ----------------------------
        predicted_b64 = _ndarray_to_base64_png(result["predicted_image"])

        return {
            "status": "success",
            "confidence": round(result["confidence"], 6),
            "flare_probability": round(result["flare_probability"], 6),
            "predicted_image_base64": predicted_b64,
        }

    except HTTPException:
        raise
    except Exception as exc:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/explain")
async def explain_prediction(request: ExplainRequest):
    """Run XAI analysis (GradCAM, attention map, uncertainty map) for the
    given inputs and return each map as a base64 PNG."""
    pipeline = _get_pipeline()

    try:
        raw_images = _load_images(request.image_paths)

        xai_maps = pipeline.explain(
            image_sequence=raw_images,
            telemetry=request.telemetry_data,
            physics=request.physics_data,
        )

        return {
            "status": "success",
            "gradcam_base64": _ndarray_map_to_base64(xai_maps["gradcam"]),
            "attention_map_base64": _ndarray_map_to_base64(xai_maps["attention_map"]),
            "uncertainty_map_base64": _ndarray_map_to_base64(xai_maps["uncertainty_map"]),
        }

    except HTTPException:
        raise
    except Exception as exc:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(exc))
