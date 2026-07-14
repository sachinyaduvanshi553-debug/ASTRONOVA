import base64
import io
import traceback
from pathlib import Path
from typing import List, Optional
from datetime import datetime

import cv2
import numpy as np
import torch
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from PIL import Image

from .inference import VisionInferencePipeline

router = APIRouter(prefix="/vision", tags=["vision"])

_pipeline: Optional[VisionInferencePipeline] = None
_load_error: Optional[str] = None
_total_inferences = 0
_last_inference_time = None

try:
    _pipeline = VisionInferencePipeline()
except Exception as exc:
    _load_error = f"{type(exc).__name__}: {exc}"
    print(f"[vision] Failed to load VisionInferencePipeline – {_load_error}")


def _get_pipeline() -> VisionInferencePipeline:
    global _total_inferences, _last_inference_time
    if _pipeline is None:
        raise HTTPException(status_code=503, detail=f"Vision model is not loaded. {_load_error or ''}")
    _total_inferences += 1
    _last_inference_time = datetime.utcnow().isoformat()
    return _pipeline


class PredictRequest(BaseModel):
    image_paths: List[str]
    telemetry_data: List[float] = []
    physics_data: List[float] = []


class ExplainRequest(BaseModel):
    image_paths: List[str]
    telemetry_data: List[float] = []
    physics_data: List[float] = []


class UncertaintyRequest(BaseModel):
    image_paths: List[str]
    telemetry_data: List[float] = []
    physics_data: List[float] = []


def _load_images(paths: List[str]) -> List[np.ndarray]:
    images: List[np.ndarray] = []
    for p in paths:
        path = Path(p)
        if not path.is_file():
            raise HTTPException(status_code=400, detail=f"Image file not found: {p}")
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            raise HTTPException(status_code=400, detail=f"Could not decode image: {p}")
        images.append(img)
    return images


def _ndarray_to_base64_png(arr: np.ndarray) -> str:
    if arr.ndim == 4:
        arr = arr[0]
    if arr.shape[0] in (1, 3):
        arr = np.transpose(arr, (1, 2, 0))
    arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    if arr.shape[2] == 1:
        arr = arr.squeeze(2)
    pil_img = Image.fromarray(arr)
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _ndarray_map_to_base64(arr: np.ndarray) -> str:
    arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    pil_img = Image.fromarray(arr, mode="L")
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


@router.get("/status")
async def get_status():
    mem_used = 0
    if torch.cuda.is_available():
        mem_used = torch.cuda.memory_allocated() / (1024 * 1024)
        
    return {
        "status": "ok" if _pipeline else "unavailable",
        "device": str(_pipeline.device) if _pipeline else None,
        "model_loaded": _pipeline is not None,
        "gpu_memory_used_mb": mem_used,
        "total_inferences_count": _total_inferences,
        "last_inference_timestamp": _last_inference_time,
        "error": _load_error
    }


@router.get("/model")
async def get_model_info():
    import json
    config_path = Path("models/vision/model_config.json")
    metadata_path = Path("models/vision/training_metadata.json")
    
    info = {"architecture": "SolarVisionPredictor (Dual Head ResNet50 + Transformer)"}
    
    if config_path.exists():
        with open(config_path, "r") as f:
            info["config"] = json.load(f)
    if metadata_path.exists():
        with open(metadata_path, "r") as f:
            info["training_metadata"] = json.load(f)
            
    if _pipeline:
        info["parameter_count"] = _pipeline.model.get_num_parameters()
        
    return info


@router.post("/predict")
async def predict_future_image(request: PredictRequest):
    pipeline = _get_pipeline()
    try:
        raw_images = _load_images(request.image_paths)
        result = pipeline.predict(
            image_sequence=raw_images,
            telemetry=request.telemetry_data,
            physics=request.physics_data,
        )

        predicted_b64 = _ndarray_to_base64_png(result["predicted_image"])

        return {
            "status": "success",
            "flare_class": result["flare_class"],
            "flare_probability": round(result["flare_probability"], 6),
            "predicted_flux": result["predicted_flux"],
            "class_probabilities": result["class_probabilities"],
            "predicted_image_base64": predicted_b64,
        }

    except HTTPException:
        raise
    except Exception as exc:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/explain")
async def explain_prediction(request: ExplainRequest):
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
            "integrated_gradients_base64": _ndarray_map_to_base64(xai_maps["integrated_gradients"]),
        }

    except HTTPException:
        raise
    except Exception as exc:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/uncertainty")
async def get_uncertainty(request: UncertaintyRequest):
    pipeline = _get_pipeline()
    try:
        raw_images = _load_images(request.image_paths)
        result = pipeline.predict_with_uncertainty(
            image_sequence=raw_images,
            telemetry=request.telemetry_data,
            physics=request.physics_data,
        )

        return {
            "status": "success",
            "confidence": result["confidence"],
            "class_uncertainty": result["class_uncertainty"],
            "flux_uncertainty": result["flux_uncertainty"],
            "pixel_variance_map_base64": _ndarray_map_to_base64(result["pixel_variance_map"]),
        }

    except HTTPException:
        raise
    except Exception as exc:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(exc))
