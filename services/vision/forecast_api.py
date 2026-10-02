"""
services/vision/forecast_api.py — ASTRONOVA Phase 14: FastAPI Vision Forecast Endpoint

Provides the /api/v1/vision/forecast endpoint for multimodal spatiotemporal
solar flare forecasting using the trained SolarImageForecaster model.

SCIENTIFIC RULES:
    - All generated images labeled: image_type = "AI_FORECAST"
    - All heatmaps labeled: "AI Attention / Model Explanation"
    - Disclaimer included in every response
    - Uncertainty estimates included when MC-Dropout available

Endpoint:
    POST /api/v1/vision/forecast
"""
from __future__ import annotations

import base64
import io
import json
import logging
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel

logger = logging.getLogger("astronova.vision.forecast_api")

router = APIRouter(prefix="/api/v1/vision", tags=["vision-forecast"])

# Global model reference
_forecaster = None
_load_error = None
_device = None

PROJECT_ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def _load_forecaster():
    """Lazy-load the SolarImageForecaster from checkpoint."""
    global _forecaster, _load_error, _device
    import sys
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    try:
        from models.vision.solar_image_forecaster import SolarImageForecaster

        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ckpt_path = PROJECT_ROOT / "checkpoints" / "vision" / "best_roc_auc.pt"

        if not ckpt_path.exists():
            # Try best_val_loss
            ckpt_path = PROJECT_ROOT / "checkpoints" / "vision" / "best_val_loss.pt"
        if not ckpt_path.exists():
            ckpt_path = PROJECT_ROOT / "checkpoints" / "vision" / "last.pt"

        if not ckpt_path.exists():
            _load_error = f"No checkpoint found at {ckpt_path.parent}"
            logger.warning("Vision forecast model not loaded: %s", _load_error)
            return

        ckpt = torch.load(ckpt_path, map_location=_device, weights_only=False)
        config = ckpt.get("config", {})
        model_config = ckpt.get("model_config", {})

        model = SolarImageForecaster(
            backbone=model_config.get("backbone", "resnet18"),
            image_size=model_config.get("image_size", config.get("model", {}).get("image_size", 256)),
            spatial_dim=config.get("model", {}).get("spatial_dim", 256),
            temporal_dim=config.get("model", {}).get("temporal_dim", 256),
            n_heads=config.get("model", {}).get("n_heads", 4),
            n_temporal_layers=config.get("model", {}).get("n_transformer_layers", 2),
        ).to(_device)

        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        _forecaster = model
        logger.info("Vision forecast model loaded from %s (epoch %d)", ckpt_path, ckpt.get("epoch", -1))
    except Exception as e:
        _load_error = f"{type(e).__name__}: {e}"
        logger.warning("Failed to load vision forecaster: %s", _load_error)


# Try loading on import
_load_forecaster()


# ---------------------------------------------------------------------------
# Request/Response models
# ---------------------------------------------------------------------------
class ForecastRequest(BaseModel):
    """Request body for the vision forecast endpoint."""
    image_paths: list[str] = []
    telemetry: list[list[float]] = []   # [[soft_flux, hard_flux], ...]
    magnetic_features: list[list[float]] = []  # [[16 features], ...]
    image_size: int = 256
    use_uncertainty: bool = False
    mc_passes: int = 20


class HorizonForecast(BaseModel):
    """Per-horizon prediction."""
    horizon: str
    horizon_display: str
    probability_C_plus: float
    probability_M_plus: float
    probability_X_plus: float
    predicted_class: str
    uncertainty_C_plus: float | None = None
    uncertainty_M_plus: float | None = None
    uncertainty_X_plus: float | None = None


class ForecastResponse(BaseModel):
    """Response body for the vision forecast endpoint."""
    status: str
    timestamp: str
    horizons: list[HorizonForecast]
    predicted_image_base64: str | None = None
    flare_heatmap_base64: str | None = None
    image_type: str = "AI_FORECAST"
    scientific_disclaimer: str = (
        "Predicted images are AI FORECAST VISUALIZATIONS, NOT actual solar observations. "
        "Classification probabilities are model estimates, not certainties. "
        "Always verify with operational NOAA/SWPC forecasts."
    )
    model_info: dict = {}


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@router.get("/health")
async def health():
    return {
        "status": "ok" if _forecaster is not None else "degraded",
        "model_loaded": _forecaster is not None,
        "load_error": _load_error,
        "device": str(_device) if _device else "unknown",
    }


@router.post("/forecast", response_model=ForecastResponse)
async def forecast(request: ForecastRequest):
    """
    Run multimodal solar flare forecast.
    
    Accepts historical solar image paths and GOES telemetry.
    Returns per-horizon flare probabilities, spatial heatmap,
    and predicted future solar image (AI FORECAST).
    """
    if _forecaster is None:
        raise HTTPException(
            status_code=503,
            detail=f"Vision forecast model not loaded. {_load_error or 'No checkpoint found.'}",
        )

    try:
        import sys
        if str(PROJECT_ROOT) not in sys.path:
            sys.path.insert(0, str(PROJECT_ROOT))

        from services.vision.preprocessing.image_loader import load_image
        from models.vision.solar_image_forecaster import HORIZONS

        HORIZON_DISPLAY = {
            "h15m": "+15 min", "h30m": "+30 min", "h60m": "+60 min",
            "h360m": "+6 hours", "h720m": "+12 hours", "h1440m": "+24 hours",
        }

        device = _device
        image_size = request.image_size

        # 1. Load images
        images = []
        for p in request.image_paths:
            img = load_image(p, expected_size=image_size, channels=3, missing_ok=True)
            if img is None:
                img = torch.zeros(3, image_size, image_size)
            else:
                if img.shape[-1] != image_size or img.shape[-2] != image_size:
                    img = torch.nn.functional.interpolate(
                        img.unsqueeze(0), size=(image_size, image_size),
                        mode="bilinear", align_corners=False
                    ).squeeze(0)
            images.append(img)

        if not images:
            images = [torch.zeros(3, image_size, image_size)]

        image_seq = torch.stack(images).unsqueeze(0).to(device)  # [1, T, C, H, W]

        # 2. Telemetry
        if request.telemetry:
            tel = torch.tensor(request.telemetry, dtype=torch.float32).unsqueeze(0).to(device)
        else:
            T = image_seq.shape[1]
            tel = torch.zeros(1, T, 2, device=device)

        # 3. Inference
        if request.use_uncertainty:
            output = _forecaster.predict_with_uncertainty(
                image_seq, tel, n_passes=request.mc_passes
            )
            std_probs = output.get("class_std", {})
        else:
            with torch.no_grad():
                output = _forecaster(image_seq, tel)
            std_probs = None

        # 4. Build response
        horizons = []
        for h in HORIZONS:
            probs = output["class_probs"][h][0].cpu().numpy()

            # Determine predicted class
            if probs[2] >= 0.5:
                pred_class = "X"
            elif probs[1] >= 0.5:
                pred_class = "M"
            elif probs[0] >= 0.5:
                pred_class = "C"
            else:
                pred_class = "quiet"

            hf = HorizonForecast(
                horizon=h,
                horizon_display=HORIZON_DISPLAY.get(h, h),
                probability_C_plus=float(probs[0]),
                probability_M_plus=float(probs[1]),
                probability_X_plus=float(probs[2]),
                predicted_class=pred_class,
            )

            if std_probs and h in std_probs:
                std = std_probs[h][0].cpu().numpy()
                hf.uncertainty_C_plus = float(std[0])
                hf.uncertainty_M_plus = float(std[1])
                hf.uncertainty_X_plus = float(std[2])

            horizons.append(hf)

        # 5. Encode predicted image as base64
        pred_img_b64 = None
        if "predicted_image" in output:
            pred_img = output["predicted_image"][0].cpu().numpy()
            pred_img = (pred_img.transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)
            try:
                from PIL import Image as PILImage
                pil_img = PILImage.fromarray(pred_img)
                buf = io.BytesIO()
                pil_img.save(buf, format="PNG")
                pred_img_b64 = base64.b64encode(buf.getvalue()).decode()
            except ImportError:
                pass

        # 6. Encode heatmap as base64
        heatmap_b64 = None
        if "flare_heatmap" in output:
            heatmap = output["flare_heatmap"][0, 0].cpu().numpy()
            heatmap_uint8 = (heatmap * 255).clip(0, 255).astype(np.uint8)
            try:
                from PIL import Image as PILImage
                pil_hm = PILImage.fromarray(heatmap_uint8, mode="L")
                buf = io.BytesIO()
                pil_hm.save(buf, format="PNG")
                heatmap_b64 = base64.b64encode(buf.getvalue()).decode()
            except ImportError:
                pass

        return ForecastResponse(
            status="success",
            timestamp=datetime.utcnow().isoformat() + "Z",
            horizons=horizons,
            predicted_image_base64=pred_img_b64,
            flare_heatmap_base64=heatmap_b64,
            image_type="AI_FORECAST",
            model_info=_forecaster.get_config() if hasattr(_forecaster, "get_config") else {},
        )

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/forecast/upload")
async def forecast_upload(file: UploadFile = File(...)):
    """
    Upload a solar image and get a forecast.
    Wraps the forecast endpoint for single-image upload.
    """
    if _forecaster is None:
        raise HTTPException(status_code=503, detail="Vision forecast model not loaded.")

    try:
        import tempfile
        content = await file.read()
        suffix = Path(file.filename or "image.jpg").suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(content)
            tmp_path = tmp.name

        request = ForecastRequest(image_paths=[tmp_path])
        result = await forecast(request)

        # Clean up
        Path(tmp_path).unlink(missing_ok=True)
        return result
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
