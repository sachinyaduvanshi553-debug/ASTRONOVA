"""
Forecasting Router.
Provides production API endpoints for:
- POST /api/v1/forecast/predict (24x16 sequence flare prediction)
- POST /api/v1/forecast/forecast (Standard forecast endpoint)
- GET /api/v1/forecast/model-info (Model metadata, architecture, features)
- GET /api/v1/forecast/metrics (Benchmark test evaluation metrics)
- GET /api/v1/forecast/health (Health check)
"""

import json
import os
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, Body, Query
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from services.forecasting.services.inference_engine import AstroNovaInferenceEngine

router = APIRouter(prefix="/api/v1/forecast", tags=["forecasting"])
inference_engine = AstroNovaInferenceEngine()


class SequenceForecastRequest(BaseModel):
    sequence: Optional[List[List[float]]] = Field(
        None, description="24-hour sequence of 16 photospheric magnetic features (24 x 16)"
    )
    features: Optional[List[float]] = Field(
        None, description="Single observation features (fallback for 1x16)"
    )
    use_ensemble: bool = Field(True, description="Whether to use multi-model ensemble")
    include_xai: bool = Field(True, description="Whether to include XAI feature & temporal attributions")


@router.post("/predict")
async def predict_flare(request: SequenceForecastRequest = Body(...)):
    """
    Predicts >= M-class solar flare probability for the next 24 hours
    given a 24-hour sequence of 16 photospheric magnetic parameters.
    """
    if request.sequence is not None and len(request.sequence) > 0:
        seq = request.sequence
    elif request.features is not None and len(request.features) > 0:
        # Replicate across 24h window if single observation provided
        feat = request.features
        seq = [feat for _ in range(24)]
    else:
        # Default nominal active region sample
        seq = np.zeros((24, 16), dtype=np.float32).tolist()

    res = inference_engine.predict(
        sequence_data=seq,
        include_xai=request.include_xai,
        use_ensemble=request.use_ensemble,
    )
    return res


@router.post("/forecast")
async def forecast_endpoint(request: SequenceForecastRequest = Body(...)):
    """
    Alias endpoint for operational flare forecasting.
    """
    return await predict_flare(request)


@router.get("/model-info")
def get_model_info():
    """
    Returns active model architecture, dataset study window, feature set, and checkpoint status.
    """
    cfg_path = "checkpoints/model_config.json"
    cfg = {}
    if os.path.exists(cfg_path):
        with open(cfg_path, "r") as f:
            cfg = json.load(f)

    split_path = "checkpoints/split_metadata.json"
    split_meta = {}
    if os.path.exists(split_path):
        with open(split_path, "r") as f:
            split_meta = json.load(f)

    return {
        "model_name": "AstroNova Research-Grade Forecaster",
        "base_paper": "Prediction of Solar Flares Using Photospheric Magnetic Field Parameters with Deep Learning (Chaudhary et al., 2026)",
        "study_period": "May 2010 – May 2018 (Solar Cycle 24)",
        "dataset_sources": [
            "SDO/HMI SHARPs (hmi.sharp)",
            "cgem.Lorentz (Integrated Lorentz Forces)",
            "GOES X-ray Flare Catalog (NCEI/NOAA)",
        ],
        "input_shape": [24, 16],
        "prediction_horizon": "24 hours",
        "target": "Binary (1: >= M-Class Flare in next 24h, 0: Quiet / < M-Class)",
        "features": inference_engine.features,
        "feature_count": len(inference_engine.features),
        "locked_threshold": inference_engine.threshold,
        "calibrator_type": "Platt Scaling" if inference_engine.calibrator else "Raw",
        "active_models": list(inference_engine.ensemble.models.keys()),
        "ensemble_weights": inference_engine.ensemble.weights,
        "dataset_splits": split_meta,
        "config": cfg,
    }


@router.get("/metrics")
def get_metrics():
    """
    Returns evaluation metrics across Baseline Transformer, Optimized Transformer, BiLSTM, and Ensemble.
    """
    csv_path = "reports/benchmark_comparison.csv"
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        return {
            "status": "success",
            "benchmark_comparison": df.to_dict(orient="records"),
        }

    return {
        "status": "pending",
        "message": "Benchmark comparison is currently computing.",
    }


@router.get("/health")
def health():
    """
    Service health check endpoint.
    """
    return {
        "status": "healthy",
        "device": str(inference_engine.device),
        "models_loaded": len(inference_engine.ensemble.models) > 0,
        "feature_count": len(inference_engine.features),
        "locked_threshold": inference_engine.threshold,
    }
