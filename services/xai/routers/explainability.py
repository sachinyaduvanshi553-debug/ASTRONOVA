"""
XAI Router for AstroNova.
Provides API endpoints for:
- POST /api/v1/xai/explain (Live Integrated Gradients and SHAP attributions for input sequence)
- GET /api/v1/xai/explain (Precomputed global XAI analysis: SHAP ranking, PDP 2D surface, waterfall)
- GET /api/v1/xai/pdp (1D and 2D Partial Dependence Plot coordinates)
- GET /api/v1/xai/health (Health check)
"""

import json
import os
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, Body
import numpy as np
from pydantic import BaseModel, Field

from services.forecasting.services.inference_engine import AstroNovaInferenceEngine

router = APIRouter(prefix="/api/v1/xai", tags=["xai"])
inference_engine = AstroNovaInferenceEngine()


class ExplainRequest(BaseModel):
    sequence: Optional[List[List[float]]] = Field(
        None, description="24-hour sequence of 16 features (24 x 16)"
    )


@router.post("/explain")
async def explain_live_prediction(request: ExplainRequest = Body(...)):
    """
    Computes real-time Integrated Gradients spatiotemporal attribution heatmap
    and feature importance ranking for the provided sequence.
    """
    if request.sequence is not None and len(request.sequence) > 0:
        seq = request.sequence
    else:
        seq = np.zeros((24, 16), dtype=np.float32).tolist()

    res = inference_engine.predict(sequence_data=seq, include_xai=True)
    return {
        "prediction_summary": {
            "flare_probability": res["flare_probability"],
            "prediction": res["prediction"],
            "classification": res["classification"],
            "confidence": res["confidence"],
        },
        "top_contributing_features": res.get("important_features", []),
        "temporal_profile": res.get("temporal_contributions", []),
        "attribution_matrix": res.get("spatiotemporal_attribution", []),
        "feature_names": inference_engine.features,
    }


@router.get("/explain")
def get_global_explanations():
    """
    Returns global SHAP rankings, PDP interaction grids, and sample waterfalls.
    """
    xai_path = "reports/xai_analysis.json"
    if os.path.exists(xai_path):
        with open(xai_path, "r") as f:
            data = json.load(f)
        return {
            "status": "success",
            "global_shap_importance": data.get("global_importance", []),
            "positive_sample_waterfall": data.get("pos_waterfall", {}),
            "negative_sample_waterfall": data.get("neg_waterfall", {}),
            "pdp_2d_totusjh_meanjzh": data.get("pdp_2d", {}),
        }

    return {
        "status": "pending",
        "message": "XAI global analysis is currently computing.",
    }


@router.get("/pdp")
def get_pdp():
    """
    Returns 1D and 2D Partial Dependence data for TOTUSJH and MEANJZH.
    """
    xai_path = "reports/xai_analysis.json"
    if os.path.exists(xai_path):
        with open(xai_path, "r") as f:
            data = json.load(f)
        return {
            "status": "success",
            "pdp_1d_totusjh": data.get("pdp_1d_totusjh", {}),
            "pdp_2d_totusjh_meanjzh": data.get("pdp_2d", {}),
        }

    return {"status": "pending", "message": "PDP computations in progress."}


@router.get("/health")
def health():
    return {
        "status": "healthy",
        "explainer_loaded": hasattr(inference_engine, "ig_explainer"),
        "feature_count": len(inference_engine.features),
    }
