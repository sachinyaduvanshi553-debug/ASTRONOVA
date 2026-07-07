from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List
import torch
import numpy as np

from .inference import VisionInferencePipeline

router = APIRouter(prefix="/vision", tags=["vision"])

# Load dummy model for API (in production, use trained checkpoint)
try:
    pipeline = VisionInferencePipeline()
except Exception as e:
    pipeline = None
    print(f"Failed to load VisionInferencePipeline: {e}")

class PredictRequest(BaseModel):
    # In a real app, these might be IDs referencing data stored in DB or blob storage
    image_sequence_ids: List[str]
    telemetry_data: List[float]
    physics_data: List[float]

@router.post("/predict")
async def predict_future_image(request: PredictRequest):
    if not pipeline:
        raise HTTPException(status_code=500, detail="Vision model not loaded")
        
    try:
        # Dummy data conversion based on request
        # In reality, fetch actual data from IDs
        dummy_images = torch.randn(1, 2, 3, 256, 256)
        dummy_telemetry = torch.tensor([request.telemetry_data[:10]], dtype=torch.float32)
        dummy_physics = torch.tensor([request.physics_data[:5]], dtype=torch.float32)
        
        # Ensure dimensions match
        if dummy_telemetry.shape[1] < 10:
            dummy_telemetry = torch.cat([dummy_telemetry, torch.zeros(1, 10 - dummy_telemetry.shape[1])], dim=1)
        if dummy_physics.shape[1] < 5:
            dummy_physics = torch.cat([dummy_physics, torch.zeros(1, 5 - dummy_physics.shape[1])], dim=1)
            
        result = pipeline.predict(dummy_images, dummy_telemetry, dummy_physics)
        
        return {
            "status": "success",
            "confidence": result['confidence'],
            "flare_probability": result['flare_probability'],
            "message": "Prediction generated successfully (dummy image data not returned over JSON for size reasons)"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
