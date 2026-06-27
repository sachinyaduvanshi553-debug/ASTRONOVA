from fastapi import APIRouter, Body
from pydantic import BaseModel, Field
from typing import List
import torch

# Import model classes
from services.solar_vision.models.convlstm import ConvLSTM
from services.solar_vision.models.unet import UNet
from services.solar_vision.models.diffusion import DiffusionModel

router = APIRouter(prefix="/solar-vision", tags=["Solar Vision"])

class VisionInput(BaseModel):
    sdo_images: List[str] = Field(..., description="List of file paths or URLs to historical SDO images")
    goes_xray: List[float] = Field(..., description="Sequence of GOES X‑ray measurements")
    solexs: List[float] = Field(..., description="SOLEXS measurement series")
    noaa_regions: List[dict] = Field(..., description="Active‑region metadata from NOAA")

class VisionPrediction(BaseModel):
    timestamps: List[str] = Field(..., description="Future time points for predictions")
    convlstm_shape: List[int] = Field(..., description="Output tensor shape from ConvLSTM")
    unet_shape: List[int] = Field(..., description="Output tensor shape from UNet")
    diffusion_shape: List[int] = Field(..., description="Output tensor shape from Diffusion model")
    confidence: List[float] = Field(..., description="Model confidence for each time slot")

@router.post("/predict", response_model=VisionPrediction)
async def predict_vision(payload: VisionInput = Body(...)):
    """Run the three placeholder models on a dummy input and return output shapes.
    A real implementation would load actual data, preprocess, and run the models.
    """
    # Create dummy batch input (batch=1, seq_len=5, channels=3, H=64, W=64)
    dummy_input = torch.randn(1, 5, 3, 64, 64)

    # ConvLSTM
    convlstm = ConvLSTM(input_dim=3, hidden_dim=[16, 32], kernel_size=[3, 3], num_layers=2)
    convlstm_out = convlstm(dummy_input)

    # UNet (expects (batch, channels, H, W)) – use first timestep
    unet = UNet(in_channels=3, out_channels=3)
    unet_out = unet(dummy_input[:, 0])

    # Diffusion model – simple pass-through
    diffusion = DiffusionModel()
    diffusion_out = diffusion(dummy_input[:, 0])

    timestamps = ["+30m", "+1h", "+6h", "+12h"]
    confidence = [0.95, 0.90, 0.85, 0.80]
    return VisionPrediction(
        timestamps=timestamps,
        convlstm_shape=list(convlstm_out.shape),
        unet_shape=list(unet_out.shape),
        diffusion_shape=list(diffusion_out.shape),
        confidence=confidence,
    )
