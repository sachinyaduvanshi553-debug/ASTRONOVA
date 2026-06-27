from datetime import datetime

from pydantic import BaseModel, Field


class ForecastRequest(BaseModel):
    lookback_minutes: int = Field(default=60, ge=10)
    horizons: list[int] = Field(default=[5, 15, 30, 60])
    model_type: str = Field(default="ensemble")

class ForecastResult(BaseModel):
    horizon_minutes: int
    probability: float = Field(..., ge=0.0, le=1.0)
    predicted_class: str
    peak_flux_estimate: float
    confidence_interval: list[float]

class NowcastResult(BaseModel):
    is_flare: bool
    goes_class: str
    confidence: float
    detection_method: str
    peak_flux: float

class SolarHazardIndex(BaseModel):
    score: float = Field(..., ge=0.0, le=1.0)
    category: str  # Safe, Moderate, High, Extreme
    components: dict[str, float]
    timestamp: datetime
