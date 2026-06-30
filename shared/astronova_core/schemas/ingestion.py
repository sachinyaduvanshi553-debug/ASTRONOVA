from datetime import datetime

from pydantic import BaseModel, Field, field_validator


class SolexsDataPoint(BaseModel):
    time: datetime
    soft_xray_flux: float = Field(..., ge=0, description="Soft X-ray flux in W/m^2")
    hard_xray_flux: float = Field(..., ge=0, description="Hard X-ray flux in W/m^2")
    energy_band_lo: float = Field(..., ge=0)
    energy_band_hi: float = Field(..., ge=0)
    quality_flag: int = Field(default=0)

    @field_validator("time")
    @classmethod
    def validate_time(cls, v: datetime) -> datetime:
        if v > datetime.utcnow():
            raise ValueError("Observation time cannot be in the future")
        return v

class IngestionRequest(BaseModel):
    filepath: str
    format: str = Field(..., pattern="^(fits|cdf|csv|json)$")

class IngestionResponse(BaseModel):
    job_id: str
    status: str
    rows_ingested: int
    errors: list[str] | None = None
