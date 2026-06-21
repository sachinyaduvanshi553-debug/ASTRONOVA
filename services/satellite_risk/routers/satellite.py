from fastapi import APIRouter, Query
from services.satellite_risk.services.risk_calculator import SatelliteRiskCalculator

router = APIRouter(prefix="/api/v1/satellite", tags=["satellite-risk"])
calculator = SatelliteRiskCalculator()

@router.get("/assess")
async def assess_satellite_risk(goes_class: str = Query(..., description="Predicted/Current GOES class (e.g. M5.2, X1.0)")):
    return calculator.calculate_satellite_risk(goes_class)

@router.get("/health")
def health():
    return {"status": "healthy"}
