from fastapi import APIRouter, Query
from services.earth_impact.services.impact_calculator import EarthImpactCalculator

router = APIRouter(prefix="/api/v1/impact", tags=["earth-impact"])
calculator = EarthImpactCalculator()

@router.get("/assess")
async def assess_impact(goes_class: str = Query(..., description="Predicted/Current GOES class (e.g. M5.2, X1.0)")):
    return calculator.calculate_impact(goes_class)

@router.get("/health")
def health():
    return {"status": "healthy"}
