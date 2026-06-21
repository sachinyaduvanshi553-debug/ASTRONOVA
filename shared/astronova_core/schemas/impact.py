from datetime import datetime
from typing import List, Dict, Optional
from pydantic import BaseModel

class RegionalRisk(BaseModel):
    region: str
    risk_score: float
    population_exposure: float
    infrastructure_risk: float

class SatelliteRisk(BaseModel):
    satellite_id: str
    orbit_type: str
    risk_score: float
    mitigation_actions: List[str]

class CommDisruptionRisk(BaseModel):
    system_type: str
    severity: str  # Low, Medium, High, Critical
    confidence: float
    affected_regions: List[str]

class EarthImpactResponse(BaseModel):
    overall_risk: float
    regions: List[RegionalRisk]
    satellites: List[SatelliteRisk]
    comms: List[CommDisruptionRisk]
    timestamp: datetime
