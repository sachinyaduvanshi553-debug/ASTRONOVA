import pandas as pd
from fastapi import APIRouter, Depends
from services.features.engineering.physics_features import PhysicsFeatures
from services.features.engineering.time_domain import TimeDomainFeatures
from sqlalchemy import desc, select
from sqlalchemy.ext.asyncio import AsyncSession

from astronova_core.database import get_db
from astronova_core.models.timeseries import SolexsObservation

router = APIRouter(prefix="/api/v1/features", tags=["features"])

@router.post("/compute")
async def compute_features(db: AsyncSession = Depends(get_db)):
    # Load recent observations
    stmt = select(SolexsObservation).order_by(desc(SolexsObservation.time)).limit(100)
    result = await db.execute(stmt)
    observations = result.scalars().all()

    if not observations:
        return {"message": "No observations found to process features"}

    data = [{
        "time": obs.time,
        "soft_xray_flux": obs.soft_xray_flux,
        "hard_xray_flux": obs.hard_xray_flux
    } for obs in observations]

    df = pd.DataFrame(data).sort_values("time")

    # Compute features
    df = TimeDomainFeatures.compute_rolling_features(df)
    df = PhysicsFeatures.compute_physics_features(df)

    latest_feat = df.iloc[-1].to_dict()
    latest_feat["time"] = latest_feat["time"].isoformat()

    return {
        "status": "computed",
        "latest_features": latest_feat
    }

@router.get("/health")
def health():
    return {"status": "healthy"}
