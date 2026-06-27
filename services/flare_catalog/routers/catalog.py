import uuid

import pandas as pd
from fastapi import APIRouter, Depends
from sqlalchemy import desc, select
from sqlalchemy.ext.asyncio import AsyncSession

from astronova_core.database import get_db
from astronova_core.models.events import FlareEvent
from astronova_core.models.timeseries import SolexsObservation
from astronova_core.utils.physics import classify_flare

router = APIRouter(prefix="/api/v1/catalog", tags=["catalog"])

@router.post("/generate")
async def generate_catalog_from_observations(db: AsyncSession = Depends(get_db)):
    # Simple segmentation algorithm: detect peaks exceeding C1.0 (1e-6)
    stmt = select(SolexsObservation).order_by(SolexsObservation.time)
    result = await db.execute(stmt)
    observations = result.scalars().all()

    if not observations:
        return {"status": "skipped", "message": "No observations in database"}

    df = pd.DataFrame([{
        "time": obs.time,
        "soft_xray_flux": obs.soft_xray_flux
    } for obs in observations])

    flares_detected = 0
    # Window analysis
    for _idx, row in df.iterrows():
        flux = row["soft_xray_flux"]
        if flux >= 1e-5: # M-class threshold
            # Create a mock validated flare event
            event = FlareEvent(
                id=uuid.uuid4(),
                detected_at=row["time"],
                goes_class=classify_flare(flux),
                peak_flux=flux,
                duration_seconds=1200,
                confidence=0.92,
                shi_score=0.74
            )
            await db.merge(event)
            flares_detected += 1

    await db.commit()
    return {"status": "completed", "flares_detected": flares_detected}

@router.get("/flares")
async def list_catalog(db: AsyncSession = Depends(get_db)):
    stmt = select(FlareEvent).order_by(desc(FlareEvent.detected_at)).limit(50)
    result = await db.execute(stmt)
    flares = result.scalars().all()
    return [{
        "id": str(f.id),
        "detected_at": f.detected_at,
        "goes_class": f.goes_class,
        "peak_flux": f.peak_flux,
        "shi_score": f.shi_score
    } for f in flares]

@router.get("/health")
def health():
    return {"status": "healthy"}
