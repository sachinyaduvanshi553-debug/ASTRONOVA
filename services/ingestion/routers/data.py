from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, desc
from astronova_core.database import get_db
from astronova_core.models.timeseries import SolexsObservation
from datetime import datetime, timedelta
from typing import List, Optional

router = APIRouter(prefix="/api/v1/data", tags=["data"])

@router.get("/observations")
async def get_observations(
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
    limit: int = Query(default=100, ge=1, le=1000),
    db: AsyncSession = Depends(get_db)
):
    if not start_time:
        start_time = datetime.utcnow() - timedelta(hours=24)
    if not end_time:
        end_time = datetime.utcnow()
        
    stmt = select(SolexsObservation).where(
        SolexsObservation.time >= start_time,
        SolexsObservation.time <= end_time
    ).order_by(desc(SolexsObservation.time)).limit(limit)
    
    result = await db.execute(stmt)
    observations = result.scalars().all()
    
    return [
        {
            "time": obs.time,
            "soft_xray_flux": obs.soft_xray_flux,
            "hard_xray_flux": obs.hard_xray_flux,
            "quality_flag": obs.quality_flag
        } for obs in observations
    ]

@router.get("/latest")
async def get_latest_observations(
    limit: int = Query(default=10, ge=1, le=100),
    db: AsyncSession = Depends(get_db)
):
    stmt = select(SolexsObservation).order_by(desc(SolexsObservation.time)).limit(limit)
    result = await db.execute(stmt)
    observations = result.scalars().all()
    
    return [
        {
            "time": obs.time,
            "soft_xray_flux": obs.soft_xray_flux,
            "hard_xray_flux": obs.hard_xray_flux,
            "quality_flag": obs.quality_flag
        } for obs in observations
    ]
