import os
import uuid

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from services.ingestion.models import IngestionJob
from services.ingestion.schemas import IngestionJobResponse
from services.ingestion.services.ingestion_service import IngestionService
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from astronova_core.database import get_db
from astronova_core.security import RoleChecker, UserRole

router = APIRouter(prefix="/api/v1/ingest", tags=["ingestion"])
ingest_service = IngestionService()
require_operator = RoleChecker([UserRole.OPERATOR, UserRole.ADMIN])

@router.post("/upload", response_model=IngestionJobResponse)
async def upload_file(
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
    current_user = Depends(require_operator)
):
    temp_dir = "/tmp/astronova_uploads"
    os.makedirs(temp_dir, exist_ok=True)
    file_path = os.path.join(temp_dir, f"{uuid.uuid4()}_{file.filename}")

    with open(file_path, "wb") as f:
        f.write(await file.read())

    file_format = file.filename.split(".")[-1]
    job = await ingest_service.ingest_file(file_path, file_format, db)
    return IngestionJobResponse(
        job_id=str(job.id),
        status=job.status,
        source_file=file.filename,
        rows_ingested=job.rows_ingested,
        started_at=job.started_at,
        completed_at=job.completed_at
    )

@router.get("/status/{job_id}", response_model=IngestionJobResponse)
async def get_job_status(job_id: str, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(IngestionJob).where(IngestionJob.id == uuid.UUID(job_id)))
    job = result.scalar_one_or_none()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return IngestionJobResponse(
        job_id=str(job.id),
        status=job.status,
        source_file=job.source_file,
        rows_ingested=job.rows_ingested,
        started_at=job.started_at,
        completed_at=job.completed_at
    )

@router.get("/health")
async def health_check():
    return {"status": "healthy", "service": "ingestion"}
