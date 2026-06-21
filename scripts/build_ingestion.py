import os

def create_file(path, content):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"Created: {path}")

# --- 1. config.py ---
create_file("services/ingestion/config.py", """from astronova_core.config import get_settings

class IngestionConfig:
    def __init__(self):
        self.settings = get_settings()
        self.upload_dir = "/app/data/uploads"
        
ingestion_config = IngestionConfig()
""")

# --- 2. requirements.txt ---
create_file("services/ingestion/requirements.txt", """fastapi>=0.115.0
uvicorn>=0.30.0
pydantic>=2.9.0
sqlalchemy>=2.0.0
asyncpg>=0.30.0
pandas>=2.2.0
numpy>=2.0.0
confluent-kafka>=2.6.0
apscheduler>=3.10.0
redis>=5.2.0
prometheus-client>=0.21.0
python-jose[cryptography]>=3.3.0
passlib[bcrypt]>=1.7.4
structlog>=24.0.0
python-multipart>=0.0.9
astronova-core
""")

# --- 3. Dockerfile ---
create_file("services/ingestion/Dockerfile", """FROM python:3.12-slim as builder

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

FROM python:3.12-slim as runtime

WORKDIR /app

COPY --from=builder /root/.local /root/.local
COPY . .

ENV PATH=/root/.local/bin:$PATH
ENV PYTHONUNBUFFERED=1

EXPOSE 8001

HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8001/api/v1/ingest/health')" || exit 1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"]
""")

# --- 4. models.py ---
create_file("services/ingestion/models.py", """from sqlalchemy import Column, String, Integer, DateTime, Text, JSON
from sqlalchemy.dialects.postgresql import UUID
import uuid
from datetime import datetime
from astronova_core.database import Base

class IngestionJob(Base):
    __tablename__ = "ingestion_jobs"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    status = Column(String(50), nullable=False, default="pending")  # pending, processing, completed, failed
    source_file = Column(String(255), nullable=False)
    format = Column(String(50), nullable=False)
    rows_ingested = Column(Integer, default=0)
    errors = Column(JSON, nullable=True)
    started_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)
""")

# --- 5. schemas.py ---
create_file("services/ingestion/schemas.py", """from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional, List

class IngestionJobResponse(BaseModel):
    job_id: str
    status: str
    source_file: str
    rows_ingested: int
    started_at: datetime
    completed_at: Optional[datetime] = None

class BulkIngestionRequest(BaseModel):
    directory_path: str

class ScheduleRequest(BaseModel):
    cron_expression: str
""")

# --- 6. services/kafka_producer.py ---
create_file("services/ingestion/services/kafka_producer.py", """from astronova_core.kafka_client import AstroNovaProducer
import json
from typing import Dict, Any

class DataProducer:
    def __init__(self):
        self.producer = AstroNovaProducer()

    def publish_observation(self, key: str, data: Dict[str, Any]) -> None:
        self.producer.send_message("astronova.raw.solexs", key=key, value=data)

    def publish_ingestion_complete(self, job_id: str, stats: Dict[str, Any]) -> None:
        self.producer.send_message("astronova.events", key=job_id, value={
            "event_type": "ingestion_complete",
            "job_id": job_id,
            "stats": stats
        })
""")

# --- 7. services/ingestion_service.py ---
create_file("services/ingestion/services/ingestion_service.py", """import os
import pandas as pd
import uuid
from datetime import datetime
from sqlalchemy.ext.asyncio import AsyncSession
from services.ingestion.models import IngestionJob
from services.ingestion.services.kafka_producer import DataProducer
from astronova_core.models.timeseries import SolexsObservation
from astronova_core.logging import get_logger

logger = get_logger("ingestion-service")

class IngestionService:
    def __init__(self):
        self.producer = DataProducer()

    async def ingest_file(self, file_path: str, file_format: str, db: AsyncSession) -> IngestionJob:
        job = IngestionJob(
            id=uuid.uuid4(),
            status="processing",
            source_file=file_path,
            format=file_format,
            started_at=datetime.utcnow()
        )
        db.add(job)
        await db.commit()
        await db.refresh(job)

        try:
            # Read file based on format
            if file_format.lower() == "csv":
                df = pd.read_csv(file_path)
            elif file_format.lower() == "json":
                df = pd.read_json(file_path)
            else:
                raise ValueError(f"Unsupported format: {file_format}")

            # Validate columns
            required_cols = ["time", "soft_xray_flux", "hard_xray_flux"]
            for col in required_cols:
                if col not in df.columns:
                    raise ValueError(f"Missing required column: {col}")

            df["time"] = pd.to_datetime(df["time"])
            
            # Store in DB and publish to Kafka
            rows_stored = 0
            for _, row in df.iterrows():
                obs_time = row["time"].to_pydatetime()
                # Store raw observation
                db_obs = SolexsObservation(
                    time=obs_time,
                    soft_xray_flux=float(row["soft_xray_flux"]),
                    hard_xray_flux=float(row["hard_xray_flux"]),
                    energy_band_lo=1.0,
                    energy_band_hi=8.0,
                    quality_flag=int(row.get("quality_flag", 0)),
                    source_file=os.path.basename(file_path)
                )
                await db.merge(db_obs)
                
                # Publish raw data to Kafka topic for Processing Service
                self.producer.publish_observation(
                    key=obs_time.isoformat(),
                    data={
                        "time": obs_time.isoformat(),
                        "soft_xray_flux": float(row["soft_xray_flux"]),
                        "hard_xray_flux": float(row["hard_xray_flux"]),
                        "quality_flag": int(row.get("quality_flag", 0)),
                        "source_file": os.path.basename(file_path)
                    }
                )
                rows_stored += 1

            # Update job status
            job.status = "completed"
            job.rows_ingested = rows_stored
            job.completed_at = datetime.utcnow()
            await db.commit()
            
            # Publish ingestion complete event
            self.producer.publish_ingestion_complete(str(job.id), {"rows_ingested": rows_stored})
            logger.info("ingestion_job_success", job_id=str(job.id), rows=rows_stored)

        except Exception as e:
            logger.error("ingestion_job_failed", job_id=str(job.id), error=str(e))
            job.status = "failed"
            job.errors = [str(e)]
            job.completed_at = datetime.utcnow()
            await db.commit()

        return job
""")

# --- 8. services/scheduler.py ---
create_file("services/ingestion/services/scheduler.py", """from apscheduler.schedulers.asyncio import AsyncScheduler
from apscheduler.schedulers.background import BackgroundScheduler
from astronova_core.logging import get_logger

logger = get_logger("ingestion-scheduler")

class IngestionScheduler:
    def __init__(self):
        self.scheduler = BackgroundScheduler()

    def start(self):
        if not self.scheduler.running:
            self.scheduler.start()
            logger.info("scheduler_started")

    def stop(self):
        if self.scheduler.running:
            self.scheduler.shutdown()
            logger.info("scheduler_stopped")
""")

# --- 9. routers/ingest.py ---
create_file("services/ingestion/routers/ingest.py", """import os
import uuid
from fastapi import APIRouter, UploadFile, File, Depends, HTTPException, status, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
from services.ingestion.services.ingestion_service import IngestionService
from services.ingestion.schemas import IngestionJobResponse, BulkIngestionRequest
from services.ingestion.models import IngestionJob
from astronova_core.database import get_db
from astronova_core.security import get_current_user, UserRole, RoleChecker
from sqlalchemy import select

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
""")

# --- 10. routers/data.py ---
create_file("services/ingestion/routers/data.py", """from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, desc
from astronova_core.database import get_db
from astronova_core.models.timeseries import SolexsObservation
from datetime import datetime, timedelta
from typing import List

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
""")

# --- 11. main.py ---
create_file("services/ingestion/main.py", """import os
from fastapi import FastAPI, Depends
from services.ingestion.routers import ingest, data
from astronova_core.logging import setup_logging
from astronova_core.metrics import metrics_router
from fastapi.middleware.cors import CORSMiddleware

setup_logging("ingestion-service")

app = FastAPI(
    title="AstroNova Ingestion Service",
    description="Service for ingesting SoLEXS/HEL1OS space weather data.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(ingest.router)
app.include_router(data.router)
app.include_router(metrics_router)

@app.get("/")
def read_root():
    return {"message": "AstroNova Ingestion Service API v1"}
""")

print("INGESTION SERVICE WRITTEN SUCCESSFULLY")
