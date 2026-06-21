from sqlalchemy import Column, String, Integer, DateTime, Text, JSON
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
