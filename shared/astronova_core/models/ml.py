from sqlalchemy import Column, DateTime, Float, String, Integer, Text, JSON
from sqlalchemy.dialects.postgresql import UUID
import uuid
from datetime import datetime
from astronova_core.database import Base

class MLModel(Base):
    __tablename__ = "ml_models"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(100), nullable=False)
    version = Column(String(50), nullable=False)
    framework = Column(String(50), nullable=False)
    metrics = Column(JSON, nullable=True)
    parameters = Column(JSON, nullable=True)
    mlflow_run_id = Column(String(100), nullable=True)
    status = Column(String(50), default="inactive")
    created_at = Column(DateTime, default=datetime.utcnow)

class Prediction(Base):
    __tablename__ = "predictions"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    model_id = Column(UUID(as_uuid=True), nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
    horizon_minutes = Column(Integer, nullable=False)
    predicted_probability = Column(Float, nullable=False)
    predicted_class = Column(String(10), nullable=False)
    confidence = Column(Float, nullable=False)
