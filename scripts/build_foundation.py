import os


def create_file(path, content):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"Created: {path}")

# --- 1. logging.py ---
create_file("shared/astronova_core/logging.py", """import logging
import sys
import json
import time
from typing import Any, Dict
import structlog

def setup_logging(service_name: str, log_level: str = "INFO") -> None:
    shared_processors = [
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
    ]

    structlog.configure(
        processors=shared_processors + [
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    formatter = structlog.stdlib.ProcessorFormatter(
        processor=structlog.processors.JSONRenderer(),
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.addHandler(handler)
    root_logger.setLevel(log_level)

def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    return structlog.get_logger(name)
""")

# --- 2. database.py ---
create_file("shared/astronova_core/database.py", """import os
from typing import AsyncGenerator
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import declarative_base
from astronova_core.config import get_settings

settings = get_settings()

engine = create_async_engine(
    settings.db.database_url,
    pool_pre_ping=True,
    echo=False
)

AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False
)

Base = declarative_base()

async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()

async def init_db() -> None:
    async with engine.begin() as conn:
        if settings.db.timescale_enabled:
            await conn.execute("CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;")
        # Tables will be managed via Alembic
""")

# --- 3. security.py ---
create_file("shared/astronova_core/security.py", """from datetime import datetime, timedelta
from typing import Optional, List, Union
from enum import Enum
from jose import jwt, JWTError
from passlib.context import CryptContext
from pydantic import BaseModel
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from astronova_core.config import get_settings
from astronova_core.exceptions import AuthenticationError, AuthorizationError

class UserRole(str, Enum):
    ADMIN = "admin"
    ANALYST = "analyst"
    OPERATOR = "operator"
    VIEWER = "viewer"
    SERVICE = "service"

class TokenData(BaseModel):
    username: Optional[str] = None
    role: Optional[UserRole] = None

settings = get_settings()
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/login", auto_error=False)

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=settings.jwt.access_token_expire_minutes))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, settings.jwt.jwt_secret_key, algorithm=settings.jwt.jwt_algorithm)

def verify_token(token: str) -> TokenData:
    try:
        payload = jwt.decode(token, settings.jwt.jwt_secret_key, algorithms=[settings.jwt.jwt_algorithm])
        username: str = payload.get("sub")
        role: str = payload.get("role")
        if username is None or role is None:
            raise AuthenticationError("Invalid token payload")
        return TokenData(username=username, role=UserRole(role))
    except JWTError:
        raise AuthenticationError("Could not validate credentials")

def get_current_user(token: str = Depends(oauth2_scheme)) -> TokenData:
    if not token:
        raise AuthenticationError("Token missing")
    return verify_token(token)

class RoleChecker:
    def __init__(self, allowed_roles: List[UserRole]):
        self.allowed_roles = allowed_roles

    def __call__(self, current_user: TokenData = Depends(get_current_user)) -> TokenData:
        if current_user.role not in self.allowed_roles:
            raise AuthorizationError("Access forbidden: Insufficient permissions")
        return current_user
""")

# --- 4. base.py ---
create_file("shared/astronova_core/models/base.py", """import uuid
from datetime import datetime
from sqlalchemy import Column, DateTime, String
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import declarative_mixin

@declarative_mixin
class UUIDMixin:
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

@declarative_mixin
class TimestampMixin:
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

@declarative_mixin
class SoftDeleteMixin:
    deleted_at = Column(DateTime, nullable=True)

    @property
    def is_deleted(self) -> bool:
        return self.deleted_at is not None
""")

# --- 5. timeseries.py ---
create_file("shared/astronova_core/models/timeseries.py", """from sqlalchemy import Column, DateTime, Float, String, Integer, Boolean
from astronova_core.database import Base

class SolexsObservation(Base):
    __tablename__ = "solexs_observations"

    time = Column(DateTime, primary_key=True)
    soft_xray_flux = Column(Float, nullable=False)
    hard_xray_flux = Column(Float, nullable=False)
    energy_band_lo = Column(Float, nullable=False)
    energy_band_hi = Column(Float, nullable=False)
    quality_flag = Column(Integer, default=0)
    source_file = Column(String(255), nullable=True)
    data_version = Column(String(50), default="1.0.0")

class ProcessedObservation(Base):
    __tablename__ = "processed_observations"

    time = Column(DateTime, primary_key=True)
    cleaned_soft_flux = Column(Float, nullable=False)
    cleaned_hard_flux = Column(Float, nullable=False)
    interpolated = Column(Boolean, default=False)
    outlier_removed = Column(Boolean, default=False)
    processing_pipeline_id = Column(String(100), nullable=False)
""")

# --- 6. events.py ---
create_file("shared/astronova_core/models/events.py", """from sqlalchemy import Column, DateTime, Float, String, Integer, Text
from sqlalchemy.dialects.postgresql import UUID
import uuid
from datetime import datetime
from astronova_core.database import Base

class FlareEvent(Base):
    __tablename__ = "flare_events"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    detected_at = Column(DateTime, default=datetime.utcnow)
    goes_class = Column(String(10), nullable=False)
    peak_flux = Column(Float, nullable=False)
    duration_seconds = Column(Integer, nullable=False)
    confidence = Column(Float, nullable=False)
    shi_score = Column(Float, nullable=False)

class Alert(Base):
    __tablename__ = "alerts"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    severity = Column(String(20), nullable=False)  # info, warning, critical, extreme
    title = Column(String(100), nullable=False)
    message = Column(Text, nullable=False)
    sent_at = Column(DateTime, default=datetime.utcnow)
    acknowledged_at = Column(DateTime, nullable=True)
    acknowledged_by = Column(String(100), nullable=True)

class AuditLog(Base):
    __tablename__ = "audit_logs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(String(100), nullable=False)
    action = Column(String(100), nullable=False)
    resource = Column(String(100), nullable=False)
    details = Column(Text, nullable=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    ip_address = Column(String(45), nullable=True)
""")

# --- 7. ml.py ---
create_file("shared/astronova_core/models/ml.py", """from sqlalchemy import Column, DateTime, Float, String, Integer, Text, JSON
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
""")

print("FOUNDATION MODULES WRITTEN SUCCESSFULLY")
