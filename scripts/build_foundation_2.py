import os

def create_file(path, content):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"Created: {path}")

# --- 1. schemas/ingestion.py ---
create_file("shared/astronova_core/schemas/ingestion.py", """from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel, Field, field_validator

class SolexsDataPoint(BaseModel):
    time: datetime
    soft_xray_flux: float = Field(..., ge=0, description="Soft X-ray flux in W/m^2")
    hard_xray_flux: float = Field(..., ge=0, description="Hard X-ray flux in W/m^2")
    energy_band_lo: float = Field(..., ge=0)
    energy_band_hi: float = Field(..., ge=0)
    quality_flag: int = Field(default=0)

    @field_validator("time")
    @classmethod
    def validate_time(cls, v: datetime) -> datetime:
        if v > datetime.utcnow():
            raise ValueError("Observation time cannot be in the future")
        return v

class IngestionRequest(BaseModel):
    filepath: str
    format: str = Field(..., pattern="^(fits|cdf|csv|json)$")

class IngestionResponse(BaseModel):
    job_id: str
    status: str
    rows_ingested: int
    errors: Optional[List[str]] = None
""")

# --- 2. schemas/forecasting.py ---
create_file("shared/astronova_core/schemas/forecasting.py", """from datetime import datetime
from typing import List, Dict, Optional
from pydantic import BaseModel, Field

class ForecastRequest(BaseModel):
    lookback_minutes: int = Field(default=60, ge=10)
    horizons: List[int] = Field(default=[5, 15, 30, 60])
    model_type: str = Field(default="ensemble")

class ForecastResult(BaseModel):
    horizon_minutes: int
    probability: float = Field(..., ge=0.0, le=1.0)
    predicted_class: str
    peak_flux_estimate: float
    confidence_interval: List[float]

class NowcastResult(BaseModel):
    is_flare: bool
    goes_class: str
    confidence: float
    detection_method: str
    peak_flux: float

class SolarHazardIndex(BaseModel):
    score: float = Field(..., ge=0.0, le=1.0)
    category: str  # Safe, Moderate, High, Extreme
    components: Dict[str, float]
    timestamp: datetime
""")

# --- 3. schemas/impact.py ---
create_file("shared/astronova_core/schemas/impact.py", """from datetime import datetime
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
""")

# --- 4. exceptions.py ---
create_file("shared/astronova_core/exceptions.py", """from fastapi import status

class AstroNovaException(Exception):
    def __init__(self, message: str, status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR):
        super().__init__(message)
        self.message = message
        self.status_code = status_code

class DataIngestionError(AstroNovaException):
    def __init__(self, message: str):
        super().__init__(message, status.HTTP_400_BAD_REQUEST)

class DataValidationError(AstroNovaException):
    def __init__(self, message: str):
        super().__init__(message, status.HTTP_422_UNPROCESSABLE_ENTITY)

class DataProcessingError(AstroNovaException):
    def __init__(self, message: str):
        super().__init__(message, status.HTTP_500_INTERNAL_SERVER_ERROR)

class ModelNotFoundError(AstroNovaException):
    def __init__(self, message: str):
        super().__init__(message, status.HTTP_444_NOT_FOUND if hasattr(status, "HTTP_444_NOT_FOUND") else 404)

class InferenceError(AstroNovaException):
    def __init__(self, message: str):
        super().__init__(message, status.HTTP_500_INTERNAL_SERVER_ERROR)

class AuthenticationError(AstroNovaException):
    def __init__(self, message: str):
        super().__init__(message, status.HTTP_401_UNAUTHORIZED)

class AuthorizationError(AstroNovaException):
    def __init__(self, message: str):
        super().__init__(message, status.HTTP_403_FORBIDDEN)

class ServiceUnavailableError(AstroNovaException):
    def __init__(self, message: str):
        super().__init__(message, status.HTTP_503_SERVICE_UNAVAILABLE)

class RateLimitError(AstroNovaException):
    def __init__(self, message: str):
        super().__init__(message, status.HTTP_429_TOO_MANY_REQUESTS)
""")

# --- 5. middleware.py ---
create_file("shared/astronova_core/middleware.py", """import time
import uuid
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from astronova_core.logging import get_logger

logger = get_logger("gateway-middleware")

class CorrelationIDMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next) -> Response:
        correlation_id = request.headers.get("X-Correlation-ID", str(uuid.uuid4()))
        request.state.correlation_id = correlation_id
        response: Response = await call_next(request)
        response.headers["X-Correlation-ID"] = correlation_id
        return response

class RequestLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next) -> Response:
        start_time = time.time()
        response = await call_next(request)
        process_time = time.time() - start_time
        logger.info(
            "request_processed",
            method=request.method,
            path=request.url.path,
            status_code=response.status_code,
            duration_ms=round(process_time * 1000, 2)
        )
        return response
""")

# --- 6. metrics.py ---
create_file("shared/astronova_core/metrics.py", """from fastapi import APIRouter
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response

metrics_router = APIRouter()

REQUEST_COUNT = Counter(
    "astronova_requests_total",
    "Total HTTP Requests",
    ["service", "method", "endpoint", "http_status"]
)

REQUEST_LATENCY = Histogram(
    "astronova_request_latency_seconds",
    "HTTP Request Latency",
    ["service", "endpoint"]
)

PREDICTION_COUNT = Counter(
    "astronova_predictions_total",
    "Total Predictions Made",
    ["model_name", "goes_class"]
)

@metrics_router.get("/metrics")
def metrics():
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
""")

# --- 7. kafka_client.py ---
create_file("shared/astronova_core/kafka_client.py", """import json
from typing import Any, Dict
from confluent_kafka import Producer, Consumer
from astronova_core.config import get_settings
from astronova_core.logging import get_logger

settings = get_settings()
logger = get_logger("kafka-client")

class AstroNovaProducer:
    def __init__(self):
        conf = {
            'bootstrap.servers': settings.kafka.bootstrap_servers,
            'client.id': 'astronova-producer'
        }
        self.producer = Producer(conf)

    def send_message(self, topic: str, key: str, value: Dict[str, Any]) -> None:
        try:
            self.producer.produce(
                topic,
                key=key,
                value=json.dumps(value).encode('utf-8'),
                callback=self._delivery_report
            )
            self.producer.poll(0)
        except Exception as e:
            logger.error("failed_to_send_kafka_msg", error=str(e), topic=topic)

    def _delivery_report(self, err, msg):
        if err is not None:
            logger.error("kafka_delivery_failed", error=str(err))
        else:
            logger.debug("kafka_delivery_success", topic=msg.topic(), partition=msg.partition())

    def flush(self):
        self.producer.flush()
""")

# --- 8. cache.py ---
create_file("shared/astronova_core/cache.py", """import json
from typing import Any, Optional
import redis.asyncio as redis
from astronova_core.config import get_settings

settings = get_settings()

class RedisCache:
    def __init__(self):
        self.client = redis.from_url(settings.redis.redis_url, decode_responses=True)

    async def get(self, key: str) -> Optional[Any]:
        value = await self.client.get(key)
        if value:
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                return value
        return None

    async def set(self, key: str, value: Any, expire_seconds: int = 3600) -> None:
        await self.client.set(key, json.dumps(value), ex=expire_seconds)

    async def delete(self, key: str) -> None:
        await self.client.delete(key)
""")

# --- 9. utils/physics.py ---
create_file("shared/astronova_core/utils/physics.py", """from typing import Dict, Tuple

GOES_THRESHOLDS: Dict[str, Tuple[float, float]] = {
    'A': (1e-8, 1e-7),
    'B': (1e-7, 1e-6),
    'C': (1e-6, 1e-5),
    'M': (1e-5, 1e-4),
    'X': (1e-4, float('inf')),
}

def classify_flare(flux: float) -> str:
    for classification, (low, high) in GOES_THRESHOLDS.items():
        if low <= flux < high:
            # Calculate GOES sub-multiplier, e.g. M5.3, X1.2
            multiplier = flux / low
            return f"{classification}{multiplier:.1f}"
    if flux < 1e-8:
        return f"A0.0"
    return "X10.0"

def compute_xray_ratio(soft_flux: float, hard_flux: float) -> float:
    if soft_flux <= 0 or hard_flux <= 0:
        return 0.0
    return soft_flux / hard_flux
""")

# --- 10. utils/data_io.py ---
create_file("shared/astronova_core/utils/data_io.py", """import pandas as pd
import json

def read_csv_solexs(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    df['time'] = pd.to_datetime(df['time'])
    return df

def read_json_data(filepath: str) -> pd.DataFrame:
    with open(filepath, 'r') as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    df['time'] = pd.to_datetime(df['time'])
    return df
""")

# --- 11. utils/time_utils.py ---
create_file("shared/astronova_core/utils/time_utils.py", """from datetime import datetime
import pandas as pd

def to_utc(dt: datetime) -> datetime:
    if dt.tzinfo is not None:
        return dt.astimezone(pd.UTC).replace(tzinfo=None)
    return dt

def resample_timeseries(df: pd.DataFrame, freq: str = '1T') -> pd.DataFrame:
    df = df.set_index('time')
    df = df.resample(freq).mean().interpolate(method='linear')
    return df.reset_index()
""")

# --- 12. setup.py ---
create_file("shared/setup.py", """from setuptools import setup, find_packages
setup(
    name='astronova-core',
    version='1.0.0',
    packages=find_packages(),
    install_requires=[
        'fastapi>=0.115.0',
        'pydantic>=2.9.0',
        'sqlalchemy>=2.0.0',
        'asyncpg>=0.30.0',
        'redis>=5.2.0',
        'confluent-kafka>=2.6.0',
        'prometheus-client>=0.21.0',
        'python-jose[cryptography]>=3.3.0',
        'passlib[bcrypt]>=1.7.4',
        'pandas>=2.2.0',
        'numpy>=2.0.0',
        'structlog>=24.0.0',
    ]
)
""")

print("FOUNDATION PART 2 COMPLETE")
