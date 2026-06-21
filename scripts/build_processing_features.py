import os

def create_file(path, content):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"Created: {path}")

# =====================================================================
# PART 1: DATA PROCESSING SERVICE
# =====================================================================

# --- 1. requirements.txt ---
create_file("services/processing/requirements.txt", """fastapi>=0.115.0
uvicorn>=0.30.0
pydantic>=2.9.0
sqlalchemy>=2.0.0
asyncpg>=0.30.0
pandas>=2.2.0
numpy>=2.0.0
scipy>=1.13.0
confluent-kafka>=2.6.0
redis>=5.2.0
prometheus-client>=0.21.0
structlog>=24.0.0
astronova-core
""")

# --- 2. main.py ---
create_file("services/processing/main.py", """from fastapi import FastAPI
from astronova_core.logging import setup_logging
from astronova_core.metrics import metrics_router

setup_logging("processing-service")

app = FastAPI(
    title="AstroNova Processing Service",
    description="Data processing and cleaning pipelines.",
    version="1.0.0"
)

app.include_router(metrics_router)

@app.get("/")
def read_root():
    return {"message": "AstroNova Processing Service API v1"}

@app.get("/api/v1/process/health")
def health():
    return {"status": "healthy"}
""")

# --- 3. Dockerfile ---
create_file("services/processing/Dockerfile", """FROM python:3.12-slim as builder
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

FROM python:3.12-slim as runtime
WORKDIR /app
COPY --from=builder /root/.local /root/.local
COPY . .
ENV PATH=/root/.local/bin:$PATH
ENV PYTHONUNBUFFERED=1
EXPOSE 8002
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8002"]
""")

# --- 4. pipelines/base.py ---
create_file("services/processing/pipelines/base.py", """from abc import ABC, abstractmethod
import pandas as pd

class BasePipeline(ABC):
    @abstractmethod
    def fit(self, df: pd.DataFrame) -> 'BasePipeline':
        pass
        
    @abstractmethod
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        pass
        
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)
""")

# --- 5. pipelines/cleaning.py ---
create_file("services/processing/pipelines/cleaning.py", """import pandas as pd
import numpy as np
from services.processing.pipelines.base import BasePipeline

class DataCleaningPipeline(BasePipeline):
    def fit(self, df: pd.DataFrame) -> 'DataCleaningPipeline':
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        # Drop duplicate records
        df = df.drop_duplicates(subset=["time"])
        # Fill missing values
        df["soft_xray_flux"] = df["soft_xray_flux"].interpolate(method="linear").ffill().bfill()
        df["hard_xray_flux"] = df["hard_xray_flux"].interpolate(method="linear").ffill().bfill()
        # Handle outliers using IQR
        for col in ["soft_xray_flux", "hard_xray_flux"]:
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            df[col] = np.clip(df[col], lower_bound, upper_bound)
        return df
""")

# --- 6. pipelines/smoothing.py ---
create_file("services/processing/pipelines/smoothing.py", """import pandas as pd
from scipy.signal import savgol_filter
from services.processing.pipelines.base import BasePipeline

class SmoothingPipeline(BasePipeline):
    def __init__(self, window_length: int = 11, polyorder: int = 3):
        self.window_length = window_length
        self.polyorder = polyorder

    def fit(self, df: pd.DataFrame) -> 'SmoothingPipeline':
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        # Apply Savitzky-Golay filter to smooth fluxes
        if len(df) >= self.window_length:
            df["soft_xray_flux"] = savgol_filter(df["soft_xray_flux"], self.window_length, self.polyorder)
            df["hard_xray_flux"] = savgol_filter(df["hard_xray_flux"], self.window_length, self.polyorder)
        return df
""")


# =====================================================================
# PART 2: FEATURE ENGINEERING SERVICE
# =====================================================================

# --- 7. requirements.txt ---
create_file("services/features/requirements.txt", """fastapi>=0.115.0
uvicorn>=0.30.0
pydantic>=2.9.0
sqlalchemy>=2.0.0
asyncpg>=0.30.0
pandas>=2.2.0
numpy>=2.0.0
scipy>=1.13.0
redis>=5.2.0
prometheus-client>=0.21.0
structlog>=24.0.0
astronova-core
""")

# --- 8. main.py ---
create_file("services/features/main.py", """from fastapi import FastAPI
from services.features.routers import features
from astronova_core.logging import setup_logging
from astronova_core.metrics import metrics_router

setup_logging("features-service")

app = FastAPI(
    title="AstroNova Feature Service",
    description="Feature engineering service for forecasting models.",
    version="1.0.0"
)

app.include_router(features.router)
app.include_router(metrics_router)

@app.get("/")
def read_root():
    return {"message": "AstroNova Feature Service API v1"}
""")

# --- 9. Dockerfile ---
create_file("services/features/Dockerfile", """FROM python:3.12-slim as builder
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

FROM python:3.12-slim as runtime
WORKDIR /app
COPY --from=builder /root/.local /root/.local
COPY . .
ENV PATH=/root/.local/bin:$PATH
ENV PYTHONUNBUFFERED=1
EXPOSE 8003
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8003"]
""")

# --- 10. engineering/time_domain.py ---
create_file("services/features/engineering/time_domain.py", """import pandas as pd
import numpy as np

class TimeDomainFeatures:
    @staticmethod
    def compute_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        # Compute rolling mean and std
        for window in [5, 15, 30]:
            df[f"soft_flux_roll_mean_{window}"] = df["soft_xray_flux"].rolling(window=window, min_periods=1).mean()
            df[f"soft_flux_roll_std_{window}"] = df["soft_xray_flux"].rolling(window=window, min_periods=1).std().fillna(0)
            df[f"hard_flux_roll_mean_{window}"] = df["hard_xray_flux"].rolling(window=window, min_periods=1).mean()
            df[f"hard_flux_roll_std_{window}"] = df["hard_xray_flux"].rolling(window=window, min_periods=1).std().fillna(0)
        
        # Flux derivatives
        df["soft_flux_gradient"] = df["soft_xray_flux"].diff().fillna(0)
        df["hard_flux_gradient"] = df["hard_xray_flux"].diff().fillna(0)
        return df
""")

# --- 11. engineering/physics_features.py ---
create_file("services/features/engineering/physics_features.py", """import pandas as pd
from astronova_core.utils.physics import compute_xray_ratio

class PhysicsFeatures:
    @staticmethod
    def compute_physics_features(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        # Compute soft/hard x-ray ratio
        df["xray_ratio"] = df.apply(
            lambda row: compute_xray_ratio(row["soft_xray_flux"], row["hard_xray_flux"]), axis=1
        )
        return df
""")

# --- 12. routers/features.py ---
create_file("services/features/routers/features.py", """from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from astronova_core.database import get_db
from services.features.engineering.time_domain import TimeDomainFeatures
from services.features.engineering.physics_features import PhysicsFeatures
from astronova_core.models.timeseries import SolexsObservation
from sqlalchemy import select, desc
import pandas as pd

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
""")

print("PROCESSING & FEATURES SERVICES WRITTEN")
