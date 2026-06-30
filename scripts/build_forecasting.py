import os


def create_file(path, content):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"Created: {path}")

# =====================================================================
# PART 1: ML MODELS
# =====================================================================

# --- 1. bilstm.py ---
create_file("ml/models/bilstm.py", """import torch
import torch.nn as nn
from typing import Dict, Any

class BiLSTMForecaster(nn.Module):
    def __init__(self, input_size: int = 10, hidden_size: int = 64, num_layers: int = 2, num_classes: int = 5):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )
        self.fc = nn.Linear(hidden_size * 2, num_classes)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch_size, seq_len, input_size]
        out, _ = self.lstm(x)
        # Take the last time step output
        out = out[:, -1, :]
        out = self.fc(out)
        return self.softmax(out)
""")

# --- 2. gru_model.py ---
create_file("ml/models/gru_model.py", """import torch
import torch.nn as nn

class GRUForecaster(nn.Module):
    def __init__(self, input_size: int = 10, hidden_size: int = 64, num_layers: int = 2, num_classes: int = 5):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_size, num_classes)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return self.softmax(out)
""")

# --- 3. transformer.py ---
create_file("ml/models/transformer.py", """import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]

class SolarTransformer(nn.Module):
    def __init__(self, input_size: int = 10, d_model: int = 64, nhead: int = 4, num_layers: int = 2, num_classes: int = 5):
        super().__init__()
        self.input_projection = nn.Linear(input_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, num_classes)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_projection(x)
        x = self.pos_encoder(x)
        out = self.transformer_encoder(x)
        out = out.mean(dim=1)  # Global average pooling
        out = self.fc(out)
        return self.softmax(out)
""")

# --- 4. nowcasting.py ---
create_file("ml/models/nowcasting.py", """import numpy as np

class ThresholdDetector:
    def __init__(self, threshold: float = 1e-5):
        self.threshold = threshold

    def detect(self, current_flux: float) -> bool:
        return current_flux >= self.threshold
""")


# =====================================================================
# PART 2: FORECASTING SERVICE
# =====================================================================

# --- 5. requirements.txt ---
create_file("services/forecasting/requirements.txt", """fastapi>=0.115.0
uvicorn>=0.30.0
pydantic>=2.9.0
sqlalchemy>=2.0.0
asyncpg>=0.30.0
torch>=2.0.0
pandas>=2.2.0
numpy>=2.0.0
redis>=5.2.0
prometheus-client>=0.21.0
structlog>=24.0.0
astronova-core
""")

# --- 6. main.py ---
create_file("services/forecasting/main.py", """from fastapi import FastAPI
from services.forecasting.routers import forecast
from astronova_core.logging import setup_logging
from astronova_core.metrics import metrics_router

setup_logging("forecasting-service")

app = FastAPI(
    title="AstroNova Forecasting Service",
    description="Service for nowcasting and multi-horizon solar flare forecasting.",
    version="1.0.0"
)

app.include_router(forecast.router)
app.include_router(metrics_router)

@app.get("/")
def read_root():
    return {"message": "AstroNova Forecasting Service API v1"}
""")

# --- 7. Dockerfile ---
create_file("services/forecasting/Dockerfile", """FROM python:3.12-slim as builder
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

FROM python:3.12-slim as runtime
WORKDIR /app
COPY --from=builder /root/.local /root/.local
COPY . .
ENV PATH=/root/.local/bin:$PATH
ENV PYTHONUNBUFFERED=1
EXPOSE 8004
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8004"]
""")

# --- 8. services/inference_engine.py ---
create_file("services/forecasting/services/inference_engine.py", """import torch
import numpy as np
from ml.models.bilstm import BiLSTMForecaster
from typing import Dict, Any

class InferenceEngine:
    def __init__(self):
        # Initialize standard model with random weights for demo/production fallback
        self.model = BiLSTMForecaster(input_size=4, hidden_size=32, num_layers=1, num_classes=5)
        self.model.eval()

    def predict(self, features: list) -> Dict[str, Any]:
        # Mock feature parsing: features should shape [batch, seq_len, num_features]
        # Here we just generate output classification probabilities
        # A, B, C, M, X flare probability
        with torch.no_grad():
            x = torch.randn(1, 10, 4)
            probs = self.model(x).squeeze().tolist()

        classes = ["A", "B", "C", "M", "X"]
        pred_class_idx = np.argmax(probs)

        return {
            "prediction": {
                "horizon_minutes": 30,
                "probabilities": dict(zip(classes, probs)),
                "predicted_class": classes[pred_class_idx],
                "confidence": float(probs[pred_class_idx])
            }
        }
""")

# --- 9. services/nowcasting.py ---
create_file("services/forecasting/services/nowcasting.py", """from ml.models.nowcasting import ThresholdDetector
from astronova_core.utils.physics import classify_flare

class NowcastingService:
    def __init__(self):
        self.detector = ThresholdDetector(threshold=1e-5) # M-class threshold

    def analyze_nowcast(self, current_flux: float) -> dict:
        is_flare = self.detector.detect(current_flux)
        goes_class = classify_flare(current_flux)

        return {
            "is_flare": is_flare,
            "goes_class": goes_class,
            "confidence": 0.95 if is_flare else 0.99,
            "detection_method": "threshold_detector",
            "peak_flux": current_flux
        }
""")

# --- 10. services/solar_hazard_index.py ---
create_file("services/forecasting/services/solar_hazard_index.py", """class SolarHazardIndexCalculator:
    @staticmethod
    def calculate_shi(probabilities: dict, gradient: float) -> dict:
        # Weighted sum: 60% X-class prob, 30% M-class prob, 10% gradient factor
        x_prob = probabilities.get("X", 0.0)
        m_prob = probabilities.get("M", 0.0)

        raw_score = (x_prob * 0.6) + (m_prob * 0.3) + min(abs(gradient) * 1e5, 0.1)
        score = min(max(raw_score, 0.0), 1.0)

        if score < 0.2:
            category = "Safe"
        elif score < 0.5:
            category = "Moderate"
        elif score < 0.8:
            category = "High"
        else:
            category = "Extreme"

        return {
            "score": score,
            "category": category,
            "components": {
                "x_flare_probability": x_prob,
                "m_flare_probability": m_prob,
                "flux_gradient_factor": min(abs(gradient) * 1e5, 0.1)
            }
        }
""")

# --- 11. routers/forecast.py ---
create_file("services/forecasting/routers/forecast.py", """from fastapi import APIRouter, HTTPException, Query
from services.forecasting.services.inference_engine import InferenceEngine
from services.forecasting.services.nowcasting import NowcastingService
from services.forecasting.services.solar_hazard_index import SolarHazardIndexCalculator
from astronova_core.schemas.forecasting import ForecastRequest

router = APIRouter(prefix="/api/v1/forecast", tags=["forecasting"])
inference_engine = InferenceEngine()
nowcast_service = NowcastingService()

@router.post("/predict")
async def get_prediction(request: ForecastRequest):
    # Retrieve model predictions
    res = inference_engine.predict([])
    return res

@router.get("/nowcast")
async def get_nowcast(current_flux: float = Query(..., description="Current observed flux (W/m^2)")):
    res = nowcast_service.analyze_nowcast(current_flux)
    return res

@router.get("/shi")
async def get_shi(current_flux: float = Query(..., description="Current observed flux (W/m^2)")):
    res = nowcast_service.analyze_nowcast(current_flux)
    pred_res = inference_engine.predict([])

    # Calculate gradient proxy
    gradient = current_flux * 0.05
    shi = SolarHazardIndexCalculator.calculate_shi(
        pred_res["prediction"]["probabilities"],
        gradient
    )
    return shi

@router.get("/health")
def health():
    return {"status": "healthy"}
""")

print("FORECASTING MODULES AND SERVICE WRITTEN")
