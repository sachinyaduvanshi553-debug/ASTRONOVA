from astronova_core.dynamic_logging import DynamicLoggingMiddleware
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from services.gateway.routers.dynamic_logging import router as dynamic_logging_router
from services.ml.inference import load_model, predict
from services.vision.api import router as vision_router

app = FastAPI(title="Solar Flare AI API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(DynamicLoggingMiddleware, service_name="main_api")

app.include_router(vision_router)
app.include_router(dynamic_logging_router)

model = None


# -----------------------------
# INPUT SCHEMA
# -----------------------------
class InputData(BaseModel):
    features: dict


# -----------------------------
# STARTUP
# -----------------------------
@app.on_event("startup")
def startup():
    global model
    try:
        model = load_model()
    except Exception as e:
        print(f"Model load warning: {e}")


# -----------------------------
# HEALTH CHECK
# -----------------------------
@app.get("/")
def home():
    return {"status": "Solar AI API Running 🚀"}


# -----------------------------
# PREDICTION ENDPOINT
# -----------------------------
@app.post("/predict")
def get_prediction(data: InputData):
    return predict(data.features)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("services.api.main:app", host="0.0.0.0", port=8013, reload=True)
