from fastapi import FastAPI
from services.xai.routers import explainability
from astronova_core.logging import setup_logging
from astronova_core.metrics import metrics_router

setup_logging("xai-service")

app = FastAPI(
    title="AstroNova XAI Service",
    description="Explainable AI service detailing model predictions.",
    version="1.0.0"
)

app.include_router(explainability.router)
app.include_router(metrics_router)

@app.get("/")
def read_root():
    return {"message": "AstroNova XAI Service API v1"}
