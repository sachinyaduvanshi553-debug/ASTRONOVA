from fastapi import FastAPI
from services.satellite_risk.routers import satellite
from astronova_core.logging import setup_logging
from astronova_core.metrics import metrics_router

setup_logging("satellite-risk-service")

app = FastAPI(
    title="AstroNova Satellite Risk Service",
    description="Service evaluating satellite drag, SEU, and TID risks.",
    version="1.0.0"
)

app.include_router(satellite.router)
app.include_router(metrics_router)

@app.get("/")
def read_root():
    return {"message": "AstroNova Satellite Risk Service API v1"}
