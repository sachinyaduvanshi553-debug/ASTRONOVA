from fastapi import FastAPI
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
