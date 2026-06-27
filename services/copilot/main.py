from fastapi import FastAPI
from services.copilot.routers import copilot

from astronova_core.logging import setup_logging
from astronova_core.metrics import metrics_router

setup_logging("copilot-service")

app = FastAPI(
    title="AstroNova LLM Copilot Service",
    description="Grounded AI Copilot for space weather operations.",
    version="1.0.0"
)

app.include_router(copilot.router)
app.include_router(metrics_router)

@app.get("/")
def read_root():
    return {"message": "AstroNova LLM Copilot Service API v1"}
