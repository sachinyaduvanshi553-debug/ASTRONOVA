"""Physics Engine Service entry point.
Provides FastAPI application exposing physics-informed calculations.
"""

from fastapi import FastAPI

from services.physics_engine.routers import physics
from shared.astronova_core.logging import setup_logging

setup_logging("physics-engine-service")

app = FastAPI(
    title="Astronova Physics Engine Service",
    description="Provides physics-informed metrics for solar flare forecasting.",
    version="0.1.0",
)

app.include_router(physics.router)


@app.get("/health")
async def health_check():
    return {"status": "ok"}
