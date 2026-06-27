'''Solar Vision Service entry point.
Provides FastAPI application that exposes vision prediction endpoints.
'''

from fastapi import FastAPI
from services.solar_vision.routers import vision

from astronova_core.logging import setup_logging

setup_logging("solar-vision-service")

app = FastAPI(
    title="Astronova Solar Vision Service",
    description="Generates future solar images using ConvLSTM, U‑Net and diffusion refinement.",
    version="0.1.0",
)

app.include_router(vision.router)

@app.get("/health")
async def health_check():
    return {"status": "ok"}
