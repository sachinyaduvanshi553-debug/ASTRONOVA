"""Solar Vision Service entry point.
Provides FastAPI application that exposes vision prediction endpoints,
now using the full VisionInferencePipeline from services/vision.
"""
from fastapi import FastAPI
from astronova_core.logging import setup_logging

# Use the fully featured vision router from services/vision
from services.vision.api import router as vision_router

setup_logging("solar-vision-service")

app = FastAPI(
    title="Astronova Solar Vision Service",
    description="Generates future solar images using multimodal ConvLSTM + ResNet50 encoder, cross-modal fusion, U-Net decoder, and diffusion refinement.",
    version="1.0.0",
)

app.include_router(vision_router)

@app.get("/health")
async def health_check():
    return {"status": "ok", "service": "solar-vision", "version": "1.0.0"}
