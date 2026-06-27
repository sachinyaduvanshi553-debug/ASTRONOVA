from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from services.ingestion.routers import data, ingest

from astronova_core.logging import setup_logging
from astronova_core.metrics import metrics_router

setup_logging("ingestion-service")

app = FastAPI(
    title="AstroNova Ingestion Service",
    description="Service for ingesting SoLEXS/HEL1OS space weather data.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(ingest.router)
app.include_router(data.router)
app.include_router(metrics_router)

@app.get("/")
def read_root():
    return {"message": "AstroNova Ingestion Service API v1"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "services.ingestion.main:app",
        host="0.0.0.0",
        port=8001,
        reload=True
    )
