from fastapi import FastAPI
from services.comms_impact.routers import comms

from astronova_core.logging import setup_logging
from astronova_core.metrics import metrics_router

setup_logging("comms-impact-service")

app = FastAPI(
    title="AstroNova Comms Impact Service",
    description="Predicts ionospheric radio and GNSS/NavIC degradation.",
    version="1.0.0"
)

app.include_router(comms.router)
app.include_router(metrics_router)

@app.get("/")
def read_root():
    return {"message": "AstroNova Comms Impact Service API v1"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "services.comms_impact.main:app",
        host="0.0.0.0",
        port=8011,
        reload=True
    )
