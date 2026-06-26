from fastapi import FastAPI
from services.features.routers import features
from astronova_core.logging import setup_logging
from astronova_core.metrics import metrics_router

setup_logging("features-service")

app = FastAPI(
    title="AstroNova Feature Service",
    description="Feature engineering service for forecasting models.",
    version="1.0.0"
)

app.include_router(features.router)
app.include_router(metrics_router)

@app.get("/")
def read_root():
    return {"message": "AstroNova Feature Service API v1"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "services.features.main:app",
        host="0.0.0.0",
        port=8003,
        reload=True
    )
