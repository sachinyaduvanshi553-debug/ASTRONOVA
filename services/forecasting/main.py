from fastapi import FastAPI
from services.forecasting.routers import forecast
from astronova_core.logging import setup_logging
from astronova_core.metrics import metrics_router

setup_logging("forecasting-service")

app = FastAPI(
    title="AstroNova Forecasting Service",
    description="Service for nowcasting and multi-horizon solar flare forecasting.",
    version="1.0.0"
)

app.include_router(forecast.router)
app.include_router(metrics_router)

@app.get("/")
def read_root():
    return {"message": "AstroNova Forecasting Service API v1"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "services.forecasting.main:app",
        host="0.0.0.0",
        port=8004,
        reload=True
    )
