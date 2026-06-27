from fastapi import FastAPI
from services.earth_impact.routers import impact

from astronova_core.logging import setup_logging
from astronova_core.metrics import metrics_router

setup_logging("earth-impact-service")

app = FastAPI(
    title="AstroNova Earth Impact Service",
    description="Service to predict ionospheric absorption and geomagnetic risks.",
    version="1.0.0"
)

app.include_router(impact.router)
app.include_router(metrics_router)

@app.get("/")
def read_root():
    return {"message": "AstroNova Earth Impact Service API v1"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "services.earth_impact.main:app",
        host="0.0.0.0",
        port=8006,
        reload=True
    )
