from fastapi import FastAPI
from services.notifications.routers import alerts

from astronova_core.logging import setup_logging
from astronova_core.metrics import metrics_router

setup_logging("notification-service")

app = FastAPI(
    title="AstroNova Notification Service",
    description="Tiered alert routing and dashboard alerts delivery.",
    version="1.0.0"
)

app.include_router(alerts.router)
app.include_router(metrics_router)

@app.get("/")
def read_root():
    return {"message": "AstroNova Notification Service API v1"}
