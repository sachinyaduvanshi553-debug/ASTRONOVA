from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from services.gateway.routers import auth, proxy, dynamic_logging
from services.vision.api import router as vision_router

from astronova_core.logging import setup_logging
from astronova_core.metrics import metrics_router
from astronova_core.dynamic_logging import DynamicLoggingMiddleware

setup_logging("gateway")

app = FastAPI(
    title="AstroNova API Gateway",
    description="Central secure entrypoint routing API traffic with Dynamic Logging & Dynamic Database capabilities.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(DynamicLoggingMiddleware, service_name="gateway")

app.include_router(auth.router)
app.include_router(proxy.router)
app.include_router(vision_router)
app.include_router(dynamic_logging.router)
app.include_router(metrics_router)

@app.get("/")
def read_root():
    return {"message": "AstroNova API Gateway API v1", "dynamic_logging": "enabled"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "services.gateway.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
