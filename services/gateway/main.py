from fastapi import FastAPI, Depends
from services.gateway.routers import auth, proxy
from astronova_core.logging import setup_logging
from astronova_core.metrics import metrics_router
from fastapi.middleware.cors import CORSMiddleware

setup_logging("gateway")

app = FastAPI(
    title="AstroNova API Gateway",
    description="Central secure entrypoint routing API traffic.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router)
app.include_router(proxy.router)
app.include_router(metrics_router)

@app.get("/")
def read_root():
    return {"message": "AstroNova API Gateway API v1"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "services.gateway.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
