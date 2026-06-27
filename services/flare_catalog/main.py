from fastapi import FastAPI
from services.flare_catalog.routers import catalog

from astronova_core.logging import setup_logging
from astronova_core.metrics import metrics_router

setup_logging("flare-catalog-service")

app = FastAPI(
    title="AstroNova Flare Catalog Service",
    description="Maintains validated solar flare catalogs.",
    version="1.0.0"
)

app.include_router(catalog.router)
app.include_router(metrics_router)

@app.get("/")
def read_root():
    return {"message": "AstroNova Flare Catalog Service API v1"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "services.flare_catalog.main:app",
        host="0.0.0.0",
        port=8014,
        reload=True
    )
