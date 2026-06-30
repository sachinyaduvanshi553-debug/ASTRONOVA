from fastapi import FastAPI
from services.rag.routers import knowledge

from astronova_core.logging import setup_logging
from astronova_core.metrics import metrics_router

setup_logging("rag-service")

app = FastAPI(
    title="AstroNova RAG Knowledge Service",
    description="Vector search knowledge base for space weather.",
    version="1.0.0"
)

app.include_router(knowledge.router)
app.include_router(metrics_router)

@app.get("/")
def read_root():
    return {"message": "AstroNova RAG Knowledge Service API v1"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "services.rag.main:app",
        host="0.0.0.0",
        port=8008,
        reload=True
    )
