from fastapi import APIRouter, Query
from services.rag.services.vector_store import SpaceWeatherVectorStore

router = APIRouter(prefix="/api/v1/rag", tags=["knowledge"])
store = SpaceWeatherVectorStore()

@router.get("/search")
async def search_knowledge(query: str = Query(..., description="Query string")):
    return store.search(query)

@router.get("/health")
def health():
    return {"status": "healthy"}
