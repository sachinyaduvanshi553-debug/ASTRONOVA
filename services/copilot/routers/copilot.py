from fastapi import APIRouter, Query
from services.copilot.services.rag_chain import SpaceWeatherRAGChain

router = APIRouter(prefix="/api/v1/copilot", tags=["copilot"])
chain = SpaceWeatherRAGChain()

@router.get("/chat")
async def chat_with_copilot(query: str = Query(..., description="User query")):
    return chain.chat(query)

@router.get("/health")
def health():
    return {"status": "healthy"}
