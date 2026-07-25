from fastapi import APIRouter
from pydantic import BaseModel
from services.copilot.services.rag_chain import SpaceWeatherRAGChain

router = APIRouter(prefix="/api/v1/copilot", tags=["copilot"])
chain = SpaceWeatherRAGChain()

class ChatRequest(BaseModel):
    query: str

@router.post("/chat")
async def chat_with_copilot(request: ChatRequest):
    # Log incoming query for debugging
    print(f"[Copilot] Received query: {request.query}")
    return chain.chat(request.query)

@router.get("/health")
def health():
    return {"status": "healthy"}
