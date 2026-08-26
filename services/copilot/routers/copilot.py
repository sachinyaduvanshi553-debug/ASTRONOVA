import traceback

from fastapi import APIRouter
from pydantic import BaseModel

from services.copilot.core.rag_chain import SpaceWeatherRAGChain

router = APIRouter(prefix="/api/v1/copilot", tags=["copilot"])

# Initialize the chain at module level — if this fails, the service won't start,
# which is the correct behaviour (fail-fast).
chain = SpaceWeatherRAGChain()


class ChatRequest(BaseModel):
    query: str


@router.post("/chat")
async def chat_with_copilot(request: ChatRequest):
    """Handle a chat query from the frontend."""
    print(f"[Copilot] Received query: {request.query}")
    try:
        result = chain.chat(request.query)
        print(f"[Copilot] Response length: {len(result.get('answer', ''))}")
        return result
    except Exception as e:
        traceback.print_exc()
        return {
            "answer": f"An internal error occurred: {e!s}. Please try again.",
            "sources": [],
        }


@router.get("/health")
def health():
    return {
        "status": "healthy",
        "gemini_available": chain.genai_client is not None,
        "ollama_url": chain.ollama_url,
    }
