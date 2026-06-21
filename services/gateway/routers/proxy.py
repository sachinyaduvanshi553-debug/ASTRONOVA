from fastapi import APIRouter, Depends, HTTPException
import httpx
from astronova_core.security import get_current_user, TokenData

router = APIRouter(prefix="/api/v1/proxy", tags=["proxy"])

# Config URL mappings
SERVICE_MAP = {
    "ingest": "http://localhost:8001",
    "forecast": "http://localhost:8004",
    "xai": "http://localhost:8005",
    "impact": "http://localhost:8006",
    "satellite": "http://localhost:8007",
    "rag": "http://localhost:8008",
    "copilot": "http://localhost:8009"
}

@router.get("/{service_name}/{endpoint}")
async def proxy_request(service_name: str, endpoint: str, current_user: TokenData = Depends(get_current_user)):
    if service_name not in SERVICE_MAP:
        raise HTTPException(status_code=404, detail="Service not found")
        
    url = f"{SERVICE_MAP[service_name]}/api/v1/{service_name}/{endpoint}"
    async with httpx.AsyncClient() as client:
        try:
            r = await client.get(url, timeout=10.0)
            return r.json()
        except Exception as e:
            raise HTTPException(status_code=502, detail=f"Bad Gateway: {str(e)}")
