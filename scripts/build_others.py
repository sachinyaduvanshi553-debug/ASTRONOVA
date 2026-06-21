import os

def create_file(path, content):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"Created: {path}")

# =====================================================================
# PART 1: RAG KNOWLEDGE SERVICE
# =====================================================================

# --- 1. requirements.txt ---
create_file("services/rag/requirements.txt", """fastapi>=0.115.0
uvicorn>=0.30.0
pydantic>=2.9.0
chromadb>=0.5.0
pandas>=2.2.0
numpy>=2.0.0
prometheus-client>=0.21.0
structlog>=24.0.0
astronova-core
""")

# --- 2. main.py ---
create_file("services/rag/main.py", """from fastapi import FastAPI
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
""")

# --- 3. Dockerfile ---
create_file("services/rag/Dockerfile", """FROM python:3.12-slim as builder
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

FROM python:3.12-slim as runtime
WORKDIR /app
COPY --from=builder /root/.local /root/.local
COPY . .
ENV PATH=/root/.local/bin:$PATH
ENV PYTHONUNBUFFERED=1
EXPOSE 8008
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8008"]
""")

# --- 4. services/vector_store.py ---
create_file("services/rag/services/vector_store.py", """from typing import List, Dict, Any

class SpaceWeatherVectorStore:
    def search(self, query: str) -> List[Dict[str, Any]]:
        # Mock semantic search results
        return [
            {
                "document_id": "doc_1",
                "title": "ISRO Aditya-L1 SoLEXS Instrument Specification",
                "text": "The Solar Low Energy X-ray Spectrometer (SoLEXS) on Aditya-L1 observes soft X-rays in 1-22 keV.",
                "score": 0.89
            },
            {
                "document_id": "doc_2",
                "title": "Space Weather Hazard Guidelines",
                "text": "Solar flares exceeding M5 class pose extreme ionospheric disruption risk for GNSS/NavIC systems.",
                "score": 0.76
            }
        ]
""")

# --- 5. routers/knowledge.py ---
create_file("services/rag/routers/knowledge.py", """from fastapi import APIRouter, Query
from services.rag.services.vector_store import SpaceWeatherVectorStore

router = APIRouter(prefix="/api/v1/rag", tags=["knowledge"])
store = SpaceWeatherVectorStore()

@router.get("/search")
async def search_knowledge(query: str = Query(..., description="Query string")):
    return store.search(query)

@router.get("/health")
def health():
    return {"status": "healthy"}
""")


# =====================================================================
# PART 2: LLM COPILOT SERVICE
# =====================================================================

# --- 6. requirements.txt ---
create_file("services/copilot/requirements.txt", """fastapi>=0.115.0
uvicorn>=0.30.0
pydantic>=2.9.0
prometheus-client>=0.21.0
structlog>=24.0.0
astronova-core
""")

# --- 7. main.py ---
create_file("services/copilot/main.py", """from fastapi import FastAPI
from services.copilot.routers import copilot
from astronova_core.logging import setup_logging
from astronova_core.metrics import metrics_router

setup_logging("copilot-service")

app = FastAPI(
    title="AstroNova LLM Copilot Service",
    description="Grounded AI Copilot for space weather operations.",
    version="1.0.0"
)

app.include_router(copilot.router)
app.include_router(metrics_router)

@app.get("/")
def read_root():
    return {"message": "AstroNova LLM Copilot Service API v1"}
""")

# --- 8. Dockerfile ---
create_file("services/copilot/Dockerfile", """FROM python:3.12-slim as builder
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

FROM python:3.12-slim as runtime
WORKDIR /app
COPY --from=builder /root/.local /root/.local
COPY . .
ENV PATH=/root/.local/bin:$PATH
ENV PYTHONUNBUFFERED=1
EXPOSE 8009
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8009"]
""")

# --- 9. services/rag_chain.py ---
create_file("services/copilot/services/rag_chain.py", """from typing import Dict, Any

class SpaceWeatherRAGChain:
    def chat(self, query: str) -> Dict[str, Any]:
        # Local RAG-grounded LLM response mock
        return {
            "answer": "Based on the Aditya-L1 SoLEXS Instrument Specifications, the spectrometer monitors solar soft X-ray flux in the 1 to 22 keV range. The anticipated solar flares could cause HF communication blackouts especially over the South-Asian sector.",
            "sources": [
                {"title": "Aditya-L1 SoLEXS specs", "chunk": "observes soft X-rays in 1-22 keV"},
                {"title": "Space Weather Hazard Guidelines", "chunk": "pose extreme ionospheric disruption risk for GNSS/NavIC"}
            ]
        }
""")

# --- 10. routers/copilot.py ---
create_file("services/copilot/routers/copilot.py", """from fastapi import APIRouter, Query
from services.copilot.services.rag_chain import SpaceWeatherRAGChain

router = APIRouter(prefix="/api/v1/copilot", tags=["copilot"])
chain = SpaceWeatherRAGChain()

@router.get("/chat")
async def chat_with_copilot(query: str = Query(..., description="User query")):
    return chain.chat(query)

@router.get("/health")
def health():
    return {"status": "healthy"}
""")


# =====================================================================
# PART 3: API GATEWAY
# =====================================================================

# --- 11. requirements.txt ---
create_file("services/gateway/requirements.txt", """fastapi>=0.115.0
uvicorn>=0.30.0
pydantic>=2.9.0
httpx>=0.27.0
redis>=5.2.0
python-jose[cryptography]>=3.3.0
passlib[bcrypt]>=1.7.4
prometheus-client>=0.21.0
structlog>=24.0.0
astronova-core
""")

# --- 12. main.py ---
create_file("services/gateway/main.py", """from fastapi import FastAPI, Depends
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
""")

# --- 13. Dockerfile ---
create_file("services/gateway/Dockerfile", """FROM python:3.12-slim as builder
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

FROM python:3.12-slim as runtime
WORKDIR /app
COPY --from=builder /root/.local /root/.local
COPY . .
ENV PATH=/root/.local/bin:$PATH
ENV PYTHONUNBUFFERED=1
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
""")

# --- 14. routers/auth.py ---
create_file("services/gateway/routers/auth.py", """from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from astronova_core.security import create_access_token, get_password_hash, verify_password, UserRole
from pydantic import BaseModel

router = APIRouter(prefix="/api/v1/auth", tags=["auth"])

class LoginResponse(BaseModel):
    access_token: str
    token_type: str
    role: str

@router.post("/login", response_model=LoginResponse)
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    # Admin mock login
    if form_data.username == "admin" and form_data.password == "admin123":
        access_token = create_access_token(data={"sub": "admin", "role": UserRole.ADMIN})
        return LoginResponse(access_token=access_token, token_type="bearer", role=UserRole.ADMIN)
    
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Incorrect username or password",
        headers={"WWW-Authenticate": "Bearer"},
    )
""")

# --- 15. routers/proxy.py ---
create_file("services/gateway/routers/proxy.py", """from fastapi import APIRouter, Depends, HTTPException
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
""")


# =====================================================================
# PART 4: NOTIFICATION SERVICE
# =====================================================================

# --- 16. requirements.txt ---
create_file("services/notifications/requirements.txt", """fastapi>=0.115.0
uvicorn>=0.30.0
pydantic>=2.9.0
prometheus-client>=0.21.0
structlog>=24.0.0
astronova-core
""")

# --- 17. main.py ---
create_file("services/notifications/main.py", """from fastapi import FastAPI
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
""")

# --- 18. Dockerfile ---
create_file("services/notifications/Dockerfile", """FROM python:3.12-slim as builder
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

FROM python:3.12-slim as runtime
WORKDIR /app
COPY --from=builder /root/.local /root/.local
COPY . .
ENV PATH=/root/.local/bin:$PATH
ENV PYTHONUNBUFFERED=1
EXPOSE 8010
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8010"]
""")

# --- 19. services/alert_manager.py ---
create_file("services/notifications/services/alert_manager.py", """from typing import Dict, Any, List

class AlertManager:
    def create_alert(self, severity: str, title: str, message: str) -> Dict[str, Any]:
        return {
            "alert_id": "alert_mock_123",
            "severity": severity,
            "title": title,
            "message": message,
            "status": "sent"
        }
""")

# --- 20. routers/alerts.py ---
create_file("services/notifications/routers/alerts.py", """from fastapi import APIRouter
from services.notifications.services.alert_manager import AlertManager
from pydantic import BaseModel

router = APIRouter(prefix="/api/v1/notify", tags=["alerts"])
manager = AlertManager()

class AlertRequest(BaseModel):
    severity: str
    title: str
    message: str

@router.post("/alert")
async def trigger_alert(req: AlertRequest):
    return manager.create_alert(req.severity, req.title, req.message)

@router.get("/health")
def health():
    return {"status": "healthy"}
""")

print("ALL OTHER SERVICES GENERATED SUCCESSFULLY")
