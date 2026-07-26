import sys
import os

# Ensure the project root and shared library are on sys.path so that
# both `services.copilot.*` and `astronova_core.*` resolve correctly,
# regardless of how uvicorn is launched.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SHARED_ROOT = os.path.join(PROJECT_ROOT, "shared")
for p in (PROJECT_ROOT, SHARED_ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# ── Import the router using the absolute package path ──
from services.copilot.routers import copilot as copilot_router

# Attempt structured logging from shared lib; fall back to stdlib
try:
    from astronova_core.logging import setup_logging
    setup_logging("copilot-service")
except Exception:
    import logging
    logging.basicConfig(level=logging.INFO)
    logging.getLogger(__name__).info("Using stdlib logging (astronova_core.logging unavailable)")

# Attempt metrics router from shared lib; skip if unavailable
metrics_router = None
try:
    from astronova_core.metrics import metrics_router as _mr
    metrics_router = _mr
except Exception:
    pass

# ── Build the FastAPI app ──────────────────────────────────────────────────
app = FastAPI(
    title="AstroNova LLM Copilot Service",
    description="Grounded AI Copilot for space weather operations.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(copilot_router.router)
if metrics_router:
    app.include_router(metrics_router)


@app.get("/")
def read_root():
    return {"message": "AstroNova LLM Copilot Service API v1"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8009,
        reload=True,
    )
