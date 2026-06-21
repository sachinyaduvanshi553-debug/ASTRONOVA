from fastapi import APIRouter
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
