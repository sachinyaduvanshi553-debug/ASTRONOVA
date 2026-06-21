from fastapi import APIRouter
from services.xai.services.shap_explainer import SHAPExplainer

router = APIRouter(prefix="/api/v1/xai", tags=["xai"])
explainer = SHAPExplainer()

@router.get("/explain")
async def get_explanation():
    return explainer.explain_prediction()

@router.get("/health")
def health():
    return {"status": "healthy"}
