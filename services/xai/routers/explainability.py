from fastapi import APIRouter
from services.xai.services.shap_explainer import SHAPExplainer
from ml.models.xgboost_model import XGBoostForecaster
import os

router = APIRouter(prefix="/api/v1/xai", tags=["xai"])

model_path = os.path.join("models", "xgboost", "model.pkl")
if os.path.exists(model_path):
    xgboost_model = XGBoostForecaster.load(model_path)
    explainer = SHAPExplainer(model=xgboost_model)
else:
    explainer = None

@router.get("/explain")
async def get_explanation():
    if explainer is None:
        return {"error": "Model not found, cannot explain."}
    return explainer.explain_prediction()

@router.get("/health")
def health():
    return {"status": "healthy", "explainer_loaded": explainer is not None}
