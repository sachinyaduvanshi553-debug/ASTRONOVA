from fastapi import FastAPI
from pydantic import BaseModel
from services.ml.inference import load_model, predict

app = FastAPI(title="Solar Flare AI API")

model = None


# -----------------------------
# INPUT SCHEMA
# -----------------------------
class InputData(BaseModel):
    features: dict


# -----------------------------
# STARTUP
# -----------------------------
@app.on_event("startup")
def startup():
    global model
    model = load_model()


# -----------------------------
# HEALTH CHECK
# -----------------------------
@app.get("/")
def home():
    return {"status": "Solar AI API Running 🚀"}


# -----------------------------
# PREDICTION ENDPOINT
# -----------------------------
@app.post("/predict")
def get_prediction(data: InputData):
    return predict(data.features)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "services.api.main:app",
        host="0.0.0.0",
        port=8013,
        reload=True
    )
