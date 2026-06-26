import joblib
import numpy as np
from pathlib import Path

MODEL_PATH = Path("models/flare_model.pkl")

model = None


# -----------------------------
# LOAD MODEL
# -----------------------------
def load_model():
    global model
    model = joblib.load(MODEL_PATH)
    print("✅ Model loaded successfully")
    return model


# -----------------------------
# PREDICT
# -----------------------------
def predict(input_data: dict):

    if model is None:
        load_model()

    features = np.array(list(input_data.values())).reshape(1, -1)
    prediction = model.predict(features)[0]

    return {
        "prediction": str(prediction)
    }