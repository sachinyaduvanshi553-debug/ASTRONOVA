from typing import Any

import numpy as np
import torch
from ml.models.bilstm import BiLSTMForecaster
from ml.models.xgboost_model import XGBoostForecaster
from ml.models.lightgbm_model import LightGBMForecaster
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

class InferenceEngine:
    def __init__(self):
        logger.info("Initializing Real Inference Engine with Ensemble Model...")
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load LSTM
        self.lstm = BiLSTMForecaster(input_size=15, num_horizons=4).to(self.device)
        try:
            self.lstm.load_state_dict(torch.load("models/lstm/best.pt", map_location=self.device))
            self.lstm.eval()
            self.has_lstm = True
        except Exception as e:
            logger.warning(f"Could not load LSTM: {e}")
            self.has_lstm = False
            
        # Load XGBoost
        try:
            self.xgb = XGBoostForecaster.load("models/xgboost/model.pkl")
            self.has_xgb = True
        except Exception as e:
            logger.warning(f"Could not load XGBoost: {e}")
            self.has_xgb = False
            
        # Load LightGBM
        try:
            self.lgb = LightGBMForecaster.load("models/lightgbm/model.pkl")
            self.has_lgb = True
        except Exception as e:
            logger.warning(f"Could not load LightGBM: {e}")
            self.has_lgb = False
            
        # Ensemble Weights
        self.weights = {"lstm": 0.3, "xgb": 0.4, "lgb": 0.3}

    def predict(self, features: np.ndarray, current_flux: float = None) -> Dict[str, Any]:
        """
        Features expected shape: [batch_size, 10, 15]
        """
        batch_size = features.shape[0]
        horizons = ["15m", "30m", "1h", "6h"]
        classes = ["A", "B", "C", "M", "X"]
        
        # LSTM Prediction
        if self.has_lstm:
            x_t = torch.tensor(features, dtype=torch.float32).to(self.device)
            with torch.no_grad():
                lstm_probs, lstm_regs = self.lstm(x_t, return_tuple=True)
                lstm_probs = lstm_probs.cpu().numpy()
                lstm_regs = lstm_regs.cpu().numpy()
        else:
            lstm_probs = np.zeros((batch_size, 4, 5))
            lstm_regs = np.zeros((batch_size, 4, 1))
            self.weights["lstm"] = 0.0
            
        # XGBoost Prediction
        if self.has_xgb:
            xgb_probs, xgb_regs = self.xgb.predict(features)
        else:
            xgb_probs = np.zeros((batch_size, 4, 5))
            xgb_regs = np.zeros((batch_size, 4, 1))
            self.weights["xgb"] = 0.0
            
        # LightGBM Prediction
        if self.has_lgb:
            lgb_probs, lgb_regs = self.lgb.predict(features)
        else:
            lgb_probs = np.zeros((batch_size, 4, 5))
            lgb_regs = np.zeros((batch_size, 4, 1))
            self.weights["lgb"] = 0.0
            
        # Normalize weights
        total_weight = sum(self.weights.values())
        if total_weight == 0:
            raise RuntimeError("No models loaded for inference.")
            
        w_lstm = self.weights["lstm"] / total_weight
        w_xgb = self.weights["xgb"] / total_weight
        w_lgb = self.weights["lgb"] / total_weight
        
        # Final Ensemble
        final_probs = (lstm_probs * w_lstm) + (xgb_probs * w_xgb) + (lgb_probs * w_lgb)
        final_regs = (lstm_regs * w_lstm) + (xgb_regs * w_xgb) + (lgb_regs * w_lgb)
        
        # Format response (for batch index 0)
        out_horizons = {}
        for h_idx, h_name in enumerate(horizons):
            probs_h = final_probs[0, h_idx, :]
            pred_class_idx = np.argmax(probs_h)
            
            out_horizons[h_name] = {
                "flare_probability": float(np.sum(probs_h[3:])), # M + X class
                "flare_class": classes[pred_class_idx],
                "probabilities": dict(zip(classes, [float(p) for p in probs_h])),
                "predicted_intensity": float(10 ** final_regs[0, h_idx, 0]),
                "confidence": float(probs_h[pred_class_idx])
            }
            
        # Overall aggregates (using 15m as immediate summary)
        return {
            "prediction": {
                "flare_probability": out_horizons["15m"]["flare_probability"],
                "flare_class": out_horizons["15m"]["flare_class"],
                "predicted_intensity": out_horizons["15m"]["predicted_intensity"],
                "confidence": out_horizons["15m"]["confidence"],
                "horizons": out_horizons
            }
        }
