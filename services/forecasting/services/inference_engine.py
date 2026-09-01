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
        if isinstance(features, list):
            if len(features) == 0:
                features = np.zeros((1, 10, 15), dtype=np.float32)
            else:
                features = np.array(features, dtype=np.float32)
        if hasattr(features, "ndim") and features.ndim == 2:
            features = np.expand_dims(features, axis=0)

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
        
        # Final Ensemble setup
        out_horizons = {}
        
        # Horizon-specific weight adjustments (LSTM is better short-term, XGB/LGB better long-term)
        horizon_modifiers = {
            0: {"lstm": 1.25, "xgb": 0.85, "lgb": 0.85}, # 15m
            1: {"lstm": 1.10, "xgb": 0.95, "lgb": 0.95}, # 30m
            2: {"lstm": 0.90, "xgb": 1.05, "lgb": 1.05}, # 1h
            3: {"lstm": 0.70, "xgb": 1.15, "lgb": 1.15}  # 6h
        }

        for h_idx, h_name in enumerate(horizons):
            # Dynamic weighting per horizon
            w_lstm_dyn = w_lstm * horizon_modifiers[h_idx]["lstm"]
            w_xgb_dyn = w_xgb * horizon_modifiers[h_idx]["xgb"]
            w_lgb_dyn = w_lgb * horizon_modifiers[h_idx]["lgb"]
            
            # Re-normalize dynamic weights
            total_dyn = w_lstm_dyn + w_xgb_dyn + w_lgb_dyn
            w_lstm_dyn /= total_dyn
            w_xgb_dyn /= total_dyn
            w_lgb_dyn /= total_dyn
            
            # Blend probabilities and regression
            probs_h = (lstm_probs[0, h_idx, :] * w_lstm_dyn) + \
                      (xgb_probs[0, h_idx, :] * w_xgb_dyn) + \
                      (lgb_probs[0, h_idx, :] * w_lgb_dyn)
            
            reg_h = (lstm_regs[0, h_idx, 0] * w_lstm_dyn) + \
                    (xgb_regs[0, h_idx, 0] * w_xgb_dyn) + \
                    (lgb_regs[0, h_idx, 0] * w_lgb_dyn)
            
            intensity = float(10 ** reg_h)
            
            # Physics-Informed Precision Enhancements:
            if current_flux is not None and current_flux > 0:
                # 1. Kinematic Bound check
                max_growth_factor = 10.0 ** ((h_idx + 1) * 0.6)
                upper_bound = current_flux * max_growth_factor
                lower_bound = current_flux * 0.05
                intensity = min(max(intensity, lower_bound), upper_bound)
                
                # 2. Persistence Blending for near-term accuracy
                alpha = max(0.05, 0.7 - (h_idx * 0.2))
                intensity = (alpha * current_flux) + ((1.0 - alpha) * intensity)

            # Enforce probability-regression coherence
            if intensity >= 1e-4: implied_class = 4
            elif intensity >= 1e-5: implied_class = 3
            elif intensity >= 1e-6: implied_class = 2
            elif intensity >= 1e-7: implied_class = 1
            else: implied_class = 0
            
            # Temperature scaling for probability calibration
            T = 0.85
            probs_h = np.exp(np.log(probs_h + 1e-10) / T)
            probs_h /= np.sum(probs_h)

            pred_class_idx = int(np.argmax(probs_h))
            if pred_class_idx != implied_class:
                probs_h[implied_class] += 0.25
                probs_h /= np.sum(probs_h)
                pred_class_idx = int(np.argmax(probs_h))
            
            out_horizons[h_name] = {
                "flare_probability": float(np.sum(probs_h[3:])), # M + X class
                "flare_class": classes[pred_class_idx],
                "probabilities": dict(zip(classes, [float(p) for p in probs_h])),
                "predicted_intensity": float(intensity),
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
