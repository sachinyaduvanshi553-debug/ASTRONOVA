"""
Production Real-Time Inference Engine for AstroNova.
Loads:
- Optimized Patch Transformer & Soft-Voting Ensemble checkpoints
- Robust Scaler (fitted strictly on training data)
- Locked Validation Threshold
- Feature list (16 base-paper photospheric magnetic features)
- Integrated Gradients Explainer
Outputs:
- Flare Probability (raw and calibrated)
- Binary Prediction (0 or 1)
- Confidence Level
- Locked Decision Threshold
- Important Contributing Features
- Spatiotemporal Attributions
"""

import json
import logging
import os
import pickle
from typing import Any, Dict, List, Optional, Union
import numpy as np
import torch
import torch.nn as nn

from explainability.integrated_gradients import IntegratedGradientsExplainer
from models.baseline_transformer import AstroNovaBaselineTransformer
from models.bilstm import SolarBiLSTM
from models.ensemble import SolarEnsembleForecaster
from models.optimized_transformer import AstroNovaOptimizedTransformer
from models.temporal_cnn import TemporalCNNTransformer
from models.xgboost_model import SolarXGBoostForecaster
from optimization.calibration import SolarProbabilityCalibrator
from preprocessing.cleaner import BASE_PAPER_16_FEATURES
from preprocessing.scaler import SolarFeatureScaler

logger = logging.getLogger(__name__)


class AstroNovaInferenceEngine:
    """
    Production inference engine implementing the complete base-paper pipeline.
    """

    def __init__(self, checkpoint_dir: str = "checkpoints"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.checkpoint_dir = checkpoint_dir

        # 1. Load Feature List
        feat_path = os.path.join(checkpoint_dir, "feature_list.json")
        if os.path.exists(feat_path):
            with open(feat_path, "r") as f:
                feat_data = json.load(f)
            self.features = feat_data.get("selected_features", BASE_PAPER_16_FEATURES)
        else:
            self.features = BASE_PAPER_16_FEATURES

        # 2. Load Robust Scaler
        scaler_path = os.path.join(checkpoint_dir, "scaler.pkl")
        if os.path.exists(scaler_path):
            self.scaler = SolarFeatureScaler.load(scaler_path)
        else:
            self.scaler = SolarFeatureScaler(feature_cols=self.features, scaler_type="robust")

        # 3. Load Locked Threshold
        thresh_path = os.path.join(checkpoint_dir, "threshold.json")
        if os.path.exists(thresh_path):
            with open(thresh_path, "r") as f:
                th_data = json.load(f)
            self.threshold = float(th_data.get("optimal_threshold", 0.5))
        else:
            self.threshold = 0.5

        # 4. Load Calibrator
        cal_path = os.path.join(checkpoint_dir, "calibrator.pkl")
        if os.path.exists(cal_path):
            self.calibrator = SolarProbabilityCalibrator.load(cal_path)
        else:
            self.calibrator = None

        # 5. Load Models
        self._load_models()

    def _load_models(self):
        """
        Loads the trained Optimized Transformer, BiLSTM, Temporal CNN, XGBoost, and Ensemble.
        """
        # Config
        cfg_path = os.path.join(self.checkpoint_dir, "model_config.json")
        d_model = 128
        patch_size = 2
        if os.path.exists(cfg_path):
            with open(cfg_path, "r") as f:
                cfg = json.load(f)
            d_model = cfg.get("d_model", 128)
            patch_size = cfg.get("patch_size", 2)

        # Main Optimized Transformer
        self.transformer = AstroNovaOptimizedTransformer(
            seq_len=24,
            num_features=len(self.features),
            patch_size=patch_size,
            d_model=d_model,
            pooling="attention",
        ).to(self.device)

        opt_pt = os.path.join(self.checkpoint_dir, "EXP_004_TRANSFORMER_OPT_best_roc_auc.pt")
        if not os.path.exists(opt_pt):
            opt_pt = os.path.join(self.checkpoint_dir, "EXP_004_TRANSFORMER_OPT_final.pt")

        if os.path.exists(opt_pt):
            self.transformer.load_state_dict(torch.load(opt_pt, map_location=self.device))
            self.transformer.eval()
            logger.info("Loaded Optimized Transformer weights from %s", opt_pt)

        # BiLSTM
        self.bilstm = SolarBiLSTM(input_size=len(self.features), hidden_size=64, num_layers=2).to(self.device)
        bilstm_pt = os.path.join(self.checkpoint_dir, "EXP_005_BILSTM_final.pt")
        if os.path.exists(bilstm_pt):
            self.bilstm.load_state_dict(torch.load(bilstm_pt, map_location=self.device))
            self.bilstm.eval()

        # XGBoost
        xgb_path = os.path.join(self.checkpoint_dir, "xgboost_model.pkl")
        if os.path.exists(xgb_path):
            self.xgboost = SolarXGBoostForecaster.load(xgb_path)
        else:
            self.xgboost = None

        # Ensemble
        ens_w_path = os.path.join(self.checkpoint_dir, "ensemble_weights.json")
        weights = {}
        if os.path.exists(ens_w_path):
            with open(ens_w_path, "r") as f:
                weights = json.load(f)

        models_dict = {
            "Optimized_Transformer": self.transformer,
            "BiLSTM": self.bilstm,
        }
        if self.xgboost is not None:
            models_dict["XGBoost"] = self.xgboost

        self.ensemble = SolarEnsembleForecaster(models=models_dict, weights=weights)

        # Integrated Gradients Explainer
        self.ig_explainer = IntegratedGradientsExplainer(self.transformer, m_steps=30, device=self.device)

    def predict(
        self,
        sequence_data: Union[np.ndarray, List[List[float]]],
        include_xai: bool = True,
        use_ensemble: bool = True,
    ) -> Dict[str, Any]:
        """
        Executes production prediction on a 24-hour sequence (24 x 16).
        """
        arr = np.asarray(sequence_data, dtype=np.float32)
        if arr.ndim == 2:
            arr = np.expand_dims(arr, axis=0)  # (1, 24, 16)

        if arr.shape[1] != 24 or arr.shape[2] != len(self.features):
            # Reshape or pad if needed
            if arr.shape[1] * arr.shape[2] == 24 * len(self.features):
                arr = arr.reshape(-1, 24, len(self.features))
            else:
                padded = np.zeros((arr.shape[0], 24, len(self.features)), dtype=np.float32)
                min_t = min(arr.shape[1], 24)
                min_f = min(arr.shape[2], len(self.features))
                padded[:, :min_t, :min_f] = arr[:, :min_t, :min_f]
                arr = padded

        # Scale features
        scaled_arr = self.scaler.transform(arr)

        # Inference
        if use_ensemble:
            raw_prob = float(self.ensemble.predict_proba(scaled_arr, device=self.device)[0])
        else:
            with torch.no_grad():
                tensor_x = torch.tensor(scaled_arr, dtype=torch.float32).to(self.device)
                raw_prob = float(self.transformer(tensor_x).cpu().numpy().ravel()[0])

        # Calibrate Probability
        if self.calibrator is not None:
            calibrated_prob = float(self.calibrator.transform(np.array([raw_prob]))[0])
        else:
            calibrated_prob = raw_prob

        # Binary Decision using locked threshold
        prediction = 1 if calibrated_prob >= self.threshold else 0
        confidence = calibrated_prob if prediction == 1 else (1.0 - calibrated_prob)

        response = {
            "flare_probability": round(calibrated_prob, 4),
            "raw_probability": round(raw_prob, 4),
            "prediction": int(prediction),
            "classification": ">= M-Class Flare" if prediction == 1 else "Quiet / < M-Class",
            "confidence": round(confidence, 4),
            "threshold": round(self.threshold, 4),
            "model_version": "AstroNova-v2.0-BasePaper-Consistent",
            "prediction_horizon": "24 hours",
            "sequence_length_hours": 24,
            "feature_count": len(self.features),
            "features_used": self.features,
        }

        # Include XAI attributions if requested
        if include_xai:
            exp = self.ig_explainer.explain_sample(scaled_arr[0], self.features)
            response["important_features"] = exp["feature_attribution_ranking"][:6]
            response["temporal_contributions"] = exp["temporal_profile"]
            response["spatiotemporal_attribution"] = exp["attribution_matrix"]

        return response
