"""
SHAP Explainer Module for Solar Flare Forecasting.
Calculates SHAP values using Kernel/Sampling/Exact explainers.
Generates:
1. Global feature importance ranking (mean absolute SHAP)
2. Beeswarm distribution data
3. Waterfall explanations for positive and negative test instances
"""

import logging
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import shap
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class SolarSHAPExplainer:
    """
    Computes SHAP value attributions for 3D temporal sequence inputs.
    """

    def __init__(
        self,
        model: Union[nn.Module, any],
        feature_names: List[str],
        background_samples: np.ndarray,
        device: str = "cpu",
    ):
        self.model = model
        self.feature_names = feature_names
        self.device = device

        # Flatten 3D background to 2D for KernelExplainer or maintain wrapper
        self.background = background_samples
        self.n_features = len(feature_names)

        # Build prediction function wrapper accepting flattened or 3D features
        def model_predict_flat(x_flat):
            # Reshape (N, 24 * 16) -> (N, 24, 16)
            N = len(x_flat)
            x_3d = x_flat.reshape(N, 24, self.n_features)
            if isinstance(self.model, nn.Module):
                self.model.eval()
                with torch.no_grad():
                    t_x = torch.tensor(x_3d, dtype=torch.float32).to(self.device)
                    return self.model(t_x).cpu().numpy().ravel()
            else:
                return self.model.predict_proba(x_3d).ravel()

        self.predict_fn = model_predict_flat
        bg_flat = background_samples.reshape(len(background_samples), -1)
        # Use shap.sample to keep background small and fast (e.g. 50 samples)
        self.explainer = shap.KernelExplainer(self.predict_fn, shap.sample(bg_flat, 30))

    def compute_shap_values(self, X_samples: np.ndarray, nsamples: int = 100) -> np.ndarray:
        """
        Computes SHAP values.
        Input X_samples shape: (N, 24, 16)
        Returns: (N, 24, 16) SHAP values tensor.
        """
        N = len(X_samples)
        X_flat = X_samples.reshape(N, -1)
        shap_flat = self.explainer.shap_values(X_flat, nsamples=nsamples)
        if isinstance(shap_flat, list):
            shap_flat = shap_flat[0]
        shap_3d = np.array(shap_flat).reshape(N, 24, self.n_features)
        return shap_3d

    def get_global_importance(self, shap_3d: np.ndarray) -> List[Dict[str, Union[str, float]]]:
        """
        Calculates mean absolute SHAP value per feature aggregated over all time steps and instances.
        """
        # Aggregate over time steps: (N, F)
        feat_shap_per_sample = np.sum(shap_3d, axis=1)
        # Mean absolute SHAP per feature across samples
        mean_abs_shap = np.mean(np.abs(feat_shap_per_sample), axis=0)

        importance_list = [
            {"feature": self.feature_names[i], "mean_abs_shap": float(mean_abs_shap[i])}
            for i in np.argsort(-mean_abs_shap)
        ]
        return importance_list

    def get_sample_waterfall(
        self,
        sample_3d: np.ndarray,
        shap_sample_3d: np.ndarray,
        base_value: float = 0.05,
    ) -> Dict[str, any]:
        """
        Generates waterfall plot decomposition for an individual test sample.
        """
        feat_vals = np.mean(sample_3d, axis=0)  # (16,)
        feat_shaps = np.sum(shap_sample_3d, axis=0)  # (16,)

        waterfall_data = [
            {
                "feature": self.feature_names[i],
                "feature_value": float(feat_vals[i]),
                "shap_value": float(feat_shaps[i]),
                "direction": "positive" if feat_shaps[i] >= 0 else "negative",
            }
            for i in np.argsort(-np.abs(feat_shaps))
        ]

        return {
            "base_value": float(base_value),
            "final_prediction": float(base_value + np.sum(feat_shaps)),
            "contributions": waterfall_data,
        }
