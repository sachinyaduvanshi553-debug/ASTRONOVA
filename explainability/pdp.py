"""
Partial Dependence Plot (PDP) Module for Solar Flare Forecasting.
Calculates:
1. 1D Partial Dependence for individual magnetic features (TOTUSJH, MEANJZH, TOTPOT, USFLUX)
2. 2D Partial Dependence for interacting feature pairs (especially TOTUSJH x MEANJZH)
3. 3D Surface grid representation
"""

import logging
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class PartialDependenceCalculator:
    """
    Computes marginal effects of photospheric magnetic field features on flare probability.
    """

    def __init__(self, model: Union[nn.Module, any], feature_names: List[str], device: str = "cpu"):
        self.model = model
        self.feature_names = feature_names
        self.device = device

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Wrapper to run forward predictions.
        """
        if isinstance(self.model, nn.Module):
            self.model.eval()
            with torch.no_grad():
                tensor_x = torch.tensor(X, dtype=torch.float32).to(self.device)
                if len(tensor_x) > 512:
                    out = []
                    for i in range(0, len(tensor_x), 512):
                        batch = tensor_x[i : i + 512]
                        p = self.model(batch).cpu().numpy().ravel()
                        out.append(p)
                    return np.concatenate(out)
                else:
                    return self.model(tensor_x).cpu().numpy().ravel()
        elif hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X).ravel()
        else:
            raise TypeError("Model must be PyTorch nn.Module or have predict_proba")

    def compute_1d_pdp(
        self,
        X_sample: np.ndarray,
        feature_name: str,
        grid_resolution: int = 25,
    ) -> Dict[str, any]:
        """
        Computes 1D partial dependence for a specific feature.
        """
        if feature_name not in self.feature_names:
            raise ValueError(f"Feature {feature_name} not found in feature list.")

        feat_idx = self.feature_names.index(feature_name)
        # Extract values for the feature across all time steps
        values = X_sample[:, :, feat_idx].ravel()
        min_v, max_v = np.percentile(values, 2), np.percentile(values, 98)
        grid_values = np.linspace(min_v, max_v, grid_resolution)

        pdp_values = []
        X_copy = X_sample.copy()

        for val in grid_values:
            X_copy[:, :, feat_idx] = val
            preds = self._predict(X_copy)
            pdp_values.append(float(np.mean(preds)))

        return {
            "feature": feature_name,
            "grid_values": [float(v) for v in grid_values],
            "pdp_values": [float(p) for p in pdp_values],
        }

    def compute_2d_pdp(
        self,
        X_sample: np.ndarray,
        feature_x: str = "TOTUSJH",
        feature_y: str = "MEANJZH",
        grid_resolution: int = 15,
    ) -> Dict[str, any]:
        """
        Computes 2D partial dependence heatmap / grid for interaction analysis.
        """
        idx_x = self.feature_names.index(feature_x)
        idx_y = self.feature_names.index(feature_y)

        vals_x = X_sample[:, :, idx_x].ravel()
        vals_y = X_sample[:, :, idx_y].ravel()

        grid_x = np.linspace(np.percentile(vals_x, 2), np.percentile(vals_x, 98), grid_resolution)
        grid_y = np.linspace(np.percentile(vals_y, 2), np.percentile(vals_y, 98), grid_resolution)

        pdp_grid = np.zeros((grid_resolution, grid_resolution), dtype=float)
        X_copy = X_sample.copy()

        for i, vx in enumerate(grid_x):
            for j, vy in enumerate(grid_y):
                X_copy[:, :, idx_x] = vx
                X_copy[:, :, idx_y] = vy
                preds = self._predict(X_copy)
                pdp_grid[j, i] = float(np.mean(preds))  # j = y axis, i = x axis

        return {
            "feature_x": feature_x,
            "feature_y": feature_y,
            "grid_x": [float(x) for x in grid_x],
            "grid_y": [float(y) for y in grid_y],
            "pdp_grid": pdp_grid.tolist(),
        }
