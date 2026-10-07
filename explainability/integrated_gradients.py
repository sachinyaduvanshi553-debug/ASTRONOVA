"""
Integrated Gradients for Spatiotemporal Deep Learning Models (Sundararajan et al., 2017).
Computes exact path-integral gradients between baseline x0 and input sample x:
IG_i(x) = (x_i - x0_i) * \int_0^1 \frac{\partial F(x0 + \alpha (x - x0))}{\partial x_i} d\alpha
Generates:
1. Feature attribution ranking
2. Temporal attribution (importance across 24 hours)
3. 2D Spatiotemporal attribution matrix (24 hours x 16 features)
"""

import logging
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class IntegratedGradientsExplainer:
    """
    Computes Integrated Gradients attributions for 3D input sequences (B, T, F).
    """

    def __init__(self, model: nn.Module, m_steps: int = 50, device: str = "cpu"):
        self.model = model
        self.m_steps = m_steps
        self.device = device
        self.model.to(device)
        self.model.eval()

    def attribute(
        self,
        x: np.ndarray,
        baseline: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Computes attribution tensor of identical shape to x: (24, 16).
        """
        if x.ndim == 3 and x.shape[0] == 1:
            x = x.squeeze(0)

        T, F = x.shape
        if baseline is None:
            baseline = np.zeros_like(x, dtype=np.float32)

        x_tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(0).to(self.device)
        base_tensor = torch.tensor(baseline, dtype=torch.float32).unsqueeze(0).to(self.device)

        # Generate linear interpolation path points: alpha in [0, 1]
        alphas = torch.linspace(0.0, 1.0, self.m_steps + 1).to(self.device)
        # Interpolated samples shape: (m_steps + 1, T, F)
        delta = x_tensor - base_tensor
        interpolated = base_tensor + alphas.view(-1, 1, 1) * delta
        interpolated.requires_grad_(True)

        # Forward pass on interpolated points
        preds = self.model(interpolated)  # (m_steps + 1, 1)
        # Sum predictions to compute gradient w.r.t interpolated inputs
        grads = torch.autograd.grad(outputs=preds.sum(), inputs=interpolated)[0]  # (m_steps + 1, T, F)

        # Approximate Riemann integral using trapezoidal rule
        avg_grads = (grads[:-1] + grads[1:]) / 2.0
        integrated_grad = avg_grads.mean(dim=0)  # (T, F)

        # Scale by delta: (x - baseline) * integrated_grad
        attributions = (delta.squeeze(0) * integrated_grad).detach().cpu().numpy()
        return attributions.astype(np.float32)

    def explain_sample(
        self,
        x: np.ndarray,
        feature_names: List[str],
        baseline: Optional[np.ndarray] = None,
    ) -> Dict[str, any]:
        """
        Computes feature importance, temporal importance, and spatiotemporal attribution heatmap.
        """
        attr_matrix = self.attribute(x, baseline=baseline)  # Shape: (24, 16)
        
        # 1. Overall feature importance: sum of absolute attributions over time
        feat_importance = np.sum(np.abs(attr_matrix), axis=0)
        feat_rank = [
            {"feature": feature_names[i], "attribution": float(feat_importance[i])}
            for i in np.argsort(-feat_importance)
        ]

        # 2. Temporal importance: sum of absolute attributions over features
        temporal_importance = np.sum(np.abs(attr_matrix), axis=1)  # Shape: (24,)
        temporal_profile = [
            {"hour": int(h + 1), "attribution": float(temporal_importance[h])}
            for h in range(len(temporal_importance))
        ]

        return {
            "feature_attribution_ranking": feat_rank,
            "temporal_profile": temporal_profile,
            "attribution_matrix": attr_matrix.tolist(),
            "feature_names": feature_names,
            "sequence_length": attr_matrix.shape[0],
        }
