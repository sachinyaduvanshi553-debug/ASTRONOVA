"""
models/vision/flare_location_head.py — Spatial flare location heatmap head.

Produces a probability heatmap over the solar disc indicating likely
future flare activity at each spatial location.

IMPORTANT SCIENTIFIC DISCLAIMER:
    This heatmap represents the model's uncertainty about WHERE a future flare
    is most likely based on historical patterns.
    The output MUST always be labeled "AI Forecast / Predicted Flare Region".
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FlareLocationHead(nn.Module):
    """
    Upsampling spatial decoder producing per-pixel flare probability heatmap.
    """

    def __init__(
        self,
        input_dim: int = 256,
        target_size: int = 256,
        hidden_dim: int = 64,
    ):
        super().__init__()
        self.target_size = target_size

        self.decoder = nn.Sequential(
            nn.Conv2d(input_dim, hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1),  # Output: [B, 1, H', W']
        )

    def forward_logits(self, spatial_maps: torch.Tensor) -> torch.Tensor:
        """Computes raw unactivated location logits interpolated to target_size."""
        x = self.decoder(spatial_maps)
        if x.shape[-1] != self.target_size or x.shape[-2] != self.target_size:
            x = F.interpolate(
                x,
                size=(self.target_size, self.target_size),
                mode="bilinear",
                align_corners=False,
            )
        return x

    def forward(self, spatial_maps: torch.Tensor) -> torch.Tensor:
        """
        Args:
            spatial_maps: [B, D, H', W'] — encoded spatial features
        Returns:
            heatmap: [B, 1, H, W] in [0, 1]
        """
        logits = self.forward_logits(spatial_maps)
        return torch.sigmoid(logits)

    def compute_centroid(self, heatmap: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute soft centroid of the heatmap."""
        B, _, H, W = heatmap.shape
        device = heatmap.device

        xs = torch.linspace(0, 1, W, device=device).view(1, 1, 1, W)
        ys = torch.linspace(0, 1, H, device=device).view(1, 1, H, 1)

        total = heatmap.sum(dim=(-2, -1), keepdim=True) + 1e-8
        cx = (heatmap * xs).sum(dim=(-2, -1)) / total.squeeze(-1).squeeze(-1)
        cy = (heatmap * ys).sum(dim=(-2, -1)) / total.squeeze(-1).squeeze(-1)

        return cx.squeeze(-1), cy.squeeze(-1)
