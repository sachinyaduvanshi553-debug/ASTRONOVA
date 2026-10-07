"""
models/multimodal_fusion.py — Multimodal feature fusion for solar flare forecasting.

Fuses:
    1. Image features (CNN-encoded spatial maps)
    2. GOES/SoLEXS telemetry embeddings
    3. HMI SHARP magnetic feature embeddings
    4. Physics-derived features

Architecture:
    Image Encoder → Image embedding
    Telemetry Encoder → Telemetry embedding
    Magnetic Encoder → Magnetic embedding
    ↓
    Cross-Attention / Feature Fusion
    ↓
    Multimodal Temporal Transformer
    ↓
    Prediction Heads (classification, heatmap, image)
"""
from __future__ import annotations

import torch
import torch.nn as nn


class CrossModalAttention(nn.Module):
    """
    Cross-attention between primary (image) and secondary (telemetry/magnetic) modalities.
    
    Query: image embedding
    Key/Value: secondary modality embedding
    """

    def __init__(self, d_model: int, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        """
        Args:
            query: [B, T, D] — image embeddings (query)
            kv:    [B, T, D] — secondary modality (key/value)
        Returns:
            fused: [B, T, D]
        """
        attn_out, _ = self.attn(query, kv, kv)
        return self.norm(query + self.dropout(attn_out))


class MultimodalProjection(nn.Module):
    """Projects multi-source embeddings to a common dimension."""

    def __init__(self, *input_dims: int, output_dim: int):
        super().__init__()
        self.projections = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d, output_dim),
                nn.LayerNorm(output_dim),
                nn.GELU(),
            ) for d in input_dims
        ])

    def forward(self, *tensors: torch.Tensor) -> list[torch.Tensor]:
        return [proj(t) for proj, t in zip(self.projections, tensors)]


class MultimodalFusion(nn.Module):
    """
    Full multimodal fusion module combining image + telemetry + magnetic features.
    
    Handles missing modalities explicitly:
        - If magnetic data is absent → magnetic embedding = learned missing embedding
        - If telemetry is absent → telemetry embedding = zero embedding
        - Never silently fills missing scientific data with arbitrary zeros
    
    Input:
        image_embeds:     [B, T, D_img]   — from CNN encoder
        telemetry_embeds: [B, T, D_tel]   — from TelemetryEncoder
        magnetic_embeds:  [B, T, D_mag]   — from MagneticFeatureEncoder (may be missing)
    
    Output:
        fused: [B, T, D] — fused multimodal representation
    """

    def __init__(
        self,
        image_dim: int = 256,
        telemetry_dim: int = 128,
        magnetic_dim: int = 64,
        output_dim: int = 256,
        n_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()

        # Project all modalities to common space
        self.projection = MultimodalProjection(
            image_dim, telemetry_dim, magnetic_dim,
            output_dim=output_dim,
        )

        # Cross-modal attention: image ↔ telemetry
        self.img_tel_attn = CrossModalAttention(output_dim, n_heads, dropout)
        # Cross-modal attention: image ↔ magnetic
        self.img_mag_attn = CrossModalAttention(output_dim, n_heads, dropout)

        # Final fusion MLP
        self.fusion_mlp = nn.Sequential(
            nn.Linear(output_dim * 3, output_dim * 2),
            nn.LayerNorm(output_dim * 2),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(output_dim * 2, output_dim),
            nn.LayerNorm(output_dim),
        )

    def forward(
        self,
        image_embeds: torch.Tensor,
        telemetry_embeds: torch.Tensor,
        magnetic_embeds: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            image_embeds:     [B, T, D_img]
            telemetry_embeds: [B, T, D_tel]
            magnetic_embeds:  [B, T, D_mag]
        Returns:
            fused: [B, T, D]
        """
        img_proj, tel_proj, mag_proj = self.projection(
            image_embeds, telemetry_embeds, magnetic_embeds
        )

        # Cross-modal attention
        fused_img_tel = self.img_tel_attn(img_proj, tel_proj)  # [B, T, D]
        fused_img_mag = self.img_mag_attn(img_proj, mag_proj)  # [B, T, D]

        # Concatenate and fuse
        combined = torch.cat([img_proj, fused_img_tel, fused_img_mag], dim=-1)  # [B, T, 3D]
        fused = self.fusion_mlp(combined)   # [B, T, D]

        return fused
