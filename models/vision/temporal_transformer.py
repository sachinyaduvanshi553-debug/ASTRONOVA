"""
models/vision/temporal_transformer.py — Temporal Transformer for solar sequence modeling.

Models time-dependent evolution of:
  - Solar active regions
  - X-ray flux intensity
  - Magnetic complexity
  - Solar morphology

Architecture:
    CNN Encoder → temporal embeddings → Transformer → forecasting heads

Input:  pooled_embeds [B, T, D] — from CNN encoder
Output: [B, D]                   — context embedding for forecasting heads
        [B, T, D]                — all temporal representations
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


class SinusoidalPositionalEncoding(nn.Module):
    """Sinusoidal temporal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, T, D]"""
        x = x + self.pe[:, : x.shape[1]]
        return self.dropout(x)


class TemporalTransformer(nn.Module):
    """
    Transformer encoder for temporal sequence of spatial embeddings.
    
    Captures long-range temporal dependencies in:
      - Solar flux evolution
      - Active region growth/decay
      - Pre-flare signatures

    Input:  [B, T, D] — sequence of spatial embeddings from CNN
    Output: [B, D]    — global context embedding (last timestep + CLS token)
            [B, T, D] — all temporal representations
    """

    def __init__(
        self,
        d_model: int = 256,
        n_heads: int = 4,
        n_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        max_seq_len: int = 64,
    ):
        super().__init__()
        self.d_model = d_model

        # CLS token for global representation
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # Positional encoding
        self.pos_enc = SinusoidalPositionalEncoding(d_model, max_len=max_seq_len + 1, dropout=dropout)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,   # Pre-norm (more stable training)
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers,
            norm=nn.LayerNorm(d_model),
        )

        # Output projection
        self.out_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [B, T, D] — sequence of temporal embeddings
        Returns:
            context: [B, D]    — CLS token output (global context)
            all_out: [B, T, D] — all temporal outputs
        """
        B, T, D = x.shape

        # Prepend CLS token
        cls = self.cls_token.expand(B, 1, D)
        x = torch.cat([cls, x], dim=1)  # [B, T+1, D]

        # Positional encoding
        x = self.pos_enc(x)

        # Transformer
        out = self.transformer(x)  # [B, T+1, D]

        context = self.out_norm(out[:, 0])   # CLS token: [B, D]
        all_out = out[:, 1:]                  # [B, T, D]

        return context, all_out


class TelemetryEncoder(nn.Module):
    """
    Encodes GOES/SoLEXS telemetry features into embedding space.
    
    Input:  [B, T, F_tel] — time series of telemetry features
    Output: [B, T, D_tel] — embedded telemetry
    """

    def __init__(self, input_dim: int = 2, hidden_dim: int = 64, output_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, T, F_tel] → [B, T, D_tel]"""
        return self.encoder(x)


class MagneticFeatureEncoder(nn.Module):
    """
    Encodes HMI SHARP magnetic features (16 parameters) into embedding space.
    
    Input:  [B, T, 16] — magnetic features per timestep
            missing_mask: [B, T] — True where magnetic data is absent
    Output: [B, T, D_mag]
    """

    def __init__(
        self,
        input_dim: int = 16,
        output_dim: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(64, output_dim),
        )
        # Zero embedding for missing data (not arbitrary imputation)
        self.missing_embedding = nn.Parameter(torch.zeros(output_dim))

    def forward(self, x: torch.Tensor, missing_mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            x:            [B, T, 16]
            missing_mask: [B, T] bool — True where data is absent
        Returns:
            [B, T, D_mag]
        """
        out = self.encoder(x)   # [B, T, D]
        if missing_mask is not None:
            # Replace missing timesteps with learned missing-data embedding
            mask = missing_mask.unsqueeze(-1).float()   # [B, T, 1]
            missing = self.missing_embedding.view(1, 1, -1).expand_as(out)
            out = out * (1 - mask) + missing * mask
        return out
