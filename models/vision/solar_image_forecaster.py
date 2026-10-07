"""
models/vision/solar_image_forecaster.py — Complete Multimodal Solar Flare Forecasting Model.

Predicts:
  1. Six-horizon flare class threshold probabilities (C+, M+, X+) via independent sigmoid heads.
  2. Spatial flare-location probability heatmaps.
  3. Horizon-conditioned predicted future solar images (AI Forecast Visualizations).
  4. Epistemic uncertainty estimation via MC-Dropout.

CRITICAL SCIENTIFIC SAFETY RULES:
  - Output images are labeled: image_type = "AI_FORECAST"
  - Never claim generated images are observed solar observations.
  - Primary forecasting task: M+ flare probability (m_plus_probability).
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .cnn_encoder import CNNSpatialEncoder
from .conv_lstm import ConvLSTM
from .temporal_transformer import TemporalTransformer
from .flare_location_head import FlareLocationHead
from .image_forecaster import ImageForecastDecoder, MultimodalFusion

HORIZONS = ["h15m", "h30m", "h60m", "h360m", "h720m", "h1440m"]
HORIZON_DISPLAY = {
    "h15m": "+15 min",
    "h30m": "+30 min",
    "h60m": "+60 min",
    "h360m": "+6 hours",
    "h720m": "+12 hours",
    "h1440m": "+24 hours",
}
MAGNETIC_16_FEATURES = [
    "ABSNJZH", "AREA_ACR", "MEANALP", "MEANJZH", "MEANPOT", "MEANSHR",
    "R_VALUE", "SAVNCPP", "TOTBSQ", "TOTFX", "TOTFY", "TOTFZ",
    "TOTPOT", "TOTUSJH", "TOTUSJZ", "USFLUX"
]


class SolarTelemetryEncoder(nn.Module):
    """Encodes GOES X-ray telemetry vectors (flux, log_flux, statistics)."""

    def __init__(self, input_dim: int = 2, output_dim: int = 64, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, output_dim),
            nn.LayerNorm(output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Accepts [B, D] or [B, T, D]
        if x.dim() == 3:
            # Average/pool across temporal dim or take last
            x = x.mean(dim=1)
        return self.net(x)


class SolarMagneticEncoder(nn.Module):
    """
    Encodes photospheric magnetic features (16 HMI SHARP parameters).
    Architecture: 16 -> 128 -> 128 with LayerNorm/GELU and missing feature mask support.
    """

    def __init__(self, input_dim: int = 16, output_dim: int = 64, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, output_dim),
            nn.LayerNorm(output_dim),
        )

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # If [B, T, 16], reduce to [B, 16]
        if x.dim() == 3:
            x = x[:, -1, :]
        if mask is not None and mask.dim() == 3:
            mask = mask[:, -1, :]
            x = x * (~mask).float()
        return self.net(x)


class SolarPhysicsEncoder(nn.Module):
    """Encodes optional space weather physics vectors."""

    def __init__(self, input_dim: int = 5, output_dim: int = 64, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, output_dim),
            nn.LayerNorm(output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.mean(dim=1)
        return self.net(x)


class HorizonClassificationHead(nn.Module):
    """
    Independent binary threshold classification head for a specific horizon.
    Outputs raw logits for [C+, M+, X+].
    """

    def __init__(self, input_dim: int = 256, hidden_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 3),  # 3 independent threshold logits (C+, M+, X+)
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = self.net(x)  # [B, 3]
        probs = torch.sigmoid(logits)  # [B, 3] in [0, 1]
        return logits, probs


class SolarImageForecaster(nn.Module):
    """
    Full Multimodal Spatiotemporal Solar Flare Forecasting Architecture.
    """

    def __init__(
        self,
        backbone: str = "resnet18",
        image_size: int = 256,
        channels: int = 3,
        pretrained_encoder: bool = False,
        temporal_model: str = "transformer",
        spatial_dim: int = 256,
        temporal_dim: int = 256,
        n_heads: int = 4,
        n_temporal_layers: int = 2,
        telemetry_dim: int = 2,
        tel_embed_dim: int = 64,
        magnetic_dim: int = 16,
        mag_embed_dim: int = 64,
        physics_dim: int = 5,
        physics_embed_dim: int = 64,
        fusion_output_dim: int = 256,
        classifier_hidden: int = 128,
        dropout: float = 0.1,
        max_seq_len: int = 24,
    ):
        super().__init__()
        self.image_size = image_size
        self.channels = channels
        self.spatial_dim = spatial_dim
        self.temporal_dim = temporal_dim
        self.fusion_output_dim = fusion_output_dim
        self.temporal_model_type = temporal_model

        # 1. Spatial CNN Encoder
        self.cnn_encoder = CNNSpatialEncoder(
            backbone=backbone,
            output_dim=spatial_dim,
            pretrained=pretrained_encoder,
            channels=channels,
        )

        # 2. Temporal Sequence Encoder
        if temporal_model == "transformer":
            self.temporal = TemporalTransformer(
                d_model=spatial_dim,
                n_heads=n_heads,
                n_layers=n_temporal_layers,
                dim_feedforward=spatial_dim * 2,
                dropout=dropout,
                max_seq_len=max_seq_len + 1,
            )
        elif temporal_model == "lstm":
            self.temporal = ConvLSTM(
                input_dim=spatial_dim,
                hidden_dim=temporal_dim,
                n_layers=2,
                dropout=dropout,
            )
        else:
            raise ValueError(f"Unknown temporal_model: {temporal_model}")

        # 3. Telemetry, Magnetic, and Physics Encoders
        self.telemetry_dim = telemetry_dim
        self.magnetic_dim = magnetic_dim
        self.physics_dim = physics_dim
        self.telemetry_encoder = SolarTelemetryEncoder(telemetry_dim, tel_embed_dim, dropout)
        self.magnetic_encoder = SolarMagneticEncoder(magnetic_dim, mag_embed_dim, dropout)
        self.physics_encoder = SolarPhysicsEncoder(physics_dim, physics_embed_dim, dropout)

        # 4. Multimodal Fusion
        self.fusion = MultimodalFusion(
            image_dim=spatial_dim,
            telemetry_dim=tel_embed_dim,
            magnetic_dim=mag_embed_dim,
            physics_dim=physics_embed_dim,
            output_dim=fusion_output_dim,
            dropout=dropout,
        )

        # 5. Six Independent Horizon Classification Heads (C+, M+, X+)
        self.classification_heads = nn.ModuleDict({
            h: HorizonClassificationHead(fusion_output_dim, classifier_hidden, dropout)
            for h in HORIZONS
        })

        # 6. Spatial Flare Location Head
        self.location_head = FlareLocationHead(
            input_dim=spatial_dim,
            target_size=image_size,
            hidden_dim=64,
        )

        # 7. Horizon-Conditioned Future Image Decoder Head
        self.image_decoder = ImageForecastDecoder(
            input_dim=spatial_dim,
            latent_dim=fusion_output_dim,
            output_channels=channels,
            target_size=image_size,
            n_horizons=len(HORIZONS),
        )

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(
        self,
        image_seq: torch.Tensor,
        telemetry: Optional[torch.Tensor] = None,
        magnetic: Optional[torch.Tensor] = None,
        physics: Optional[torch.Tensor] = None,
        mag_missing: Optional[torch.Tensor] = None,
        forecast_horizon: str = "h60m",
    ) -> Dict[str, Any]:
        """
        Args:
            image_seq: [B, T, C, H, W]
            telemetry: [B, T, telemetry_dim] or [B, telemetry_dim] or None
            magnetic:  [B, T, 16] or [B, 16] or None
            physics:   [B, 5] or None
            mag_missing: [B, T] or None
            forecast_horizon: standard horizon code (e.g. "h60m")
        """
        B, T, C, H, W = image_seq.shape
        device = image_seq.device

        # 1. Spatial Encoding
        spatial_maps, pooled_embeds = self.cnn_encoder(image_seq)
        last_spatial = spatial_maps[:, -1]  # [B, spatial_dim, H', W']
        last_image_emb = pooled_embeds[:, -1]  # [B, spatial_dim]

        # 2. Temporal Modeling
        if self.temporal_model_type == "transformer":
            temporal_ctx, _ = self.temporal(pooled_embeds)  # [B, spatial_dim]
        else:
            last_h, _ = self.temporal(spatial_maps)
            temporal_ctx = self.global_pool(last_h).view(B, -1)

        # 3. Telemetry Encoding (with zero-fallback)
        if telemetry is None:
            telemetry = torch.zeros(B, self.telemetry_dim, device=device)
        tel_emb = self.telemetry_encoder(telemetry)

        # 4. Magnetic Encoding (with zero-fallback and mask)
        if magnetic is None:
            magnetic = torch.zeros(B, self.magnetic_dim, device=device)
            if mag_missing is None:
                mag_missing = torch.ones(B, dtype=torch.bool, device=device)
        mag_emb = self.magnetic_encoder(magnetic, mag_missing)

        # 5. Physics Encoding (with zero-fallback)
        if physics is None:
            physics = torch.zeros(B, self.physics_dim, device=device)
        phys_emb = self.physics_encoder(physics)

        # 6. Multimodal Fusion
        fused_latent = self.fusion(temporal_ctx, tel_emb, mag_emb, phys_emb)  # [B, fusion_output_dim]

        # 7. Six Horizon Classification Heads
        class_logits: Dict[str, torch.Tensor] = {}
        class_probs: Dict[str, torch.Tensor] = {}
        c_plus_probs: Dict[str, torch.Tensor] = {}
        m_plus_probs: Dict[str, torch.Tensor] = {}
        x_plus_probs: Dict[str, torch.Tensor] = {}

        for h, head in self.classification_heads.items():
            logits, probs = head(fused_latent)
            class_logits[h] = logits
            class_probs[h] = probs
            c_plus_probs[h] = probs[:, 0]
            m_plus_probs[h] = probs[:, 1]  # Primary M+ task!
            x_plus_probs[h] = probs[:, 2]

        # 8. Spatial Flare Location Prediction
        location_logits = self.location_head.forward_logits(last_spatial)
        location_heatmap = torch.sigmoid(location_logits)  # [B, 1, H, W] in [0, 1]

        # 9. Future Image Decoding for requested horizon and all horizons
        predicted_future_image = self.image_decoder(last_spatial, horizon=forecast_horizon)
        future_images: Dict[str, torch.Tensor] = {
            h: self.image_decoder(last_spatial, horizon=h)
            for h in HORIZONS
        }

        return {
            "class_logits": class_logits,
            "class_probs": class_probs,
            "c_plus_probability": c_plus_probs,
            "m_plus_probability": m_plus_probs,  # Primary scientific task
            "x_plus_probability": x_plus_probs,
            "location_logits": location_logits,
            "location_heatmap": location_heatmap,
            "flare_heatmap": location_heatmap,  # compatibility alias
            "predicted_future_image": predicted_future_image,
            "predicted_image": predicted_future_image,  # compatibility alias
            "future_images": future_images,
            "image_type": "AI_FORECAST",
            "forecast_horizon": forecast_horizon,
            "latent": fused_latent,
            "spatial_maps": last_spatial,
        }

    def forecast_image(self, image_seq: torch.Tensor, horizon: str = "h60m") -> torch.Tensor:
        """Convenience method to generate an AI forecast image for a specific horizon."""
        spatial_maps, _ = self.cnn_encoder(image_seq)
        last_spatial = spatial_maps[:, -1]
        return self.image_decoder(last_spatial, horizon=horizon)

    @contextmanager
    def mc_dropout_mode(self):
        """Activates dropout during evaluation while keeping normalization layers fixed."""
        self.train()
        for m in self.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm)):
                m.eval()
        try:
            yield
        finally:
            self.eval()

    def predict_with_uncertainty(
        self,
        image_seq: torch.Tensor,
        telemetry: Optional[torch.Tensor] = None,
        magnetic: Optional[torch.Tensor] = None,
        physics: Optional[torch.Tensor] = None,
        mag_missing: Optional[torch.Tensor] = None,
        n_passes: int = 20,
    ) -> Dict[str, Any]:
        """MC-Dropout epistemic uncertainty estimation."""
        self.eval()
        all_probs: Dict[str, List[torch.Tensor]] = {h: [] for h in HORIZONS}
        all_heatmaps: List[torch.Tensor] = []
        all_images: List[torch.Tensor] = []

        with self.mc_dropout_mode():
            with torch.no_grad():
                for _ in range(n_passes):
                    out = self.forward(
                        image_seq=image_seq,
                        telemetry=telemetry,
                        magnetic=magnetic,
                        physics=physics,
                        mag_missing=mag_missing,
                    )
                    for h in HORIZONS:
                        all_probs[h].append(out["class_probs"][h])
                    all_heatmaps.append(out["location_heatmap"])
                    all_images.append(out["predicted_future_image"])

        mean_probs, std_probs = {}, {}
        for h in HORIZONS:
            stk = torch.stack(all_probs[h], dim=0)  # [N, B, 3]
            mean_probs[h] = stk.mean(dim=0)
            std_probs[h] = stk.std(dim=0)

        mean_heatmap = torch.stack(all_heatmaps, dim=0).mean(dim=0)
        mean_image = torch.stack(all_images, dim=0).mean(dim=0)
        std_image = torch.stack(all_images, dim=0).std(dim=0)

        return {
            "class_probs": mean_probs,
            "class_std": std_probs,
            "location_heatmap": mean_heatmap,
            "flare_heatmap": mean_heatmap,
            "predicted_future_image": mean_image,
            "predicted_image": mean_image,
            "uncertainty_image": std_image,
            "image_type": "AI_FORECAST",
            "n_mc_passes": n_passes,
        }

    def get_config(self) -> Dict[str, Any]:
        return {
            "model_class": "SolarImageForecaster",
            "backbone": self.cnn_encoder.backbone_name,
            "image_size": self.image_size,
            "channels": self.channels,
            "spatial_dim": self.spatial_dim,
            "temporal_model": self.temporal_model_type,
            "horizons": HORIZONS,
            "forecast_classes": ["C_plus", "M_plus", "X_plus"],
            "image_type": "AI_FORECAST",
            "scientific_disclaimer": (
                "Predicted images are AI Forecast Visualizations, NOT actual solar observations. "
                "Always label: AI Forecast / Predicted Solar Flare Visualization"
            ),
        }
