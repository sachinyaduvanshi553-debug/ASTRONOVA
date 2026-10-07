"""
models/vision/image_forecaster.py — Multimodal fusion, horizon-conditioned future image decoder, and loss functions.

SCIENTIFIC DISCLAIMER:
    All predicted future images are AI FORECAST VISUALIZATIONS.
    They are NEVER observed solar images.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

HORIZONS = ["h15m", "h30m", "h60m", "h360m", "h720m", "h1440m"]


class SSIMLoss(nn.Module):
    """
    Structural Similarity Index (SSIM) loss for image quality.
    Loss = 1.0 - mean(SSIM).
    """

    def __init__(self, window_size: int = 11, channels: int = 3):
        super().__init__()
        self.window_size = window_size
        self.channels = channels
        self.register_buffer("window", self._create_window(window_size, channels))

    def _gaussian_kernel(self, size: int, sigma: float = 1.5) -> torch.Tensor:
        coords = torch.arange(size).float() - size // 2
        kernel = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        return kernel / kernel.sum()

    def _create_window(self, size: int, channels: int) -> torch.Tensor:
        k1d = self._gaussian_kernel(size)
        k2d = k1d.unsqueeze(1) * k1d.unsqueeze(0)
        window = k2d.unsqueeze(0).unsqueeze(0).repeat(channels, 1, 1, 1)
        return window

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        C = pred.shape[1]
        window = self.window.to(dtype=pred.dtype, device=pred.device)
        pad = self.window_size // 2

        mu_x = F.conv2d(pred, window, padding=pad, groups=C)
        mu_y = F.conv2d(target, window, padding=pad, groups=C)
        mu_xx = mu_x ** 2
        mu_yy = mu_y ** 2
        mu_xy = mu_x * mu_y

        sigma_xx = F.conv2d(pred ** 2, window, padding=pad, groups=C) - mu_xx
        sigma_yy = F.conv2d(target ** 2, window, padding=pad, groups=C) - mu_yy
        sigma_xy = F.conv2d(pred * target, window, padding=pad, groups=C) - mu_xy

        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        numerator = (2 * mu_xy + C1) * (2 * sigma_xy + C2)
        denominator = (mu_xx + mu_yy + C1) * (sigma_xx + sigma_yy + C2)
        ssim_map = numerator / (denominator + 1e-8)

        return 1.0 - ssim_map.mean()


class MultimodalFusion(nn.Module):
    """
    Multimodal feature fusion:
    Image + Temporal + Telemetry + Magnetic + Physics embeddings -> Fused Latent.
    Structure: concat -> LayerNorm -> Linear -> GELU -> Dropout -> Linear.
    """

    def __init__(
        self,
        image_dim: int = 256,
        telemetry_dim: int = 64,
        magnetic_dim: int = 64,
        physics_dim: int = 64,
        output_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        in_features = image_dim + telemetry_dim + magnetic_dim + physics_dim
        self.norm = nn.LayerNorm(in_features)
        self.fc1 = nn.Linear(in_features, output_dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(output_dim, output_dim)

    def forward(
        self,
        image_emb: torch.Tensor,
        telemetry_emb: torch.Tensor,
        magnetic_emb: torch.Tensor,
        physics_emb: torch.Tensor,
    ) -> torch.Tensor:
        # Concatenate all modal representations: [B, in_features]
        x = torch.cat([image_emb, telemetry_emb, magnetic_emb, physics_emb], dim=-1)
        x = self.norm(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        return x


class ImageForecastDecoder(nn.Module):
    """
    Decodes spatial features + horizon condition into a predicted future solar image [B, 3, H, W] in [0, 1].
    """

    def __init__(
        self,
        input_dim: int = 256,
        latent_dim: int = 256,
        output_channels: int = 3,
        target_size: int = 256,
        n_horizons: int = 6,
    ):
        super().__init__()
        self.target_size = target_size
        self.horizon_embedding = nn.Embedding(n_horizons, 32)
        self.horizon_map = {h: i for i, h in enumerate(HORIZONS)}

        # Channel projection combining spatial maps and horizon condition
        self.proj = nn.Sequential(
            nn.Conv2d(input_dim + 32, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.GELU(),
        )

        # Upsampling stages
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.GELU(),
            nn.Conv2d(16, output_channels, kernel_size=3, padding=1),
        )

    def forward(self, spatial_map: torch.Tensor, horizon: str = "h60m") -> torch.Tensor:
        B, C, H, W = spatial_map.shape
        h_idx = self.horizon_map.get(horizon, 2)  # default h60m
        h_idx_tensor = torch.full((B,), h_idx, dtype=torch.long, device=spatial_map.device)
        h_emb = self.horizon_embedding(h_idx_tensor)  # [B, 32]
        h_emb_spatial = h_emb.view(B, 32, 1, 1).expand(B, 32, H, W)

        fused = torch.cat([spatial_map, h_emb_spatial], dim=1)  # [B, C+32, H, W]
        x = self.proj(fused)
        out = self.decoder(x)

        if out.shape[-1] != self.target_size or out.shape[-2] != self.target_size:
            out = F.interpolate(
                out,
                size=(self.target_size, self.target_size),
                mode="bilinear",
                align_corners=False,
            )

        return torch.sigmoid(out)


class ImageForecastLoss(nn.Module):
    """
    Composite Loss for the complete multimodal solar forecasting pipeline:
      L_total = lambda_cls * L_cls + lambda_image * L_image + lambda_location * L_location
    """

    def __init__(
        self,
        lambda_cls: float = 1.0,
        lambda_image: float = 0.5,
        lambda_location: float = 0.5,
        lambda_c: float = 0.5,
        lambda_m: float = 1.0,
        lambda_x: float = 0.75,
        focal_gamma: float = 2.0,
        focal_alpha: float = 0.25,
        ssim_weight: float = 0.5,
        channels: int = 3,
    ):
        super().__init__()
        self.lambda_cls = lambda_cls
        self.lambda_image = lambda_image
        self.lambda_location = lambda_location
        self.class_weights = torch.tensor([lambda_c, lambda_m, lambda_x])
        self.focal_gamma = focal_gamma
        self.focal_alpha = focal_alpha
        self.ssim_weight = ssim_weight

        self.l1 = nn.L1Loss()
        self.mse = nn.MSELoss()
        self.ssim = SSIMLoss(channels=channels)

    def _focal_bce_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Focal loss for multi-label threshold forecasting [C+, M+, X+]."""
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        probs = torch.sigmoid(logits)
        pt = probs * targets + (1 - probs) * (1 - targets)
        focal = self.focal_alpha * ((1 - pt) ** self.focal_gamma) * bce

        weights = self.class_weights.to(logits.device).view(1, 3)
        weighted_focal = focal * weights
        return weighted_focal.mean()

    def _dice_loss(self, pred: torch.Tensor, target: torch.Tensor, smooth: float = 1e-5) -> torch.Tensor:
        intersection = (pred * target).sum(dim=(-2, -1))
        union = pred.sum(dim=(-2, -1)) + target.sum(dim=(-2, -1))
        dice = (2.0 * intersection + smooth) / (union + smooth)
        return 1.0 - dice.mean()

    def forward(
        self,
        predictions: Dict[str, Any],
        targets: Dict[str, Any],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        device = next(
            (v.device for v in predictions.get("class_logits", {}).values() if isinstance(v, torch.Tensor)),
            torch.device("cpu"),
        )
        loss_components: Dict[str, float] = {}

        # 1. Classification Loss (Averaged across the 6 horizons)
        cls_loss = torch.tensor(0.0, device=device)
        n_horizons = 0
        pred_logits = predictions.get("class_logits", {})
        true_labels = targets.get("labels", {})

        for h in HORIZONS:
            if h in pred_logits and h in true_labels:
                h_logits = pred_logits[h]
                h_target = true_labels[h].to(device)
                h_loss = self._focal_bce_loss(h_logits, h_target)
                cls_loss = cls_loss + h_loss
                loss_components[f"cls_{h}"] = h_loss.item()
                n_horizons += 1

        if n_horizons > 0:
            cls_loss = cls_loss / n_horizons
        loss_components["cls_total"] = cls_loss.item()

        # 2. Genuine Future Image Loss (ONLY computed when genuine target exists!)
        img_loss = torch.tensor(0.0, device=device)
        n_img_targets = 0
        future_images = predictions.get("future_images", {})
        target_images = targets.get("future_images", {})
        target_avail = targets.get("target_available", {})

        for h in HORIZONS:
            avail = target_avail.get(h, False)
            if isinstance(avail, torch.Tensor):
                has_avail = bool(avail.any().item())
            else:
                has_avail = bool(avail)

            if has_avail and h in target_images and target_images[h] is not None:
                p_img = future_images.get(h, predictions.get("predicted_future_image"))
                t_img = target_images[h].to(device)
                if p_img is not None:
                    if isinstance(avail, torch.Tensor) and not avail.all():
                        mask = avail.to(device)
                        p_img = p_img[mask]
                        t_img = t_img[mask]
                    if p_img.shape[0] > 0:
                        l1_val = self.l1(p_img, t_img)
                        ssim_val = self.ssim(p_img, t_img)
                        h_img_loss = l1_val + self.ssim_weight * ssim_val
                        img_loss = img_loss + h_img_loss
                        n_img_targets += 1
                        loss_components[f"img_{h}"] = h_img_loss.item()

        if n_img_targets > 0:
            img_loss = img_loss / n_img_targets
        loss_components["img_total"] = img_loss.item()

        # 3. Spatial Location Head Loss (Only when spatial mask ground truth is available)
        loc_loss = torch.tensor(0.0, device=device)
        spatial_target = targets.get("spatial_mask")
        if spatial_target is not None and "location_logits" in predictions:
            loc_logits = predictions["location_logits"]
            st = spatial_target.to(device)
            bce_loc = F.binary_cross_entropy_with_logits(loc_logits, st)
            dice_loc = self._dice_loss(torch.sigmoid(loc_logits), st)
            loc_loss = bce_loc + dice_loc
            loss_components["location_bce"] = bce_loc.item()
            loss_components["location_dice"] = dice_loc.item()
        loss_components["location_total"] = loc_loss.item()

        # Total Loss
        total_loss = (
            self.lambda_cls * cls_loss
            + self.lambda_image * img_loss
            + self.lambda_location * loc_loss
        )
        loss_components["total_loss"] = total_loss.item()

        return total_loss, loss_components
