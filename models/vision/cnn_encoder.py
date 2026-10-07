"""
models/vision/cnn_encoder.py — CNN spatial encoder for solar images.

Supports:
    - ResNet18 (lighter, faster, CPU-friendly)
    - ResNet50 (more capacity, recommended for larger datasets)

Input:  [B, T, C, H, W] — batch of image sequences
Output: [B, T, D, H', W'] — spatial feature maps per timestep
        [B, T, D]         — global pooled embeddings per timestep

PRETRAINED WEIGHTS NOTE:
    ImageNet pretraining is documented but scientifically debatable for solar imagery.
    SDO images have very different statistics from natural images.
    Default: pretrained=False for clean scientific baselines.
    Set pretrained=True only for transfer-learning experiments and document clearly.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torchvision.models as tvm


class CNNSpatialEncoder(nn.Module):
    """
    Per-frame CNN encoder for solar image sequences.
    
    Processes each timestep independently, then temporal model fuses across time.
    
    Args:
        backbone: "resnet18" or "resnet50"
        output_dim: dimension of output embedding D
        pretrained: use ImageNet pretrained weights
                    (document in experiment config if True)
        channels: input channels (3=RGB, 1=grayscale)
    
    Input:  [B, T, C, H, W]
    Output: (spatial_maps, pooled_embeds)
            spatial_maps:   [B, T, D, H', W']
            pooled_embeds:  [B, T, D]
    """

    def __init__(
        self,
        backbone: str = "resnet18",
        output_dim: int = 256,
        pretrained: bool = False,
        channels: int = 3,
    ):
        super().__init__()
        self.backbone_name = backbone
        self.output_dim = output_dim
        self.pretrained = pretrained

        # Build backbone
        weights = None
        if backbone == "resnet18":
            if pretrained:
                weights = tvm.ResNet18_Weights.IMAGENET1K_V1
            base = tvm.resnet18(weights=weights)
            base_out_channels = 512
        elif backbone == "resnet50":
            if pretrained:
                weights = tvm.ResNet50_Weights.IMAGENET1K_V2
            base = tvm.resnet50(weights=weights)
            base_out_channels = 2048
        else:
            raise ValueError(f"Unsupported backbone: {backbone}. Use 'resnet18' or 'resnet50'")

        # Adapt first conv for arbitrary channel count
        if channels != 3:
            base.conv1 = nn.Conv2d(
                channels, 64, kernel_size=7, stride=2, padding=3, bias=False
            )
            nn.init.kaiming_normal_(base.conv1.weight, mode="fan_out", nonlinearity="relu")

        # Remove final classification layers, keep feature extractor
        self.feature_extractor = nn.Sequential(
            base.conv1, base.bn1, base.relu, base.maxpool,
            base.layer1, base.layer2, base.layer3, base.layer4,
        )

        # Project to output_dim
        self.spatial_proj = nn.Sequential(
            nn.Conv2d(base_out_channels, output_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(output_dim),
            nn.ReLU(inplace=True),
        )
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        if pretrained:
            import logging
            logging.getLogger(__name__).warning(
                "Using ImageNet pretrained weights for %s. "
                "Document this decision in experiment config. "
                "Solar imagery differs substantially from ImageNet distribution.",
                backbone,
            )

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [B, T, C, H, W]
        Returns:
            spatial_maps:  [B, T, D, H', W']
            pooled_embeds: [B, T, D]
        """
        B, T, C, H, W = x.shape

        # Process all frames in batch for efficiency
        x_flat = x.view(B * T, C, H, W)
        feats = self.feature_extractor(x_flat)   # [B*T, base_C, H', W']
        feats = self.spatial_proj(feats)           # [B*T, D, H', W']

        _, D, Hp, Wp = feats.shape

        spatial_maps = feats.view(B, T, D, Hp, Wp)
        pooled = self.global_pool(feats).view(B, T, D)

        return spatial_maps, pooled

    def get_parameter_count(self) -> dict[str, int]:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable}
