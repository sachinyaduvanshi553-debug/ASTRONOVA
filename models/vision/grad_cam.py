"""
models/vision/grad_cam.py — Gradient-weighted Class Activation Mapping (Grad-CAM)
for solar flare forecasting model.

Generates visual explanations showing which regions of the solar image
most influenced the model's flare prediction.

SCIENTIFIC DISCLAIMER:
    Grad-CAM heatmaps show model attention, NOT physical causation.
    They indicate which image regions the model weighted most heavily
    for its prediction — this is an AI explanation, not a physics derivation.
    
    Always label: "AI Attention / Model Explanation Heatmap"

Reference: Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks
           via Gradient-based Localization" (ICCV 2017)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class GradCAM:
    """
    Grad-CAM for SolarImageForecaster.
    
    Hooks into the CNN encoder's last convolutional layer to capture
    activations and gradients, then computes class-discriminative heatmaps.
    
    Usage:
        cam = GradCAM(model, target_layer="cnn_encoder.feature_extractor")
        heatmap = cam.generate(image_seq, telemetry, horizon="h60m", class_idx=1)
        cam.remove_hooks()
    """

    def __init__(self, model: nn.Module, target_layer: str | None = None):
        self.model = model
        self.gradients = None
        self.activations = None
        self._hooks = []

        # Find target layer
        if target_layer is None:
            target_layer = "cnn_encoder.feature_extractor"

        layer = self._get_layer(model, target_layer)
        if layer is None:
            raise ValueError(f"Could not find layer: {target_layer}")

        # Register hooks
        self._hooks.append(
            layer.register_forward_hook(self._forward_hook)
        )
        self._hooks.append(
            layer.register_full_backward_hook(self._backward_hook)
        )

    def _get_layer(self, model: nn.Module, layer_path: str) -> nn.Module | None:
        """Navigate nested module path like 'cnn_encoder.feature_extractor'."""
        parts = layer_path.split(".")
        current = model
        for part in parts:
            if hasattr(current, part):
                current = getattr(current, part)
            else:
                return None
        return current

    def _forward_hook(self, module, input, output):
        self.activations = output.detach()

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(
        self,
        image_seq: torch.Tensor,
        telemetry: torch.Tensor,
        magnetic: torch.Tensor | None = None,
        mag_missing: torch.Tensor | None = None,
        horizon: str = "h60m",
        class_idx: int = 1,  # 0=C+, 1=M+, 2=X+
        target_size: int | None = None,
    ) -> np.ndarray:
        """
        Generate Grad-CAM heatmap.
        
        Args:
            image_seq:  [B, T, C, H, W]
            telemetry:  [B, T, 2]
            horizon:    target horizon key
            class_idx:  0=C+, 1=M+, 2=X+
            target_size: output heatmap size (default: image_seq H)
        
        Returns:
            heatmap: numpy array [B, H, W] in [0, 1]
                     LABELED: "AI Attention / Model Explanation Heatmap"
        """
        self.model.eval()
        image_seq = image_seq.requires_grad_(True)

        # Forward pass
        output = self.model(image_seq, telemetry, magnetic, mag_missing)
        logits = output["class_logits"][horizon]  # [B, 3]

        # Select target class
        target = logits[:, class_idx]  # [B]

        # Backward pass
        self.model.zero_grad()
        target.sum().backward(retain_graph=True)

        if self.gradients is None or self.activations is None:
            raise RuntimeError("Grad-CAM hooks did not capture gradients/activations")

        # Compute weights: global average pool of gradients
        # activations: [B*T, C_feat, H', W']
        # gradients:   [B*T, C_feat, H', W']
        weights = self.gradients.mean(dim=(-2, -1), keepdim=True)  # [B*T, C, 1, 1]
        cam = (weights * self.activations).sum(dim=1, keepdim=True)  # [B*T, 1, H', W']
        cam = F.relu(cam)  # Only positive contributions

        # Use last frame's CAM (most relevant for prediction)
        B = image_seq.shape[0]
        T = image_seq.shape[1]
        cam = cam.view(B, T, 1, cam.shape[-2], cam.shape[-1])
        cam = cam[:, -1]  # [B, 1, H', W'] — last timestep

        # Resize
        if target_size is None:
            target_size = image_seq.shape[-1]
        cam = F.interpolate(cam, size=(target_size, target_size), mode="bilinear", align_corners=False)

        # Normalize per sample to [0, 1]
        cam = cam.squeeze(1)  # [B, H, W]
        for i in range(B):
            vmin, vmax = cam[i].min(), cam[i].max()
            if vmax - vmin > 1e-8:
                cam[i] = (cam[i] - vmin) / (vmax - vmin)

        return cam.detach().cpu().numpy()

    def remove_hooks(self):
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()

    def __del__(self):
        self.remove_hooks()


def overlay_cam_on_image(
    image: np.ndarray,
    cam: np.ndarray,
    alpha: float = 0.5,
    colormap: str = "jet",
) -> np.ndarray:
    """
    Overlay Grad-CAM heatmap on original image.
    
    Args:
        image: [H, W, 3] in [0, 1] or [0, 255]
        cam:   [H, W] in [0, 1]
        alpha: overlay transparency
    
    Returns:
        overlaid: [H, W, 3] in [0, 255] uint8
        MUST be labeled: "AI Attention / Model Explanation"
    """
    try:
        import cv2
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)

        # Resize cam to match image
        cam_resized = cv2.resize(cam.astype(np.float32), (image.shape[1], image.shape[0]))
        cam_uint8 = (cam_resized * 255).astype(np.uint8)

        heatmap = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

        overlaid = (alpha * heatmap + (1 - alpha) * image).clip(0, 255).astype(np.uint8)
        return overlaid
    except ImportError:
        # Fallback: simple red channel overlay
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        overlay = image.copy()
        overlay[:, :, 0] = np.clip(
            overlay[:, :, 0].astype(np.float32) + cam * 128, 0, 255
        ).astype(np.uint8)
        return overlay
