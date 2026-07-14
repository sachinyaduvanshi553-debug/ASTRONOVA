import torch
import torch.nn as nn
import numpy as np
import cv2
from typing import Dict, Any, Optional


class UncertaintyEngine:
    """
    Estimates uncertainty for solar flare predictions using authentic Monte Carlo Dropout.
    Provides pixel-level uncertainty, credible intervals, and class entropy.
    """
    def __init__(self, model: nn.Module, n_samples: int = 20, device: str = 'cpu'):
        self.model = model
        self.n_samples = n_samples
        self.device = device

    def _enable_dropout(self):
        """Set model to train mode (enables dropout) but disable BN updates."""
        self.model.train()
        for module in self.model.modules():
            if isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.BatchNorm1d):
                module.eval()

    def _disable_dropout(self):
        """Set model back to eval mode."""
        self.model.eval()

    def compute_pixel_uncertainty(self, images: torch.Tensor, telemetry: torch.Tensor, physics: torch.Tensor) -> Dict[str, Any]:
        """
        Run n_samples MC dropout passes to compute predictive uncertainty.
        """
        self._enable_dropout()
        
        preds_images = []
        preds_flux = []
        preds_class_probs = []

        with torch.no_grad():
            for _ in range(self.n_samples):
                out = self.model(images, telemetry, physics)
                # Ensure we handle both dictionary outputs and direct tensor outputs for backward compat
                if isinstance(out, dict):
                    preds_images.append(out['predicted_image'].cpu().numpy())
                    preds_flux.append(out['reg_output'].cpu().numpy())
                    preds_class_probs.append(out['class_probs'].cpu().numpy())
                else:
                    preds_images.append(out.cpu().numpy())
                    # Dummy values if model doesn't have heads yet
                    preds_flux.append(np.zeros((images.shape[0], 1)))
                    preds_class_probs.append(np.ones((images.shape[0], 5)) * 0.2)
                    
        self._disable_dropout()

        # Stack to (N, B, C, H, W)
        preds_images = np.stack(preds_images)
        preds_flux = np.stack(preds_flux)
        preds_class_probs = np.stack(preds_class_probs)
        
        # We process the first item in the batch
        img_stack = preds_images[:, 0] # (N, C, H, W)
        
        mean_image = np.mean(img_stack, axis=0) # (C, H, W)
        pixel_variance = np.var(img_stack, axis=0) # (C, H, W)
        # Average variance across channels
        pixel_variance_spatial = np.mean(pixel_variance, axis=0) # (H, W)
        pixel_std = np.std(img_stack, axis=0) # (C, H, W)
        
        # Confidence score (inverse of normalized variance)
        mean_var = np.mean(pixel_variance_spatial)
        confidence = float(1.0 / (1.0 + mean_var))
        
        # Credible intervals (2.5th and 97.5th percentiles)
        credible_low = np.percentile(img_stack, 2.5, axis=0)
        credible_high = np.percentile(img_stack, 97.5, axis=0)
        
        # Flux uncertainty
        flux_stack = preds_flux[:, 0]
        flux_uncertainty = float(np.std(flux_stack))
        mean_flux = float(np.mean(flux_stack))
        
        # Class uncertainty (entropy of mean probabilities)
        mean_probs = np.mean(preds_class_probs[:, 0], axis=0)
        entropy = -np.sum(mean_probs * np.log(mean_probs + 1e-8))
        class_uncertainty = float(entropy)
        
        return {
            "mean_image": mean_image,
            "pixel_variance": pixel_variance_spatial,
            "pixel_std": pixel_std,
            "confidence": confidence,
            "credible_interval_low": credible_low,
            "credible_interval_high": credible_high,
            "class_uncertainty": class_uncertainty,
            "flux_uncertainty": flux_uncertainty,
            "mean_class_probs": mean_probs,
            "mean_flux": mean_flux
        }

    def compute_class_uncertainty(self, images: torch.Tensor, telemetry: torch.Tensor, physics: torch.Tensor) -> Dict[str, Any]:
        """Wrapper for compute_pixel_uncertainty that returns just the class info."""
        res = self.compute_pixel_uncertainty(images, telemetry, physics)
        return {"entropy": res["class_uncertainty"]}

    def generate_uncertainty_heatmap(self, pixel_variance: np.ndarray, original_image: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Generate colored heatmap of uncertainty overlaid on original image.
        """
        # Normalize variance map to 0-255
        var_norm = (pixel_variance - np.min(pixel_variance)) / (np.max(pixel_variance) - np.min(pixel_variance) + 1e-8)
        var_uint8 = np.uint8(var_norm * 255)
        
        # Apply JET colormap (Red = high uncertainty, Blue = low)
        heatmap = cv2.applyColorMap(var_uint8, cv2.COLORMAP_JET)
        
        if original_image is not None:
            # Ensure original image is same size and 3 channels
            if len(original_image.shape) == 2:
                original_image = cv2.cvtColor(original_image, cv2.COLOR_GRAY2BGR)
            elif original_image.shape[2] == 4:
                original_image = cv2.cvtColor(original_image, cv2.COLOR_BGRA2BGR)
            
            orig_resized = cv2.resize(original_image, (heatmap.shape[1], heatmap.shape[0]))
            
            # Overlay
            alpha = 0.5
            overlay = cv2.addWeighted(heatmap, alpha, orig_resized, 1 - alpha, 0)
            return overlay
            
        return heatmap
