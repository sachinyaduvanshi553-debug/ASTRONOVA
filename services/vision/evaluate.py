"""
Evaluation metrics for the ASTRONOVA solar prediction pipeline.

Includes PSNR, MAE, RMSE, structural SSIM (Wang et al. 2004) implemented
from scratch with Gaussian-windowed PyTorch convolutions, and FID-like
feature extraction using the project's ResNet50 backbone.
"""

import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Basic pixel-level metrics
# ---------------------------------------------------------------------------

def calculate_psnr(pred: torch.Tensor, target: torch.Tensor,
                   data_range: float = 1.0) -> torch.Tensor:
    """Peak Signal-to-Noise Ratio (higher is better)."""
    mse = torch.mean((pred - target) ** 2)
    if mse == 0:
        return torch.tensor(float('inf'))
    return 20.0 * torch.log10(torch.tensor(data_range, device=mse.device) / torch.sqrt(mse))


def calculate_mae(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Mean Absolute Error (lower is better)."""
    return torch.mean(torch.abs(pred - target)).item()


def calculate_rmse(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Root Mean Squared Error (lower is better)."""
    return torch.sqrt(torch.mean((pred - target) ** 2)).item()


# ---------------------------------------------------------------------------
# Per-channel analysis helpers
# ---------------------------------------------------------------------------

def calculate_per_channel_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    data_range: float = 1.0,
) -> Dict[str, List[float]]:
    """Compute PSNR, MAE, RMSE, and SSIM independently for each channel.

    Parameters
    ----------
    pred, target : torch.Tensor
        Tensors of shape ``(C, H, W)`` or ``(B, C, H, W)``.
    data_range : float
        Dynamic range of the pixel values (1.0 for [0, 1]).

    Returns
    -------
    dict
        ``{'psnr': [...], 'mae': [...], 'rmse': [...], 'ssim': [...]}``
        where each list has one entry per channel.
    """
    if pred.dim() == 3:
        pred = pred.unsqueeze(0)
        target = target.unsqueeze(0)

    num_channels = pred.shape[1]
    results: Dict[str, List[float]] = {
        'psnr': [], 'mae': [], 'rmse': [], 'ssim': [],
    }

    for c in range(num_channels):
        p = pred[:, c:c + 1, :, :]
        t = target[:, c:c + 1, :, :]
        results['psnr'].append(calculate_psnr(p, t, data_range).item())
        results['mae'].append(calculate_mae(p, t))
        results['rmse'].append(calculate_rmse(p, t))
        results['ssim'].append(calculate_ssim(p, t, data_range=data_range).item())

    return results


# ---------------------------------------------------------------------------
# SSIM — Wang et al. 2004 (from-scratch, Gaussian window via PyTorch conv)
# ---------------------------------------------------------------------------

def _fspecial_gaussian(size: int = 11, sigma: float = 1.5,
                       channels: int = 1,
                       device: Optional[torch.device] = None) -> torch.Tensor:
    """Create a 2-D Gaussian kernel identical to MATLAB's fspecial('gaussian').

    Returns a ``(channels, 1, size, size)`` tensor suitable for use as a
    depthwise convolution weight.
    """
    coords = torch.arange(size, dtype=torch.float32, device=device) - (size - 1) / 2.0
    g = torch.exp(-(coords ** 2) / (2.0 * sigma ** 2))
    g_2d = g.unsqueeze(1) * g.unsqueeze(0)          # outer product → 2-D
    g_2d = g_2d / g_2d.sum()                         # normalise
    return g_2d.unsqueeze(0).unsqueeze(0).repeat(channels, 1, 1, 1)


def calculate_ssim(
    pred: torch.Tensor,
    target: torch.Tensor,
    data_range: float = 1.0,
    window_size: int = 11,
    sigma: float = 1.5,
    size_average: bool = True,
) -> torch.Tensor:
    """Structural Similarity Index (SSIM) — Wang et al., IEEE TIP 2004.

    Implemented from scratch using depthwise Gaussian-windowed convolutions in
    PyTorch.  No external libraries are required.

    Parameters
    ----------
    pred, target : torch.Tensor
        Images of shape ``(B, C, H, W)`` in ``[0, data_range]``.
    data_range : float
        Dynamic range of pixel values (e.g. 1.0 or 255.0).
    window_size : int
        Side length of the square Gaussian window (default 11).
    sigma : float
        Standard deviation of the Gaussian kernel (default 1.5).
    size_average : bool
        If ``True`` return the scalar mean SSIM; otherwise return the
        per-pixel SSIM map averaged over the batch.

    Returns
    -------
    torch.Tensor
        Scalar SSIM value (or per-pixel map).
    """
    if pred.dim() == 3:
        pred = pred.unsqueeze(0)
        target = target.unsqueeze(0)

    channels = pred.shape[1]
    window = _fspecial_gaussian(window_size, sigma, channels, pred.device).to(pred.dtype)
    pad = window_size // 2

    # Stability constants (as in the original paper)
    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2

    # Windowed statistics via depthwise convolution
    mu_pred = F.conv2d(pred, window, padding=pad, groups=channels)
    mu_target = F.conv2d(target, window, padding=pad, groups=channels)

    mu_pred_sq = mu_pred ** 2
    mu_target_sq = mu_target ** 2
    mu_pred_target = mu_pred * mu_target

    sigma_pred_sq = F.conv2d(pred * pred, window, padding=pad, groups=channels) - mu_pred_sq
    sigma_target_sq = F.conv2d(target * target, window, padding=pad, groups=channels) - mu_target_sq
    sigma_pred_target = F.conv2d(pred * target, window, padding=pad, groups=channels) - mu_pred_target

    # SSIM formula
    numerator = (2.0 * mu_pred_target + C1) * (2.0 * sigma_pred_target + C2)
    denominator = (mu_pred_sq + mu_target_sq + C1) * (sigma_pred_sq + sigma_target_sq + C2)
    ssim_map = numerator / denominator

    if size_average:
        return ssim_map.mean()
    return ssim_map.mean(dim=0)


# ---------------------------------------------------------------------------
# FID-like feature extraction using the project's ResNet50 encoder
# ---------------------------------------------------------------------------

def calculate_fid_features(
    images: torch.Tensor,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Extract 2048-d features from images using a ResNet50 backbone.

    These features can be used to compute Fréchet Inception Distance (FID) or
    similar distribution-level metrics between real and generated solar images.

    The function uses the same ResNet50 architecture employed by
    ``services.vision.encoder.ImageEncoder`` but takes the activations right
    before the final classification layer (global-average-pooled 2048-d
    vectors), matching the standard FID protocol.

    Parameters
    ----------
    images : torch.Tensor
        Batch of images ``(B, C, H, W)`` in ``[0, 1]``.  If *C* == 1 the
        channel is replicated to 3 so the backbone can process it.
    device : torch.device, optional
        Device to run inference on.  Defaults to CUDA if available.

    Returns
    -------
    torch.Tensor
        Feature matrix of shape ``(B, 2048)``.
    """
    import torchvision.models as models

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Build a headless ResNet50 with global average pooling → 2048-d
    resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    # Remove the final FC layer; keep everything up to and including avgpool
    backbone = nn.Sequential(*list(resnet.children())[:-1])  # ends at AdaptiveAvgPool2d
    backbone = backbone.to(device).eval()

    if images.dim() == 3:
        images = images.unsqueeze(0)

    # Replicate single-channel solar images to 3 channels for ResNet
    if images.shape[1] == 1:
        images = images.repeat(1, 3, 1, 1)

    images = images.to(device, dtype=torch.float32)

    with torch.no_grad():
        features = backbone(images)  # (B, 2048, 1, 1)
    return features.flatten(start_dim=1)  # (B, 2048)


def calculate_fid(
    real_features: torch.Tensor,
    generated_features: torch.Tensor,
    eps: float = 1e-6,
) -> float:
    """Compute Fréchet Inception Distance between two feature sets.

    Uses the closed-form formula:
        FID = ||μ_r − μ_g||² + Tr(Σ_r + Σ_g − 2 (Σ_r Σ_g)^{1/2})

    Parameters
    ----------
    real_features, generated_features : torch.Tensor
        Feature matrices of shape ``(N, D)`` (e.g. from ``calculate_fid_features``).
    eps : float
        Small value added to diagonal for numerical stability.

    Returns
    -------
    float
        The FID score (lower is better).
    """
    mu_r = real_features.mean(dim=0).cpu().numpy()
    mu_g = generated_features.mean(dim=0).cpu().numpy()

    # Covariance matrices (unbiased)
    r_np = real_features.cpu().numpy()
    g_np = generated_features.cpu().numpy()
    sigma_r = np.cov(r_np, rowvar=False)
    sigma_g = np.cov(g_np, rowvar=False)

    diff = mu_r - mu_g
    # Matrix square root via eigenvalue decomposition (avoids scipy dependency)
    product = sigma_r @ sigma_g
    eigvals, eigvecs = np.linalg.eigh(product)
    eigvals = np.maximum(eigvals, 0.0)  # clip small negatives from numerics
    sqrt_product = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T

    fid = float(diff @ diff + np.trace(sigma_r + sigma_g - 2.0 * sqrt_product))
    return fid


# ---------------------------------------------------------------------------
# Evaluator class
# ---------------------------------------------------------------------------

class Evaluator:
    """End-to-end evaluation harness for ASTRONOVA prediction models."""

    def __init__(self, model: nn.Module,
                 device: Optional[str] = None):
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)
        self.model = model.to(self.device)

    # ----- core evaluation loop -------------------------------------------

    def evaluate_dataset(
        self,
        dataloader,
        data_range: float = 1.0,
    ) -> Dict[str, float]:
        """Run the model over *dataloader* and return averaged metrics.

        Returns PSNR, MAE, RMSE, **and SSIM** for every sample.
        """
        self.model.eval()
        metrics: Dict[str, List[float]] = {
            'psnr': [], 'mae': [], 'rmse': [], 'ssim': [],
        }

        with torch.no_grad():
            for batch in dataloader:
                images = batch['image'].to(self.device, dtype=torch.float32)
                telemetry = batch['telemetry'].to(self.device, dtype=torch.float32)
                physics = batch['physics'].to(self.device, dtype=torch.float32)
                targets = batch['target'].to(self.device, dtype=torch.float32)

                preds = self.model(images, telemetry, physics)

                for i in range(preds.shape[0]):
                    p = preds[i].unsqueeze(0)
                    t = targets[i].unsqueeze(0)
                    metrics['psnr'].append(calculate_psnr(p, t, data_range).item())
                    metrics['mae'].append(calculate_mae(p, t))
                    metrics['rmse'].append(calculate_rmse(p, t))
                    metrics['ssim'].append(calculate_ssim(p, t, data_range=data_range).item())

        return {k: float(np.mean(v)) for k, v in metrics.items()}

    # ----- detailed report ------------------------------------------------

    def generate_evaluation_report(
        self,
        dataloader,
        data_range: float = 1.0,
    ) -> Dict:
        """Return a comprehensive evaluation report.

        The report dict contains:
        - ``per_sample``: lists of per-sample metric values.
        - ``per_channel``: lists of per-channel metric dicts.
        - ``summary``: mean, std, min, max for every metric.
        """
        self.model.eval()
        per_sample: Dict[str, List[float]] = {
            'psnr': [], 'mae': [], 'rmse': [], 'ssim': [],
        }
        per_channel_records: List[Dict[str, List[float]]] = []

        with torch.no_grad():
            for batch in dataloader:
                images = batch['image'].to(self.device, dtype=torch.float32)
                telemetry = batch['telemetry'].to(self.device, dtype=torch.float32)
                physics = batch['physics'].to(self.device, dtype=torch.float32)
                targets = batch['target'].to(self.device, dtype=torch.float32)

                preds = self.model(images, telemetry, physics)

                for i in range(preds.shape[0]):
                    p = preds[i].unsqueeze(0)
                    t = targets[i].unsqueeze(0)

                    psnr_val = calculate_psnr(p, t, data_range).item()
                    mae_val = calculate_mae(p, t)
                    rmse_val = calculate_rmse(p, t)
                    ssim_val = calculate_ssim(p, t, data_range=data_range).item()

                    per_sample['psnr'].append(psnr_val)
                    per_sample['mae'].append(mae_val)
                    per_sample['rmse'].append(rmse_val)
                    per_sample['ssim'].append(ssim_val)

                    # Per-channel breakdown
                    ch_metrics = calculate_per_channel_metrics(p, t, data_range)
                    per_channel_records.append(ch_metrics)

        # Build summary statistics
        summary: Dict[str, Dict[str, float]] = {}
        for key, values in per_sample.items():
            arr = np.array(values)
            summary[key] = {
                'mean': float(np.mean(arr)),
                'std': float(np.std(arr)),
                'min': float(np.min(arr)),
                'max': float(np.max(arr)),
                'median': float(np.median(arr)),
            }

        return {
            'per_sample': per_sample,
            'per_channel': per_channel_records,
            'summary': summary,
            'num_samples': len(per_sample['psnr']),
        }
