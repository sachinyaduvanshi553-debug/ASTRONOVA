import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def calculate_psnr(pred: torch.Tensor, target: torch.Tensor, data_range: float = 1.0) -> torch.Tensor:
    mse = torch.mean((pred - target) ** 2)
    if mse == 0:
        return torch.tensor(float('inf'))
    return 20.0 * torch.log10(torch.tensor(data_range, device=mse.device) / torch.sqrt(mse))

def calculate_mae(pred: torch.Tensor, target: torch.Tensor) -> float:
    return torch.mean(torch.abs(pred - target)).item()

def calculate_rmse(pred: torch.Tensor, target: torch.Tensor) -> float:
    return torch.sqrt(torch.mean((pred - target) ** 2)).item()


def calculate_per_channel_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    data_range: float = 1.0,
) -> Dict[str, List[float]]:
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

def _fspecial_gaussian(size: int = 11, sigma: float = 1.5,
                       channels: int = 1,
                       device: Optional[torch.device] = None) -> torch.Tensor:
    coords = torch.arange(size, dtype=torch.float32, device=device) - (size - 1) / 2.0
    g = torch.exp(-(coords ** 2) / (2.0 * sigma ** 2))
    g_2d = g.unsqueeze(1) * g.unsqueeze(0)
    g_2d = g_2d / g_2d.sum()
    return g_2d.unsqueeze(0).unsqueeze(0).repeat(channels, 1, 1, 1)

def calculate_ssim(
    pred: torch.Tensor,
    target: torch.Tensor,
    data_range: float = 1.0,
    window_size: int = 11,
    sigma: float = 1.5,
    size_average: bool = True,
) -> torch.Tensor:
    if pred.dim() == 3:
        pred = pred.unsqueeze(0)
        target = target.unsqueeze(0)

    channels = pred.shape[1]
    window = _fspecial_gaussian(window_size, sigma, channels, pred.device).to(pred.dtype)
    pad = window_size // 2

    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2

    mu_pred = F.conv2d(pred, window, padding=pad, groups=channels)
    mu_target = F.conv2d(target, window, padding=pad, groups=channels)

    mu_pred_sq = mu_pred ** 2
    mu_target_sq = mu_target ** 2
    mu_pred_target = mu_pred * mu_target

    sigma_pred_sq = F.conv2d(pred * pred, window, padding=pad, groups=channels) - mu_pred_sq
    sigma_target_sq = F.conv2d(target * target, window, padding=pad, groups=channels) - mu_target_sq
    sigma_pred_target = F.conv2d(pred * target, window, padding=pad, groups=channels) - mu_pred_target

    numerator = (2.0 * mu_pred_target + C1) * (2.0 * sigma_pred_target + C2)
    denominator = (mu_pred_sq + mu_target_sq + C1) * (sigma_pred_sq + sigma_target_sq + C2)
    ssim_map = numerator / denominator

    if size_average:
        return ssim_map.mean()
    return ssim_map.mean(dim=0)


def calculate_fid_features(images: torch.Tensor, device: Optional[torch.device] = None) -> torch.Tensor:
    import torchvision.models as models
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    backbone = nn.Sequential(*list(resnet.children())[:-1])
    backbone = backbone.to(device).eval()

    if images.dim() == 3:
        images = images.unsqueeze(0)

    if images.shape[1] == 1:
        images = images.repeat(1, 3, 1, 1)

    images = images.to(device, dtype=torch.float32)

    with torch.no_grad():
        features = backbone(images)
    return features.flatten(start_dim=1)


def calculate_fid(real_features: torch.Tensor, generated_features: torch.Tensor, eps: float = 1e-6) -> float:
    mu_r = real_features.mean(dim=0).cpu().numpy()
    mu_g = generated_features.mean(dim=0).cpu().numpy()

    r_np = real_features.cpu().numpy()
    g_np = generated_features.cpu().numpy()
    
    # Avoid nan in cov when batch size is 1
    if r_np.shape[0] <= 1 or g_np.shape[0] <= 1:
        return 0.0
        
    sigma_r = np.cov(r_np, rowvar=False)
    sigma_g = np.cov(g_np, rowvar=False)

    diff = mu_r - mu_g
    product = sigma_r @ sigma_g
    eigvals, eigvecs = np.linalg.eigh(product)
    eigvals = np.maximum(eigvals, 0.0)
    sqrt_product = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T

    fid = float(diff @ diff + np.trace(sigma_r + sigma_g - 2.0 * sqrt_product))
    return fid


class Evaluator:
    def __init__(self, model: nn.Module, device: Optional[str] = None):
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)
        self.model = model.to(self.device)

    def evaluate_dataset(self, dataloader, data_range: float = 1.0) -> Dict[str, float]:
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

                out = self.model(images, telemetry, physics)
                preds = out['predicted_image'] if isinstance(out, dict) else out

                for i in range(preds.shape[0]):
                    p = preds[i].unsqueeze(0)
                    t = targets[i].unsqueeze(0)
                    metrics['psnr'].append(calculate_psnr(p, t, data_range).item())
                    metrics['mae'].append(calculate_mae(p, t))
                    metrics['rmse'].append(calculate_rmse(p, t))
                    metrics['ssim'].append(calculate_ssim(p, t, data_range=data_range).item())

        return {k: float(np.mean(v)) if len(v) > 0 else 0.0 for k, v in metrics.items()}

    def generate_evaluation_report(self, dataloader, data_range: float = 1.0) -> Dict:
        self.model.eval()
        per_sample: Dict[str, List[float]] = {'psnr': [], 'mae': [], 'rmse': [], 'ssim': []}
        per_channel_records: List[Dict[str, List[float]]] = []
        
        all_class_preds = []
        all_class_targets = []

        with torch.no_grad():
            for batch in dataloader:
                images = batch['image'].to(self.device, dtype=torch.float32)
                telemetry = batch['telemetry'].to(self.device, dtype=torch.float32)
                physics = batch['physics'].to(self.device, dtype=torch.float32)
                targets = batch['target'].to(self.device, dtype=torch.float32)
                
                # Check if we have classification targets
                has_classes = 'flare_class' in batch

                out = self.model(images, telemetry, physics)
                
                if isinstance(out, dict):
                    preds = out['predicted_image']
                    if has_classes and 'class_probs' in out:
                        c_preds = out['class_probs'].argmax(dim=-1).cpu().numpy()
                        c_targets = batch['flare_class'].cpu().numpy()
                        all_class_preds.extend(c_preds)
                        all_class_targets.extend(c_targets)
                else:
                    preds = out

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

                    ch_metrics = calculate_per_channel_metrics(p, t, data_range)
                    per_channel_records.append(ch_metrics)

        summary: Dict[str, Dict[str, float]] = {}
        for key, values in per_sample.items():
            arr = np.array(values) if values else np.array([0.0])
            summary[key] = {
                'mean': float(np.mean(arr)),
                'std': float(np.std(arr)),
                'min': float(np.min(arr)),
                'max': float(np.max(arr)),
                'median': float(np.median(arr)),
            }

        report = {
            'per_sample': per_sample,
            'per_channel': per_channel_records,
            'summary': summary,
            'num_samples': len(per_sample['psnr']),
        }
        
        # Classification metrics
        if all_class_preds and all_class_targets:
            from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
            y_pred = np.array(all_class_preds)
            y_true = np.array(all_class_targets)
            
            report['accuracy'] = float(accuracy_score(y_true, y_pred))
            report['macro_f1'] = float(f1_score(y_true, y_pred, average='macro'))
            report['confusion_matrix'] = confusion_matrix(y_true, y_pred).tolist()

        return report
