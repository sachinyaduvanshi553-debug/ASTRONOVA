import torch
import numpy as np

# We would typically use torchmetrics or similar libraries for SSIM, PSNR, FID
# Here we define basic functions or wrappers to compute these metrics

def calculate_psnr(pred, target, data_range=1.0):
    mse = torch.mean((pred - target) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * torch.log10(data_range / torch.sqrt(mse))

def calculate_mae(pred, target):
    return torch.mean(torch.abs(pred - target)).item()

def calculate_rmse(pred, target):
    return torch.sqrt(torch.mean((pred - target) ** 2)).item()

class Evaluator:
    def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.device = device

    def evaluate_dataset(self, dataloader):
        self.model.eval()
        metrics = {'psnr': [], 'mae': [], 'rmse': []}
        
        with torch.no_grad():
            for batch in dataloader:
                images = batch['image'].to(self.device, dtype=torch.float32)
                telemetry = batch['telemetry'].to(self.device, dtype=torch.float32)
                physics = batch['physics'].to(self.device, dtype=torch.float32)
                targets = batch['target'].to(self.device, dtype=torch.float32)
                
                preds = self.model(images, telemetry, physics)
                
                # Compute metrics
                for i in range(len(preds)):
                    metrics['psnr'].append(calculate_psnr(preds[i], targets[i]).item())
                    metrics['mae'].append(calculate_mae(preds[i], targets[i]))
                    metrics['rmse'].append(calculate_rmse(preds[i], targets[i]))
                    
        return {k: np.mean(v) for k, v in metrics.items()}
