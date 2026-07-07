import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class CombinedLoss(nn.Module):
    """
    Combined loss function for solar image prediction.
    Includes MSE, MAE, and Perceptual Loss (VGG16-based).
    VGG inputs are properly normalized with ImageNet statistics.
    """
    def __init__(self, alpha=1.0, beta=1.0, gamma=0.1):
        super(CombinedLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        
        self.mse = nn.MSELoss()
        self.mae = nn.L1Loss()
        
        # Perceptual Loss using VGG16 features (up to relu3_3)
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features
        self.vgg_subset = nn.Sequential(*list(vgg)[:16]).eval()
        for param in self.vgg_subset.parameters():
            param.requires_grad = False

        # ImageNet normalization constants for VGG preprocessing
        self.register_buffer('vgg_mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('vgg_std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def _normalize_for_vgg(self, x):
        """Normalize tensor to ImageNet statistics expected by VGG."""
        # Clamp to [0, 1] first if the data might be outside this range
        x = torch.clamp(x, 0.0, 1.0)
        return (x - self.vgg_mean.to(x.device)) / (self.vgg_std.to(x.device) + 1e-8)

    def forward(self, pred, target):
        loss_mse = self.mse(pred, target)
        loss_mae = self.mae(pred, target)
        
        # Perceptual Loss — normalize inputs for VGG with ImageNet stats
        pred_vgg = self._normalize_for_vgg(pred)
        target_vgg = self._normalize_for_vgg(target)
        
        pred_features = self.vgg_subset(pred_vgg)
        target_features = self.vgg_subset(target_vgg)
        loss_perceptual = self.mse(pred_features, target_features)
        
        total_loss = self.alpha * loss_mse + self.beta * loss_mae + self.gamma * loss_perceptual
        return total_loss, {
            'mse': loss_mse.item(),
            'mae': loss_mae.item(),
            'perceptual': loss_perceptual.item()
        }
