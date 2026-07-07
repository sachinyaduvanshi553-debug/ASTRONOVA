import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class CombinedLoss(nn.Module):
    """
    Combined loss function for image prediction.
    Includes MSE, MAE, and Perceptual Loss (LPIPS/VGG).
    """
    def __init__(self, alpha=1.0, beta=1.0, gamma=0.1):
        super(CombinedLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        
        self.mse = nn.MSELoss()
        self.mae = nn.L1Loss()
        
        # Perceptual Loss using VGG16 features
        vgg = models.vgg16(pretrained=True).features
        self.vgg_subset = nn.Sequential(*list(vgg)[:16]).eval()
        for param in self.vgg_subset.parameters():
            param.requires_grad = False

    def forward(self, pred, target):
        loss_mse = self.mse(pred, target)
        loss_mae = self.mae(pred, target)
        
        # Perceptual Loss
        # Normalize inputs for VGG if necessary, assuming inputs are [0,1] or standard
        # In practice we should normalize with ImageNet mean/std
        pred_features = self.vgg_subset(pred)
        target_features = self.vgg_subset(target)
        loss_perceptual = self.mse(pred_features, target_features)
        
        total_loss = self.alpha * loss_mse + self.beta * loss_mae + self.gamma * loss_perceptual
        return total_loss, {
            'mse': loss_mse.item(),
            'mae': loss_mae.item(),
            'perceptual': loss_perceptual.item()
        }
