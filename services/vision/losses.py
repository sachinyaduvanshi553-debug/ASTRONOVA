import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

def _fspecial_gaussian(size: int = 11, sigma: float = 1.5, channels: int = 1, device=None) -> torch.Tensor:
    coords = torch.arange(size, dtype=torch.float32, device=device) - (size - 1) / 2.0
    g = torch.exp(-(coords ** 2) / (2.0 * sigma ** 2))
    g_2d = g.unsqueeze(1) * g.unsqueeze(0)
    g_2d = g_2d / g_2d.sum()
    return g_2d.unsqueeze(0).unsqueeze(0).repeat(channels, 1, 1, 1)

class SSIMLoss(nn.Module):
    """SSIM-based loss. 1 - SSIM."""
    def __init__(self, window_size=11, sigma=1.5):
        super().__init__()
        self.window_size = window_size
        self.sigma = sigma

    def forward(self, pred, target):
        channels = pred.shape[1]
        window = _fspecial_gaussian(self.window_size, self.sigma, channels, pred.device).to(pred.dtype)
        pad = self.window_size // 2

        C1 = 0.0001
        C2 = 0.0009

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
        
        return 1.0 - ssim_map.mean()


class DiceLoss(nn.Module):
    """Dice loss for binary active region segmentation proxy."""
    def __init__(self, threshold=0.5, smooth=1e-5):
        super().__init__()
        self.threshold = threshold
        self.smooth = smooth

    def forward(self, pred, target):
        # Soft binarization approximation using sigmoid
        pred_bin = torch.sigmoid(10 * (pred - self.threshold))
        target_bin = (target > self.threshold).float()
        
        intersection = (pred_bin * target_bin).sum(dim=(2, 3))
        union = pred_bin.sum(dim=(2, 3)) + target_bin.sum(dim=(2, 3))
        
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice.mean()


class PhysicsConsistencyLoss(nn.Module):
    """Penalizes predictions where mean image brightness diverges from GOES flux."""
    def forward(self, pred, log_goes_flux):
        # We only apply this if log_goes_flux is valid (not all zeros or NaNs)
        valid_mask = ~torch.isnan(log_goes_flux) & (log_goes_flux != 0)
        if not valid_mask.any():
            return torch.tensor(0.0, device=pred.device, requires_grad=True)
            
        pred_valid = pred[valid_mask]
        goes_valid = log_goes_flux[valid_mask]
        
        # Grayscale proxy
        gray = pred_valid.mean(dim=1) # (B, H, W)
        active_mask = (gray > 0.8).float()
        
        # brightness_ratio = active_mean * active_fraction
        # Avoid division by zero
        active_sum = (gray * active_mask).sum(dim=(1,2))
        active_count = active_mask.sum(dim=(1,2)) + 1e-8
        active_mean = active_sum / active_count
        active_fraction = active_count / (gray.shape[1] * gray.shape[2])
        
        brightness_ratio = active_mean * active_fraction
        
        # estimated_flux = 1e-8 * 10^(brightness_ratio * 4.3)
        # log10(estimated) = -8 + brightness_ratio * 4.3
        log_estimated = -8.0 + brightness_ratio * 4.3
        
        return F.l1_loss(log_estimated, goes_valid)


class TemporalSmoothnessLoss(nn.Module):
    """Penalizes predictions that are temporally inconsistent with last input frame."""
    def forward(self, pred, last_input_frame):
        return F.mse_loss(pred, last_input_frame)


class SolarVisionLoss(nn.Module):
    """Complete composite loss for solar image prediction."""
    def __init__(self, w_mse=1.0, w_l1=1.0, w_ssim=0.5, w_perceptual=0.1, 
                 w_dice=0.3, w_cls=2.0, w_reg=1.5, w_physics=0.5, w_temporal=0.2):
        super().__init__()
        self.weights = {
            'mse': w_mse, 'l1': w_l1, 'ssim': w_ssim, 'perceptual': w_perceptual,
            'dice': w_dice, 'cls': w_cls, 'reg': w_reg, 'physics': w_physics, 'temporal': w_temporal
        }
        
        self.mse = nn.MSELoss()
        self.l1 = nn.L1Loss()
        self.ssim = SSIMLoss()
        self.dice = DiceLoss()
        self.cls = nn.CrossEntropyLoss()
        self.reg = nn.HuberLoss()
        self.physics = PhysicsConsistencyLoss()
        self.temporal = TemporalSmoothnessLoss()
        
        # Perceptual Loss setup
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features
        self.vgg_subset = nn.Sequential(*list(vgg)[:16]).eval()
        for param in self.vgg_subset.parameters():
            param.requires_grad = False
            
        self.register_buffer('vgg_mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('vgg_std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def _normalize_for_vgg(self, x):
        x = torch.clamp(x, 0.0, 1.0)
        return (x - self.vgg_mean.to(x.device)) / (self.vgg_std.to(x.device) + 1e-8)

    def forward(self, predictions: dict, targets: dict) -> tuple[torch.Tensor, dict]:
        pred_img = predictions.get('predicted_image')
        target_img = targets.get('target_image')
        
        # Optional targets with defaults to zero if missing
        flare_class = targets.get('flare_class', torch.zeros(pred_img.shape[0], dtype=torch.long, device=pred_img.device))
        log_flux = targets.get('log_flux', torch.zeros((pred_img.shape[0], 1), dtype=torch.float32, device=pred_img.device))
        last_input = targets.get('last_input_image', torch.zeros_like(pred_img))
        log_goes_flux = targets.get('log_goes_flux', torch.zeros(pred_img.shape[0], device=pred_img.device))
        
        loss_dict = {}
        total_loss = 0.0
        
        # 1. Pixel losses
        if target_img is not None and pred_img is not None:
            l_mse = self.mse(pred_img, target_img)
            l_l1 = self.l1(pred_img, target_img)
            l_ssim = self.ssim(pred_img, target_img)
            l_dice = self.dice(pred_img, target_img)
            
            # Perceptual
            p_vgg = self._normalize_for_vgg(pred_img)
            t_vgg = self._normalize_for_vgg(target_img)
            l_perceptual = self.mse(self.vgg_subset(p_vgg), self.vgg_subset(t_vgg))
            
            loss_dict['mse'] = l_mse.item()
            loss_dict['l1'] = l_l1.item()
            loss_dict['ssim'] = l_ssim.item()
            loss_dict['dice'] = l_dice.item()
            loss_dict['perceptual'] = l_perceptual.item()
            
            total_loss += (self.weights['mse']*l_mse + self.weights['l1']*l_l1 + 
                          self.weights['ssim']*l_ssim + self.weights['perceptual']*l_perceptual + 
                          self.weights['dice']*l_dice)
                          
        # 2. Heads
        class_logits = predictions.get('class_logits')
        if class_logits is not None:
            l_cls = self.cls(class_logits, flare_class)
            loss_dict['cls'] = l_cls.item()
            total_loss += self.weights['cls'] * l_cls
            
        reg_output = predictions.get('reg_output')
        if reg_output is not None:
            l_reg = self.reg(reg_output, log_flux.view_as(reg_output))
            loss_dict['reg'] = l_reg.item()
            total_loss += self.weights['reg'] * l_reg
            
        # 3. Physics & Temporal
        if pred_img is not None:
            l_physics = self.physics(pred_img, log_goes_flux)
            l_temporal = self.temporal(pred_img, last_input)
            loss_dict['physics'] = l_physics.item()
            loss_dict['temporal'] = l_temporal.item()
            
            total_loss += self.weights['physics'] * l_physics + self.weights['temporal'] * l_temporal
            
        return total_loss, loss_dict

# Alias for backward compatibility
CombinedLoss = SolarVisionLoss
