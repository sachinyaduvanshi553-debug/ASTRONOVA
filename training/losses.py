"""
Scientific Loss Functions for Extreme Solar Flare Class Imbalance.
Implements:
1. Weighted Binary Cross-Entropy (Weighted BCE)
2. Focal Loss (Lin et al.)
3. Class-Balanced Loss (Cui et al.)
4. Combined Weighted BCE + Focal Loss
"""

import math
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightedBCELoss(nn.Module):
    """
    Weighted Binary Cross Entropy Loss.
    Applies positive class weight pos_weight.
    """

    def __init__(self, pos_weight: float = 10.0, eps: float = 1e-7):
        super().__init__()
        self.pos_weight = pos_weight
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = torch.clamp(pred, self.eps, 1.0 - self.eps)
        loss = -(self.pos_weight * target * torch.log(pred) + (1.0 - target) * torch.log(1.0 - pred))
        return loss.mean()


class FocalLoss(nn.Module):
    """
    Binary Focal Loss for hard example mining and class imbalance:
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """

    def __init__(self, gamma: float = 2.0, alpha: float = 0.75, eps: float = 1e-7):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = torch.clamp(pred, self.eps, 1.0 - self.eps)
        # p_t is the probability of the true class
        p_t = target * pred + (1.0 - target) * (1.0 - pred)
        alpha_t = target * self.alpha + (1.0 - target) * (1.0 - self.alpha)

        focal_weight = alpha_t * torch.pow((1.0 - p_t), self.gamma)
        loss = -focal_weight * torch.log(p_t)
        return loss.mean()


class ClassBalancedLoss(nn.Module):
    """
    Class-Balanced Loss based on Effective Number of Samples (Cui et al., 2019).
    E_n = (1 - beta^N) / (1 - beta)
    """

    def __init__(self, beta: float = 0.999, num_pos: int = 100, num_neg: int = 2000, eps: float = 1e-7):
        super().__init__()
        self.eps = eps
        eff_pos = (1.0 - (beta ** max(num_pos, 1))) / (1.0 - beta)
        eff_neg = (1.0 - (beta ** max(num_neg, 1))) / (1.0 - beta)
        
        weight_pos = 1.0 / max(eff_pos, 1e-6)
        weight_neg = 1.0 / max(eff_neg, 1e-6)
        
        # Normalize weights
        total_w = weight_pos + weight_neg
        self.w_pos = weight_pos / total_w
        self.w_neg = weight_neg / total_w

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = torch.clamp(pred, self.eps, 1.0 - self.eps)
        loss = -(self.w_pos * target * torch.log(pred) + self.w_neg * (1.0 - target) * torch.log(1.0 - pred))
        return loss.mean()


class CombinedBCEFocalLoss(nn.Module):
    """
    Hybrid loss combining Weighted BCE and Focal Loss for robust boundary learning.
    """

    def __init__(self, pos_weight: float = 8.0, gamma: float = 2.0, alpha: float = 0.75, bce_weight: float = 0.5):
        super().__init__()
        self.bce = WeightedBCELoss(pos_weight=pos_weight)
        self.focal = FocalLoss(gamma=gamma, alpha=alpha)
        self.bce_weight = bce_weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        l_bce = self.bce(pred, target)
        l_focal = self.focal(pred, target)
        return self.bce_weight * l_bce + (1.0 - self.bce_weight) * l_focal
