"""
models/vision/flare_classifier.py — Per-horizon flare classification heads.

Outputs independent binary probabilities for each forecast horizon and class:
    P(C+) — C-class or higher flare
    P(M+) — M-class or higher flare
    P(X+) — X-class flare

6 forecast horizons: +15m, +30m, +60m, +6h, +12h, +24h

Uses sigmoid outputs for independent thresholds (not softmax).
This allows P(C+) >= P(M+) >= P(X+) to be enforced via calibration,
but they are trained independently to allow flexibility.
"""
from __future__ import annotations

import torch
import torch.nn as nn

HORIZONS = ["h15m", "h30m", "h60m", "h360m", "h720m", "h1440m"]
N_CLASSES = 3  # C+, M+, X+


class HorizonClassificationHead(nn.Module):
    """Classification head for a single horizon."""

    def __init__(self, input_dim: int, hidden_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, N_CLASSES),
            # NO sigmoid here — applied in loss + inference separately
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, D] — context embedding
        Returns:
            logits: [B, 3] — (C+, M+, X+) logits
        """
        return self.head(x)


class MultiHorizonFlareClassifier(nn.Module):
    """
    Independent classification head for each of the 6 horizons.
    
    Input:  context [B, D] — from temporal model
    Output: dict of {horizon_key: logits [B, 3]}
            and probs {horizon_key: probs [B, 3]}
    
    Horizon keys: h15m, h30m, h60m, h360m, h720m, h1440m
    Class indices: 0=C+, 1=M+, 2=X+
    """

    def __init__(
        self,
        input_dim: int = 256,
        hidden_dim: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.heads = nn.ModuleDict({
            h: HorizonClassificationHead(input_dim, hidden_dim, dropout)
            for h in HORIZONS
        })

    def forward(self, context: torch.Tensor) -> tuple[dict, dict]:
        """
        Args:
            context: [B, D]
        Returns:
            logits: dict[str, Tensor[B, 3]]
            probs:  dict[str, Tensor[B, 3]]
        """
        logits, probs = {}, {}
        for h, head in self.heads.items():
            l = head(context)           # [B, 3]
            logits[h] = l
            probs[h] = torch.sigmoid(l)  # Independent sigmoid per class
        return logits, probs

    def predict_class(self, probs: dict[str, torch.Tensor], horizon: str = "h60m") -> list[str]:
        """Convert probabilities to GOES class strings."""
        p = probs.get(horizon, torch.zeros(1, 3))
        classes = []
        for i in range(p.shape[0]):
            if p[i, 2] >= 0.5:
                classes.append("X")
            elif p[i, 1] >= 0.5:
                classes.append("M")
            elif p[i, 0] >= 0.5:
                classes.append("C")
            else:
                classes.append("quiet")
        return classes
