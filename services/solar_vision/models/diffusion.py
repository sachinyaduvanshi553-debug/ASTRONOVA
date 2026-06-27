import torch
import torch.nn as nn


class DiffusionModel(nn.Module):
    """A minimal diffusion refinement placeholder.
    In a full implementation this would perform iterative denoising of an image.
    For now the forward pass simply returns the input tensor unchanged.
    """
    def __init__(self) -> None:
        super().__init__()
        # No learnable parameters for the stub

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the input as‑is.

        Args:
            x: Tensor representing a batch of images (B, C, H, W).
        Returns:
            torch.Tensor: Unchanged input tensor.
        """
        return x
