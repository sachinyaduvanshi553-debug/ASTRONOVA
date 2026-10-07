"""
models/vision/conv_lstm.py — ConvLSTM for spatiotemporal solar sequence modeling.

Processes sequences of spatial feature maps and captures temporal evolution
of solar active regions, intensity, and morphology.

Input:  [B, T, C, H, W] — batch of spatial feature map sequences
Output: [B, C, H, W]    — final hidden state (future spatial representation)
        [B, T, C, H, W] — all hidden states

Reference: Shi et al., "Convolutional LSTM Network: A Machine Learning Approach 
for Precipitation Nowcasting" (NeurIPS 2015)
"""
from __future__ import annotations

import torch
import torch.nn as nn


class ConvLSTMCell(nn.Module):
    """Single ConvLSTM cell."""

    def __init__(self, input_dim: int, hidden_dim: int, kernel_size: int = 3, bias: bool = True):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        pad = kernel_size // 2

        # Gates: input, forget, cell, output — all in one convolution
        self.conv = nn.Conv2d(
            input_dim + hidden_dim,
            4 * hidden_dim,
            kernel_size=kernel_size,
            padding=pad,
            bias=bias,
        )
        # Initialize forget gate bias to 1 (helps with long sequences)
        if bias:
            nn.init.constant_(self.conv.bias[hidden_dim: 2 * hidden_dim], 1.0)

    def forward(
        self,
        x: torch.Tensor,
        h_prev: torch.Tensor,
        c_prev: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x:      [B, input_dim, H, W]
            h_prev: [B, hidden_dim, H, W]
            c_prev: [B, hidden_dim, H, W]
        Returns:
            h_next, c_next: [B, hidden_dim, H, W]
        """
        combined = torch.cat([x, h_prev], dim=1)
        gates = self.conv(combined)

        i, f, g, o = torch.chunk(gates, 4, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        g = torch.tanh(g)
        o = torch.sigmoid(o)

        c_next = f * c_prev + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next

    def init_hidden(self, batch_size: int, h: int, w: int, device: torch.device):
        return (
            torch.zeros(batch_size, self.hidden_dim, h, w, device=device),
            torch.zeros(batch_size, self.hidden_dim, h, w, device=device),
        )


class ConvLSTM(nn.Module):
    """
    Multi-layer ConvLSTM for spatiotemporal sequence modeling.
    
    Input:  [B, T, C_in, H, W]
    Output: last_hidden [B, hidden_dim, H, W]
            all_hidden  [B, T, hidden_dim, H, W]
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        kernel_size: int = 3,
        n_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim

        self.cells = nn.ModuleList()
        for i in range(n_layers):
            in_dim = input_dim if i == 0 else hidden_dim
            self.cells.append(ConvLSTMCell(in_dim, hidden_dim, kernel_size))

        self.dropout = nn.Dropout2d(p=dropout)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [B, T, C_in, H, W]
        Returns:
            last_hidden: [B, hidden_dim, H, W]
            all_hidden:  [B, T, hidden_dim, H, W]
        """
        B, T, C, H, W = x.shape
        device = x.device

        # Initialize hidden states
        states = [cell.init_hidden(B, H, W, device) for cell in self.cells]

        all_h = []
        for t in range(T):
            inp = x[:, t]  # [B, C, H, W]
            for layer_idx, cell in enumerate(self.cells):
                h, c = states[layer_idx]
                h, c = cell(inp, h, c)
                if layer_idx < self.n_layers - 1:
                    h = self.dropout(h)
                states[layer_idx] = (h, c)
                inp = h
            all_h.append(h)

        all_hidden = torch.stack(all_h, dim=1)   # [B, T, hidden_dim, H, W]
        last_hidden = all_h[-1]                   # [B, hidden_dim, H, W]

        return last_hidden, all_hidden
