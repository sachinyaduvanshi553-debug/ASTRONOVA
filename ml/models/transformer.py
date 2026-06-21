import torch
import torch.nn as nn
import math
from typing import Tuple, Union

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]

class SolarTransformer(nn.Module):
    """
    Transformer-based solar flare forecaster with MC Dropout uncertainty estimation
    and dual heads (classification + regression) for multiple horizons.
    """
    def __init__(
        self, 
        input_size: int = 7, 
        d_model: int = 64, 
        nhead: int = 4, 
        num_layers: int = 2, 
        num_classes: int = 5,
        num_horizons: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_horizons = num_horizons
        
        self.input_projection = nn.Linear(input_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            batch_first=True,
            dropout=dropout
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.dropout = nn.Dropout(p=dropout)
        self.fc_class = nn.Linear(d_model, num_horizons * num_classes)
        self.fc_reg = nn.Linear(d_model, num_horizons * 1)

    def forward(self, x: torch.Tensor, return_tuple: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        # x: [batch_size, seq_len, input_size]
        x = self.input_projection(x)
        x = self.pos_encoder(x)
        out = self.transformer_encoder(x)
        out = self.dropout(out)
        
        out = out.mean(dim=1)  # Global average pooling
        
        # Classification logits and probabilities
        class_logits = self.fc_class(out).view(-1, self.num_horizons, self.num_classes)
        class_probs = torch.softmax(class_logits, dim=-1)
        
        # Regression output (flux value)
        reg_out = self.fc_reg(out).view(-1, self.num_horizons, 1)
        
        if return_tuple:
            return class_probs, reg_out
        else:
            # Backward compatibility default: return last horizon classification probabilities
            return class_probs[:, -1, :]

    def predict_with_uncertainty(self, x: torch.Tensor, n_samples: int = 20) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Runs multiple forward passes with dropout enabled to compute MC uncertainty.
        """
        self.train() # Keep dropout active
        
        all_probs = []
        all_regs = []
        
        with torch.no_grad():
            for _ in range(n_samples):
                probs, reg = self.forward(x, return_tuple=True)
                all_probs.append(probs.unsqueeze(0))
                all_regs.append(reg.unsqueeze(0))
                
        stacked_probs = torch.cat(all_probs, dim=0)
        stacked_regs = torch.cat(all_regs, dim=0)
        
        mean_probs = torch.mean(stacked_probs, dim=0)
        mean_reg = torch.mean(stacked_regs, dim=0)
        std_reg = torch.std(stacked_regs, dim=0)
        
        self.eval() # Restore to eval mode
        return mean_probs, mean_reg, std_reg
