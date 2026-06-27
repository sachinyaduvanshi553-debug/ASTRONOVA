
import torch
import torch.nn as nn


class GRUForecaster(nn.Module):
    """
    Bidirectional GRU solar flare forecaster with MC Dropout uncertainty estimation
    and dual heads (classification + regression) for multiple horizons.
    """
    def __init__(
        self,
        input_size: int = 7,
        hidden_size: int = 64,
        num_layers: int = 2,
        num_classes: int = 5,
        num_horizons: int = 4,
        dropout: float = 0.3
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_horizons = num_horizons

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        self.dropout = nn.Dropout(p=dropout)
        self.fc_class = nn.Linear(hidden_size * 2, num_horizons * num_classes)
        self.fc_reg = nn.Linear(hidden_size * 2, num_horizons * 1)

    def forward(self, x: torch.Tensor, return_tuple: bool = False) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        # x: [batch_size, seq_len, input_size]
        out, _ = self.gru(x)
        out = self.dropout(out)

        # Take the last time step output
        out = out[:, -1, :]

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

    def predict_with_uncertainty(self, x: torch.Tensor, n_samples: int = 20) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
