import torch
import torch.nn as nn

class ConvLSTMCell(nn.Module):
    """A single ConvLSTM cell.
    Args:
        input_dim: Number of channels of input tensor.
        hidden_dim: Number of channels of hidden state.
        kernel_size: Size of the convolutional kernel.
        bias: Whether to add the bias term.
    """
    def __init__(self, input_dim, hidden_dim, kernel_size, bias=True):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(in_channels=input_dim + hidden_dim,
                              out_channels=4 * hidden_dim,
                              kernel_size=kernel_size,
                              padding=padding,
                              bias=bias)
        self.hidden_dim = hidden_dim

    def forward(self, x, hidden):
        h_cur, c_cur = hidden
        combined = torch.cat([x, h_cur], dim=1)
        conv_output = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(conv_output, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)
        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next

class ConvLSTM(nn.Module):
    """Stacked ConvLSTM for sequence modelling.
    Args:
        input_dim: Channels of input frames.
        hidden_dim: List of hidden dimensions for each layer.
        kernel_size: List of kernel sizes for each layer.
        num_layers: Number of ConvLSTM layers.
    """
    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers, bias=True):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = self._ensure_list(hidden_dim, num_layers)
        self.kernel_size = self._ensure_list(kernel_size, num_layers)
        self.num_layers = num_layers
        self.cells = nn.ModuleList([
            ConvLSTMCell(
                input_dim=self.input_dim if i == 0 else self.hidden_dim[i-1],
                hidden_dim=self.hidden_dim[i],
                kernel_size=self.kernel_size[i],
                bias=bias)
            for i in range(num_layers)
        ])

    @staticmethod
    def _ensure_list(param, num_layers):
        if isinstance(param, (list, tuple)):
            return list(param)
        return [param] * num_layers

    def forward(self, x, hidden_state=None):
        """Input shape: (batch, seq_len, channels, height, width)"""
        b, seq_len, _, h, w = x.size()
        if hidden_state is None:
            hidden_state = self._init_hidden(b, (h, w))
        cur_input = x
        for layer_idx in range(self.num_layers):
            h, c = hidden_state[layer_idx]
            output_inner = []
            for t in range(seq_len):
                h, c = self.cells[layer_idx](cur_input[:, t], (h, c))
                output_inner.append(h)
            cur_input = torch.stack(output_inner, dim=1)
        return cur_input

    def _init_hidden(self, batch_size, image_size):
        height, width = image_size
        init = []
        device = next(self.parameters()).device
        for hidden_dim in self.hidden_dim:
            h = torch.zeros(batch_size, hidden_dim, height, width, device=device)
            c = torch.zeros(batch_size, hidden_dim, height, width, device=device)
            init.append((h, c))
        return init
