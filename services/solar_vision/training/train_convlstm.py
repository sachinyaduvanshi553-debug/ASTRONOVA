import argparse
from pathlib import Path

import torch
import torch.nn as nn

# Import the ConvLSTM model from the project
from models.convlstm import ConvLSTM


def get_dummy_dataset(num_samples: int, seq_len: int, channels: int, height: int, width: int):
    """Generate a dummy dataset of random tensors for demonstration purposes.
    In a real scenario, replace this with a proper Dataset class that loads solar image sequences.
    """
    for _ in range(num_samples):
        # Input shape: (batch, seq_len, channels, height, width)
        x = torch.randn(1, seq_len, channels, height, width)
        # Target could be the next frame or a segmentation mask; here we use random tensor
        y = torch.randn(1, seq_len, channels, height, width)
        yield x, y

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ConvLSTM(
        input_dim=args.input_dim,
        hidden_dim=args.hidden_dim,
        kernel_size=args.kernel_size,
        num_layers=args.num_layers,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    model.train()
    for epoch in range(1, args.epochs + 1):
        epoch_loss = 0.0
        for x, y in get_dummy_dataset(
            num_samples=args.samples_per_epoch,
            seq_len=args.seq_len,
            channels=args.input_dim,
            height=args.height,
            width=args.width,
        ):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Epoch {epoch}/{args.epochs} - Loss: {epoch_loss/args.samples_per_epoch:.6f}")

    # Save checkpoint
    checkpoint_dir = Path(args.output_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / "convlstm_checkpoint.pt"
    torch.save(model.state_dict(), checkpoint_path)
    print(f"Model checkpoint saved to {checkpoint_path}")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ConvLSTM model for Solar Vision Engine")
    parser.add_argument("--input-dim", type=int, default=3, help="Number of input channels (e.g., RGB)")
    parser.add_argument("--hidden-dim", type=int, nargs="+", default=[64, 64], help="Hidden dimensions per ConvLSTM layer")
    parser.add_argument("--kernel-size", type=int, nargs="+", default=[3, 3], help="Kernel size per ConvLSTM layer")
    parser.add_argument("--num-layers", type=int, default=2, help="Number of ConvLSTM layers")
    parser.add_argument("--seq-len", type=int, default=5, help="Length of input sequence")
    parser.add_argument("--height", type=int, default=128, help="Spatial height of input frames")
    parser.add_argument("--width", type=int, default=128, help="Spatial width of input frames")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--samples-per-epoch", type=int, default=20, help="Number of dummy samples per epoch")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--output-dir", type=str, default="./models/checkpoints", help="Directory to save model checkpoints")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    train(args)
