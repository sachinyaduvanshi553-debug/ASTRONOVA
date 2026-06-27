import argparse
import torch
from pathlib import Path

# Import the ConvLSTM model stub and a dummy checkpoint loader
from models.convlstm import ConvLSTM

def load_model(checkpoint_path: str, args) -> ConvLSTM:
    model = ConvLSTM(
        input_dim=args.input_dim,
        hidden_dim=args.hidden_dim,
        kernel_size=args.kernel_size,
        num_layers=args.num_layers,
    )
    if Path(checkpoint_path).exists():
        state_dict = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        model.load_state_dict(state_dict)
        print(f"Loaded checkpoint from {checkpoint_path}")
    else:
        print(f"Checkpoint not found at {checkpoint_path}, using random weights.")
    model.eval()
    return model

def predict_future(model: ConvLSTM, input_seq: torch.Tensor, steps: int) -> torch.Tensor:
    """Generate future frames by repeatedly feeding the last output as next input.
    This is a simplistic rollout for demonstration.
    """
    preds = []
    cur_seq = input_seq.clone()
    for _ in range(steps):
        with torch.no_grad():
            out = model(cur_seq)
        # Take the last frame as the next input frame
        next_frame = out[:, -1:]
        preds.append(next_frame)
        # Append to sequence and drop the oldest frame to keep length constant
        cur_seq = torch.cat([cur_seq[:, 1:], next_frame], dim=1)
    return torch.cat(preds, dim=1)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference with ConvLSTM model for Solar Vision")
    parser.add_argument("--checkpoint", type=str, default="./models/checkpoints/convlstm_checkpoint.pt",
                        help="Path to model checkpoint file")
    parser.add_argument("--input-dim", type=int, default=3, help="Number of input channels")
    parser.add_argument("--hidden-dim", type=int, nargs="+", default=[64, 64], help="Hidden dims per layer")
    parser.add_argument("--kernel-size", type=int, nargs="+", default=[3, 3], help="Kernel sizes per layer")
    parser.add_argument("--num-layers", type=int, default=2, help="Number of ConvLSTM layers")
    parser.add_argument("--seq-len", type=int, default=5, help="Length of input sequence")
    parser.add_argument("--height", type=int, default=128, help="Spatial height of frames")
    parser.add_argument("--width", type=int, default=128, help="Spatial width of frames")
    parser.add_argument("--future-steps", type=int, default=3, help="How many future frames to predict")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    # Create a dummy input sequence matching the expected shape
    dummy_input = torch.randn(1, args.seq_len, args.input_dim, args.height, args.width)
    model = load_model(args.checkpoint, args)
    future = predict_future(model, dummy_input, args.future_steps)
    print(f"Predicted future shape: {future.shape}")
    # Optionally, save the predictions
    out_path = Path("./models/predictions")
    out_path.mkdir(parents=True, exist_ok=True)
    torch.save(future, out_path / "future_predictions.pt")
    print(f"Saved predictions to {out_path / 'future_predictions.pt'}")
