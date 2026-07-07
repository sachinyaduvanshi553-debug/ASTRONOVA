import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import os

class Trainer:
    """
    Handles the training loop with mixed precision and basic checkpointing.
    """
    def __init__(self, model, criterion, optimizer, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.criterion = criterion.to(device)
        self.optimizer = optimizer
        self.device = device
        self.scaler = torch.cuda.amp.GradScaler() if 'cuda' in device else None

    def train_epoch(self, dataloader):
        self.model.train()
        total_loss = 0
        metrics_sum = {'mse': 0, 'mae': 0, 'perceptual': 0}
        
        pbar = tqdm(dataloader, desc="Training")
        for batch in pbar:
            images = batch['image'].to(self.device, dtype=torch.float32)
            telemetry = batch['telemetry'].to(self.device, dtype=torch.float32)
            physics = batch['physics'].to(self.device, dtype=torch.float32)
            targets = batch['target'].to(self.device, dtype=torch.float32)
            
            self.optimizer.zero_grad()
            
            if self.scaler:
                with torch.cuda.amp.autocast():
                    preds = self.model(images, telemetry, physics)
                    loss, loss_dict = self.criterion(preds, targets)
                
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                preds = self.model(images, telemetry, physics)
                loss, loss_dict = self.criterion(preds, targets)
                loss.backward()
                self.optimizer.step()
                
            total_loss += loss.item()
            for k, v in loss_dict.items():
                metrics_sum[k] += v
                
            pbar.set_postfix({'loss': loss.item(), 'mse': loss_dict['mse']})
            
        avg_loss = total_loss / len(dataloader)
        avg_metrics = {k: v / len(dataloader) for k, v in metrics_sum.items()}
        return avg_loss, avg_metrics

    def evaluate(self, dataloader):
        self.model.eval()
        total_loss = 0
        metrics_sum = {'mse': 0, 'mae': 0, 'perceptual': 0}
        
        with torch.no_grad():
            for batch in dataloader:
                images = batch['image'].to(self.device, dtype=torch.float32)
                telemetry = batch['telemetry'].to(self.device, dtype=torch.float32)
                physics = batch['physics'].to(self.device, dtype=torch.float32)
                targets = batch['target'].to(self.device, dtype=torch.float32)
                
                preds = self.model(images, telemetry, physics)
                loss, loss_dict = self.criterion(preds, targets)
                
                total_loss += loss.item()
                for k, v in loss_dict.items():
                    metrics_sum[k] += v
                    
        avg_loss = total_loss / len(dataloader)
        avg_metrics = {k: v / len(dataloader) for k, v in metrics_sum.items()}
        return avg_loss, avg_metrics

    def save_checkpoint(self, path, epoch, loss):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
        }, path)

if __name__ == "__main__":
    # Dry-run execution
    from .model import MultimodalSolarModel
    from .losses import CombinedLoss
    from .dataset import SolarImageDataset
    
    print("Initializing authentic dataset and model for dry-run...")
    dataset = SolarImageDataset(
        image_dir="DATA/events/flare_sequences",
        goes_csv_path="DATA/cleaned/goes/goes_xrs_oct2024_jan2025.csv",
        sequence_length=2
    )
    
    if len(dataset) == 0:
        print("Dataset is empty! Make sure download_sample_data.py has finished running.")
    else:
        # Create DataLoader
        loader = DataLoader(dataset, batch_size=2, shuffle=False)
        
        model = MultimodalSolarModel(pretrained_encoder=False)
        criterion = CombinedLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        
        trainer = Trainer(model, criterion, optimizer, device='cpu')
        print(f"Running 1 epoch dry-run on {len(dataset)} authentic samples...")
        loss, metrics = trainer.train_epoch(loader)
        print(f"Dry-run completed successfully. Loss: {loss:.4f}, Metrics: {metrics}")
