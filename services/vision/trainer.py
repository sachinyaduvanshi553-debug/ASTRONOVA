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
    
    print("Initializing dummy dataset and model for dry-run...")
    dataset = SolarImageDataset(image_dir="dummy", sequence_length=2)
    # create a dummy dataloader returning one random batch
    class DummyLoader:
        def __init__(self):
            self.batch = {
                'image': torch.randn(2, 2, 3, 256, 256),
                'telemetry': torch.randn(2, 10),
                'physics': torch.randn(2, 5),
                'target': torch.randn(2, 3, 256, 256)
            }
        def __iter__(self):
            yield self.batch
        def __len__(self):
            return 1
            
    loader = DummyLoader()
    model = MultimodalSolarModel(pretrained_encoder=False)
    criterion = CombinedLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    trainer = Trainer(model, criterion, optimizer, device='cpu')
    print("Running 1 epoch dry-run...")
    loss, metrics = trainer.train_epoch(loader)
    print(f"Dry-run completed successfully. Loss: {loss:.4f}, Metrics: {metrics}")
