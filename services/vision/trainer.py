import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Optional, Tuple, Dict
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from .data.loader import SolarDataModule
from .model import SolarVisionPredictor
from .losses import SolarVisionLoss


class VisionTrainer:
    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        grad_accum_steps: int = 4,
        max_grad_norm: float = 1.0,
        checkpoint_dir: str = 'models/vision',
        use_amp: bool = True,
        early_stopping_patience: int = 10,
        log_dir: str = 'runs/vision',
    ):
        self.model = model.to(device)
        self.criterion = criterion.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.grad_accum_steps = grad_accum_steps
        self.max_grad_norm = max_grad_norm
        self.checkpoint_dir = checkpoint_dir
        self.use_amp = use_amp and 'cuda' in device
        self.early_stopping_patience = early_stopping_patience
        
        self.scaler = torch.cuda.amp.GradScaler() if self.use_amp else None
        self.writer = SummaryWriter(log_dir)
        
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.global_step = 0

    def train_epoch(self, loader: DataLoader) -> Tuple[float, Dict[str, float]]:
        self.model.train()
        total_loss = 0
        metrics_sum = {}
        
        self.optimizer.zero_grad()
        pbar = tqdm(loader, desc="Training")
        
        for batch_idx, batch in enumerate(pbar):
            images = batch['image'].to(self.device, dtype=torch.float32)
            telemetry = batch['telemetry'].to(self.device, dtype=torch.float32)
            physics = batch['physics'].to(self.device, dtype=torch.float32)
            
            targets = {
                'target_image': batch['target'].to(self.device, dtype=torch.float32),
                'flare_class': batch['flare_class'].to(self.device, dtype=torch.long),
                'log_flux': batch['log_flux'].to(self.device, dtype=torch.float32),
                'last_input_image': images[:, -1, :, :, :],
                'log_goes_flux': telemetry[:, 1] # xrsb is at index 1 usually
            }
            
            # Forward + AMP
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    preds = self.model(images, telemetry, physics)
                    loss, loss_dict = self.criterion(preds, targets)
                
                scaled_loss = loss / self.grad_accum_steps
                self.scaler.scale(scaled_loss).backward()
                
            else:
                preds = self.model(images, telemetry, physics)
                loss, loss_dict = self.criterion(preds, targets)
                scaled_loss = loss / self.grad_accum_steps
                scaled_loss.backward()
            
            # Accumulate metrics
            total_loss += loss.item()
            for k, v in loss_dict.items():
                metrics_sum[k] = metrics_sum.get(k, 0) + v
                
            # Optimizer step with accumulation
            if (batch_idx + 1) % self.grad_accum_steps == 0 or (batch_idx + 1) == len(loader):
                if self.use_amp:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.optimizer.step()
                    
                self.optimizer.zero_grad()
                
                if self.scheduler is not None:
                    self.scheduler.step()
                
                # Logging
                self.writer.add_scalar('train/loss', loss.item(), self.global_step)
                for k, v in loss_dict.items():
                    self.writer.add_scalar(f'train/{k}_loss', v, self.global_step)
                if self.scheduler:
                    self.writer.add_scalar('train/lr', self.scheduler.get_last_lr()[0], self.global_step)
                    
                # Log images every 50 steps
                if self.global_step % 50 == 0:
                    self.writer.add_images('train/target_images', targets['target_image'][:4], self.global_step)
                    self.writer.add_images('train/predicted_images', preds['predicted_image'][:4].detach(), self.global_step)
                
                self.global_step += 1
                
            pbar.set_postfix({'loss': loss.item(), 'cls': loss_dict.get('cls', 0), 'reg': loss_dict.get('reg', 0)})
            
        avg_loss = total_loss / len(loader)
        avg_metrics = {k: v / len(loader) for k, v in metrics_sum.items()}
        return avg_loss, avg_metrics

    def validate(self, loader: DataLoader) -> Tuple[float, Dict[str, float]]:
        self.model.eval()
        total_loss = 0
        metrics_sum = {}
        
        with torch.no_grad():
            for batch in loader:
                images = batch['image'].to(self.device, dtype=torch.float32)
                telemetry = batch['telemetry'].to(self.device, dtype=torch.float32)
                physics = batch['physics'].to(self.device, dtype=torch.float32)
                
                targets = {
                    'target_image': batch['target'].to(self.device, dtype=torch.float32),
                    'flare_class': batch['flare_class'].to(self.device, dtype=torch.long),
                    'log_flux': batch['log_flux'].to(self.device, dtype=torch.float32),
                    'last_input_image': images[:, -1, :, :, :],
                    'log_goes_flux': telemetry[:, 1]
                }
                
                preds = self.model(images, telemetry, physics)
                loss, loss_dict = self.criterion(preds, targets)
                
                total_loss += loss.item()
                for k, v in loss_dict.items():
                    metrics_sum[k] = metrics_sum.get(k, 0) + v
                    
        avg_loss = total_loss / len(loader)
        avg_metrics = {k: v / len(loader) for k, v in metrics_sum.items()}
        return avg_loss, avg_metrics

    def save_checkpoint(self, path: str, epoch: int, metrics: dict):
        state = {
            'epoch': epoch,
            'model_state_dict': self.model.module.state_dict() if isinstance(self.model, nn.DataParallel) else self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics,
            'global_step': self.global_step,
        }
        if self.scheduler:
            state['scheduler_state_dict'] = self.scheduler.state_dict()
        if self.scaler:
            state['scaler_state_dict'] = self.scaler.state_dict()
            
        torch.save(state, path)

    def load_checkpoint(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        model_state = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
        model_state.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        if self.scaler and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
            
        self.global_step = checkpoint.get('global_step', 0)
        return checkpoint.get('epoch', 0)

    def export_best_model(self):
        best_path = os.path.join(self.checkpoint_dir, 'best_checkpoint.pt')
        export_path = os.path.join(self.checkpoint_dir, 'best.pt')
        if os.path.exists(best_path):
            checkpoint = torch.load(best_path, map_location='cpu')
            # Just save the weights for inference
            torch.save({'model_state_dict': checkpoint['model_state_dict']}, export_path)

    def train(self, train_loader: DataLoader, val_loader: DataLoader, epochs: int):
        for epoch in range(epochs):
            print(f"Epoch {epoch+1}/{epochs}")
            train_loss, train_metrics = self.train_epoch(train_loader)
            val_loss, val_metrics = self.validate(val_loader)
            
            print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            for k in val_metrics:
                print(f"  {k}: Train {train_metrics.get(k, 0):.4f} | Val {val_metrics[k]:.4f}")
                
            self.writer.add_scalar('val/loss', val_loss, epoch)
            for k, v in val_metrics.items():
                self.writer.add_scalar(f'val/{k}_loss', v, epoch)
                
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                self.save_checkpoint(os.path.join(self.checkpoint_dir, 'best_checkpoint.pt'), epoch, val_metrics)
                print(">>> Saved new best model!")
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.early_stopping_patience:
                    print(f"Early stopping triggered after {epoch+1} epochs.")
                    break
                    
        self.export_best_model()


class MultiGPUTrainer(VisionTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs!")
            self.model = nn.DataParallel(self.model)


def build_trainer_from_config(config: dict) -> Tuple[VisionTrainer, SolarDataModule]:
    # Factory logic to construct everything
    data_module = SolarDataModule(
        data_dir=config.get('data_dir', 'data'),
        goes_csv=config.get('goes_csv', 'data/cleaned/goes/goes_xrs.csv'),
        seq_len=config.get('seq_len', 4),
        horizon=config.get('horizon', 60),
        batch_size=config.get('batch_size', 4),
    )
    
    model = SolarVisionPredictor(pretrained_encoder=not config.get('no_pretrained', False))
    criterion = SolarVisionLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.get('lr', 1e-4), weight_decay=1e-4)
    
    # Cosine annealing with warmup could be added here
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.get('epochs', 50))
    
    trainer = MultiGPUTrainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        grad_accum_steps=config.get('grad_accum', 4),
    )
    
    return trainer, data_module


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default='data')
    parser.add_argument('--goes-csv', default='data/cleaned/goes/goes_xrs.csv')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--seq-len', type=int, default=4)
    parser.add_argument('--horizon', type=int, default=60)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--grad-accum', type=int, default=4)
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--no-pretrained', action='store_true')
    args = parser.parse_args()
    
    config = vars(args)
    trainer, data_module = build_trainer_from_config(config)
    
    # Note: data setup must happen first
    # data_module.setup()
    # if args.resume: trainer.load_checkpoint(args.resume)
    # trainer.train(data_module.train_dataloader(), data_module.val_dataloader(), args.epochs)
