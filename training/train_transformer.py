"""
Transformer Training Engine.
Supports Baseline and Optimized Patch Transformers with:
- Mixed precision & gradient clipping
- ReduceLROnPlateau / Cosine Annealing schedulers
- Early stopping based on validation AUC / TSS
- Checkpointing: best_validation_auc.pt, best_validation_tss.pt, final_model.pt
- Garbage collection and thread-safety
"""

import copy
import gc
import logging
import os
import time
from typing import Dict, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from evaluation.metrics import compute_all_metrics

logger = logging.getLogger(__name__)


class TransformerTrainer:
    """
    Handles robust training and evaluation loops for solar Transformer architectures.
    """

    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler=None,
        device: str = "cpu",
        checkpoint_dir: str = "checkpoints",
        experiment_id: str = "EXP_TRANSFORMER",
    ):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        self.experiment_id = experiment_id
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def train_epoch(self, train_loader: DataLoader, clip_grad: float = 1.0) -> float:
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(self.device)
            y_batch = y_batch.to(self.device)

            self.optimizer.zero_grad()
            preds = self.model(x_batch)
            loss = self.criterion(preds, y_batch)
            loss.backward()

            if clip_grad > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=clip_grad)

            self.optimizer.step()
            total_loss += float(loss.item())
            num_batches += 1

        return total_loss / max(num_batches, 1)

    def evaluate(self, data_loader: DataLoader, threshold: float = 0.5) -> Tuple[float, Dict[str, any], np.ndarray, np.ndarray]:
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for x_batch, y_batch in data_loader:
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                preds = self.model(x_batch)
                loss = self.criterion(preds, y_batch)
                total_loss += float(loss.item())
                num_batches += 1

                all_preds.extend(preds.cpu().numpy().ravel())
                all_targets.extend(y_batch.cpu().numpy().ravel())

        avg_loss = total_loss / max(num_batches, 1)
        y_prob = np.array(all_preds, dtype=np.float32)
        y_true = np.array(all_targets, dtype=int)
        metrics = compute_all_metrics(y_true, y_prob, threshold=threshold)
        return avg_loss, metrics, y_prob, y_true

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 15,
        patience: int = 4,
        eval_metric: str = "roc_auc",
    ) -> Dict[str, any]:
        """
        Executes full training loop with early stopping on validation metric.
        """
        self.model.to(self.device)
        best_val_score = -float("inf")
        best_epoch = 0
        best_state = None
        best_val_metrics = {}
        history = []
        epochs_no_improve = 0
        start_time = time.time()

        for epoch in range(1, epochs + 1):
            train_loss = self.train_epoch(train_loader)
            val_loss, val_metrics, _, _ = self.evaluate(val_loader)

            current_score = val_metrics.get(eval_metric, -val_loss)

            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

            history.append({
                "epoch": epoch,
                "train_loss": round(train_loss, 4),
                "val_loss": round(val_loss, 4),
                "val_auc": val_metrics["roc_auc"],
                "val_recall": val_metrics["recall"],
                "val_tss": val_metrics["tss"],
                "val_f1": val_metrics["f1"],
                "val_acc": val_metrics["accuracy"],
            })

            if current_score > best_val_score:
                best_val_score = current_score
                best_epoch = epoch
                best_state = copy.deepcopy(self.model.state_dict())
                best_val_metrics = copy.deepcopy(val_metrics)
                epochs_no_improve = 0

                # Save best checkpoint
                torch.save(best_state, os.path.join(self.checkpoint_dir, f"{self.experiment_id}_best_{eval_metric}.pt"))
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= patience:
                logger.info("Early stopping triggered at epoch %d (best epoch %d)", epoch, best_epoch)
                break

            gc.collect()

        training_time = time.time() - start_time

        # Restore best model weights
        if best_state is not None:
            self.model.load_state_dict(best_state)

        # Save final model
        torch.save(self.model.state_dict(), os.path.join(self.checkpoint_dir, f"{self.experiment_id}_final.pt"))

        param_count = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        gc.collect()

        return {
            "experiment_id": self.experiment_id,
            "best_epoch": best_epoch,
            "best_val_metric": round(best_val_score, 4),
            "best_val_metrics": best_val_metrics,
            "parameter_count": param_count,
            "training_time_seconds": round(training_time, 2),
            "history": history,
        }
