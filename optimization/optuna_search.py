"""
Hyperparameter Optimization using Optuna.
Tunes learning rate, weight decay, dropout, embedding dimensions, attention heads,
patch sizes, and loss parameters.
Trained strictly on TRAIN set, evaluated strictly on VALIDATION set.
Test set remains completely untouched.
"""

import json
import logging
import os
from typing import Dict, List, Optional, Tuple
import numpy as np
import optuna
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from datasets.flare_dataset import SolarFlareDataset
from evaluation.metrics import compute_all_metrics
from models.optimized_transformer import AstroNovaOptimizedTransformer
from training.losses import FocalLoss, WeightedBCELoss

# Suppress verbose Optuna logs
optuna.logging.set_verbosity(optuna.logging.WARNING)
logger = logging.getLogger(__name__)


def objective(
    trial: optuna.Trial,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    device: str = "cpu",
    epochs_per_trial: int = 6,
) -> float:
    """
    Optuna objective function maximizing Validation ROC-AUC or TSS.
    """
    # 1. Hyperparameter suggestions
    lr = trial.suggest_float("lr", 1e-4, 1e-3, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True)
    dropout = trial.suggest_float("dropout", 0.1, 0.3)
    d_model = trial.suggest_categorical("d_model", [128, 192])
    nhead = trial.suggest_categorical("nhead", [4, 8])
    num_layers = trial.suggest_categorical("num_layers", [2, 3])
    patch_size = trial.suggest_categorical("patch_size", [2, 4])
    loss_type = trial.suggest_categorical("loss_type", ["weighted_bce", "focal"])

    # 2. Build model
    model = AstroNovaOptimizedTransformer(
        seq_len=24,
        num_features=X_train.shape[2],
        patch_size=patch_size,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dim_feedforward=d_model * 2,
        dropout=dropout,
        pooling="attention",
        pos_encoding_type="learnable",
    ).to(device)

    # 3. Loss function
    if loss_type == "weighted_bce":
        pos_weight = trial.suggest_float("pos_weight", 4.0, 12.0)
        criterion = WeightedBCELoss(pos_weight=pos_weight)
    else:
        gamma = trial.suggest_float("gamma", 1.5, 3.0)
        alpha = trial.suggest_float("alpha", 0.5, 0.85)
        criterion = FocalLoss(gamma=gamma, alpha=alpha)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # 4. DataLoaders
    train_loader = DataLoader(SolarFlareDataset(X_train, y_train), batch_size=128, shuffle=True)
    val_loader = DataLoader(SolarFlareDataset(X_val, y_val), batch_size=256, shuffle=False)

    # 5. Training loop
    for epoch in range(epochs_per_trial):
        model.train()
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            pred = model(x_batch)
            loss = criterion(pred, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

    # 6. Evaluate on Validation set
    model.eval()
    val_preds, val_targets = [], []
    with torch.no_grad():
        for x_batch, y_batch in val_loader:
            x_batch = x_batch.to(device)
            pred = model(x_batch).cpu().numpy().ravel()
            val_preds.extend(pred)
            val_targets.extend(y_batch.numpy().ravel())

    metrics = compute_all_metrics(val_targets, val_preds, threshold=0.5)
    # Objective is validation ROC-AUC + TSS
    score = metrics["roc_auc"] + 0.5 * max(metrics["tss"], 0.0)
    return float(score)


class HyperparameterSearch:
    """
    Manages hyperparameter search studies using Optuna.
    """

    def __init__(self, n_trials: int = 15, study_name: str = "astronova_hparam_search"):
        self.n_trials = n_trials
        self.study_name = study_name
        self.study: Optional[optuna.Study] = None

    def run_search(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        device: str = "cpu",
    ) -> Dict[str, any]:
        """
        Executes the Optuna study.
        """
        self.study = optuna.create_study(direction="maximize", study_name=self.study_name)
        self.study.optimize(
            lambda trial: objective(trial, X_train, y_train, X_val, y_val, device=device),
            n_trials=self.n_trials,
        )

        best_params = self.study.best_params
        best_val_score = self.study.best_value
        logger.info("Optuna best validation score: %.4f with params: %s", best_val_score, best_params)

        os.makedirs("checkpoints", exist_ok=True)
        with open("checkpoints/best_hparams.json", "w") as f:
            json.dump(best_params, f, indent=2)

        return best_params
