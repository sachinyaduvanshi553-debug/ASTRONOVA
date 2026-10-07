"""
PyTorch Dataset and DataLoader for Solar Flare Time-Series Sequences.
Supports fast tensor indexing, pinned memory, and custom batching.
"""

from typing import Optional, Tuple
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class SolarFlareDataset(Dataset):
    """
    Encapsulates (N, 24, 16) solar flare sequences and binary labels.
    """

    def __init__(
        self,
        sequences: np.ndarray,
        labels: np.ndarray,
        transform=None,
    ):
        self.sequences = torch.tensor(sequences, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.sequences[idx]
        y = self.labels[idx]
        if self.transform:
            x = self.transform(x)
        return x, y


def create_dataloaders(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    batch_size: int = 64,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Factory function for Train, Val, and Test DataLoaders.
    """
    train_dataset = SolarFlareDataset(X_train, y_train)
    val_dataset = SolarFlareDataset(X_val, y_val)
    test_dataset = SolarFlareDataset(X_test, y_test)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    return train_loader, val_loader, test_loader
