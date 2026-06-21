import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from ml.models.bilstm import BiLSTMForecaster
import os

# Define Dataset class for time-series sequences
class SolarFluxDataset(Dataset):
    def __init__(self, data, seq_len=10, horizon=5):
        self.seq_len = seq_len
        self.horizon = horizon
        self.X, self.y = self._prepare_sequences(data)

    def _prepare_sequences(self, data):
        X, y = [], []
        # Normalizing fluxes for better convergence
        flux = data['soft_xray_flux'].values
        # Log scaling: transform W/m^2 to a scaled range
        flux_scaled = np.log10(flux + 1e-9)
        
        for i in range(len(data) - self.seq_len - self.horizon):
            X.append(flux_scaled[i : i + self.seq_len])
            # Target is the classification of the future flux (A=0, B=1, C=2, M=3, X=4)
            future_flux = flux[i + self.seq_len + self.horizon - 1]
            if future_flux < 1e-8:
                cls = 0
            elif future_flux < 1e-7:
                cls = 1
            elif future_flux < 1e-6:
                cls = 2
            elif future_flux < 1e-5:
                cls = 3
            else:
                cls = 4
            y.append(cls)
            
        return torch.tensor(X, dtype=torch.float32).unsqueeze(-1), torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def train_model(data_path: str, epochs: int = 5):
    print(f"Loading real dataset from {data_path}...")
    if not os.path.exists(data_path):
        print(f"Data path {data_path} does not exist. Please run the data collector first.")
        return

    df = pd.read_csv(data_path)
    print(f"Dataset contains {len(df)} records.")

    # Split into train and validation sets
    train_size = int(len(df) * 0.8)
    train_df = df.iloc[:train_size]
    val_df = df.iloc[train_size:]

    # Prepare datasets and dataloaders
    train_dataset = SolarFluxDataset(train_df)
    val_dataset = SolarFluxDataset(val_df)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # Initialize model (input size = 1 feature: soft_xray_flux)
    model = BiLSTMForecaster(input_size=1, hidden_size=32, num_layers=2, num_classes=5)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    print("Starting model training...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()

        accuracy = 100 * correct / total
        print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(train_loader):.4f} | Train Accuracy: {accuracy:.2f}%")

    # Evaluate on validation data
    model.eval()
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            outputs = model(batch_x)
            _, predicted = torch.max(outputs, 1)
            val_total += batch_y.size(0)
            val_correct += (predicted == batch_y).sum().item()
            
    val_accuracy = 100 * val_correct / val_total
    print(f"Validation Accuracy: {val_accuracy:.2f}%")

    # Save trained model weights
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/bilstm_aditya_real.pt")
    print("Model saved to models/bilstm_aditya_real.pt")

if __name__ == "__main__":
    # Train using the fetched real-time GOES dataset as a test run
    train_model("data/sample/real_time_goes.csv", epochs=3)
