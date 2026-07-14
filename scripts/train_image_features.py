import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import logging

# Ensure root directory is on the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from ml.data.image_features_dataset import ImageFeaturesDataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ImageFeaturesMLP(nn.Module):
    def __init__(self, input_size=2560, hidden_size=512, num_classes=5):
        super(ImageFeaturesMLP, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size // 2, num_classes)
        )
        
    def forward(self, x):
        return self.network(x)

def train_model(csv_131: str, csv_193: str, epochs: int = 5):
    logger.info("Initializing dataset...")
    try:
        dataset = ImageFeaturesDataset(csv_131, csv_193)
    except FileNotFoundError as e:
        logger.error(e)
        return
        
    if len(dataset) == 0:
        logger.error("Merged dataset is empty. Check if flare_id and timestamp match across files.")
        return
        
    logger.info(f"Dataset ready. Total samples: {len(dataset)}")
    
    # Split into train and validation sets
    train_size = int(len(dataset) * 0.8)
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    model = ImageFeaturesMLP(input_size=2560, num_classes=5)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    logger.info("Starting training...")
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
            
        train_acc = 100 * correct / total
        logger.info(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(train_loader):.4f} | Train Acc: {train_acc:.2f}%")
        
    # Evaluate
    model.eval()
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            outputs = model(batch_x)
            _, predicted = torch.max(outputs, 1)
            val_total += batch_y.size(0)
            val_correct += (predicted == batch_y).sum().item()
            
    if val_total > 0:
        val_acc = 100 * val_correct / val_total
        logger.info(f"Validation Accuracy: {val_acc:.2f}%")
        
    # Save model
    os.makedirs("models/image_features", exist_ok=True)
    model_path = "models/image_features/mlp_model.pt"
    torch.save(model.state_dict(), model_path)
    logger.info(f"Model saved to {model_path}")

if __name__ == "__main__":
    csv_131 = "data/features/spectral/image_features_131.csv"
    csv_193 = "data/features/spectral/image_features_193.csv"
    train_model(csv_131, csv_193, epochs=3)
