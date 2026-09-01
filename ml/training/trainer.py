import argparse
import json
import logging
import os

import torch
import torch.nn as nn
from ml.data.dataset import RealGoesDataset
from ml.models.lightgbm_model import LightGBMForecaster
from torch.utils.data import DataLoader

from ml.models.bilstm import BiLSTMForecaster
from ml.models.gru_model import GRUForecaster
from ml.models.transformer import SolarTransformer
from ml.models.xgboost_model import XGBoostForecaster

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def train_deep_model(
    model_name: str,
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 15,
):
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=3)

    ce_loss = nn.CrossEntropyLoss()
    mse_loss = nn.MSELoss()

    best_val_loss = float("inf")
    patience = 5
    patience_counter = 0

    os.makedirs(f"models/{model_name}", exist_ok=True)

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for batch_x, batch_y_class, batch_y_reg in train_loader:
            optimizer.zero_grad()
            class_preds, reg_preds = model(batch_x, return_tuple=True)

            # Aggregate loss over all horizons
            loss = 0.0
            for h in range(class_preds.shape[1]):
                l_c = ce_loss(class_preds[:, h, :], batch_y_class[:, h])
                l_r = mse_loss(reg_preds[:, h, 0], batch_y_reg[:, h, 0])
                loss += 0.7 * l_c + 0.3 * l_r

            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y_class, batch_y_reg in val_loader:
                class_preds, reg_preds = model(batch_x, return_tuple=True)
                loss = 0.0
                for h in range(class_preds.shape[1]):
                    l_c = ce_loss(class_preds[:, h, :], batch_y_class[:, h])
                    l_r = mse_loss(reg_preds[:, h, 0], batch_y_reg[:, h, 0])
                    loss += 0.7 * l_c + 0.3 * l_r
                val_loss += loss.item()

        val_loss /= len(val_loader)
        scheduler.step(val_loss)

        logger.info(
            f"[{model_name}] Epoch {epoch + 1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), f"models/{model_name}/best.pt")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info("Early stopping triggered.")
                break

    # Load best weights
    model.load_state_dict(torch.load(f"models/{model_name}/best.pt"))
    with open(f"models/{model_name}/metrics.json", "w") as f:
        json.dump({"val_loss": best_val_loss}, f)


def train_tabular_model(model_name: str, model, dataset: RealGoesDataset):
    logger.info(f"Extracting features for {model_name}...")
    X = dataset.X.numpy()
    y_class = dataset.y_class.numpy()
    y_reg = dataset.y_reg.numpy()

    os.makedirs(f"models/{model_name}", exist_ok=True)
    model.fit(X, y_class, y_reg)
    model.save(f"models/{model_name}/model.pkl")

    with open(f"models/{model_name}/metrics.json", "w") as f:
        json.dump({"status": "trained"}, f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["lstm", "gru", "transformer", "xgboost", "lightgbm"],
    )
    parser.add_argument("--data", type=str, default="data/sample/real_time_goes.csv")
    args = parser.parse_args()

    logger.info("Loading dataset...")
    dataset = RealGoesDataset(args.data)

    if args.model in ["xgboost", "lightgbm"]:
        model = XGBoostForecaster() if args.model == "xgboost" else LightGBMForecaster()
        train_tabular_model(args.model, model, dataset)
    else:
        # PyTorch DataLoaders
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

        if args.model == "lstm":
            model = BiLSTMForecaster()
        elif args.model == "gru":
            model = GRUForecaster()
        elif args.model == "transformer":
            model = SolarTransformer()

        train_deep_model(args.model, model, train_loader, val_loader)

    logger.info(f"Model {args.model} trained and saved successfully.")


if __name__ == "__main__":
    main()
