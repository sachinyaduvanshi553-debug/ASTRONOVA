import argparse
import logging

import numpy as np
import torch
from ml.data.dataset import RealGoesDataset
from ml.models.lightgbm_model import LightGBMForecaster
from torch.utils.data import DataLoader

from ml.models.bilstm import BiLSTMForecaster
from ml.models.xgboost_model import XGBoostForecaster
from ml.training.metrics import calculate_classification_metrics, calculate_regression_metrics

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def evaluate_model(model_name: str, model, data_loader: DataLoader):
    model.eval()
    all_class_preds, all_class_probs, all_class_true = [], [], []
    all_reg_preds, all_reg_true = [], []

    with torch.no_grad():
        for batch_x, batch_y_class, batch_y_reg in data_loader:
            class_probs, reg_preds = model(batch_x, return_tuple=True)

            # Use horizon 0 (15m) for simplified reporting
            probs = class_probs[:, 0, :].numpy()
            preds = np.argmax(probs, axis=-1)

            all_class_probs.extend(probs)
            all_class_preds.extend(preds)
            all_class_true.extend(batch_y_class[:, 0].numpy())

            all_reg_preds.extend(reg_preds[:, 0, 0].numpy())
            all_reg_true.extend(batch_y_reg[:, 0, 0].numpy())

    c_metrics = calculate_classification_metrics(
        np.array(all_class_true), np.array(all_class_preds), np.array(all_class_probs)
    )
    r_metrics = calculate_regression_metrics(np.array(all_reg_true), np.array(all_reg_preds))

    c_metrics.update(r_metrics)
    return c_metrics


def evaluate_tabular(model_name: str, model, dataset: RealGoesDataset):
    X = dataset.X.numpy()
    y_class_true = dataset.y_class.numpy()[:, 0]
    y_reg_true = dataset.y_reg.numpy()[:, 0, 0]

    probs_out, regs_out = model.predict(X)

    probs = probs_out[:, 0, :]
    preds = np.argmax(probs, axis=-1)
    regs = regs_out[:, 0, 0]

    c_metrics = calculate_classification_metrics(y_class_true, preds, probs)
    r_metrics = calculate_regression_metrics(y_reg_true, regs)

    c_metrics.update(r_metrics)
    return c_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--data", type=str, default="data/sample/real_time_goes.csv")
    args = parser.parse_args()

    dataset = RealGoesDataset(args.data)

    if args.model in ["xgboost", "lightgbm"]:
        if args.model == "xgboost":
            model = XGBoostForecaster.load(f"models/{args.model}/model.pkl")
        else:
            model = LightGBMForecaster.load(f"models/{args.model}/model.pkl")
        metrics = evaluate_tabular(args.model, model, dataset)
    else:
        if args.model == "lstm":
            model = BiLSTMForecaster()

        model.load_state_dict(torch.load(f"models/{args.model}/best.pt"))
        loader = DataLoader(dataset, batch_size=32, shuffle=False)
        metrics = evaluate_model(args.model, model, loader)

    print(f"\n--- Metrics for {args.model.upper()} (15m horizon) ---")
    for k, v in metrics.items():
        print(f"{k.upper()}: {v:.4f}")
