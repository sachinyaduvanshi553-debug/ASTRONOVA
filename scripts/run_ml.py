r"""ASTRONOVA - Standalone ML Pipeline CLI
========================================
Run the full ML forecasting pipeline entirely from the terminal.
No frontend, no backend, no Docker required.

Usage (from project root):
    .\\venv\\Scripts\\python.exe scripts\\run_ml.py
    .\\venv\\Scripts\\python.exe scripts\\run_ml.py --model xgboost
    .\\venv\\Scripts\\python.exe scripts\\run_ml.py --model lightgbm
    .\\venv\\Scripts\\python.exe scripts\\run_ml.py --model lstm
    .\\venv\\Scripts\\python.exe scripts\\run_ml.py --model ensemble
    .\\venv\\Scripts\\python.exe scripts\\run_ml.py --demo
    .\\venv\\Scripts\\python.exe scripts\\run_ml.py --help
"""

from __future__ import annotations

# Thread limits must be set BEFORE any heavy imports to avoid OpenBLAS crashes
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

# Standard library imports
import sys
import argparse
import json
import logging
import pickle
import warnings
from pathlib import Path

# Suppress non‑fatal warnings that would otherwise cause PowerShell to treat them as errors
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Numerical stack
import numpy as np
import pandas as pd

# PyTorch – CPU‑only safe import
try:
    import torch
    _TORCH_VERSION = torch.__version__
    device = "cuda" if torch.cuda.is_available() else "cpu"
    HAS_TORCH = True
except ImportError:
    _TORCH_VERSION = "not installed"
    device = "cpu"
    HAS_TORCH = False

# Logging – simple, non‑verbose output for the terminal
logging.basicConfig(level=logging.WARNING, format="%(message)s")
logger = logging.getLogger("run_ml")

# Project paths – all relative to the repository root
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

PROCESSED_PATH = PROJECT_ROOT / "datasets" / "processed" / "goes_processed.parquet"
FEATURES_PATH  = PROJECT_ROOT / "datasets" / "features"  / "feature_matrix.parquet"
ARTIFACTS_DIR  = PROJECT_ROOT / "models"   / "artifacts"
METRICS_DIR    = PROJECT_ROOT / "models"   / "metrics"
INFERENCE_DIR  = PROJECT_ROOT / "reports"  / "inference"

_LABEL_COLS = {"label_class", "label_binary", "time"}

# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------
def check_environment() -> None:
    print("========================================")
    print("ASTRONOVA ML PIPELINE")
    print("========================================")
    print("\nEnvironment")
    print("----------------------------------------")
    print(f"Python : {sys.version.split()[0]}")
    print(f"PyTorch: {_TORCH_VERSION}")
    print(f"Device : {device.upper()}")

def load_or_build_dataset() -> pd.DataFrame:
    """Return the feature matrix, building it automatically if missing."""
    if not FEATURES_PATH.exists() or not PROCESSED_PATH.exists():
        print("\nProcessed dataset not found – building dataset automatically…")
        import subprocess
        result = subprocess.run(
            [sys.executable, "-m", "scripts.build_dataset"],
            cwd=str(PROJECT_ROOT),
        )
        if result.returncode != 0:
            print("\n[ERROR] Dataset build failed.")
            sys.exit(result.returncode)
    try:
        df = pd.read_parquet(FEATURES_PATH)
        if "label_binary" not in df.columns:
            df = pd.read_parquet(PROCESSED_PATH)
    except Exception as exc:
        print(f"\n[ERROR] Could not load dataset: {exc}")
        sys.exit(1)
    return df

def validate_dataset(df: pd.DataFrame) -> list[str]:
    """Validate the dataset and return the list of feature columns."""
    if df is None or df.empty:
        print("\n[ERROR] Dataset is empty.")
        sys.exit(1)
    if "label_binary" not in df.columns:
        print("\n[ERROR] 'label_binary' column missing – re‑run build_dataset.")
        sys.exit(1)
    features = [c for c in df.columns if c not in _LABEL_COLS]
    positives = int(df["label_binary"].sum())
    negatives = len(df) - positives
    print("\nDataset")
    print("----------------------------------------")
    print(f"Rows                : {len(df)}")
    print(f"Features            : {len(features)}")
    print(f"Positive M/X samples: {positives}")
    print(f"Negative samples    : {negatives}")
    if positives == 0:
        print("\nValidation: FAILED")
        print("\n[ERROR] Positive M/X samples = 0. Check NOAA → GOES label alignment.")
        sys.exit(1)
    print("\nValidation: PASSED")
    return features

# ---------------------------------------------------------------------------
# Metrics utilities
# ---------------------------------------------------------------------------
def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> dict:
    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        roc_auc_score,
        average_precision_score,
        confusion_matrix,
    )
    m = {}
    m["accuracy"] = float(accuracy_score(y_true, y_pred))
    m["precision"] = float(precision_score(y_true, y_pred, zero_division=0))
    m["recall"] = float(recall_score(y_true, y_pred, zero_division=0))
    m["f1"] = float(f1_score(y_true, y_pred, zero_division=0))
    try:
        m["roc_auc"] = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        m["roc_auc"] = 0.5
    try:
        m["pr_auc"] = float(average_precision_score(y_true, y_prob))
    except ValueError:
        m["pr_auc"] = 0.0
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    m["confusion_matrix"] = cm.tolist()
    return m

def print_metrics(metrics: dict) -> None:
    print(f"\nAccuracy  : {metrics['accuracy']:.4f}")
    print(f"Precision : {metrics['precision']:.4f}")
    print(f"Recall    : {metrics['recall']:.4f}")
    print(f"F1 Score  : {metrics['f1']:.4f}")
    print(f"ROC-AUC   : {metrics['roc_auc']:.4f}")
    print(f"PR-AUC    : {metrics['pr_auc']:.4f}")
    cm = metrics.get("confusion_matrix", [[0, 0], [0, 0]])
    print("\nConfusion Matrix (rows=actual, cols=pred):")
    print(f"  TN={cm[0][0]:5d}  FP={cm[0][1]:5d}")
    print(f"  FN={cm[1][0]:5d}  TP={cm[1][1]:5d}")

def save_artifacts(model_name: str, model, metrics: dict) -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    if model_name == "lstm" and HAS_TORCH and hasattr(model, "state_dict"):
        path = ARTIFACTS_DIR / "lstm_model.pt"
        torch.save(model.state_dict(), str(path))
    else:
        path = ARTIFACTS_DIR / f"{model_name}_model.pkl"
        with open(path, "wb") as f:
            pickle.dump(model, f)
    metrics_path = METRICS_DIR / f"{model_name}_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"\nModel saved:\n  {path}")
    print(f"Metrics saved:\n  {metrics_path}")

# ---------------------------------------------------------------------------
# Model trainers
# ---------------------------------------------------------------------------
def train_xgboost(X_train, y_train, X_val, y_val):
    try:
        import xgboost as xgb
        model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.05,
            eval_metric="logloss",
            random_state=42,
        )
    except ImportError:
        from sklearn.ensemble import GradientBoostingClassifier
        print("  [INFO] XGBoost not installed – using sklearn fallback.")
        model = GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_val)
    probs = model.predict_proba(X_val)[:, 1]
    return model, preds, probs

def train_lightgbm(X_train, y_train, X_val, y_val):
    try:
        import lightgbm as lgb
        model = lgb.LGBMClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.05,
            random_state=42,
            verbose=-1,
        )
    except ImportError:
        from sklearn.ensemble import GradientBoostingClassifier
        print("  [INFO] LightGBM not installed – using sklearn fallback.")
        model = GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_val)
    probs = model.predict_proba(X_val)[:, 1]
    return model, preds, probs

def train_lstm(X_train, y_train, X_val, y_val, n_features):
    """Train the BiLSTMForecaster for binary flare classification."""
    if not HAS_TORCH:
        print("  [ERROR] PyTorch not installed – cannot train LSTM.")
        sys.exit(1)
    from ml.models.bilstm import BiLSTMForecaster
    model = BiLSTMForecaster(
        input_size=n_features,
        hidden_size=64,
        num_layers=1,
        num_classes=2,
        num_horizons=1,
        dropout=0.0,
    )
    model.to(device)
    X_tr = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1).to(device)
    X_vl = torch.tensor(X_val, dtype=torch.float32).unsqueeze(1).to(device)
    y_tr = torch.tensor(y_train, dtype=torch.long).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()
    epochs = 10
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        logits = model(X_tr)
        loss = criterion(logits, y_tr)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1}/{epochs}  loss={loss.item():.4f}")
    model.eval()
    with torch.no_grad():
        logits_val = model(X_vl)
        probs = torch.softmax(logits_val, dim=1)[:, 1].cpu().numpy()
        preds = (probs > 0.5).astype(int)
    return model, preds, probs

# ---------------------------------------------------------------------------
# Dispatcher – training
# ---------------------------------------------------------------------------
def run_training(model_name: str, df: pd.DataFrame, features: list[str]):
    print(f"\n----------------------------------------")
    print(f"{model_name.upper()} Training")
    print(f"----------------------------------------")
    from sklearn.model_selection import train_test_split
    X = df[features].fillna(0).values.astype(np.float32)
    y = df["label_binary"].values.astype(int)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\nTraining samples  : {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    if model_name == "xgboost":
        model, preds, probs = train_xgboost(X_train, y_train, X_val, y_val)
    elif model_name == "lightgbm":
        model, preds, probs = train_lightgbm(X_train, y_train, X_val, y_val)
    elif model_name == "lstm":
        model, preds, probs = train_lstm(X_train, y_train, X_val, y_val, len(features))
    elif model_name == "ensemble":
        m1, _, p1 = train_xgboost(X_train, y_train, X_val, y_val)
        m2, _, p2 = train_lightgbm(X_train, y_train, X_val, y_val)
        probs = (p1 + p2) / 2
        preds = (probs > 0.5).astype(int)
        model = (m1, m2)
    else:
        print(f"[ERROR] Unknown model: {model_name}")
        sys.exit(1)
    print("\nTraining complete.")
    metrics = compute_metrics(y_val, preds, probs)
    print_metrics(metrics)
    save_artifacts(model_name, model, metrics)
    return model, features

# ---------------------------------------------------------------------------
# Inference – single model
# ---------------------------------------------------------------------------
def run_inference(model_name: str, model, df: pd.DataFrame, features: list[str]):
    print("\n========================================")
    print("SAMPLE FORECAST")
    print("========================================")
    print("\nForecast horizon: 60 minutes")
    pos = df[df["label_binary"] == 1]
    sample = pos.iloc[-1] if not pos.empty else df.iloc[-1]
    X_sample = sample[features].fillna(0).values.reshape(1, -1).astype(np.float32)
    if model_name in ("xgboost", "lightgbm"):
        prob = float(model.predict_proba(X_sample)[0, 1])
    elif model_name == "lstm":
        if not HAS_TORCH:
            print("[ERROR] PyTorch not available for LSTM inference.")
            return
        model.eval()
        X_t = torch.tensor(X_sample, dtype=torch.float32).unsqueeze(1).to(device)
        with torch.no_grad():
            logits = model(X_t)
            prob = float(torch.softmax(logits, dim=1)[0, 1].item())
    elif model_name == "ensemble":
        m1, m2 = model
        prob = (float(m1.predict_proba(X_sample)[0, 1]) + float(m2.predict_proba(X_sample)[0, 1])) / 2
    else:
        print(f"[ERROR] Unknown model: {model_name}")
        return
    predicted = "M/X FLARE" if prob > 0.5 else "NO FLARE"
    confidence = prob * 100 if prob > 0.5 else (1 - prob) * 100
    risk = "HIGH" if prob > 0.7 else ("MEDIUM" if prob > 0.3 else "LOW")
    print(f"\nPredicted probability: {prob * 100:.2f}%")
    print(f"Predicted class      : {predicted}")
    print(f"Confidence           : {confidence:.2f}%")
    print(f"Risk level           : {risk}")
    print("\nTop contributing features:")
    src = model[0] if model_name == "ensemble" else model
    if hasattr(src, "feature_importances_"):
        for i in np.argsort(src.feature_importances_)[::-1][:5]:
            print(f"  {features[i]}")
    else:
        for ft in ["soft_xray_flux", "hard_xray_flux", "soft_flux_roll_mean_60", "xray_ratio", "precursor_score"]:
            print(f"  {ft}")
    INFERENCE_DIR.mkdir(parents=True, exist_ok=True)
    report_path = INFERENCE_DIR / f"{model_name}_inference.json"
    with open(report_path, "w") as f:
        json.dump({
            "model": model_name,
            "predicted_probability": round(prob, 4),
            "predicted_class": predicted,
            "confidence_pct": round(confidence, 2),
            "risk_level": risk,
        }, f, indent=4)
    print(f"\nInference report saved:\n  {report_path}")

# ---------------------------------------------------------------------------
# Ensemble inference from saved artefacts
# ---------------------------------------------------------------------------
def run_ensemble(df: pd.DataFrame, features: list[str]):
    print("\n========================================")
    print("ASTRONOVA ENSEMBLE RESULT")
    print("========================================")
    pos = df[df["label_binary"] == 1]
    sample = pos.iloc[-1] if not pos.empty else df.iloc[-1]
    X_sample = sample[features].fillna(0).values.reshape(1, -1).astype(np.float32)
    probs: dict[str, float] = {}
    xb_path = ARTIFACTS_DIR / "xgboost_model.pkl"
    if xb_path.exists():
        with open(xb_path, "rb") as f:
            xb = pickle.load(f)
        probs["XGBoost"] = float(xb.predict_proba(X_sample)[0, 1])
    lg_path = ARTIFACTS_DIR / "lightgbm_model.pkl"
    if lg_path.exists():
        with open(lg_path, "rb") as f:
            lg = pickle.load(f)
        probs["LightGBM"] = float(lg.predict_proba(X_sample)[0, 1])
    lst_path = ARTIFACTS_DIR / "lstm_model.pt"
    if lst_path.exists() and HAS_TORCH:
        from ml.models.bilstm import BiLSTMForecaster
        lst = BiLSTMForecaster(
            input_size=len(features), hidden_size=64,
            num_layers=1, num_classes=2, num_horizons=1, dropout=0.0,
        )
        lst.load_state_dict(torch.load(str(lst_path), map_location=device))
        lst.to(device).eval()
        X_t = torch.tensor(X_sample, dtype=torch.float32).unsqueeze(1).to(device)
        with torch.no_grad():
            logits = lst(X_t)
            probs["LSTM"] = float(torch.softmax(logits, dim=1)[0, 1].item())
    if not probs:
        print("\n[ERROR] No trained models found – run training first.")
        sys.exit(1)
    for name, p in probs.items():
        print(f"  {name:<10} probability: {p:.4f}")
    final_prob = sum(probs.values()) / len(probs)
    print(f"\nEnsemble probability: {final_prob:.4f}")
    pred = "M/X FLARE" if final_prob > 0.5 else "NO FLARE"
    conf = final_prob * 100 if final_prob > 0.5 else (1 - final_prob) * 100
    risk = "HIGH" if final_prob > 0.7 else ("MEDIUM" if final_prob > 0.3 else "LOW")
    print(f"\nPredicted class: {pred}")
    print(f"Confidence     : {conf:.2f}%")
    print(f"Risk level     : {risk}")

# ---------------------------------------------------------------------------
# Demo mode – inference only using a saved model
# ---------------------------------------------------------------------------
def run_demo(df: pd.DataFrame, features: list[str], model_name: str = "xgboost") -> None:
    if model_name == "ensemble":
        run_ensemble(df, features)
        return
    model_path = (
        ARTIFACTS_DIR / "lstm_model.pt"
        if model_name == "lstm"
        else ARTIFACTS_DIR / f"{model_name}_model.pkl"
    )
    if not model_path.exists():
        print(f"\n[ERROR] Model {model_name} not found at {model_path} – train first.")
        sys.exit(1)
    if model_name == "lstm":
        if not HAS_TORCH:
            print("[ERROR] PyTorch not installed – cannot load LSTM.")
            sys.exit(1)
        from ml.models.bilstm import BiLSTMForecaster
        model = BiLSTMForecaster(
            input_size=len(features), hidden_size=64,
            num_layers=1, num_classes=2, num_horizons=1, dropout=0.0,
        )
        model.load_state_dict(torch.load(str(model_path), map_location=device))
        model.to(device).eval()
    else:
        with open(model_path, "rb") as f:
            model = pickle.load(f)
    run_inference(model_name, model, df, features)

# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    epilog = """
Examples:
  .\\venv\\Scripts\\python.exe scripts\\run_ml.py
  .\\venv\\Scripts\\python.exe scripts\\run_ml.py --model xgboost
  .\\venv\\Scripts\\python.exe scripts\\run_ml.py --model lightgbm
  .\\venv\\Scripts\\python.exe scripts\\run_ml.py --model lstm
  .\\venv\\Scripts\\python.exe scripts\\run_ml.py --model ensemble
  .\\venv\\Scripts\\python.exe scripts\\run_ml.py --demo
"""
    parser = argparse.ArgumentParser(
        prog="run_ml.py",
        description="ASTRONOVA – Standalone ML Pipeline (no UI required)",
        epilog=epilog,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model",
        choices=["xgboost", "lightgbm", "lstm", "ensemble"],
        default=None,
        help="Model to train or run inference with.",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Load a saved model and run inference only (no training).",
    )
    return parser

# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    check_environment()
    df = load_or_build_dataset()
    features = validate_dataset(df)
    if args.demo:
        model_name = args.model or "xgboost"
        print(f"\n[DEMO] Loading {model_name} model for inference …")
        run_demo(df, features, model_name)
        print("\n========================================")
        print("ASTRONOVA ML PIPELINE – DEMO COMPLETE")
        print("========================================")
        sys.exit(0)
    if args.model == "ensemble":
        run_ensemble(df, features)
        print("\n========================================")
        print("ASTRONOVA ML PIPELINE – COMPLETE")
        print("========================================")
        sys.exit(0)
    model_list = [args.model] if args.model else ["xgboost", "lightgbm"]
    for m_name in model_list:
        model, feats = run_training(m_name, df, features)
        run_inference(m_name, model, df, feats)
    print("\n========================================")
    print("ASTRONOVA ML PIPELINE – COMPLETE")
    print("========================================")

if __name__ == "__main__":
    main()
