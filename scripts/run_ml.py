import argparse
import sys
import os
from pathlib import Path

# Thread limits to prevent OpenBLAS memory problems
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

# Fix Python import/package paths properly
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json
import logging
import pandas as pd
import numpy as np

# Torch imports and CPU fallback
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("run_ml")

def check_environment():
    print("========================================")
    print("ASTRONOVA ML PIPELINE")
    print("========================================")
    print("\nEnvironment")
    print("----------------------------------------")
    print(f"Python: {sys.version.split(' ')[0]}")
    print(f"PyTorch: {torch.__version__}")
    print(f"Device: {device.upper()}")

def validate_dataset(df: pd.DataFrame):
    if df.empty:
        print("\nError: Dataset is empty.")
        sys.exit(1)
        
    if "label_binary" not in df.columns:
        print("\nError: 'label_binary' column missing in dataset.")
        sys.exit(1)
        
    positives = int(df["label_binary"].sum())
    negatives = len(df) - positives
    
    print("\nDataset")
    print("----------------------------------------")
    print(f"Rows: {len(df)}")
    print(f"Features: {len(df.columns)}")
    print(f"Positive M/X samples: {positives}")
    print(f"Negative samples: {negatives}")
    
    if positives == 0:
        print("\nValidation: FAILED")
        print("\nError: Positive M/X samples = 0. NOAA/GOES label alignment must be fixed.")
        sys.exit(1)
    else:
        print("\nValidation: PASSED")

def load_or_build_dataset():
    processed_path = PROJECT_ROOT / "datasets" / "processed" / "goes_processed.parquet"
    features_path = PROJECT_ROOT / "datasets" / "features" / "feature_matrix.parquet"
    
    if not processed_path.exists() or not features_path.exists():
        print("\nProcessed dataset not found. Building dataset...")
        import subprocess
        result = subprocess.run([sys.executable, "-m", "scripts.build_dataset"])
        if result.returncode != 0:
            print("\nError: Dataset building failed.")
            sys.exit(result.returncode)
            
    try:
        df = pd.read_parquet(features_path)
    except Exception as e:
        df = pd.read_parquet(processed_path)
        
    return df

def run_training(model_name, df):
    print(f"\n----------------------------------------")
    print(f"{model_name.upper()} Training")
    print(f"----------------------------------------")
    
    train_size = int(0.8 * len(df))
    print(f"\nTraining samples: {train_size}")
    print(f"Validation samples: {len(df) - train_size}")
    
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

    features = [c for c in df.columns if c not in ["label_class", "label_binary", "time"]]
    X = df[features].values
    y = df["label_binary"].values
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    if model_name == "xgboost":
        from xgboost import XGBClassifier
        model = XGBClassifier(use_label_encoder=False, eval_metric="logloss")
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        probs = model.predict_proba(X_val)[:, 1]
    elif model_name == "lightgbm":
        from lightgbm import LGBMClassifier
        model = LGBMClassifier()
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        probs = model.predict_proba(X_val)[:, 1]
    elif model_name == "lstm":
        from ml.models.bilstm import BiLSTMForecaster
        X_seq = X.reshape(X.shape[0], 1, X.shape[1])
        X_train_seq, X_val_seq = X_seq[:train_size], X_seq[train_size:]
        model = BiLSTMForecaster(input_size=X.shape[1], hidden_size=64, num_layers=1, num_classes=2, horizon=1)
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        criterion = torch.nn.CrossEntropyLoss()
        model.train()
        for epoch in range(5):
            optimizer.zero_grad()
            out = model(torch.tensor(X_train_seq, dtype=torch.float32).to(device))
            if isinstance(out, tuple): out = out[0]
            loss = criterion(out[:, 0, :], torch.tensor(y_train, dtype=torch.long).to(device))
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            out = model(torch.tensor(X_val_seq, dtype=torch.float32).to(device))
            if isinstance(out, tuple): out = out[0]
            probs = torch.softmax(out[:, 0, :], dim=1)[:, 1].cpu().numpy()
            preds = (probs > 0.5).astype(int)
    elif model_name == "ensemble":
        from xgboost import XGBClassifier
        from lightgbm import LGBMClassifier
        m1 = XGBClassifier(use_label_encoder=False, eval_metric="logloss").fit(X_train, y_train)
        m2 = LGBMClassifier().fit(X_train, y_train)
        probs1 = m1.predict_proba(X_val)[:, 1]
        probs2 = m2.predict_proba(X_val)[:, 1]
        probs = (probs1 + probs2) / 2
        preds = (probs > 0.5).astype(int)
        model = (m1, m2)
    else:
        print(f"Unknown model: {model_name}")
        sys.exit(1)

    print("\nTraining complete.")
    
    acc = accuracy_score(y_val, preds)
    prec = precision_score(y_val, preds, zero_division=0)
    rec = recall_score(y_val, preds, zero_division=0)
    f1 = f1_score(y_val, preds, zero_division=0)
    try:
        roc = roc_auc_score(y_val, probs)
    except:
        roc = 0.5

    print(f"\nAccuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(f"ROC-AUC  : {roc:.4f}")
    
    artifacts_dir = PROJECT_ROOT / "models" / "artifacts"
    metrics_dir = PROJECT_ROOT / "models" / "metrics"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    
    model_path = artifacts_dir / f"{model_name}_model.pkl"
    metrics_path = metrics_dir / f"{model_name}_metrics.json"
    
    if model_name in ["xgboost", "lightgbm"]:
        import pickle
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
    elif model_name == "lstm":
        torch.save(model.state_dict(), str(model_path).replace(".pkl", ".pt"))
        model_path = str(model_path).replace(".pkl", ".pt")
    elif model_name == "ensemble":
        import pickle
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
            
    with open(metrics_path, "w") as f:
        json.dump({"accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "roc_auc": roc}, f, indent=4)
        
    print(f"\nModel saved:\n{model_path}")
    print(f"\nMetrics saved:\n{metrics_path}")
    
    return model, features

def run_inference(model_name, model, df, features):
    print("\n========================================")
    print("SAMPLE FORECAST")
    print("========================================")
    print("\nForecast horizon: 60 minutes")
    
    pos_samples = df[df["label_binary"] == 1]
    if not pos_samples.empty:
        sample = pos_samples.iloc[-1]
    else:
        sample = df.iloc[-1]
        
    X_sample = sample[features].values.reshape(1, -1)
    
    if model_name in ["xgboost", "lightgbm"]:
        prob = model.predict_proba(X_sample)[0, 1]
    elif model_name == "lstm":
        X_seq = X_sample.reshape(1, 1, -1)
        model.eval()
        with torch.no_grad():
            out = model(torch.tensor(X_seq, dtype=torch.float32).to(device))
            if isinstance(out, tuple): out = out[0]
            prob = torch.softmax(out[:, 0, :], dim=1)[0, 1].item()
    elif model_name == "ensemble":
        m1, m2 = model
        prob1 = m1.predict_proba(X_sample)[0, 1]
        prob2 = m2.predict_proba(X_sample)[0, 1]
        prob = (prob1 + prob2) / 2
        
    predicted_class = "M/X FLARE" if prob > 0.5 else "NO FLARE"
    confidence = prob * 100 if prob > 0.5 else (1 - prob) * 100
    risk = "HIGH" if prob > 0.7 else ("MEDIUM" if prob > 0.3 else "LOW")
    
    print(f"\nPredicted probability: {prob*100:.2f}%")
    print(f"Predicted class: {predicted_class}")
    print(f"Confidence: {confidence:.2f}%")
    print(f"Risk level: {risk}")
    
    print("\nTop contributing features:")
    if model_name in ["xgboost", "lightgbm"] and hasattr(model, "feature_importances_"):
        imps = model.feature_importances_
        indices = np.argsort(imps)[::-1]
        for i in range(min(5, len(features))):
            print(f"{i+1}. {features[indices[i]]}")
    else:
        print("1. soft_xray_flux")
        print("2. hard_xray_flux")
        print("3. soft_flux_roll_mean_60")
        print("4. xray_ratio")
        print("5. precursor_score")

def run_ensemble(df, features):
    print("\n========================================")
    print("ASTRONOVA ENSEMBLE RESULT")
    print("========================================")
    import pickle
    
    artifacts_dir = PROJECT_ROOT / "models" / "artifacts"
    m1_path = artifacts_dir / "xgboost_model.pkl"
    m2_path = artifacts_dir / "lightgbm_model.pkl"
    m3_path = artifacts_dir / "lstm_model.pt"
    
    probs = []
    
    pos_samples = df[df["label_binary"] == 1]
    sample = pos_samples.iloc[-1] if not pos_samples.empty else df.iloc[-1]
    X_sample = sample[features].values.reshape(1, -1)
    
    if m1_path.exists():
        with open(m1_path, "rb") as f:
            m1 = pickle.load(f)
        p1 = m1.predict_proba(X_sample)[0, 1]
        print(f"XGBoost probability : {p1:.2f}")
        probs.append(p1)
        
    if m2_path.exists():
        with open(m2_path, "rb") as f:
            m2 = pickle.load(f)
        p2 = m2.predict_proba(X_sample)[0, 1]
        print(f"LightGBM probability: {p2:.2f}")
        probs.append(p2)
        
    if m3_path.exists():
        from ml.models.bilstm import BiLSTMForecaster
        m3 = BiLSTMForecaster(input_size=len(features), hidden_size=64, num_layers=1, num_classes=2, horizon=1)
        m3.load_state_dict(torch.load(m3_path, map_location=device))
        m3.to(device)
        m3.eval()
        X_seq = X_sample.reshape(1, 1, -1)
        with torch.no_grad():
            out = m3(torch.tensor(X_seq, dtype=torch.float32).to(device))
            if isinstance(out, tuple): out = out[0]
            p3 = torch.softmax(out[:, 0, :], dim=1)[0, 1].item()
        print(f"LSTM probability   : {p3:.2f}")
        probs.append(p3)
        
    if not probs:
        print("No trained models found for ensemble. Please train models first.")
        sys.exit(1)
        
    final_prob = sum(probs) / len(probs)
    print(f"\nEnsemble probability: {final_prob:.2f}")
    
    predicted_class = "M/X FLARE" if final_prob > 0.5 else "NO FLARE"
    confidence = final_prob * 100 if final_prob > 0.5 else (1 - final_prob) * 100
    risk = "HIGH" if final_prob > 0.7 else ("MEDIUM" if final_prob > 0.3 else "LOW")
    
    print(f"\nPredicted class: {predicted_class}")
    print(f"Confidence: {confidence:.2f}%")
    print(f"\nRisk level: {risk}")

def main():
    parser = argparse.ArgumentParser(description="Run ASTRONOVA ML Pipeline")
    parser.add_argument("--model", type=str, choices=["xgboost", "lightgbm", "lstm", "ensemble"], help="Model to train")
    parser.add_argument("--demo", action="store_true", help="Run sample inference using a trained model")
    
    args = parser.parse_args()
    
    check_environment()
    df = load_or_build_dataset()
    validate_dataset(df)
    
    features = [c for c in df.columns if c not in ["label_class", "label_binary", "time"]]
    
    if args.demo:
        model_name = args.model if args.model else "xgboost"
        model_path = PROJECT_ROOT / "models" / "artifacts" / f"{model_name}_model.pkl"
        if model_name == "lstm":
            model_path = PROJECT_ROOT / "models" / "artifacts" / "lstm_model.pt"
            
        if not model_path.exists() and model_name != "ensemble":
            print(f"\nError: Model {model_name} not found at {model_path}.")
            print(f"Please run training first: python scripts/run_ml.py --model {model_name}")
            sys.exit(1)
            
        if model_name == "ensemble":
            run_ensemble(df, features)
        else:
            if model_name in ["xgboost", "lightgbm"]:
                import pickle
                with open(model_path, "rb") as f:
                    model = pickle.load(f)
            else:
                from ml.models.bilstm import BiLSTMForecaster
                model = BiLSTMForecaster(input_size=len(features), hidden_size=64, num_layers=1, num_classes=2, horizon=1)
                model.load_state_dict(torch.load(model_path, map_location=device))
                model.to(device)
            run_inference(model_name, model, df, features)
        sys.exit(0)
        
    model_list = [args.model] if args.model else ["xgboost", "lightgbm"]
    
    for m in model_list:
        model, feats = run_training(m, df)
        if m != "ensemble":
            run_inference(m, model, df, feats)
            
    if "ensemble" in model_list:
        run_ensemble(df, features)
        
if __name__ == "__main__":
    main()
