import pandas as pd
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# -----------------------------
# PATH
# -----------------------------
DATA_PATH = Path("data/fused/train.csv")
MODEL_PATH = Path("models/flare_model.pkl")


# -----------------------------
# LOAD DATA
# -----------------------------
def load_data():
    print("📥 Loading training data...")
    df = pd.read_csv(DATA_PATH)
    return df


# -----------------------------
# PREPARE DATA
# -----------------------------
def prepare_data(df):

    print("⚙️ Preparing data...")

    # Drop timestamp safely
    if "timestamp" in df.columns:
        df = df.drop(columns=["timestamp"])

    # Target column (adjust if your dataset differs)
    target_col = "flare_class"

    df = df.dropna()

    X = df.drop(columns=[target_col])
    y = df[target_col]

    return train_test_split(X, y, test_size=0.2, random_state=42)


# -----------------------------
# TRAIN MODEL
# -----------------------------
def train_model(X_train, X_test, y_train, y_test):

    print("🤖 Training model...")

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    acc = model.score(X_test, y_test)
    print(f"✅ Accuracy: {acc:.4f}")

    return model


# -----------------------------
# SAVE MODEL
# -----------------------------
def save_model(model):
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"💾 Model saved at: {MODEL_PATH}")


# -----------------------------
# RUN
# -----------------------------
def run():
    df = load_data()
    X_train, X_test, y_train, y_test = prepare_data(df)
    model = train_model(X_train, X_test, y_train, y_test)
    save_model(model)


if __name__ == "__main__":
    run()