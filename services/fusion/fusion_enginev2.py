import pandas as pd
import numpy as np
from pathlib import Path

BASE = Path("data/cleaned")

GOES_PATH = BASE / "goes/goes_xrs_oct2024_jan2025.csv"
HELIOS_PATH = BASE / "helios/helios_filtered.parquet"
SOLEXS_PATH = BASE / "solexs/solexs_ml_ready.csv"
NOAA_PATH = BASE / "noaa_labels/noaa_flares.csv"

OUTPUT_PATH = Path("data/fused/fusion_v2.csv")


# -----------------------------
# 1. LOAD DATA
# -----------------------------
def load_data():
    goes = pd.read_csv(GOES_PATH)
    helios = pd.read_parquet(HELIOS_PATH)
    solexs = pd.read_csv(SOLEXS_PATH)
    noaa = pd.read_csv(NOAA_PATH)
    return goes, helios, solexs, noaa


# -----------------------------
# 2. STANDARDIZE TIME
# -----------------------------
def get_time_col(df):
    return [c for c in df.columns if "time" in c.lower()][0]


def standardize(df):
    t = get_time_col(df)
    df[t] = pd.to_datetime(df[t], errors="coerce")
    df = df.dropna(subset=[t])
    return df.rename(columns={t: "timestamp"}).sort_values("timestamp")


# -----------------------------
# 3. FEATURE ENGINEERING
# -----------------------------
def engineer_goes(goes):
    flux_cols = [c for c in goes.columns if "flux" in c.lower()]

    if len(flux_cols) > 0:
        col = flux_cols[0]
        goes["goes_flux_rate"] = goes[col].diff().fillna(0)
        goes["goes_flux_norm"] = (goes[col] - goes[col].mean()) / (goes[col].std() + 1e-6)

    return goes


def engineer_helios(helios):
    num_cols = helios.select_dtypes(include=np.number).columns

    if len(num_cols) > 0:
        col = num_cols[0]
        helios["helios_grad"] = helios[col].diff().fillna(0)

    return helios


def engineer_solexs(solexs):
    num_cols = solexs.select_dtypes(include=np.number).columns
    solexs["solexs_mean"] = solexs[num_cols].mean(axis=1)
    return solexs


# -----------------------------
# 4. ASOF MERGE
# -----------------------------
def asof(base, other, window="10min"):
    return pd.merge_asof(
        base.sort_values("timestamp"),
        other.sort_values("timestamp"),
        on="timestamp",
        direction="nearest",
        tolerance=pd.Timedelta(window)
    )


# -----------------------------
# 5. FLARE PROBABILITY SCORE
# -----------------------------
def compute_risk(df):
    """
    Physics-inspired risk score (NOT ML yet)
    """

    cols = df.select_dtypes(include=np.number).columns

    score = np.zeros(len(df))

    for c in cols:
        norm = (df[c] - df[c].mean()) / (df[c].std() + 1e-6)
        score += np.clip(norm, -2, 2)

    score = score / len(cols)

    # squash to 0-1
    df["flare_risk_score"] = 1 / (1 + np.exp(-score))

    return df


# -----------------------------
# 6. FUSION ENGINE V2
# -----------------------------
def build_fusion_v2(goes, helios, solexs, noaa):

    print("🔗 Starting V2 fusion...")

    # --- Feature engineering first ---
    goes = engineer_goes(goes)
    helios = engineer_helios(helios)
    solexs = engineer_solexs(solexs)

    # --- Base = GOES ---
    fused = goes.copy()

    print("➕ Merging HELIOS...")
    fused = asof(fused, helios, "10min")

    print("➕ Merging SOLEXS...")
    fused = asof(fused, solexs, "10min")

    print("➕ Merging NOAA labels...")
    fused = asof(fused, noaa, "30min")

    # --- Physics-aware scoring ---
    print("🧠 Computing flare risk score...")
    fused = compute_risk(fused)

    # --- Clean ---
    fused = fused.fillna(method="ffill").fillna(method="bfill")
    fused = fused.drop_duplicates(subset=["timestamp"])

    return fused


# -----------------------------
# 7. SPLIT
# -----------------------------
def split(df):
    idx = int(len(df) * 0.8)
    return df.iloc[:idx], df.iloc[idx:]


# -----------------------------
# 8. RUN
# -----------------------------
def run():
    goes, helios, solexs, noaa = load_data()

    goes = standardize(goes)
    helios = standardize(helios)
    solexs = standardize(solexs)
    noaa = standardize(noaa)

    fused = build_fusion_v2(goes, helios, solexs, noaa)

    train, test = split(fused)

    Path("data/fused").mkdir(exist_ok=True)

    fused.to_csv("data/fused/fusion_v2.csv", index=False)
    train.to_csv("data/fused/train_v2.csv", index=False)
    test.to_csv("data/fused/test_v2.csv", index=False)

    print("✅ V2 Fusion complete!")
    print("📦 Saved: fusion_v2.csv")


if __name__ == "__main__":
    run()