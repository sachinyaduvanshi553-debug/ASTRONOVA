import pandas as pd
import numpy as np
from pathlib import Path
import pyarrow.dataset as ds

# -----------------------------
# PATHS
# -----------------------------
BASE = Path("data/cleaned")

GOES_PATH = BASE / "goes/goes_xrs_oct2024_jan2025.csv"
HELIOS_PATH = BASE / "helios/helios_filtered.parquet"
SOLEXS_PATH = BASE / "solexs/solexs_ml_ready.csv"
NOAA_PATH = BASE / "noaa_labels/noaa_flares.csv"

OUTPUT_PATH = Path("data/fused/fusion_dataset.csv")


# -----------------------------
# 1. LOAD DATA (FIXED MEMORY SAFE)
# -----------------------------
def load_data():
    print("📥 Loading data...")

    goes = pd.read_csv(GOES_PATH)

    # 🔥 MEMORY SAFE HELIOS LOADING
    dataset = ds.dataset(HELIOS_PATH, format="parquet")
    helios = dataset.to_table(
        columns=["timestamp", "count_rate", "stat_err", "snr"]
    ).to_pandas(split_blocks=True)

    # 🔥 reduce memory footprint (important)
    helios = helios.iloc[::10].reset_index(drop=True)

    solexs = pd.read_csv(SOLEXS_PATH)
    noaa = pd.read_csv(NOAA_PATH)

    return goes, helios, solexs, noaa


# -----------------------------
# 2. STANDARDIZE TIME
# -----------------------------
def standardize_time(df, col):
    df = df.copy(deep=False)

    df[col] = pd.to_datetime(df[col], errors="coerce", utc=True)

    # 🔥 FORCE UNIFORM TYPE (CRITICAL FIX)
    df[col] = df[col].astype("datetime64[ns, UTC]")

    df = df.loc[df[col].notna()].reset_index(drop=True)

    return df


# -----------------------------
# 3. PREPROCESS
# -----------------------------
def preprocess(goes, helios, solexs, noaa):

    print("⚙️ Preprocessing...")

    # ---- GOES ----
    time_col_goes = [c for c in goes.columns if "time" in c.lower()][0]
    goes = standardize_time(goes, time_col_goes)
    goes = goes.rename(columns={time_col_goes: "timestamp"}).sort_values("timestamp")

    # ---- HELIOS ----
    time_col_hel = [c for c in helios.columns if "time" in c.lower()][0]
    helios = standardize_time(helios, time_col_hel)
    helios = helios.rename(columns={time_col_hel: "timestamp"}).sort_values("timestamp")

    # ---- SOLEXS ----
    time_col_sol = [c for c in solexs.columns if "time" in c.lower()][0]
    solexs = standardize_time(solexs, time_col_sol)
    solexs = solexs.rename(columns={time_col_sol: "timestamp"}).sort_values("timestamp")

    # ---- NOAA ----
    time_col_noaa = [c for c in noaa.columns if "time" in c.lower()][0]
    noaa = standardize_time(noaa, time_col_noaa)
    noaa = noaa.rename(columns={time_col_noaa: "timestamp"}).sort_values("timestamp")

    return goes, helios, solexs, noaa


# -----------------------------
# 4. TIME MERGE
# -----------------------------
def time_merge(base, other, window="10min"):
    return pd.merge_asof(
        base.sort_values("timestamp"),
        other.sort_values("timestamp"),
        on="timestamp",
        direction="nearest",
        tolerance=pd.Timedelta(window)
    )


# -----------------------------
# 5. FUSION ENGINE
# -----------------------------
def build_fusion(goes, helios, solexs, noaa):

    print("🔗 Starting fusion process...")

    fused = goes.copy()

    print("➕ Merging HELIOS...")
    fused = time_merge(fused, helios, "10min")

    print("➕ Merging SOLEXS...")
    fused = time_merge(fused, solexs, "10min")

    print("➕ Merging NOAA labels...")
    fused = time_merge(fused, noaa, "30min")

    # cleanup
    fused = fused.sort_values("timestamp")
    fused = fused.drop_duplicates(subset=["timestamp"])

    # memory safe fill
    fused = fused.ffill().bfill()

    return fused


# -----------------------------
# 6. TRAIN / TEST SPLIT
# -----------------------------
def split_data(df):
    split_idx = int(len(df) * 0.8)
    return df.iloc[:split_idx], df.iloc[split_idx:]


# -----------------------------
# 7. RUN PIPELINE
# -----------------------------
def run():

    goes, helios, solexs, noaa = load_data()
    goes, helios, solexs, noaa = preprocess(goes, helios, solexs, noaa)

    fused = build_fusion(goes, helios, solexs, noaa)

    train, test = split_data(fused)

    Path("data/fused").mkdir(parents=True, exist_ok=True)

    fused.to_csv(OUTPUT_PATH, index=False)
    train.to_csv("data/fused/train.csv", index=False)
    test.to_csv("data/fused/test.csv", index=False)

    print("✅ Fusion complete!")
    print("📦 Saved: data/fused/")


# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":
    run()