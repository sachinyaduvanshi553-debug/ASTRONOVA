import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

SOLEXS_PATH = ROOT / "data" / "cleaned" / "solexs" / "solexs_ml_ready_v1.csv"


def load_solexs():
    print("Loading SOLEXS...")

    cols = [
        "date",
        "flux",
        "rolling_mean_60",
        "rolling_std_60",
        "burst_intensity",
        "peak_flux_60",
        "background_flux",
        "snr",
        "volatility"
    ]

    df = pd.read_csv(
        SOLEXS_PATH,
        usecols=cols,
        parse_dates=["date"],
        low_memory=True
    )

    df = df.rename(columns={"date": "timestamp"})
    df = df.sort_values("timestamp")

    return df


def main():
    df = load_solexs()

    print("\nSOLEXS loaded successfully!")
    print("Shape:", df.shape)
    print(df.head())


if __name__ == "__main__":
    main()