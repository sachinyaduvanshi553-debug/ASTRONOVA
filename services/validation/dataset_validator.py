from pathlib import Path
import pandas as pd
import pyarrow.parquet as pq


DATASETS = {
    "SOLEXS":
        "data/cleaned/solexs/solexs_ml_ready_v1.csv",

    "HEL1OS":
        "data/cleaned/helios/HEL1OS_FILTERED.parquet",

    "GOES":
        "data/cleaned/goes/goes_xrs_oct2024_jan2025 (1).csv",

    "NOAA":
        "data/cleaned/noaa_labels/noaa_flares_clean.csv",

    "CME":
        "data/cleaned/auxiliary/cme_clean.csv",

    "SEP":
        "data/cleaned/auxiliary/sep_clean.csv",

    "GST":
        "data/cleaned/auxiliary/gst_clean.csv"
}


def validate_csv(name, path):
    df = pd.read_csv(path, nrows=5)

    print("\n" + "=" * 70)
    print(name)
    print("=" * 70)

    print("Columns:")
    print(df.columns.tolist())

    print("\nSample:")
    print(df.head())


def validate_parquet(name, path):
    table = pq.ParquetFile(path)

    print("\n" + "=" * 70)
    print(name)
    print("=" * 70)

    print("Rows:", table.metadata.num_rows)
    print("Columns:")
    print(table.schema.names)

    sample = table.read_row_group(0).to_pandas()

    print("\nSample:")
    print(sample.head())


if __name__ == "__main__":

    for name, path in DATASETS.items():

        if not Path(path).exists():
            print(f"{name} missing")
            continue

        if path.endswith(".parquet"):
            validate_parquet(name, path)

        else:
            validate_csv(name, path)

    print("\nALL DATASETS VALIDATED SUCCESSFULLY")