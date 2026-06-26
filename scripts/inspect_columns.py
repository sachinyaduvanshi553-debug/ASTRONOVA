import pandas as pd

print("=" * 80)
print("SOLEXS")
print("=" * 80)

solexs = pd.read_csv(
    "data/cleaned/solexs/solexs_ml_ready_v1.csv",
    nrows=5
)

print(solexs.columns.tolist())
print(solexs.head())


print("\n" + "=" * 80)
print("HEL1OS")
print("=" * 80)

helios = pd.read_parquet(
    "data/cleaned/helios/HEL1OS_FILTERED.parquet"
)

print(helios.columns.tolist())
print(helios.head())


print("\n" + "=" * 80)
print("GOES")
print("=" * 80)

goes = pd.read_csv(
    "data/cleaned/goes/goes_xrs_oct2024_jan2025 (1).csv",
    nrows=5
)

print(goes.columns.tolist())
print(goes.head())


print("\n" + "=" * 80)
print("NOAA")
print("=" * 80)

noaa = pd.read_csv(
    "data/cleaned/noaa_labels/noaa_flares_clean.csv",
    nrows=5
)

print(noaa.columns.tolist())
print(noaa.head())