"""
prepare_kaggle_dataset.py — Packages local datasets and images into a Kaggle Dataset archive.

Creates `kaggle_export/astronova_dataset.zip` containing:
  - Raw / sample GOES X-ray data
  - Solar EUV observation images (I1.jpg - I13.jpg)
  - Processed parquet feature datasets
  - Model configs (configs/*.yaml)
  - Dataset manifests and metadata

Usage:
    python kaggle/prepare_kaggle_dataset.py
"""
import os
import shutil
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPORT_DIR = PROJECT_ROOT / "kaggle_export"
DATASET_ZIP = EXPORT_DIR / "astronova_dataset.zip"


def create_kaggle_dataset_bundle():
    print("==================================================")
    print("ASTRONOVA KAGGLE DATASET PACKAGER")
    print("==================================================")

    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    temp_staging = EXPORT_DIR / "astronova_dataset"
    if temp_staging.exists():
        shutil.rmtree(temp_staging)
    temp_staging.mkdir(parents=True, exist_ok=True)

    # 1. Copy sample solar images
    images_dst = temp_staging / "images"
    images_dst.mkdir(exist_ok=True)
    for img_path in PROJECT_ROOT.glob("I*.jpg"):
        shutil.copy2(img_path, images_dst / img_path.name)
        print(f"Copied image: {img_path.name}")

    if (PROJECT_ROOT / "BG IMAGE.jpg").exists():
        shutil.copy2(PROJECT_ROOT / "BG IMAGE.jpg", images_dst / "BG IMAGE.jpg")

    # 2. Copy configs
    configs_src = PROJECT_ROOT / "configs"
    if configs_src.exists():
        configs_dst = temp_staging / "configs"
        shutil.copytree(configs_src, configs_dst, dirs_exist_ok=True)
        print("Copied configs/ directory.")

    # 3. Copy datasets
    datasets_src = PROJECT_ROOT / "datasets"
    if datasets_src.exists():
        datasets_dst = temp_staging / "datasets"
        shutil.copytree(datasets_src, datasets_dst, dirs_exist_ok=True)
        print("Copied datasets/ directory.")

    # 4. Copy sample GOES CSV data if present
    sample_data = PROJECT_ROOT / "data" / "sample"
    if sample_data.exists():
        shutil.copytree(sample_data, temp_staging / "data" / "sample", dirs_exist_ok=True)
        print("Copied data/sample/ directory.")

    # 5. Copy dataset metadata
    metadata_src = PROJECT_ROOT / "kaggle" / "dataset-metadata.json"
    if metadata_src.exists():
        shutil.copy2(metadata_src, temp_staging / "dataset-metadata.json")

    # Create Zip Archive
    print(f"\nCompressing dataset into {DATASET_ZIP} ...")
    with zipfile.ZipFile(DATASET_ZIP, "w", zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(temp_staging):
            for file in files:
                file_path = Path(root) / file
                arcname = file_path.relative_to(temp_staging)
                zipf.write(file_path, arcname)

    # Clean up staging
    shutil.rmtree(temp_staging)
    size_mb = os.path.getsize(DATASET_ZIP) / (1024 * 1024)
    print(f"[SUCCESS] Kaggle Dataset Archive generated: {DATASET_ZIP} ({size_mb:.2f} MB)")
    print("\nTo upload to Kaggle via CLI:")
    print(f"  kaggle datasets create -p {EXPORT_DIR}")
    print("Or drag-and-drop the zip directly into https://www.kaggle.com/datasets/new")


if __name__ == "__main__":
    create_kaggle_dataset_bundle()
