# 🌌 AstroNova ML Training on Kaggle

Everything needed to train the **AstroNova ML & Deep Learning models** on **Kaggle GPU (T4 / P100)** is ready in the [`kaggle/`](file:///kaggle/) directory.

### 🚀 Quickstart (3 Easy Steps)
1. Open [kaggle.com/code](https://www.kaggle.com/code) -> Click **New Notebook** -> **File** -> **Upload Notebook**.
2. Select [`kaggle/astronova_solar_flare_ml_kaggle.ipynb`](file:///kaggle/astronova_solar_flare_ml_kaggle.ipynb).
3. Set Accelerator to **GPU T4 x2** (right sidebar) and click **Run All**.

### 📦 Key Files
- [`kaggle/astronova_solar_flare_ml_kaggle.ipynb`](file:///kaggle/astronova_solar_flare_ml_kaggle.ipynb): Master Kaggle Jupyter Notebook with complete training loops, metrics (TSS, HSS, Brier, ROC-AUC), GradCAM visualizations, and auto-export zip.
- [`kaggle/train_kaggle.py`](file:///kaggle/train_kaggle.py): Standalone Python training script.
- [`kaggle/prepare_kaggle_dataset.py`](file:///kaggle/prepare_kaggle_dataset.py): Bundles solar images and telemetry into a Kaggle Dataset.
- [`docs/KAGGLE_TRAINING_GUIDE.md`](file:///docs/KAGGLE_TRAINING_GUIDE.md): Full step-by-step documentation.
