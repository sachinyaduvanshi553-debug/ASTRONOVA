# 🚀 AstroNova ML Training on Kaggle Guide

This guide provides step-by-step instructions for training the **AstroNova Solar Flare ML Pipeline** (Time-Series, Solar Vision, Multimodal Fusion, and Explainable AI) on **Kaggle GPU environments** (NVIDIA T4 / P100).

---

## 📋 What is Included for Kaggle

| Component | File Path | Description |
| :--- | :--- | :--- |
| **Interactive Master Notebook** | [`kaggle/astronova_solar_flare_ml_kaggle.ipynb`](file:///kaggle/astronova_solar_flare_ml_kaggle.ipynb) | Complete end-to-end training & visual notebook ready for 1-click import into Kaggle. |
| **Standalone Script Engine** | [`kaggle/train_kaggle.py`](file:///kaggle/train_kaggle.py) | Standalone script for Kaggle script runs or batch training CLI. |
| **Dataset Packager** | [`kaggle/prepare_kaggle_dataset.py`](file:///kaggle/prepare_kaggle_dataset.py) | Utility to bundle raw images (`I1.jpg`-`I13.jpg`), GOES data, and configs into a Kaggle Dataset. |
| **Kernel Metadata** | [`kaggle/kernel-metadata.json`](file:///kaggle/kernel-metadata.json) | Metadata configuration for Kaggle CLI integration (`kaggle kernels push`). |
| **Kaggle Requirements** | [`kaggle/requirements_kaggle.txt`](file:///kaggle/requirements_kaggle.txt) | Dependency specification optimized for Kaggle kernels without conflicts. |

---

## ⚡ Method 1: Interactive Training via Kaggle Web UI (Recommended)

### Step 1: Create a New Notebook on Kaggle
1. Go to [kaggle.com/code](https://www.kaggle.com/code) and click **"New Notebook"**.
2. In the top navigation menu, click **File** -> **Upload Notebook**.
3. Select [`kaggle/astronova_solar_flare_ml_kaggle.ipynb`](file:///kaggle/astronova_solar_flare_ml_kaggle.ipynb) from your local project repository.

### Step 2: Enable GPU Hardware Acceleration
1. In the right-hand sidebar under **Notebook Options** -> **Accelerator**, select **GPU T4 x2** or **GPU P100**.
2. Ensure **Internet** is toggled **ON** (for package installations).

### Step 3: Run All Cells
1. Click **Run All** (or `Shift + Enter` cell by cell).
2. The notebook will automatically:
   - Verify GPU acceleration and mixed-precision (AMP).
   - Train time-series models (**BiLSTM**, **Solar Transformer**) across multi-horizons ($+15\text{m}, +30\text{m}, +1\text{h}, +6\text{h}$).
   - Train the computer vision multimodal model (**SolarImageForecaster** with CNN spatial encoder + ConvLSTM / Temporal Transformer + Focal Loss + Spatial Heatmap Decoder).
   - Compute space weather operational metrics (**TSS**, **HSS**, **Brier Score**, **ROC-AUC**).
   - Render GradCAM spatial risk heatmaps and multi-horizon probability curves.
   - Package all trained weights (`.pt`, `.pkl`) and metrics into `/kaggle/working/astronova_checkpoints.zip`.

### Step 4: Download Checkpoints
1. Go to the **Output** tab in the right sidebar.
2. Click the three dots next to `astronova_checkpoints.zip` and select **Download**.

---

## 💻 Method 2: Automated Kaggle CLI Push

If you have the Kaggle CLI installed (`pip install kaggle`) and your `kaggle.json` API token configured:

```bash
# 1. Package the dataset (optional if using built-in generator)
python kaggle/prepare_kaggle_dataset.py

# 2. Push notebook to Kaggle and execute on GPU
kaggle kernels push -p kaggle/

# 3. Monitor execution status
kaggle kernels status astronova-user/astronova-solar-flare-ml

# 4. Download output checkpoints when finished
kaggle kernels output astronova-user/astronova-solar-flare-ml -p ./checkpoints_kaggle/
```

---

## 📊 Models Trained on Kaggle

### 1. Solar Flare Time-Series Forecasters
- **BiLSTM Forecaster**: 2-layer Bidirectional LSTM with MC-Dropout epistemic uncertainty estimation and multi-horizon dual heads (classification + peak flux regression).
- **Solar Transformer**: Multi-head self-attention encoder with sinusoidal temporal embeddings.
- **XGBoost & LightGBM**: Gradient-boosted decision trees over 15 physical & time-domain features.

### 2. Solar Vision & Multimodal Deep Learning
- **SolarImageForecaster**:
  - **Spatial CNN Encoder**: Multi-scale feature extraction on SDO/AIA 193Å, 211Å, and 171Å EUV images.
  - **Temporal Sequence Modeling**: ConvLSTM & Bidirectional LSTM tracking active region emergence and flux emergence.
  - **Multi-Horizon Classification Head**: Simultaneous predictions at $+15\text{m}, +30\text{m}, +60\text{m}, +6\text{h}, +12\text{h}, +24\text{h}$.
  - **Focal Loss**: Combats extreme class rarity ($C+$, $M+$, $X+$).
  - **Spatial Flare Location Decoder**: Heatmap localization using BCE + Dice loss.

### 3. Space Weather Verification Metrics Computed
- **True Skill Statistic (TSS)**: $TSS = TPR - FPR = \frac{TP}{TP+FN} - \frac{FP}{FP+TN}$
- **Heidke Skill Score (HSS)**: Operational skill score relative to random chance.
- **Brier Score (BS)**: Probabilistic reliability calibration score.
- **False Alarm Ratio (FAR)**: $\frac{FP}{TP+FP}$
- **ROC-AUC & PR-AUC**: Area under ROC and Precision-Recall curves.

---

## 🔄 Deploying Trained Kaggle Checkpoints to AstroNova Local Workspace

Once you download `astronova_checkpoints.zip` from Kaggle:
1. Extract the contents into your local repository `models/` or `checkpoints/` folder:
   ```bash
   unzip astronova_checkpoints.zip -d models/
   ```
2. Your local AstroNova FastAPI services and Frontend Dashboard will immediately pick up the trained model weights.
