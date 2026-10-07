# 🌌 ASTRONOVA Vision Architecture Audit Report

**Date:** 2026-10-03  
**Status:** Audit Complete  
**Scope:** Multimodal Solar Flare Image Forecasting Pipeline (`services/vision`, `models/vision`, `scripts/`, `configs/`, `datasets/`)

---

## 1. Current Dataset Schema

The existing image sequence dataset is structured under `datasets/image_sequences/`:
- **Training Manifest:** `datasets/image_sequences/train_sequences.json` (14 sequences) + `train_metadata.csv`
- **Validation Manifest:** `datasets/image_sequences/validation_sequences.json` (3 sequences) + `validation_metadata.csv`
- **Test Manifest:** `datasets/image_sequences/test_sequences.json` (3 sequences) + `test_metadata.csv`

Each entry in the dataset contains:
- `anchor_timestamp`: ISO 8601 string representing the end of the historical observation window $T_0$ (e.g. `"2024-10-02T00:00:00+00:00"`)
- `anchor_idx`: Integer index in the global observation stream
- `window_start`: ISO 8601 string representing the start of the historical window $T_0 - W$
- `image_paths`: List of 25 image paths (e.g., `datasets/raw/sdo_images/synthetic/synthetic_20241001_000000.png` through `synthetic_20241002_000000.png`)
- `timestamps`: List of 25 timestamps corresponding to the image sequence at 1-hour cadence
- `soft_xray_flux`: List of 25 GOES soft X-ray flux values ($1-8 \text{ \AA}$)
- `log_soft_flux`: List of 25 $\log_{10}(\text{flux})$ values
- `n_images`: 25
- `labels`: Dictionary of 6 forecasting horizons: `"h15m"`, `"h30m"`, `"h60m"`, `"h360m"`, `"h720m"`, `"h1440m"`
  - Each horizon dictionary contains: `{"C_plus": bool, "M_plus": bool, "X_plus": bool, "max_class": str, "n_events": int}`
- `active_region_id`: Integer NOAA active region ID (e.g. 13780)
- `synthetic`: Boolean flag indicating synthetic smoke test data

---

## 2. Tensor Shapes & Feature Dimensions

| Component | Raw Shape / Format | Model Input Tensor Shape | Notes |
| :--- | :--- | :--- | :--- |
| **Image Sequence** | 25 file paths (PNG/JPG) | `[B, T, 3, H, W]` (e.g. `[2, 4, 3, 128, 128]` for smoke test) | Channels = 3 (RGB/EUV composites), Resized & normalized |
| **Telemetry** | `soft_xray_flux`, `log_soft_flux` | `[B, T, 2]` or `[B, telemetry_dim]` | Soft flux + $\log_{10}(\text{flux})$ per timestep |
| **Magnetic Features** | HMI SHARP 16-parameter schema | `[B, T, 16]` or `[B, 16]` | Production schema: `[ABSNJZH, AREA_ACR, MEANALP, MEANJZH, MEANPOT, MEANSHR, R_VALUE, SAVNCPP, TOTBSQ, TOTFX, TOTFY, TOTFZ, TOTPOT, TOTUSJH, TOTUSJZ, USFLUX]`. Zero-masked when absent. |
| **Physics Features** | NOAA/CME/SEP catalog parameters | `[B, physics_dim]` (dim=5 in older service) | Explicitly masked with zero tensor when unavailable. |
| **Labels** | 6 horizons $\times$ 3 binary flags | `dict[horizon -> [B, 3]]` | Sigmoids for independent threshold classification ($C+$, $M+$, $X+$). |

---

## 3. Six Standard Forecasting Horizons

The dataset strictly complies with the 6 standard operational horizons:
1. `h15m` = $+15$ minutes
2. `h30m` = $+30$ minutes
3. `h60m` = $+60$ minutes ($+1$ hour) — **Primary M+ benchmark horizon**
4. `h360m` = $+6$ hours
5. `h720m` = $+12$ hours
6. `h1440m` = $+24$ hours

---

## 4. Target Image Availability & Leakage Risks

### Current Proxy Target Problem:
In earlier scripts, `target_image = image_seq[:, -1]` was used as a proxy target.
**Leakage Risk:** Using the last input frame as the target frame is a trivial self-reconstruction task that causes target leakage and does not test future forecasting.

### Resolution:
We introduce `models/vision/future_target_resolver.py` which:
1. Reads the anchor timestamp $T_0$ for the sequence.
2. Calculates target timestamps: $T_0 + 15\text{m}, T_0 + 30\text{m}, T_0 + 60\text{m}, T_0 + 360\text{m}, T_0 + 720\text{m}, T_0 + 1440\text{m}$.
3. Searches available observed frames within a strict tolerance window.
4. If a genuine future frame exists: assigns it as the target for that horizon and sets `target_available[horizon] = True`.
5. If no future frame exists (e.g. sequence at the end of the observation catalog): sets `future_images[horizon] = None` and `target_available[horizon] = False`.
6. Explicitly rejects timestamps $\le T_0$ from being future targets.
7. Only computes image reconstruction loss against verified future targets where `target_available[horizon] == True`.

---

## 5. Architecture Mismatches & Audit Findings

1. **`SolarImagePreprocessor` import location:** `services/vision/preprocessing/__init__.py` did not export `SolarImagePreprocessor` (which lived in `services/vision/preprocessing.py`), breaking legacy `services/vision/dataset.py` and `tests/vision/test_vision.py`.
2. **Missing unified future target handling:** `train_image_forecaster.py` and `evaluate_image_forecaster.py` needed complete modernization to integrate `FutureTargetResolver`, per-horizon future image decoders, and multi-modal feature masks (handling optional magnetic and telemetry vectors).
3. **Model output contract:** Ensure `SolarImageForecaster` returns `class_logits`, `class_probs`, `c_plus_probability`, `m_plus_probability`, `x_plus_probability`, `location_logits`, `location_heatmap`, `predicted_future_image`, `future_images` (per horizon), and metadata with `image_type: "AI_FORECAST"`.
4. **VGG / Perceptual safety:** Default `pretrained_encoder=False` and disable online weight downloading during smoke test.

---

## 6. Exact Implementation Plan

1. **Fix backward compatibility in `services/vision/preprocessing/__init__.py`** to export `SolarImagePreprocessor` and `synchronize_data`.
2. **Implement `models/vision/future_target_resolver.py`** for timestamp-based genuine future image matching.
3. **Implement/Enhance `models/vision/solar_image_forecaster.py` & `models/vision/image_forecaster.py`**:
   - CNN spatial encoder (ResNet18 / custom conv, spatial map + pooled embeddings).
   - Temporal Transformer / ConvLSTM encoder.
   - Telemetry encoder (configurable input dim).
   - Magnetic encoder (16 SHARP features with missing masks).
   - Physics encoder (configurable with zero masking).
   - Multimodal fusion layer.
   - Six independent 3-output sigmoid classification heads ($C+, M+, X+$).
   - Spatial flare-location head (predicting $\mathbb{R}^{B \times 1 \times H' \times W'}$ logits and sigmoid heatmaps).
   - Horizon-conditioned future image decoder head (predicting $\mathbb{R}^{B \times 3 \times H \times W}$ in $[0, 1]$).
   - Composite loss `ImageForecastLoss` (BCE/Focal + SSIM/L1 + BCE/Dice for spatial when target present).
4. **Create `models/vision/dataset_adapter.py`** to seamlessly adapt sequence JSONs and resolve future targets.
5. **Rewrite `scripts/train_image_forecaster.py`** with complete CLI options, AMP support, smoke-test overrides, metric tracking, and checkpoint saving (`best_roc_auc.pt`, `best_val_loss.pt`, `smoke_test.pt`).
6. **Rewrite `scripts/evaluate_image_forecaster.py`** to compute ROC-AUC, PR-AUC, TSS, HSS, Brier, ECE, MAE, RMSE, SSIM, PSNR on chronological splits.
7. **Add comprehensive compatibility test suite** in `tests/vision/test_image_forecaster_compatibility.py`.
8. **Execute pytest & 1-epoch smoke test** to verify zero errors.
9. **Generate smoke test report** at `reports/vision_smoke_test_report.md`.
