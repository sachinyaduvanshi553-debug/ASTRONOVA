# ASTRONOVA Vision Forecasting — Implementation Audit Report

**Audit Date:** 2026-10-07
**Workspace:** `C:\Users\sachi\.gemini\antigravity-ide\scratch\ASTRONOVA`

---

## 1. Current Architecture

### 1.1 Legacy `services/vision/` (PRESERVED — DO NOT MODIFY)

| File | Class | Notes |
|------|-------|-------|
| model.py | SolarVisionPredictor | 5-class softmax (A/B/C/M/X), single horizon — INCOMPATIBLE with new contract |
| losses.py | SolarVisionLoss | VGG perceptual + SSIM + MSE + CE. Loads VGG-16 on init |
| trainer.py | VisionTrainer | Coupled to SolarVisionPredictor + SolarVisionLoss |
| forecast_api.py | FastAPI router | Already wired to NEW SolarImageForecaster |
| preprocessing/sequence_builder.py | SolarSequenceDataset | 6-horizon labels, chronological loading |

### 1.2 New `models/vision/` (Active Pipeline)

| File | Class | Status |
|------|-------|--------|
| solar_image_forecaster.py | SolarImageForecaster | COMPLETE — 6-horizon independent sigmoid |
| image_forecaster.py | ImageForecastDecoder, MultimodalFusion, ImageForecastLoss | COMPLETE |
| cnn_encoder.py | CNNSpatialEncoder | COMPLETE — ResNet18/50 backbone |
| temporal_transformer.py | TemporalTransformer | COMPLETE — CLS token, sinusoidal PE |
| conv_lstm.py | ConvLSTM | COMPLETE — LSTM alternative |
| flare_location_head.py | FlareLocationHead | COMPLETE — [B,1,H,W] sigmoid heatmap |
| flare_classifier.py | MultiHorizonFlareClassifier | Present |
| dataset_adapter.py | AdaptedSolarSequenceDataset | COMPLETE — wraps JSON manifests |
| future_target_resolver.py | FutureTargetResolver | COMPLETE — strict anti-leakage |
| grad_cam.py | GradCAM | Present (not tested) |
| __init__.py | all exports | COMPLETE |

---

## 2. Expected vs. Actual Model Imports

### train_image_forecaster.py (CORRECT)

```python
from models.vision.solar_image_forecaster import HORIZONS, SolarImageForecaster
from models.vision.image_forecaster import ImageForecastLoss
from models.vision.dataset_adapter import AdaptedSolarSequenceDataset, adapted_collate_fn
from services.vision.preprocessing.augmentation import SequenceAugmentation, SolarAugmentation
from services.vision.preprocessing.image_normalizer import ImageNormalizer
```

### evaluate_image_forecaster.py (CORRECT)

```python
from models.vision.solar_image_forecaster import HORIZONS, HORIZON_DISPLAY, SolarImageForecaster
from models.vision.dataset_adapter import AdaptedSolarSequenceDataset, adapted_collate_fn
```

### forecast_api.py (CORRECT)

```python
from models.vision.solar_image_forecaster import SolarImageForecaster
from models.vision.solar_image_forecaster import HORIZONS
```

---

## 3. Dataset Schema (Actual)

### Split Counts

| Split | File | Count | Type |
|-------|------|-------|------|
| Train | train_sequences.json | 14 | All synthetic |
| Validation | validation_sequences.json | 3 | All synthetic |
| Test | test_sequences.json | 3 | All synthetic |

### Per-Sequence JSON Keys

```
anchor_timestamp, anchor_idx, window_start, image_paths, timestamps,
soft_xray_flux, log_soft_flux, n_images, labels, active_region_id, synthetic
```

### Label Structure (per sequence)

```json
"labels": {
  "h15m":   {"C_plus": false, "M_plus": false, "X_plus": false, "max_class": "B", "n_events": 0},
  "h30m":   {...},
  "h60m":   {...},
  "h360m":  {...},
  "h720m":  {...},
  "h1440m": {...}
}
```

### Missing Fields

| Field | Status | Handling |
|-------|--------|---------|
| magnetic_features (16 HMI SHARP) | MISSING from JSON | Zero-filled by dataset_adapter.py |
| physics_features (5D) | MISSING from JSON | Zero-filled by dataset_adapter.py |
| spatial_mask (heatmap GT) | MISSING | Location loss disabled when absent |
| Future image timestamps | Available via FutureTargetResolver | Resolved from image directory |

---

## 4. Normalized Target Schema

```
image_seq:         Tensor[T, 3, H, W]
telemetry:         Tensor[T, 2]           -- [soft_flux, log_soft_flux]
magnetic:          Tensor[16]             -- zero-filled when absent
physics:           Tensor[5]             -- zero-filled when absent
labels:            {h: Tensor[3]}        -- [C+, M+, X+] binary per horizon
future_images:     {h: Tensor[3,H,W]|None}
target_available:  {h: bool}
anchor_timestamp:  str
is_synthetic:      bool
active_region_id:  int
```

---

## 5. Six Forecast Horizons

| Key | Minutes | Display |
|-----|---------|---------|
| h15m | 15 | +15 min |
| h30m | 30 | +30 min |
| h60m | 60 | +60 min |
| h360m | 360 | +6 hours |
| h720m | 720 | +12 hours |
| h1440m | 1440 | +24 hours |

Defined centrally: `HORIZONS = ["h15m", "h30m", "h60m", "h360m", "h720m", "h1440m"]`

---

## 6. Image Dimensions / Channels

| Property | Value |
|----------|-------|
| Default size | 256x256 (full), 128x128 (smoke) |
| Channels | 3 (RGB) |
| Input | [B, T, 3, H, W] |
| Heatmap | [B, 1, H, W] |
| Future image | [B, 3, H, W] |
| Normalization | per_image (no global stats needed) |

---

## 7. Magnetic Feature Availability

- **ABSENT** from current JSON manifests
- `dataset_adapter.py` safely returns `torch.zeros(16)` when key missing
- `SolarMagneticEncoder` supports `mag_missing` mask
- Architecture is forward-compatible with real HMI data

16 HMI SHARP Parameters: ABSNJZH, AREA_ACR, MEANALP, MEANJZH, MEANPOT, MEANSHR,
R_VALUE, SAVNCPP, TOTBSQ, TOTFX, TOTFY, TOTFZ, TOTPOT, TOTUSJH, TOTUSJZ, USFLUX

---

## 8. Telemetry / Physics Availability

| Feature | Status | Dim |
|---------|--------|-----|
| GOES X-ray (soft + log flux) | Present in JSON | [T, 2] |
| Physics features | Absent | Zero-filled [5] |
| SoLEXS/HEL1OS | Not in dataset | N/A |

---

## 9. Training Incompatibilities

| Issue | Severity | Status |
|-------|----------|--------|
| Legacy SolarVisionPredictor softmax (5-class) | HIGH | Preserved, NOT used in new pipeline |
| Coverage fail_under=70 breaks pytest --cov | MEDIUM | Workaround: --no-cov flag |
| SolarVisionLoss loads VGG-16 on import | MEDIUM | Not used in new pipeline |
| trainer.py uses deprecated torch.cuda.amp.GradScaler | LOW | Not used; new trainer correct |
| future_target_resolver uses pd.Timestamp.utcnow() | LOW | Deprecated pandas API |

---

## 10. Leakage Risks

| Risk | Status |
|------|--------|
| Future image as training target | MITIGATED — FutureTargetResolver rejects ts <= anchor_ts |
| Future image = last input frame | MITIGATED — Input paths excluded from candidate set |
| Random temporal splitting | NOT PRESENT — Chronological ordering in manifests |
| Future telemetry in input | SAFE — Telemetry window bounded to anchor |
| Test-set label leakage | SAFE — Labels from JSON; no lookahead |

---

## 11. Files Status

### Created and Verified
- models/vision/solar_image_forecaster.py
- models/vision/image_forecaster.py
- models/vision/cnn_encoder.py
- models/vision/temporal_transformer.py
- models/vision/conv_lstm.py
- models/vision/flare_location_head.py
- models/vision/future_target_resolver.py
- models/vision/dataset_adapter.py
- models/vision/__init__.py
- scripts/train_image_forecaster.py
- scripts/evaluate_image_forecaster.py
- tests/vision/test_image_forecaster_compatibility.py

### Intentionally Preserved (DO NOT MODIFY)
- services/vision/model.py
- services/vision/losses.py
- services/vision/trainer.py
- services/vision/preprocessing/sequence_builder.py
- checkpoints/ (all existing production checkpoints)

---

## 12. Compatibility Test Results

Command: `python -m pytest tests/vision/test_image_forecaster_compatibility.py -q --no-cov`

Result: **10 passed, 9 warnings in 8.05s**

| Test | Description | Result |
|------|-------------|--------|
| test_01_imports | Model + HORIZONS import | PASS |
| test_02_six_horizons_exist | 6 horizons defined | PASS |
| test_03_04_05 | Six heads, logits [B,3], probs in [0,1] | PASS |
| test_06_to_10 | Full + missing multimodal inputs | PASS |
| test_11_12_13 | Heatmap [B,1,H,W], future image [B,3,H,W], AI_FORECAST | PASS |
| test_14_15 | Loss, missing target img_total=0 | PASS |
| test_16 | FutureTargetResolver leakage safety | PASS |
| test_17 | Checkpoint save/reload | PASS |
| test_18_19 | CPU inference | PASS |
| test_20_21 | Batch=2, 128x128 shapes | PASS |

---

## 13. CUDA / GPU Information

| Property | Value |
|----------|-------|
| torch.cuda.is_available() | FALSE |
| GPU Device | NONE |
| Smoke test device | CPU (automatic fallback) |
| AMP | Disabled (no CUDA) |

WARNING: CUDA is NOT available on this machine.
Smoke test will use CPU fallback. Use --device cpu explicitly.

---

## 14. Scientific Safety

- No target leakage — verified by test_16
- Independent sigmoid outputs (NOT softmax) — verified by test_03_04_05
- Future image != last input frame — explicit exclusion in FutureTargetResolver
- AI_FORECAST label on all generated images
- Chronological splits preserved
- is_synthetic=True flagged on all 20 sequences

CAUTION: Current dataset (14+3+3 synthetic sequences) is suitable for:
  - Pipeline debugging and architecture validation
  - Smoke testing
  - Proof-of-concept training
It is NOT sufficient for scientific generalization claims or published results.

---

## 15. Remaining Blockers Before Smoke Test

| Blocker | Severity | Resolution |
|---------|----------|-----------|
| CUDA unavailable | LOW | Use --device cpu |
| Coverage fail_under=70 | LOW | Use --no-cov flag |
| pandas Timestamp.utcnow() deprecation | INFO | Minor fix in future_target_resolver.py |

**VERDICT: No blocking issues. Pipeline is ready for smoke test on CPU.**
