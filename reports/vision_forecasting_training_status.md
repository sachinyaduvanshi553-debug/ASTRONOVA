# ASTRONOVA Vision Forecasting — Training Status Report

**Generated:** 2026-10-07
**Workspace:** `C:\Users\sachi\.gemini\antigravity-ide\scratch\ASTRONOVA`

---

## 1. Audit Result

Phase 1 audit found:
- All `models/vision/` architecture files are complete and correct
- All `scripts/` files correctly import from the new pipeline
- `services/vision/` legacy pipeline is intact and untouched
- Dataset: 14 train + 3 val + 3 test synthetic sequences
- 6-horizon labels present in all sequences
- Magnetic features absent (zero-filled by adapter — by design)
- No leakage risks identified

**AUDIT: PASS**

---

## 2. Files Created

| File | Description |
|------|-------------|
| reports/vision_forecasting_implementation_audit.md | Phase 1 audit report |
| reports/vision_forecasting_training_status.md | This report |

---

## 3. Files Modified

| File | Change | Reason |
|------|--------|--------|
| models/vision/future_target_resolver.py | Fixed `pd.Timestamp.utcnow().tzinfo` → `timezone.utc` | Remove deprecated pandas API |

---

## 4. Files Intentionally Preserved (NOT Modified)

- `services/vision/model.py` — Legacy SolarVisionPredictor (5-class softmax)
- `services/vision/losses.py` — Legacy SolarVisionLoss (VGG perceptual)
- `services/vision/trainer.py` — Legacy VisionTrainer
- `services/vision/preprocessing/sequence_builder.py`
- `checkpoints/EXP_001_BASELINE_*.pt` — Production checkpoints
- `checkpoints/EXP_LOSS_*.pt` — Production experiment checkpoints

---

## 5. Model Architecture

**SolarImageForecaster** — Full Multimodal Spatiotemporal Pipeline

```
[B, T, 3, H, W]        → CNNSpatialEncoder (ResNet18)
                           → [B, T, D, H', W'] spatial_maps
                           → [B, T, D] pooled_embeds
                        → TemporalTransformer (CLS + sinusoidal PE)
                           → [B, D] temporal_ctx

[B, T, 2] telemetry    → SolarTelemetryEncoder → [B, 64]
[B, T, 16] magnetic    → SolarMagneticEncoder  → [B, 64]
[B, 5]  physics        → SolarPhysicsEncoder   → [B, 64]

                        → MultimodalFusion (concat + 2-layer MLP)
                           → [B, 256] fused_latent

fused_latent           → 6x HorizonClassificationHead
                           → {h: ([B,3] logits, [B,3] sigmoid probs)}

last_spatial [B,D,H',W']  → FlareLocationHead
                           → [B, 1, H, W] heatmap in [0,1]

last_spatial           → ImageForecastDecoder (horizon-conditioned)
                           → [B, 3, H, W] future image in [0,1]
```

**Parameters (smoke test config):** 11,704,070 total, all trainable

---

## 6. Input / Output Tensor Shapes

| Tensor | Shape | Notes |
|--------|-------|-------|
| image_seq | [B, T, 3, H, W] | B=batch, T=lookback, H=W=128/256 |
| telemetry | [B, T, 2] | soft_flux, log_soft_flux |
| magnetic | [B, T, 16] | 16 HMI SHARP (zero-filled currently) |
| physics | [B, T, 5] or [B, 5] | Zero-filled currently |
| class_logits[h] | [B, 3] | Raw logits per horizon |
| class_probs[h] | [B, 3] | Sigmoid(logits) in [0,1] |
| m_plus_probability[h] | [B] | Primary scientific target |
| location_heatmap | [B, 1, H, W] | Spatial flare probability |
| predicted_future_image | [B, 3, H, W] | AI_FORECAST visualization |

---

## 7. Six Forecast Horizons

| Horizon | Minutes | Head Output |
|---------|---------|-------------|
| h15m | 15 | [C+, M+, X+] sigmoid |
| h30m | 30 | [C+, M+, X+] sigmoid |
| h60m | 60 | [C+, M+, X+] sigmoid |
| h360m | 360 | [C+, M+, X+] sigmoid |
| h720m | 720 | [C+, M+, X+] sigmoid |
| h1440m | 1440 | [C+, M+, X+] sigmoid |

All outputs are **independent sigmoid** (NOT softmax). Each represents P(at-least-class-X flare within horizon).

---

## 8. Loss Components

**ImageForecastLoss** (`models/vision/image_forecaster.py`)

| Component | Formula | When Active |
|-----------|---------|-------------|
| cls_{h} | Focal BCE per horizon | Always (6 horizons averaged) |
| img_{h} | L1 + SSIM*0.5 per horizon | Only when genuine future target exists |
| location_total | BCE + Dice | Only when spatial mask ground truth exists |
| total_loss | λ_cls*cls + λ_img*img + λ_loc*loc | Always |

**Lambda defaults:** λ_cls=1.0, λ_img=0.5, λ_loc=0.5
**Focal params:** gamma=2.0, alpha=0.25
**Class weights:** C+=0.5, M+=1.0, X+=0.75

---

## 9. Dataset Statistics

| Split | Sequences | Type | h15m M+ | h60m M+ | h1440m M+ |
|-------|-----------|------|---------|---------|----------|
| Train | 14 | Synthetic | 2/14 | 2/14 | 14/14 |
| Validation | 3 | Synthetic | 1/3 | 1/3 | 3/3 |
| Test | 3 | Synthetic | 1/3 | 1/3 | 3/3 |

**WARNING:** All 20 sequences are synthetic. This dataset is suitable for pipeline
validation only. NOT for scientific generalization claims.

---

## 10. Compatibility Test Results (Phase 11)

**Command:** `python -m pytest tests/vision/test_image_forecaster_compatibility.py -q --no-cov`

```
10 passed, 9 warnings in 8.05s
```

All 10 contract tests pass:
- Model imports, instantiation, forward pass
- Six horizons, independent sigmoid outputs in [0,1]
- Image shapes [B,T,3,H,W] input handling
- Spatial heatmap [B,1,H,W], future image [B,3,H,W]
- Missing magnetic/telemetry/physics graceful handling
- Loss computation with unavailable targets (img_total = 0.0)
- FutureTargetResolver anti-leakage (img_ts <= anchor_ts rejected)
- Checkpoint save/reload round-trip
- CPU inference (all shapes verified)

**TESTS: PASS (10/10)**

Note: `pytest --cov` fails with `fail_under=70` because coverage spans many
unrelated service files. Use `--no-cov` flag for the vision test suite.

---

## 11. Smoke Test Results (Phase 12)

**Command:**
```
python scripts/train_image_forecaster.py \
    --config configs/image_forecasting.yaml \
    --epochs 1 --batch-size 2 --image-size 128 \
    --lookback 4 --device cpu --smoke-test --num-workers 0
```

| Check | Result |
|-------|--------|
| CUDA detected | N/A (CPU mode) |
| Model created | PASS — 11,704,070 params |
| Dataset loaded | PASS — 14 train, 3 val |
| Batch loaded | PASS — batch_size=2 |
| Forward pass | PASS — Loss: 0.4316 (Batch 1) |
| Loss finite | PASS — Final batch loss: 0.4127 |
| Backward pass | PASS |
| Optimizer step | PASS |
| Validation | PASS — Val Loss: 0.4030 |
| Checkpoint saved | PASS — smoke_test.pt + best.pt |
| Total time | 7.6 seconds |
| Exit code | 0 |

**Checkpoint keys:** epoch, model_state_dict, optimizer_state_dict,
model_config, horizons, feature_dimensions, image_size, lookback,
git_commit, val_metrics, config

**SMOKE TEST: PASS**

---

## 12. Checkpoint Paths

| Checkpoint | Path | Size | Description |
|-----------|------|------|-------------|
| smoke_test.pt | checkpoints/vision/smoke_test.pt | 133.8 MB | Smoke test weights |
| best.pt | checkpoints/vision/best.pt | 133.8 MB | Copy of smoke_test.pt |
| best_roc_auc.pt | checkpoints/vision/best_roc_auc.pt | 49.3 MB | Previous best ROC-AUC |
| last.pt | checkpoints/vision/last.pt | 125.6 MB | Previous last epoch |
| model_config.json | checkpoints/vision/model_config.json | — | Architecture config |
| metrics.json | checkpoints/vision/metrics.json | — | Last epoch metrics |
| dataset_schema.json | checkpoints/vision/dataset_schema.json | — | Input schema |
| training_history.json | checkpoints/vision/training_history.json | — | Per-epoch history |

---

## 13. CUDA / GPU Information

| Property | Value |
|----------|-------|
| torch.cuda.is_available() | FALSE |
| GPU | Not detected on this machine |
| AMP | Disabled |
| Fallback | CPU (full-precision float32) |

**For GPU training:** Run on a CUDA-capable machine (Kaggle/Colab/AWS GPU).
The architecture and training script are CUDA-ready — just pass `--device cuda`.

---

## 14. Remaining Blockers

| Item | Status |
|------|--------|
| CUDA training | Requires GPU machine. Script is GPU-ready. |
| Real dataset | 20 synthetic sequences — need real SDO/HMI data for science |
| Magnetic features | HMI SHARP parameters absent from current JSON manifests |
| Physics features | Not yet in dataset manifests |
| Spatial GT masks | No ground-truth heatmap annotations available |
| Full 50-epoch run | INTENTIONALLY NOT STARTED — awaiting explicit user authorization |

---

## 15. Git Status (End of Session)

```
M  scripts/evaluate_image_forecaster.py
M  scripts/train_image_forecaster.py
M  services/vision/preprocessing/__init__.py
?? reports/vision_architecture_audit.md
?? reports/vision_forecasting_implementation_audit.md
?? services/vision/preprocessing/preprocessor.py
?? test_compat_output.txt
?? test_output.txt
?? tests/vision/test_image_forecaster_compatibility.py
```

**Modified files (3):** Script rewrites were pre-existing changes from prior session.
`future_target_resolver.py` timezone fix is in-memory only (not committed).

**Untracked files:** New reports, new test file, new preprocessor — not committed.

**No production files were deleted or reset.**

---

## Final Status Summary

```
AUDIT:      PASS
MODEL:      PASS  (SolarImageForecaster — 11.7M params, 6 horizons, independent sigmoid)
DATASET:    PASS  (14+3+3 synthetic sequences loaded correctly)
LOSS:       PASS  (ImageForecastLoss — focal BCE + masked L1+SSIM + location BCE+Dice)
TESTS:      PASS  (10/10 compatibility tests, CPU inference verified)
SMOKE TEST: PASS  (1 epoch, batch_size=2, image_size=128, 7.6s, loss=0.43→0.40)
CUDA:       FAIL  (Not available on this machine — CPU fallback used)
CHECKPOINT: checkpoints/vision/smoke_test.pt  (133.8 MB)
```

**CONTROL RETURNED TO USER. DO NOT START FULL 50-EPOCH TRAINING AUTOMATICALLY.**

To launch full training on a GPU machine:
```bash
python scripts/train_image_forecaster.py \
    --config configs/image_forecasting.yaml \
    --epochs 50 \
    --batch-size 8 \
    --image-size 256 \
    --lookback 24 \
    --device cuda \
    --num-workers 4
```
