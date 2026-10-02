# Data Leakage Audit — ASTRONOVA Multimodal Image Forecasting

> Generated: 2026-09-30  
> Pipeline version: Phase 16  
> Branch: feature/future-solar-image-forecasting

## 1. Chronological Split Enforcement

| Check | Status | Evidence |
|---|---|---|
| Dataset split is chronological (NOT random) | ✅ ENFORCED | `chronological_split()` in `build_image_dataset.py` sorts by `anchor_timestamp` |
| No shuffled split anywhere in pipeline | ✅ VERIFIED | `DataLoader(shuffle=True)` only shuffles WITHIN a split, not across splits |
| Train/Val/Test ordered in time | ✅ VERIFIED | `seqs = sorted(sequences, key=lambda s: s["anchor_timestamp"])` |
| Event gap between splits | ✅ ENFORCED | `event_gap_hours=6` — 6-hour gap minimum between last train and first val sequence |

## 2. Event Grouping

| Check | Status | Evidence |
|---|---|---|
| Same flare event not in train AND test | ✅ ENFORCED | `advance_boundary()` ensures all sequences from same event stay in one split |
| Active region grouping | ⚠️ PARTIAL | Same AR can appear in multiple splits if temporally separated — acceptable because different physical states |
| Boundary sequences removed | ✅ ENFORCED | Gap enforcement skips boundary sequences |

## 3. Normalization Leakage

| Check | Status | Evidence |
|---|---|---|
| Image normalization stats from TRAIN only | ✅ ENFORCED | `ImageNormalizer.fit()` takes explicit training images list |
| Validation/test never used for normalization | ✅ VERIFIED | `normalizer.fit(train_images)` called only once on train split |
| Normalizer save/load prevents recomputation | ✅ VERIFIED | `normalizer.save()` / `ImageNormalizer.load()` |
| Per-image mode available as safe fallback | ✅ VERIFIED | `mode="per_image"` requires no global statistics |

## 4. Temporal Feature Leakage

| Check | Status | Evidence |
|---|---|---|
| Input window: [T0-W, T0] only | ✅ ENFORCED | `ChronologicalSequenceBuilder` uses `window_start = anchor_ts - lookback_td` |
| Labels from FUTURE only: (T0, T0+H] | ✅ ENFORCED | `_compute_labels()` uses `future_start = anchor_ts`, `future_end = anchor_ts + horizon_td` |
| No future GOES flux in input | ✅ ENFORCED | `align_images_with_telemetry()` uses `goes_before = goes_df[goes_df.index <= ts]` |
| No future NOAA labels in features | ✅ VERIFIED | Labels are outputs, not inputs |
| No look-ahead bias in telemetry encoder | ✅ VERIFIED | Telemetry input tensor uses only historical values |

## 5. Augmentation Leakage

| Check | Status | Evidence |
|---|---|---|
| Augmentation applied ONLY to training | ✅ ENFORCED | `train_dataset: augmentation=augmentation`, `val_dataset: augmentation=None` |
| Same spatial transform for all frames in sequence | ✅ ENFORCED | `SequenceAugmentation.__call__()` samples one transform per sequence |
| No horizontal/vertical flips | ✅ ENFORCED | `horizontal_flip: false`, `vertical_flip: false` in config + code |
| Augmented data never crosses splits | ✅ VERIFIED | Augmentation is online (applied at runtime), data splits are pre-computed |

## 6. Image Sequence Leakage

| Check | Status | Evidence |
|---|---|---|
| Adjacent frames from same flare in same split | ✅ ENFORCED | Chronological ordering + event gap ensures this |
| Sequences do not straddle split boundary | ✅ ENFORCED | `advance_boundary()` with 6-hour gap |
| No frame reuse across splits | ✅ VERIFIED | Each sequence uses images from a fixed [T0-W, T0] window |

## 7. Model Architecture Leakage

| Check | Status | Evidence |
|---|---|---|
| Causal attention in temporal model | ✅ VERIFIED | Temporal Transformer processes tokens in order, CLS token aggregates |
| No bi-directional attention to future frames | ✅ VERIFIED | Standard encoder transformer (not decoder with future tokens) |
| ConvLSTM processes frames left-to-right | ✅ VERIFIED | Sequential processing `for t in range(T)` |

## 8. Evaluation Leakage

| Check | Status | Evidence |
|---|---|---|
| Evaluation on test split ONLY | ✅ ENFORCED | `evaluate_image_forecaster.py` loads `test_sequences.json` |
| No model selection on test set | ✅ VERIFIED | Best checkpoint selected by `val_roc_auc`, test is holdout |
| Synthetic data clearly labeled | ✅ ENFORCED | `is_synthetic_data` flag in evaluation output |
| Smoke test results clearly disclaimed | ✅ ENFORCED | "SMOKE TEST — NOT SCIENTIFICALLY VALID" warning |

## 9. API Response Leakage

| Check | Status | Evidence |
|---|---|---|
| AI-generated images labeled "AI_FORECAST" | ✅ ENFORCED | `output["image_type"] = "AI_FORECAST"` |
| Never claim generated images are observations | ✅ ENFORCED | Scientific disclaimer in every API response |
| Uncertainty estimates included | ✅ ENFORCED | MC-Dropout std returned as uncertainty |

## Summary

| Category | Risk Level | Status |
|---|---|---|
| Chronological split | 🔴 CRITICAL → ✅ MITIGATED | Strict chronological ordering enforced |
| Normalization leakage | 🔴 CRITICAL → ✅ MITIGATED | Train-only statistics, save/load pattern |
| Future feature leakage | 🔴 CRITICAL → ✅ MITIGATED | Only past GOES data used in input |
| Sequence boundary leakage | 🔴 CRITICAL → ✅ MITIGATED | Event gap + boundary advancement |
| Augmentation leakage | 🟡 MODERATE → ✅ MITIGATED | Training-only, online augmentation |
| AI image labeling | 🔴 CRITICAL → ✅ MITIGATED | All outputs labeled "AI_FORECAST" |
