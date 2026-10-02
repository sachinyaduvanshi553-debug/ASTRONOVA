# AstroNova Image Forecasting Pre-flight Audit Report

> Generated: 2026-09-30 | Branch: feature/future-solar-image-forecasting

## 1. Existing Image Datasets
- datasets/raw/sdo_images/: NOT PRESENT
- I1.jpg-I13.jpg (root+frontend/public): UI-ONLY gallery images, NOT training data
- Real data confirmed at OneDrive: C:\Users\sachi\OneDrive\Documents\ASTRONOVA\DATA\

## 2. Real Data Location (OneDrive)
- cleaned/goes/goes_xrs_oct2024_jan2025.csv  (GOES XRS Oct2024-Jan2025)
- cleaned/helios/HEL1OS_FILTERED.parquet     (Aditya-L1 HEL1OS)
- cleaned/noaa_labels/noaa_flares_clean.csv  (NOAA flare catalog)
- cleaned/solexs/solexs_ml_ready_v1.csv      (Aditya-L1 SoLEXS)
- cleaned/auxiliary/ (CME, GST, SEP catalogs)
- events/flare_sequences/                    (SDO image sequences - status unknown)

## 3. Image Dimensions
- SDO AIA browse: 512x512, 1024x1024, 4096x4096 (JPEG)
- Pipeline target: 256x256 (configurable 128/256/512)

## 4. Missing Components (Critical)
1. services/vision/data/ directory (import fails)
2. configs/image_pipeline.yaml
3. configs/image_forecasting.yaml
4. Per-horizon flare classification heads (6 horizons x 3 classes)
5. Spatial flare location heatmap head
6. Chronological dataset splitter with event-grouped leakage prevention
7. scripts/build_image_dataset.py
8. scripts/train_image_forecaster.py
9. scripts/evaluate_image_forecaster.py
10. models/registry/
11. datasets/manifests/image_dataset_v1.json
12. checkpoints/vision/

## 5. Data Leakage Risks
- CRITICAL: Random image splitting (adjacent frames from same flare in train+test)
- CRITICAL: Normalization on full dataset (must use train split only)
- CRITICAL: Sequence overlap at train/val/test boundary
- CRITICAL: Future telemetry as input (T0+H features must NOT be in input)

## 6. Existing Working Components
- services/vision/model.py: SolarVisionPredictor (ResNet50+Transformer+Decoder)
- services/vision/encoder.py: ImageEncoder, TemporalEncoder, PhysicsEncoder
- services/vision/decoder.py: ImageDecoder
- services/vision/fusion.py: FusionNetwork (cross-attention)
- services/vision/preprocessing.py: resize, normalize, augment
- POST /api/v1/forecast/predict: FUNCTIONAL
- Next.js Solar Vision tab: FUNCTIONAL

## 7. Implementation Order
PHASE 1: Audit (this document)
PHASE 2: build_image_dataset.py + chronological splitter
PHASE 3: configs + preprocessing pipeline
PHASE 4: CNN encoder extensions
PHASE 5: ConvLSTM + Temporal Transformer
PHASE 6: Per-horizon classification heads
PHASE 7: Spatial heatmap head
PHASE 8: Multimodal fusion
PHASE 9: Future image decoder
PHASE 10: MC-Dropout uncertainty
PHASE 11: Grad-CAM / XAI
PHASE 12: train_image_forecaster.py
PHASE 13: evaluate_image_forecaster.py
PHASE 14: FastAPI /api/v1/vision/forecast
PHASE 15: Frontend updates
PHASE 16: End-to-end test
