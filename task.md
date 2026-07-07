# AstroNova Build Tasks

## Phase 1: Foundation
- [x] Implementation plan
- [x] Project root config files (pyproject.toml, .env.example, Makefile)
- [x] Shared library (astronova_core)
- [x] Database models + Alembic migrations

## Phase 2: Data Pipeline Services
- [x] Ingestion Service
- [x] Processing Service
- [x] Feature Engineering Service

## Phase 3: ML Services
- [x] Nowcasting Service
- [x] Forecasting Service (PyTorch Lightning models)
- [x] XAI Service

## Phase 4: Intelligence Services
- [x] Earth Impact Service
- [x] Satellite Risk Service
- [x] Historical Similarity Engine

## Phase 5: RAG + LLM
- [x] RAG Knowledge Service
- [x] LLM Copilot Service

## Phase 6: Gateway + Notifications
- [x] API Gateway
- [x] Notification Service

## Phase 7: Frontend
- [x] Next.js 15 Dashboard

## Phase 8: DevOps + MLOps
- [x] Dockerfiles
- [x] Docker Compose
- [x] Kubernetes manifests
- [x] GitHub Actions CI/CD
- [x] Prometheus + Grafana

## Phase 9: Testing + Docs
- [x] Test suites
- [x] Architecture diagrams
- [x] Documentation
- [x] Synthetic data generator

## Phase 10: Solar Vision Module Integration
- [x] services/vision/ full multimodal pipeline (encoder + fusion + decoder + XAI)
- [x] services/vision/__init__.py package init
- [x] VisionInferencePipeline (MC-Dropout, flare probability, GradCAM hooks)
- [x] XAIVisualizer (GradCAM, attention maps, uncertainty)
- [x] Comprehensive metrics (SSIM, PSNR, MAE, FID, flare F1/TSS)
- [x] Vision API router (/vision/predict, /vision/explain, /vision/health)
- [x] solar_vision service upgraded to use full pipeline
- [x] Frontend: Solar Vision Module tab added to dashboard
  - [x] Animated solar disc (WebGL-style canvas with corona, granules, AR spots)
  - [x] GradCAM heatmap canvas animation
  - [x] MC-Dropout uncertainty ring animation
  - [x] Multi-horizon prediction timeline (LineChart)
  - [x] Cross-instrument radar performance chart
  - [x] Active region analysis with vision-derived probabilities
  - [x] Image quality metrics panel (SSIM, PSNR, MAE, FID)
  - [x] Fusion module pipeline status breakdown
  - [x] Run Prediction button with 2.2s inference simulation
