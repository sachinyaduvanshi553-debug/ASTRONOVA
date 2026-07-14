# 🌟 AstroNova — AI-Powered Solar Flare Forecasting System

<p align="center">
  <img src="frontend/public/ASTRONOVA.png" alt="AstroNova Banner" width="100%">
</p>

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-green.svg)](https://fastapi.tiangolo.com/)
[![PostgreSQL + TimescaleDB](https://img.shields.io/badge/TimescaleDB-2.x-orange.svg)](https://www.timescale.com/)
[![Apache Kafka](https://img.shields.io/badge/Kafka-3.x-red.svg)](https://kafka.apache.org/)
[![MLflow](https://img.shields.io/badge/MLflow-2.x-blue.svg)](https://mlflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Docker](https://img.shields.io/badge/Docker-Compose-blue.svg)](https://docs.docker.com/compose/)
[![Kubernetes](https://img.shields.io/badge/Kubernetes-Ready-326CE5.svg)](https://kubernetes.io/)
[![Coverage](https://img.shields.io/badge/coverage-85%25-brightgreen.svg)]()
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/charliermarsh/ruff)

> **AstroNova** is a production-grade, microservices-based AI platform for real-time solar flare detection, X-ray flux forecasting, and space weather impact assessment — built for ISRO's SOLEXS payload on the XPoSat mission.

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Architecture](#-architecture)
- [Modules](#-modules)
- [Technology Stack](#-technology-stack)
- [Quick Start](#-quick-start)
- [Configuration](#-configuration)
- [API Reference](#-api-reference)
- [ML Pipeline](#-ml-pipeline)
- [Deployment](#-deployment)
- [Testing](#-testing)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🔭 Overview

AstroNova ingests real-time X-ray flux telemetry from ISRO's **SOLEXS (Solar X-ray Spectrometer)** instrument aboard **XPoSat**, applies multi-model AI forecasting, and delivers:

- ⚡ **Real-time solar flare detection** (sub-60s latency)
- 📈 **Multi-horizon flux forecasting** (5 min to 24 hr)
- 🌍 **Earth impact assessment** with regional risk mapping
- 🛰️ **Satellite operational risk scoring**
- 🤖 **AI Copilot** with RAG-powered space weather Q&A
- 🔔 **Multi-channel alert system** (email, webhook, SMS)
- 📊 **Interactive dashboards** with explainable AI
- 🌌 **Generative Solar Vision** (ConvLSTM + Diffusion) predicting future flare morphology and structure

### Key Capabilities

| Capability | Description | Latency |
|---|---|---|
| Flare Detection | CNN + LSTM ensemble nowcasting | < 30s |
| Short-term Forecast | 5-min to 3-hr horizon | < 5s |
| Long-term Forecast | 6-hr to 24-hr horizon | < 10s |
| Impact Assessment | Earth + satellite risk scoring | < 15s |
| RAG Q&A | Context-aware space weather Q&A | < 3s |
| Alert Delivery | Email + webhook notifications | < 5s |

## 🟢 Current Project Status & Roadmap

The AstroNova system has successfully completed its core ML and Scientific V2 Verification Sprint.

### ✅ Currently Implemented (V2 Core)
- **Data & Features**: Validated physics-informed feature engineering and synthetic/real GOES dataset pipelines.
- **ML Models**: BiLSTM, XGBoost, and LightGBM ensemble models have been trained and generalized well (Generalization gap < 2%).
- **Explainable AI (XAI)**: Integrated Gradients and SHAP provide real-time feature importance and satisfy the strict physical consistency constraints.
- **API & Inference**: FastAPI endpoints (`/predict`, `/nowcast`, `/shi`, `/simulate`) are fully operational, deterministic, and latency-optimized.
- **Predictive Solar Vision**: A completely operational end-to-end multimodal AI platform converting historical SDO sequences + telemetry into future structural predictions and precise classifications via a Dual-Head ResNet50 + Transformer Refiner architecture, with full ONNX export and XAI overlays.

### 🚀 Future Roadmap (Once Project is Complete)
- **Production Infrastructure**: Docker configuration (`Dockerfile`/`docker-compose`), Kubernetes deployment, and structured JSON logging.
- **Dynamic Logging & Hindsight Data Engine**: Seamless integration with **Postman** to dynamically log all API requests and system telemetry into TimescaleDB. This data will be automatically curated into a **"Hindsight" Dataset** alongside actual observed outcomes, powering a continuous auto-retraining pipeline to adapt to new solar cycles.

---

## 🏗️ Architecture

```
SOLEXS Telemetry
      |
      v
[Ingestion Service] ---> Kafka: astronova.raw.solexs
      |                           |
      |               [Processing Service]
      |                     | Clean, normalize, interpolate
      |                     v
      |               Kafka: astronova.processed
      |                     |
      |               [Feature Service]
      |                     | Statistical + physics features
      |                     v
      |               Kafka: astronova.features
      |                           |
      |              +------------+
      |              |            |
      |     [Forecast Service]  [Nowcast]
      |              | LSTM/CNN/XGBoost ensemble
      |              v
      |        Kafka: astronova.predictions
      |              |
      |     +--------+----------+
      |     |                   |
      | [Earth Impact]  [Satellite Risk]
      |     |                   |
      |     +--------+----------+
      |              |
      |     [Notification Service]
      |              | Email/Webhook/SMS
      |              v
      |           Alerts
      |
      +---> [RAG Service + Copilot] <-- ChromaDB + Ollama
```

---

## 📦 Modules

### Core Services

| Service | Port | Description |
|---|---|---|
| `gateway` | 8000 | API gateway with auth, rate limiting, routing |
| `ingestion` | 8001 | SOLEXS telemetry ingestion and validation |
| `processing` | 8002 | Signal cleaning, normalization, interpolation |
| `features` | 8003 | Feature extraction (physics + ML features) |
| `forecasting` | 8004 | Multi-model ensemble forecasting |
| `xai` | 8005 | Explainability (SHAP, attention, LIME) |
| `earth-impact` | 8006 | Earth impact and regional risk assessment |
| `satellite-risk` | 8007 | Satellite operational risk scoring |
| `rag` | 8008 | RAG pipeline with ChromaDB + Ollama |
| `copilot` | 8009 | AI copilot chat interface |
| `notifications` | 8010 | Multi-channel alert delivery |

### ML Pipeline (ml/)

| Component | Description |
|---|---|
| `ml/models/lstm_forecaster.py` | Bidirectional LSTM with attention |
| `ml/models/cnn_detector.py` | 1D-CNN flare detector |
| `ml/models/xgboost_classifier.py` | XGBoost GOES-class classifier |
| `ml/models/transformer_model.py` | Temporal Fusion Transformer |
| `ml/models/ensemble.py` | Weighted ensemble combiner |
| `ml/training/trainer.py` | MLflow-integrated training pipeline |
| `ml/training/hyperopt.py` | Optuna hyperparameter optimization |
| `ml/evaluation/metrics.py` | Solar-specific evaluation metrics |
| `ml/data/generators.py` | Synthetic data generation for testing |

### Shared Library (shared/astronova_core/)

| Module | Description |
|---|---|
| `config.py` | Pydantic v2 settings management |
| `logging.py` | Structured JSON logging |
| `database.py` | Async SQLAlchemy + TimescaleDB |
| `security.py` | JWT + RBAC authentication |
| `kafka_client.py` | Kafka producer/consumer utilities |
| `cache.py` | Redis caching + pub/sub |
| `metrics.py` | Prometheus metrics |
| `middleware.py` | FastAPI middleware stack |
| `exceptions.py` | Custom exception hierarchy |
| `models/` | SQLAlchemy ORM models |
| `schemas/` | Pydantic v2 request/response schemas |
| `utils/physics.py` | Solar physics calculations |
| `utils/data_io.py` | FITS/CDF/CSV data readers |
| `utils/time_utils.py` | Time series utilities |

---

## 🛠️ Technology Stack

### Backend and APIs
- **FastAPI 0.115+** - Async REST APIs
- **Pydantic v2** - Data validation and serialization
- **SQLAlchemy 2.x** - Async ORM

### Databases and Storage
- **PostgreSQL 16 + TimescaleDB 2.x** - Time-series data
- **Redis 7.x** - Caching, sessions, pub/sub
- **ChromaDB** - Vector store for RAG

### Messaging
- **Apache Kafka 3.x** - Event streaming backbone
- **Confluent Kafka Python** - Client library

### ML / AI
- **PyTorch 2.x** - Deep learning (LSTM, CNN, Transformer)
- **XGBoost 2.x** - Gradient boosting
- **MLflow 2.x** - Experiment tracking and model registry
- **Optuna** - Hyperparameter optimization
- **SHAP** - Model explainability
- **Ollama + LLaMA 3.2** - Local LLM for copilot
- **LangChain** - RAG orchestration

### Infrastructure
- **Docker + Docker Compose** - Local development
- **Kubernetes + Helm** - Production deployment
- **Prometheus + Grafana** - Observability
- **NGINX** - Reverse proxy

### Data Science
- **Astropy 6.x** - FITS file handling
- **Pandas 2.x** - Data manipulation
- **NumPy 2.x** - Numerical computing
- **SpacePy** - CDF file reading

---

## 🚀 Quick Start

### Prerequisites

- Docker 24+ and Docker Compose 2.20+
- Python 3.12+
- Make (GNU Make)
- 16 GB RAM recommended
- 50 GB disk space

### 1. Clone and Configure

```bash
git clone https://github.com/isro/astronova.git
cd astronova

# Copy environment variables
cp .env.example .env

# Edit configuration (set secrets, endpoints)
nano .env
```

### 2. Start Infrastructure

```bash
# Start all infrastructure services
make docker-up

# This starts: PostgreSQL/TimescaleDB, Redis, Kafka, MLflow, ChromaDB, Ollama
```

### 3. Initialize Database

```bash
# Run database migrations
make migrate

# Seed with sample data (optional)
make seed-data
```

### 4. Start Services

```bash
# Start all microservices in development mode
make dev

# Or start individual services
make dev-ingestion
make dev-forecasting
```

### 5. Access the Platform

| Service | URL |
|---|---|
| API Gateway | http://localhost:8000 |
| API Docs (Swagger) | http://localhost:8000/docs |
| MLflow UI | http://localhost:5000 |
| Grafana Dashboard | http://localhost:3000 |
| Kafka UI | http://localhost:8080 |
| ChromaDB | http://localhost:8001 |

### 6. Generate Test Data

```bash
# Generate synthetic SOLEXS telemetry data
make generate-data
```

---
## 📂 Project Setup

Below are step‑by‑step setup instructions for each major component of AstroNova.

### Frontend (`frontend/`)

```bash
cd frontend
npm install
npm run dev   # starts Vite dev server at http://localhost:5173
```

### Machine Learning (`ml/`)

```bash
cd ml
# Install Python dependencies
pip install -r requirements.txt   # or use poetry/uv
# Run training or inference scripts
python -m ml.training.trainer   # example training
```

### Services (`services/`)

```bash
cd services
# Install Python dependencies
pip install -r requirements.txt
# Start individual micro‑services (example)
make dev-gateway      # API gateway on 8000
make dev-ingestion    # ingestion service on 8001
# ... repeat for other services as needed
```

### Shared Library (`shared/`)

```bash
cd shared
pip install -e .   # editable install for shared utilities
```

### Monitoring (`monitoring/`)

```bash
cd monitoring
docker compose up -d   # brings up Prometheus, Grafana, etc.
```

### Docker Compose (all)

```bash
make docker-up   # starts all containers: PostgreSQL, Redis, Kafka, MLflow, ChromaDB, Ollama, etc.
```

## 📸 Screenshots

The `public/` folder contains UI screenshots. Below are previews:

![Screenshot 1](public/Screenshot%202026-06-28%20140130.png)
![Screenshot 2](public/Screenshot%202026-06-28%20140140.png)
![Screenshot 3](public/Screenshot%202026-06-28%20140153.png)
![Screenshot 4](public/Screenshot%202026-06-28%20140200.png)
![Screenshot 5](public/Screenshot%202026-06-28%20140225.png)

---

## 🔄 Data Pipeline Flowchart

```mermaid
flowchart TD
    classDef implemented fill:#d4edda,stroke:#28a745,stroke-width:2px,color:#000;
    classDef planned fill:#fff3cd,stroke:#ffc107,stroke-width:2px,stroke-dasharray: 5 5,color:#000;

    subgraph Legend
        L1[Currently Implemented]:::implemented
        L2[Planned / Future]:::planned
    end

    A["SOLEXS Telemetry\nSoft X-ray 1-15 keV"]:::implemented --> I["Ingestion Service\nPort 8001"]:::implemented
    B["HEL1OS Telemetry\nHard X-ray 10-150 keV"]:::implemented --> I
    C["GOES XRF Reference\nNOAA Real-time Feed"]:::implemented --> I

    I -->|"Kafka: astronova.raw.solexs"| P["Processing Service\nPort 8002"]:::implemented
    P -->|"Clean + Normalize + Interpolate"| P
    P -->|"Kafka: astronova.processed"| F["Feature Service\nPort 8003"]:::implemented

    F -->|"Physics + ML Features"| F
    F -->|"Kafka: astronova.features"| FC["Forecasting Service\nPort 8004"]:::implemented

    FC --> M1["BiLSTM"]:::implemented
    FC --> M2["CNN Nowcaster"]:::implemented
    FC --> M3["TFT"]:::implemented
    FC --> M4["XGBoost"]:::implemented
    FC --> ENS["Weighted Ensemble"]:::implemented
    M1 & M2 & M3 & M4 --> ENS

    ENS -->|"Kafka: astronova.predictions"| EI["Earth Impact\nPort 8006"]:::implemented
    ENS --> SR["Satellite Risk\nPort 8007"]:::implemented
    ENS --> XAI["XAI Service\nPort 8005"]:::implemented
    ENS --> NOT["Notifications\nPort 8010"]:::implemented

    XAI -->|"SHAP + LIME + Attention"| DB["Dashboard\nFrontend"]:::implemented
    EI --> DB
    SR --> DB

    DB --> COP["LLM Copilot\nPort 8009"]:::implemented
    COP <--> RAG["RAG Service\nChromaDB + LLaMA 3.2"]:::implemented

    %% Dynamic Logging & Hindsight Feature
    POSTMAN["Postman / API Clients"]:::implemented -->|"API Requests"| GATEWAY["API Gateway"]:::implemented
    GATEWAY -->|"Dynamic Logging"| TDB[("TimescaleDB\nTelemetry & Predictions")]:::planned
    TDB -->|"Hindsight Pipeline"| AUTORETRAIN["Continuous Auto-Retraining"]:::planned
    AUTORETRAIN -.->|"Model Updates"| FC
```

---

## 🏗️ Full System Architecture

```mermaid
graph TB
    classDef implemented fill:#d4edda,stroke:#28a745,stroke-width:2px,color:#000;
    classDef planned fill:#fff3cd,stroke:#ffc107,stroke-width:2px,stroke-dasharray: 5 5,color:#000;

    subgraph Legend
        L1[Currently Implemented]:::implemented
        L2[Planned / Future]:::planned
    end

    subgraph INSTRUMENTS["Aditya-L1 Instruments"]
        SOLEXS["SOLEXS\nSoft X-ray 1-15 keV"]:::implemented
        HEL1OS["HEL1OS\nHard X-ray 10-150 keV"]:::implemented
    end

    subgraph EXTERNAL["External Data Sources"]
        GOES["GOES XRF\nNOAA Real-time"]:::implemented
        NOAACAT["NOAA Flare Catalog"]:::implemented
        NASACME["NASA CME Catalog"]:::implemented
        POSTMAN["Postman / API Clients"]:::implemented
    end

    subgraph GATEWAY["API Gateway (8000)"]
        AUTH["JWT + RBAC Auth"]:::implemented
        RATELIMIT["Rate Limiter"]:::implemented
        ROUTER["API Router"]:::implemented
    end

    subgraph PIPELINE["Data Pipeline"]
        ING["Ingestion (8001)"]:::implemented
        PROC["Processing (8002)"]:::implemented
        FEAT["Feature Engineering (8003)"]:::implemented
    end

    subgraph KAFKA["Apache Kafka Event Bus"]
        K1["astronova.raw.solexs"]:::implemented
        K2["astronova.processed"]:::implemented
        K3["astronova.features"]:::implemented
        K4["astronova.predictions"]:::implemented
    end

    subgraph AICORE["AI Core"]
        FORE["Forecasting (8004)\nBiLSTM + CNN + TFT + XGBoost"]:::implemented
        XAI["XAI Service (8005)\nSHAP + LIME + Attention"]:::implemented
    end

    subgraph HINDSIGHT["Dynamic Logging & Hindsight"]
        DYNLOG["Dynamic API Logger"]:::planned
        AUTORETRAIN["Continuous Auto-Retraining"]:::planned
    end

    subgraph INTEL["Intelligence Services"]
        EI["Earth Impact (8006)"]:::implemented
        SR["Satellite Risk (8007)"]:::implemented
        NOT["Notifications (8010)"]:::implemented
    end

    subgraph LLMRAG["LLM + RAG"]
        RAG["RAG Service (8008)\nChromaDB"]:::implemented
        COP["Copilot (8009)\nLLaMA 3.2 via Ollama"]:::implemented
    end

    subgraph STORAGE["Storage Layer"]
        PG["PostgreSQL 16\n+ TimescaleDB"]:::implemented
        REDIS["Redis 7\nCache + PubSub"]:::implemented
        CHROMA["ChromaDB\nVector Store"]:::implemented
    end

    subgraph FRONTEND["Frontend"]
        DASH["Next.js 15 Dashboard\nReal-time Charts + Maps"]:::implemented
    end

    subgraph OBS["Observability"]
        PROM["Prometheus"]:::planned
        GRAF["Grafana"]:::planned
        MLFLOW["MLflow\nExperiment Tracking"]:::implemented
    end

    INSTRUMENTS --> ING
    EXTERNAL --> ING
    POSTMAN --> GATEWAY
    GATEWAY --> DYNLOG
    DYNLOG --> PG
    PG -->|"Hindsight Dataset"| AUTORETRAIN
    AUTORETRAIN -.->|"Update Weights"| FORE

    ING --> K1 --> PROC --> K2 --> FEAT --> K3 --> FORE
    FORE --> K4 --> EI & SR & NOT
    FORE --> XAI
    XAI & EI & SR --> DASH
    DASH --> COP
    COP <--> RAG
    RAG <--> CHROMA
    PIPELINE & AICORE & INTEL --> PG
    GATEWAY --> PIPELINE
    OBS -.-> AICORE & PIPELINE & INTEL
```

---

## ⚙️ Configuration

All configuration is managed via environment variables. Copy `.env.example` to `.env` and update values.

### Critical Settings

```bash
# Security - MUST change in production
JWT_SECRET_KEY=your-256-bit-secret-key

# Database
DATABASE_URL=postgresql+asyncpg://astronova:password@localhost:5432/astronova

# Kafka
KAFKA_BOOTSTRAP_SERVERS=localhost:9092

# AI Services
OLLAMA_MODEL=llama3.2:3b
MLFLOW_TRACKING_URI=http://localhost:5000
```

---

## 📡 API Reference

### Authentication

```bash
# Get access token
POST /api/v1/auth/token

{
  "username": "admin",
  "password": "password"
}
```

### Forecasting

```bash
# Request solar flare forecast
POST /api/v1/forecast

{
  "lookback_minutes": 60,
  "horizons": [5, 15, 30, 60, 180, 360],
  "model_type": "ensemble"
}
```

---

## 🤖 ML Pipeline

### Training a New Model

```bash
# Run training pipeline
make train MODEL=lstm_forecaster

# View results in MLflow
open http://localhost:5000
```

### Evaluation Metrics

| Metric | Description | Target |
|---|---|---|
| TSS | True Skill Score | > 0.7 |
| HSS | Heidke Skill Score | > 0.6 |
| FAR | False Alarm Rate | < 0.3 |
| POD | Probability of Detection | > 0.8 |
| RMSE | Root Mean Square Error | < 0.15 |
| Brier Score | Probabilistic accuracy | < 0.2 |

---

## 🚢 Deployment

### Docker Compose (Development)

```bash
make docker-up       # Start all services
make docker-down     # Stop all services
make docker-build    # Rebuild images
make docker-logs     # Tail all logs
```

### Kubernetes (Production)

```bash
# Apply all K8s manifests
make k8s-apply

kubectl get pods -n astronova
kubectl get services -n astronova
```

---

## 🧪 Testing

```bash
# Run all tests
make test

# Run with coverage
make test-coverage

# Run specific service tests
make test-service SERVICE=forecasting
```

---

## 📄 License

MIT License — see [LICENSE](LICENSE)

---

*AstroNova — Watching the Sun, Protecting the Earth*

