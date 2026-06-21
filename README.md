# 🌟 AstroNova — AI-Powered Solar Flare Forecasting System

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

### Key Capabilities

| Capability | Description | Latency |
|---|---|---|
| Flare Detection | CNN + LSTM ensemble nowcasting | < 30s |
| Short-term Forecast | 5-min to 3-hr horizon | < 5s |
| Long-term Forecast | 6-hr to 24-hr horizon | < 10s |
| Impact Assessment | Earth + satellite risk scoring | < 15s |
| RAG Q&A | Context-aware space weather Q&A | < 3s |
| Alert Delivery | Email + webhook notifications | < 5s |

---

## Architecture

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
