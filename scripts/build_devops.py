import os


def create_file(path, content):
    dirname = os.path.dirname(path)
    if dirname:
        os.makedirs(dirname, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"Created: {path}")

# --- 1. docker-compose.yml ---
create_file("docker-compose.yml", """version: '3.8'

services:
  postgres:
    image: timescale/timescaledb:latest-pg16
    environment:
      POSTGRES_DB: astronova
      POSTGRES_USER: astronova
      POSTGRES_PASSWORD: changeme_secure_password
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./scripts/init_db.sql:/docker-entrypoint-initdb.d/01_init.sql
    ports:
      - "5432:5432"
    networks:
      - astronova-net

  redis:
    image: redis:7-alpine
    command: redis-server --appendonly yes
    volumes:
      - redis_data:/data
    ports:
      - "6379:6379"
    networks:
      - astronova-net

  prometheus:
    image: prom/prometheus:latest
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
    ports:
      - "9090:9090"
    networks:
      - astronova-net

  gateway:
    build: ./services/gateway
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql+asyncpg://astronova:changeme_secure_password@postgres:5432/astronova
      - REDIS_URL=redis://redis:6379/0
      - JWT_SECRET_KEY=your-super-secret-jwt-key-change-in-production
      - JWT_ALGORITHM=HS256
    depends_on:
      - postgres
      - redis
    networks:
      - astronova-net

  ingestion:
    build: ./services/ingestion
    ports:
      - "8001:8001"
    environment:
      - DATABASE_URL=postgresql+asyncpg://astronova:changeme_secure_password@postgres:5432/astronova
      - REDIS_URL=redis://redis:6379/0
    depends_on:
      - postgres
      - redis
    networks:
      - astronova-net

  forecasting:
    build: ./services/forecasting
    ports:
      - "8004:8004"
    environment:
      - DATABASE_URL=postgresql+asyncpg://astronova:changeme_secure_password@postgres:5432/astronova
      - REDIS_URL=redis://redis:6379/0
    depends_on:
      - postgres
      - redis
    networks:
      - astronova-net

volumes:
  postgres_data:
  redis_data:

networks:
  astronova-net:
    driver: bridge
""")

# --- 2. scripts/init_db.sql ---
create_file("scripts/init_db.sql", """-- Enable TimescaleDB extension
CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;

-- Create observations table
CREATE TABLE IF NOT EXISTS solexs_observations (
    time TIMESTAMP WITHOUT TIME ZONE NOT NULL,
    soft_xray_flux DOUBLE PRECISION NOT NULL,
    hard_xray_flux DOUBLE PRECISION NOT NULL,
    energy_band_lo DOUBLE PRECISION NOT NULL,
    energy_band_hi DOUBLE PRECISION NOT NULL,
    quality_flag INTEGER DEFAULT 0,
    source_file VARCHAR(255),
    data_version VARCHAR(50) DEFAULT '1.0.0'
);

-- Convert to hypertable for time-series optimization
SELECT create_hypertable('solexs_observations', 'time', if_not_exists => TRUE);

-- Create ingestion jobs table
CREATE TABLE IF NOT EXISTS ingestion_jobs (
    id UUID PRIMARY KEY,
    status VARCHAR(50) NOT NULL,
    source_file VARCHAR(255) NOT NULL,
    format VARCHAR(50) NOT NULL,
    rows_ingested INTEGER DEFAULT 0,
    errors JSON,
    started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP
);
""")

# --- 3. monitoring/prometheus.yml ---
create_file("monitoring/prometheus.yml", """global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'gateway'
    static_configs:
      - targets: ['gateway:8000']
  - job_name: 'ingestion'
    static_configs:
      - targets: ['ingestion:8001']
  - job_name: 'forecasting'
    static_configs:
      - targets: ['forecasting:8004']
""")

# --- 4. scripts/generate_synthetic_data.py ---
create_file("scripts/generate_synthetic_data.py", """import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

def generate_data(num_points: int = 1440):
    start_time = datetime.utcnow() - timedelta(days=1)
    times = [start_time + timedelta(minutes=i) for i in range(num_points)]

    # Simulate quiescent solar background flux with periodic fluctuations
    base_flux = 1e-8
    noise = np.random.normal(0, 0.2 * base_flux, num_points)
    soft_flux = base_flux + noise + (np.sin(np.linspace(0, 4*np.pi, num_points)) * 0.1 * base_flux)

    # Inject a simulated M-class solar flare
    flare_start = int(num_points * 0.4)
    flare_peak = int(num_points * 0.45)
    flare_end = int(num_points * 0.6)

    for i in range(flare_start, flare_peak):
        factor = (i - flare_start) / (flare_peak - flare_start)
        soft_flux[i] += 1.5e-5 * (factor ** 2) # Rise phase

    for i in range(flare_peak, flare_end):
        factor = (flare_end - i) / (flare_end - flare_peak)
        soft_flux[i] += 1.5e-5 * (factor ** 3) # Decay phase

    hard_flux = soft_flux * 0.1 + np.random.normal(0, 0.01 * base_flux, num_points)

    df = pd.DataFrame({
        "time": times,
        "soft_xray_flux": np.clip(soft_flux, 1e-9, None),
        "hard_xray_flux": np.clip(hard_flux, 1e-10, None),
        "quality_flag": [0] * num_points
    })

    os.makedirs("data/sample", exist_ok=True)
    df.to_csv("data/sample/synthetic_solexs.csv", index=False)
    print("Generated synthetic data in data/sample/synthetic_solexs.csv")

if __name__ == "__main__":
    generate_data()
""")

# --- 5. .github/workflows/ci.yml ---
create_file(".github/workflows/ci.yml", """name: AstroNova CI

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  build-and-test:
    runs-on: ubuntu-latest

    steps:
    - uses: actions/checkout@v4

    - name: Set up Python 3.12
      uses: actions/setup-python@v5
      with:
        python-version: '3.12'

    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install ruff pytest pytest-asyncio

    - name: Lint with Ruff
      run: |
        ruff check .

    - name: Run unit tests
      run: |
        pytest shared/ services/ --ignore=shared/setup.py
""")

# --- 6. scripts/setup.sh ---
create_file("scripts/setup.sh", """#!/bin/bash
echo "Setting up AstroNova Space Weather Platform..."
python scripts/generate_synthetic_data.py
echo "Setup complete. Run 'docker-compose up --build' to start."
""")

print("DEVOPS AND SCRIPTS GENERATED SUCCESSFULLY")
