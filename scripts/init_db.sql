-- Enable TimescaleDB extension
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
