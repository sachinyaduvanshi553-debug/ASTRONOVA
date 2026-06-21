"""
Configuration Management for AstroNova
=======================================
Pydantic v2 Settings with environment variable support for all
AstroNova microservices. Uses a nested settings approach where
each subsystem has its own settings class composed into AppSettings.

Usage:
    from astronova_core.config import get_settings

    settings = get_settings()
    db_url = settings.database.url
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any

from pydantic import AnyUrl, Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Database Settings
# ──────────────────────────────────────────────────────────────────────────────


class DatabaseSettings(BaseSettings):
    """PostgreSQL / TimescaleDB connection settings."""

    model_config = SettingsConfigDict(env_prefix="POSTGRES_", env_file=".env", extra="ignore")

    host: str = Field(default="localhost", description="PostgreSQL host")
    port: int = Field(default=5432, ge=1, le=65535, description="PostgreSQL port")
    db: str = Field(default="astronova", description="Database name")
    user: str = Field(default="astronova", description="Database user")
    password: SecretStr = Field(default=SecretStr("changeme"), description="Database password")

    # Overrides the assembled URL if provided
    url: str = Field(
        default="",
        alias="DATABASE_URL",
        description="Full async database URL. Overrides individual fields if set.",
    )

    timescale_enabled: bool = Field(
        default=True,
        alias="TIMESCALE_ENABLED",
        description="Enable TimescaleDB extension",
    )

    # Connection pool settings
    pool_size: int = Field(default=10, description="SQLAlchemy pool size")
    max_overflow: int = Field(default=20, description="SQLAlchemy max overflow")
    pool_timeout: int = Field(default=30, description="SQLAlchemy pool timeout seconds")
    pool_recycle: int = Field(default=1800, description="Connection recycle interval seconds")
    echo: bool = Field(default=False, description="Echo SQL queries (dev only)")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    @model_validator(mode="after")
    def assemble_database_url(self) -> "DatabaseSettings":
        """Build the async database URL from individual components if not provided."""
        if not self.url:
            pwd = self.password.get_secret_value()
            self.url = (
                f"postgresql+asyncpg://{self.user}:{pwd}"
                f"@{self.host}:{self.port}/{self.db}"
            )
        return self

    @field_validator("port")
    @classmethod
    def validate_port(cls, v: int) -> int:
        """Ensure port is in valid range."""
        if not (1 <= v <= 65535):
            raise ValueError(f"Port must be 1–65535, got {v}")
        return v

    def get_sync_url(self) -> str:
        """Return a synchronous psycopg2 URL for Alembic migrations."""
        return self.url.replace("postgresql+asyncpg://", "postgresql+psycopg2://")


# ──────────────────────────────────────────────────────────────────────────────
# Redis Settings
# ──────────────────────────────────────────────────────────────────────────────


class RedisSettings(BaseSettings):
    """Redis connection and cache settings."""

    model_config = SettingsConfigDict(
        env_prefix="REDIS_",
        env_file=".env",
        extra="ignore",
        populate_by_name=True,
    )

    host: str = Field(default="localhost", description="Redis host")
    port: int = Field(default=6379, ge=1, le=65535, description="Redis port")
    password: SecretStr = Field(default=SecretStr(""), description="Redis password")
    url: str = Field(default="redis://localhost:6379/0", description="Redis connection URL")
    db_index: int = Field(default=0, ge=0, le=15, description="Redis database index")

    # Cache defaults
    default_ttl: int = Field(default=300, description="Default cache TTL in seconds")
    max_connections: int = Field(default=50, description="Max connections in pool")
    socket_timeout: float = Field(default=5.0, description="Socket timeout seconds")
    socket_connect_timeout: float = Field(default=5.0, description="Connect timeout seconds")
    retry_on_timeout: bool = Field(default=True, description="Retry on timeout")

    @model_validator(mode="after")
    def assemble_redis_url(self) -> "RedisSettings":
        """Build Redis URL if not provided."""
        if not self.url or self.url == "redis://localhost:6379/0":
            pwd = self.password.get_secret_value()
            if pwd:
                self.url = f"redis://:{pwd}@{self.host}:{self.port}/{self.db_index}"
            else:
                self.url = f"redis://{self.host}:{self.port}/{self.db_index}"
        return self


# ──────────────────────────────────────────────────────────────────────────────
# Kafka Settings
# ──────────────────────────────────────────────────────────────────────────────


class KafkaSettings(BaseSettings):
    """Apache Kafka producer/consumer settings."""

    model_config = SettingsConfigDict(
        env_prefix="KAFKA_",
        env_file=".env",
        extra="ignore",
        populate_by_name=True,
    )

    bootstrap_servers: str = Field(
        default="localhost:9092",
        description="Comma-separated list of Kafka broker addresses",
    )
    topic_raw_data: str = Field(
        default="astronova.raw.solexs",
        description="Topic for raw SOLEXS telemetry",
    )
    topic_processed: str = Field(
        default="astronova.processed",
        description="Topic for processed observations",
    )
    topic_features: str = Field(
        default="astronova.features",
        description="Topic for extracted features",
    )
    topic_alerts: str = Field(
        default="astronova.alerts",
        description="Topic for alert messages",
    )
    topic_predictions: str = Field(
        default="astronova.predictions",
        description="Topic for model predictions",
    )
    consumer_group: str = Field(
        default="astronova-consumers",
        description="Default consumer group ID",
    )

    # Producer settings
    producer_acks: str = Field(default="all", description="Producer ack level")
    producer_retries: int = Field(default=5, description="Producer retry count")
    producer_batch_size: int = Field(default=16384, description="Producer batch size bytes")
    producer_linger_ms: int = Field(default=10, description="Producer linger time ms")
    producer_compression_type: str = Field(
        default="snappy", description="Producer compression type"
    )
    producer_max_block_ms: int = Field(
        default=60000, description="Max time to block on send"
    )

    # Consumer settings
    consumer_auto_offset_reset: str = Field(
        default="earliest", description="Consumer auto offset reset"
    )
    consumer_enable_auto_commit: bool = Field(
        default=False, description="Enable auto commit (prefer manual)"
    )
    consumer_max_poll_records: int = Field(
        default=500, description="Max records per poll"
    )
    consumer_session_timeout_ms: int = Field(
        default=30000, description="Consumer session timeout ms"
    )

    @field_validator("producer_acks")
    @classmethod
    def validate_acks(cls, v: str) -> str:
        """Validate producer acknowledgment level."""
        allowed = {"0", "1", "all", "-1"}
        if v not in allowed:
            raise ValueError(f"acks must be one of {allowed}, got '{v}'")
        return v

    def get_bootstrap_servers_list(self) -> list[str]:
        """Return bootstrap servers as a list."""
        return [s.strip() for s in self.bootstrap_servers.split(",")]


# ──────────────────────────────────────────────────────────────────────────────
# JWT / Security Settings
# ──────────────────────────────────────────────────────────────────────────────


class JWTSettings(BaseSettings):
    """JWT token and security settings."""

    model_config = SettingsConfigDict(
        env_prefix="JWT_",
        env_file=".env",
        extra="ignore",
        populate_by_name=True,
    )

    secret_key: SecretStr = Field(
        default=SecretStr("change-me-in-production-use-256-bit-key"),
        description="JWT signing secret key",
    )
    algorithm: str = Field(default="HS256", description="JWT signing algorithm")
    access_token_expire_minutes: int = Field(
        default=60, ge=1, description="Access token TTL in minutes"
    )
    refresh_token_expire_days: int = Field(
        default=7, ge=1, description="Refresh token TTL in days"
    )

    @field_validator("algorithm")
    @classmethod
    def validate_algorithm(cls, v: str) -> str:
        """Ensure algorithm is one of the supported HMAC algorithms."""
        allowed = {"HS256", "HS384", "HS512", "RS256", "RS384", "RS512"}
        if v not in allowed:
            raise ValueError(f"JWT algorithm must be one of {allowed}")
        return v


# ──────────────────────────────────────────────────────────────────────────────
# MLflow Settings
# ──────────────────────────────────────────────────────────────────────────────


class MLflowSettings(BaseSettings):
    """MLflow experiment tracking and model registry settings."""

    model_config = SettingsConfigDict(
        env_prefix="MLFLOW_",
        env_file=".env",
        extra="ignore",
        populate_by_name=True,
    )

    tracking_uri: str = Field(
        default="http://localhost:5000",
        description="MLflow tracking server URI",
    )
    experiment_name: str = Field(
        default="astronova-forecasting",
        description="Default MLflow experiment name",
    )
    model_registry_uri: str = Field(
        default="http://localhost:5000",
        description="MLflow model registry URI",
    )
    artifact_root: str = Field(
        default="/app/mlflow/artifacts",
        description="MLflow artifact storage root",
    )
    run_name_prefix: str = Field(
        default="astronova",
        description="Prefix for MLflow run names",
    )


# ──────────────────────────────────────────────────────────────────────────────
# ChromaDB Settings
# ──────────────────────────────────────────────────────────────────────────────


class ChromaSettings(BaseSettings):
    """ChromaDB vector store settings for RAG pipeline."""

    model_config = SettingsConfigDict(
        env_prefix="CHROMA_",
        env_file=".env",
        extra="ignore",
        populate_by_name=True,
    )

    host: str = Field(default="localhost", description="ChromaDB host")
    port: int = Field(default=8000, ge=1, le=65535, description="ChromaDB port")
    collection_flare_events: str = Field(
        default="flare_events",
        description="Collection name for flare event embeddings",
    )
    collection_knowledge: str = Field(
        default="space_weather_knowledge",
        description="Collection name for space weather knowledge base",
    )
    embedding_dimension: int = Field(
        default=768, description="Embedding vector dimension"
    )
    distance_function: str = Field(
        default="cosine",
        description="Distance function for similarity search",
    )

    def get_server_url(self) -> str:
        """Return full ChromaDB server URL."""
        return f"http://{self.host}:{self.port}"


# ──────────────────────────────────────────────────────────────────────────────
# Ollama Settings
# ──────────────────────────────────────────────────────────────────────────────


class OllamaSettings(BaseSettings):
    """Ollama local LLM settings."""

    model_config = SettingsConfigDict(
        env_prefix="OLLAMA_",
        env_file=".env",
        extra="ignore",
        populate_by_name=True,
    )

    base_url: str = Field(
        default="http://localhost:11434",
        description="Ollama API base URL",
    )
    model: str = Field(
        default="llama3.2:3b",
        description="Default Ollama chat model",
    )
    embedding_model: str = Field(
        default="nomic-embed-text",
        description="Ollama embedding model",
    )
    timeout: int = Field(default=120, description="Request timeout in seconds")
    max_tokens: int = Field(default=2048, description="Max tokens to generate")
    temperature: float = Field(default=0.1, ge=0.0, le=2.0, description="Generation temperature")
    top_p: float = Field(default=0.9, ge=0.0, le=1.0, description="Top-p sampling")
    context_window: int = Field(default=8192, description="Context window size in tokens")


# ──────────────────────────────────────────────────────────────────────────────
# Service URL Settings
# ──────────────────────────────────────────────────────────────────────────────


class ServiceURLSettings(BaseSettings):
    """Internal service URL registry."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore", populate_by_name=True)

    ingestion_service_url: str = Field(
        default="http://ingestion:8001", alias="INGESTION_SERVICE_URL"
    )
    processing_service_url: str = Field(
        default="http://processing:8002", alias="PROCESSING_SERVICE_URL"
    )
    features_service_url: str = Field(
        default="http://features:8003", alias="FEATURES_SERVICE_URL"
    )
    forecasting_service_url: str = Field(
        default="http://forecasting:8004", alias="FORECASTING_SERVICE_URL"
    )
    xai_service_url: str = Field(
        default="http://xai:8005", alias="XAI_SERVICE_URL"
    )
    earth_impact_service_url: str = Field(
        default="http://earth-impact:8006", alias="EARTH_IMPACT_SERVICE_URL"
    )
    satellite_risk_service_url: str = Field(
        default="http://satellite-risk:8007", alias="SATELLITE_RISK_SERVICE_URL"
    )
    rag_service_url: str = Field(
        default="http://rag:8008", alias="RAG_SERVICE_URL"
    )
    copilot_service_url: str = Field(
        default="http://copilot:8009", alias="COPILOT_SERVICE_URL"
    )
    notification_service_url: str = Field(
        default="http://notifications:8010", alias="NOTIFICATION_SERVICE_URL"
    )
    gateway_url: str = Field(
        default="http://gateway:8000", alias="GATEWAY_URL"
    )


# ──────────────────────────────────────────────────────────────────────────────
# SMTP / Notification Settings
# ──────────────────────────────────────────────────────────────────────────────


class NotificationSettings(BaseSettings):
    """Email and webhook notification settings."""

    model_config = SettingsConfigDict(
        env_prefix="SMTP_", env_file=".env", extra="ignore", populate_by_name=True
    )

    host: str = Field(default="smtp.gmail.com", description="SMTP host")
    port: int = Field(default=587, description="SMTP port")
    user: str = Field(default="", description="SMTP username")
    password: SecretStr = Field(default=SecretStr(""), description="SMTP password")
    use_tls: bool = Field(default=True, description="Use TLS")
    alert_email_recipients: str = Field(
        default="admin@isro.gov.in",
        alias="ALERT_EMAIL_RECIPIENTS",
        description="Comma-separated list of alert email recipients",
    )
    webhook_url: str = Field(
        default="", alias="WEBHOOK_URL", description="Webhook URL for alerts"
    )

    def get_recipients_list(self) -> list[str]:
        """Return alert recipients as a list."""
        return [r.strip() for r in self.alert_email_recipients.split(",") if r.strip()]


# ──────────────────────────────────────────────────────────────────────────────
# Scheduler Settings
# ──────────────────────────────────────────────────────────────────────────────


class SchedulerSettings(BaseSettings):
    """APScheduler cron schedule settings."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore", populate_by_name=True)

    ingestion_cron: str = Field(
        default="*/5 * * * *",
        alias="INGESTION_CRON",
        description="Cron expression for data ingestion",
    )
    retraining_cron: str = Field(
        default="0 2 * * 0",
        alias="RETRAINING_CRON",
        description="Cron expression for model retraining (weekly)",
    )


# ──────────────────────────────────────────────────────────────────────────────
# Top-level App Settings
# ──────────────────────────────────────────────────────────────────────────────


class AppSettings(BaseSettings):
    """
    Top-level application settings composing all subsystem settings.

    All environment variables are loaded from .env file and system environment.
    Settings are cached after first load — use get_settings() to obtain them.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        populate_by_name=True,
    )

    # Application identity
    app_name: str = Field(default="AstroNova", description="Application name")
    app_version: str = Field(default="1.0.0", description="Application version")
    app_env: str = Field(
        default="development",
        alias="APP_ENV",
        description="Deployment environment (development|staging|production)",
    )
    debug: bool = Field(default=False, alias="DEBUG", description="Enable debug mode")
    log_level: str = Field(
        default="INFO", alias="LOG_LEVEL", description="Logging level"
    )
    cors_origins: str = Field(
        default="http://localhost:3000",
        alias="CORS_ORIGINS",
        description="Comma-separated allowed CORS origins",
    )
    rate_limit_per_minute: int = Field(
        default=100,
        alias="RATE_LIMIT_PER_MINUTE",
        description="Max requests per minute per client",
    )

    # File system paths
    data_dir: str = Field(
        default="/app/data", alias="DATA_DIR", description="Root data directory"
    )
    model_dir: str = Field(
        default="/app/models", alias="MODEL_DIR", description="Trained models directory"
    )
    log_dir: str = Field(
        default="/app/logs", alias="LOG_DIR", description="Log files directory"
    )

    # Subsystem settings (instantiated lazily via properties)
    _database: DatabaseSettings | None = None
    _redis: RedisSettings | None = None
    _kafka: KafkaSettings | None = None
    _jwt: JWTSettings | None = None
    _mlflow: MLflowSettings | None = None
    _chroma: ChromaSettings | None = None
    _ollama: OllamaSettings | None = None
    _services: ServiceURLSettings | None = None
    _notifications: NotificationSettings | None = None
    _scheduler: SchedulerSettings | None = None

    @property
    def database(self) -> DatabaseSettings:
        """Return (cached) database settings."""
        if self._database is None:
            self._database = DatabaseSettings()
        return self._database

    @property
    def redis(self) -> RedisSettings:
        """Return (cached) Redis settings."""
        if self._redis is None:
            self._redis = RedisSettings()
        return self._redis

    @property
    def kafka(self) -> KafkaSettings:
        """Return (cached) Kafka settings."""
        if self._kafka is None:
            self._kafka = KafkaSettings()
        return self._kafka

    @property
    def jwt(self) -> JWTSettings:
        """Return (cached) JWT settings."""
        if self._jwt is None:
            self._jwt = JWTSettings()
        return self._jwt

    @property
    def mlflow(self) -> MLflowSettings:
        """Return (cached) MLflow settings."""
        if self._mlflow is None:
            self._mlflow = MLflowSettings()
        return self._mlflow

    @property
    def chroma(self) -> ChromaSettings:
        """Return (cached) ChromaDB settings."""
        if self._chroma is None:
            self._chroma = ChromaSettings()
        return self._chroma

    @property
    def ollama(self) -> OllamaSettings:
        """Return (cached) Ollama settings."""
        if self._ollama is None:
            self._ollama = OllamaSettings()
        return self._ollama

    @property
    def services(self) -> ServiceURLSettings:
        """Return (cached) service URL settings."""
        if self._services is None:
            self._services = ServiceURLSettings()
        return self._services

    @property
    def notifications(self) -> NotificationSettings:
        """Return (cached) notification settings."""
        if self._notifications is None:
            self._notifications = NotificationSettings()
        return self._notifications

    @property
    def scheduler(self) -> SchedulerSettings:
        """Return (cached) scheduler settings."""
        if self._scheduler is None:
            self._scheduler = SchedulerSettings()
        return self._scheduler

    @field_validator("app_env")
    @classmethod
    def validate_environment(cls, v: str) -> str:
        """Ensure environment is one of the allowed values."""
        allowed = {"development", "staging", "production", "testing"}
        if v.lower() not in allowed:
            raise ValueError(f"app_env must be one of {allowed}, got '{v}'")
        return v.lower()

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """Ensure log level is a valid Python logging level."""
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        v_upper = v.upper()
        if v_upper not in allowed:
            raise ValueError(f"log_level must be one of {allowed}, got '{v}'")
        return v_upper

    def get_cors_origins_list(self) -> list[str]:
        """Return CORS origins as a list."""
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    def is_production(self) -> bool:
        """Return True if running in production."""
        return self.app_env == "production"

    def is_development(self) -> bool:
        """Return True if running in development."""
        return self.app_env == "development"

    def to_safe_dict(self) -> dict[str, Any]:
        """Return settings as dict with secrets redacted (safe for logging)."""
        raw = self.model_dump()
        # Remove private underscore fields
        raw = {k: v for k, v in raw.items() if not k.startswith("_")}
        return raw


# ──────────────────────────────────────────────────────────────────────────────
# Cached settings factory
# ──────────────────────────────────────────────────────────────────────────────


@lru_cache(maxsize=1)
def get_settings() -> AppSettings:
    """
    Return the application settings singleton.

    Settings are loaded once from environment variables and .env file,
    then cached for the lifetime of the process. In tests, use:

        from unittest.mock import patch
        with patch("astronova_core.config.get_settings") as mock_settings:
            ...

    Returns:
        AppSettings: The cached application settings instance.
    """
    logger.debug("Loading AstroNova settings from environment")
    settings = AppSettings()
    logger.info(
        "Settings loaded",
        extra={
            "app_env": settings.app_env,
            "debug": settings.debug,
            "log_level": settings.log_level,
        },
    )
    return settings


def clear_settings_cache() -> None:
    """
    Clear the settings cache (useful for testing).

    This forces the next call to get_settings() to reload from environment.
    """
    get_settings.cache_clear()
    logger.debug("Settings cache cleared")
