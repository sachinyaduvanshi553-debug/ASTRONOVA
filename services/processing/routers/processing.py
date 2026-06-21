"""
Processing Router
=================
REST API endpoints for the AstroNova Data Processing Service.

Endpoints
---------
POST /api/v1/process/run          – Trigger processing for a time range
GET  /api/v1/process/status/{id}  – Poll processing job status
GET  /api/v1/process/pipelines    – List available pipelines and their configs
POST /api/v1/process/configure    – Update pipeline configuration at runtime
GET  /api/v1/process/health       – Router-level health check
"""

from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import structlog
from fastapi import APIRouter, BackgroundTasks, HTTPException, Query, status
from pydantic import BaseModel, Field, field_validator

log = structlog.get_logger(__name__)

router = APIRouter(prefix="/process", tags=["Processing"])

# ---------------------------------------------------------------------------
# In-memory job registry (replace with Redis/DB in production)
# ---------------------------------------------------------------------------
_job_registry: Dict[str, Dict[str, Any]] = {}


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------
class TimeRange(BaseModel):
    """UTC time range for a processing run."""

    start: datetime = Field(..., description="Start of the processing window (UTC)")
    end: datetime = Field(..., description="End of the processing window (UTC)")
    pipeline_ids: Optional[List[str]] = Field(
        default=None,
        description="Pipeline IDs to run. None → run all enabled pipelines.",
    )
    instrument: str = Field(
        default="solexs",
        description="Instrument identifier: solexs | helios",
    )
    priority: int = Field(default=5, ge=1, le=10, description="Job priority 1-10")

    @field_validator("end")
    @classmethod
    def end_after_start(cls, v: datetime, info: Any) -> datetime:
        if "start" in info.data and v <= info.data["start"]:
            raise ValueError("end must be strictly after start")
        return v


class ProcessingJobResponse(BaseModel):
    """Response returned when a processing job is accepted."""

    job_id: str
    status: str
    message: str
    submitted_at: datetime
    estimated_duration_seconds: Optional[float] = None


class JobStatus(BaseModel):
    """Detailed job status."""

    job_id: str
    status: str  # queued | running | completed | failed | cancelled
    progress_pct: float = Field(ge=0, le=100)
    submitted_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error: Optional[str] = None
    stats: Optional[Dict[str, Any]] = None


class PipelineInfo(BaseModel):
    """Metadata about a registered pipeline."""

    pipeline_id: str
    name: str
    description: str
    enabled: bool
    version: str
    parameters: Dict[str, Any]


class PipelineConfigUpdate(BaseModel):
    """Payload for updating pipeline configuration."""

    pipeline_id: str = Field(..., description="Pipeline to update")
    parameters: Dict[str, Any] = Field(..., description="New parameter overrides")
    enabled: Optional[bool] = Field(
        default=None, description="Enable or disable the pipeline"
    )


# ---------------------------------------------------------------------------
# Available pipelines registry (in-memory, seeded at startup)
# ---------------------------------------------------------------------------
_PIPELINES: Dict[str, Dict[str, Any]] = {
    "cleaning": {
        "pipeline_id": "cleaning",
        "name": "Data Cleaning Pipeline",
        "description": (
            "Removes duplicates, fills missing values, detects & handles "
            "outliers, validates physical constraints, and fixes timestamp gaps."
        ),
        "enabled": True,
        "version": "1.0.0",
        "parameters": {
            "missing_strategy": "interpolate",
            "outlier_method": "iqr",
            "outlier_action": "cap",
            "iqr_threshold": 1.5,
            "zscore_threshold": 3.0,
            "max_gap_minutes": 5,
        },
    },
    "smoothing": {
        "pipeline_id": "smoothing",
        "name": "Noise Smoothing Pipeline",
        "description": (
            "Applies Savitzky-Golay, wavelet denoising, Gaussian and moving-average "
            "filters to reduce instrumental noise while preserving flare morphology."
        ),
        "enabled": True,
        "version": "1.0.0",
        "parameters": {
            "methods": ["savgol", "wavelet"],
            "savgol_window": 11,
            "savgol_polyorder": 3,
            "wavelet": "db4",
            "wavelet_level": 3,
            "wavelet_threshold": "soft",
            "gaussian_sigma": 2.0,
            "moving_average_window": 5,
        },
    },
    "normalization": {
        "pipeline_id": "normalization",
        "name": "Flux Normalization Pipeline",
        "description": (
            "Applies log10 transform (log10(flux + 1e-12)), Z-score normalization, "
            "min-max scaling and robust scaling. Stores scaler parameters for "
            "inverse transform during prediction."
        ),
        "enabled": True,
        "version": "1.0.0",
        "parameters": {
            "method": "log_then_zscore",
            "log_base": 10,
            "log_offset": 1e-12,
            "feature_range": [0.0, 1.0],
        },
    },
    "resampling": {
        "pipeline_id": "resampling",
        "name": "Time-Series Resampling Pipeline",
        "description": (
            "Resamples raw data to a uniform cadence, synchronises SoLEXS and "
            "HELIOS channels, and interpolates short gaps."
        ),
        "enabled": True,
        "version": "1.0.0",
        "parameters": {
            "target_freq": "1T",
            "max_gap_samples": 5,
            "interpolation_method": "cubic",
        },
    },
}


# ---------------------------------------------------------------------------
# Background worker (simplified – real impl uses Celery/Ray)
# ---------------------------------------------------------------------------
async def _run_processing_job(
    job_id: str,
    time_range: TimeRange,
) -> None:
    """
    Background task that executes the processing pipelines.
    Updates the job registry with progress and results.
    """
    _job_registry[job_id]["status"] = "running"
    _job_registry[job_id]["started_at"] = datetime.now(timezone.utc)

    log.info(
        "processing_job.started",
        job_id=job_id,
        start=str(time_range.start),
        end=str(time_range.end),
    )

    try:
        # Determine which pipelines to run
        pipeline_ids = time_range.pipeline_ids or list(_PIPELINES.keys())
        enabled_ids = [
            pid for pid in pipeline_ids if _PIPELINES.get(pid, {}).get("enabled", False)
        ]

        total = len(enabled_ids)
        stats: Dict[str, Any] = {"pipelines_run": [], "rows_processed": 0}

        for idx, pid in enumerate(enabled_ids, start=1):
            # Simulate pipeline execution (replace with actual pipeline calls)
            await asyncio.sleep(0.1)

            progress = round((idx / total) * 100, 1)
            _job_registry[job_id]["progress_pct"] = progress

            stats["pipelines_run"].append(pid)
            log.info(
                "processing_job.pipeline_done",
                job_id=job_id,
                pipeline=pid,
                progress=progress,
            )

        stats["rows_processed"] = 1440  # placeholder: 1 day × 1-min cadence
        _job_registry[job_id]["status"] = "completed"
        _job_registry[job_id]["completed_at"] = datetime.now(timezone.utc)
        _job_registry[job_id]["progress_pct"] = 100.0
        _job_registry[job_id]["stats"] = stats

        log.info("processing_job.completed", job_id=job_id, stats=stats)

    except Exception as exc:  # noqa: BLE001
        _job_registry[job_id]["status"] = "failed"
        _job_registry[job_id]["error"] = str(exc)
        _job_registry[job_id]["completed_at"] = datetime.now(timezone.utc)
        log.error("processing_job.failed", job_id=job_id, error=str(exc))


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@router.post(
    "/run",
    response_model=ProcessingJobResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Trigger processing for a time range",
)
async def run_processing(
    body: TimeRange,
    background_tasks: BackgroundTasks,
) -> ProcessingJobResponse:
    """
    Submit a new processing job for the specified UTC time range.

    The job is executed asynchronously. Use `GET /status/{job_id}` to
    track progress.

    **Pipeline order**: cleaning → smoothing → normalization → resampling
    """
    job_id = str(uuid.uuid4())
    submitted_at = datetime.now(timezone.utc)

    # Rough estimation: 1 second per day of data, minimum 1 s
    delta_days = max(
        (body.end - body.start).total_seconds() / 86400, 1 / 1440
    )
    estimated = max(round(delta_days * 1.0, 2), 0.1)

    _job_registry[job_id] = {
        "job_id": job_id,
        "status": "queued",
        "progress_pct": 0.0,
        "submitted_at": submitted_at,
        "started_at": None,
        "completed_at": None,
        "error": None,
        "stats": None,
        "request": body.model_dump(),
    }

    background_tasks.add_task(_run_processing_job, job_id, body)

    log.info(
        "processing_job.accepted",
        job_id=job_id,
        start=str(body.start),
        end=str(body.end),
        instrument=body.instrument,
    )

    return ProcessingJobResponse(
        job_id=job_id,
        status="queued",
        message=(
            f"Processing job accepted. Instrument: {body.instrument}. "
            f"Pipelines: {body.pipeline_ids or 'all'}."
        ),
        submitted_at=submitted_at,
        estimated_duration_seconds=estimated,
    )


@router.get(
    "/status/{job_id}",
    response_model=JobStatus,
    summary="Get processing job status",
)
async def get_job_status(job_id: str) -> JobStatus:
    """
    Return the current status of a processing job.

    **Possible statuses**: `queued` | `running` | `completed` | `failed` | `cancelled`
    """
    if job_id not in _job_registry:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job '{job_id}' not found.",
        )
    job = _job_registry[job_id]
    return JobStatus(
        job_id=job_id,
        status=job["status"],
        progress_pct=job["progress_pct"],
        submitted_at=job["submitted_at"],
        started_at=job["started_at"],
        completed_at=job["completed_at"],
        error=job.get("error"),
        stats=job.get("stats"),
    )


@router.get(
    "/pipelines",
    response_model=List[PipelineInfo],
    summary="List available processing pipelines",
)
async def list_pipelines(
    enabled_only: bool = Query(default=False, description="Filter to enabled pipelines only"),
) -> List[PipelineInfo]:
    """
    Return metadata for all registered data processing pipelines,
    including their current configuration parameters.
    """
    results = list(_PIPELINES.values())
    if enabled_only:
        results = [p for p in results if p["enabled"]]
    return [PipelineInfo(**p) for p in results]


@router.post(
    "/configure",
    response_model=PipelineInfo,
    summary="Update pipeline configuration",
)
async def configure_pipeline(body: PipelineConfigUpdate) -> PipelineInfo:
    """
    Update the runtime configuration of a pipeline.

    Changes take effect immediately for new jobs. Running jobs are unaffected.

    Example body:
    ```json
    {
        "pipeline_id": "cleaning",
        "parameters": {"iqr_threshold": 2.0, "max_gap_minutes": 10},
        "enabled": true
    }
    ```
    """
    if body.pipeline_id not in _PIPELINES:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Pipeline '{body.pipeline_id}' not found.",
        )

    pipeline = _PIPELINES[body.pipeline_id]
    pipeline["parameters"].update(body.parameters)
    if body.enabled is not None:
        pipeline["enabled"] = body.enabled

    log.info(
        "pipeline.config_updated",
        pipeline_id=body.pipeline_id,
        new_params=body.parameters,
        enabled=body.enabled,
    )

    return PipelineInfo(**pipeline)


@router.get(
    "/health",
    summary="Router-level health check",
    tags=["Health"],
)
async def processing_health() -> Dict[str, Any]:
    """
    Lightweight health check that confirms the processing router is reachable
    and returns the count of jobs currently tracked.
    """
    queued = sum(1 for j in _job_registry.values() if j["status"] == "queued")
    running = sum(1 for j in _job_registry.values() if j["status"] == "running")
    completed = sum(1 for j in _job_registry.values() if j["status"] == "completed")
    failed = sum(1 for j in _job_registry.values() if j["status"] == "failed")

    return {
        "status": "ok",
        "pipelines_registered": len(_PIPELINES),
        "jobs": {
            "queued": queued,
            "running": running,
            "completed": completed,
            "failed": failed,
            "total": len(_job_registry),
        },
    }
