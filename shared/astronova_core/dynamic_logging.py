import asyncio
import json
import logging
import time
import uuid
from collections import deque
from datetime import datetime
from typing import Any, Optional

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from astronova_core.logging import get_logger

logger = get_logger("dynamic-logging")

LOG_LEVEL_MAP = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}


class DynamicLoggingManager:
    """Manages dynamic runtime log levels and in-memory log buffer."""

    def __init__(self, max_buffer_size: int = 1000):
        self._log_levels: dict[str, str] = {
            "global": "INFO",
            "gateway": "INFO",
            "ingestion": "INFO",
            "forecasting": "INFO",
            "copilot": "INFO",
            "physics_engine": "INFO",
            "solar_vision": "INFO",
        }
        self._buffer: deque[dict[str, Any]] = deque(maxlen=max_buffer_size)

    def set_level(self, service_name: str, level: str) -> str:
        level_upper = level.upper()
        if level_upper not in LOG_LEVEL_MAP:
            raise ValueError(f"Invalid log level: {level}. Allowed: {list(LOG_LEVEL_MAP.keys())}")
        
        self._log_levels[service_name] = level_upper
        
        # Update python root logger level if global or matching
        py_level = LOG_LEVEL_MAP[level_upper]
        logging.getLogger().setLevel(py_level)
        logger.info("dynamic_log_level_changed", service=service_name, level=level_upper)
        return level_upper

    def get_level(self, service_name: str = "global") -> str:
        return self._log_levels.get(service_name, self._log_levels.get("global", "INFO"))

    def get_all_levels(self) -> dict[str, str]:
        return dict(self._log_levels)

    def push_log(
        self,
        service_name: str,
        level: str,
        category: str,
        message: str,
        method: Optional[str] = None,
        path: Optional[str] = None,
        status_code: Optional[int] = None,
        duration_ms: Optional[float] = None,
        sql_query: Optional[str] = None,
        correlation_id: Optional[str] = None,
        extra_data: Optional[Any] = None,
    ) -> dict[str, Any]:
        log_entry = {
            "id": str(uuid.uuid4()),
            "timestamp": datetime.utcnow().isoformat(),
            "service_name": service_name,
            "level": level.upper(),
            "category": category,
            "message": message,
            "method": method,
            "path": path,
            "status_code": status_code,
            "duration_ms": duration_ms,
            "sql_query": sql_query,
            "correlation_id": correlation_id,
            "extra_data": json.dumps(extra_data) if isinstance(extra_data, (dict, list)) else (str(extra_data) if extra_data else None),
        }
        self._buffer.appendleft(log_entry)
        return log_entry

    def get_recent_logs(
        self,
        service_name: Optional[str] = None,
        level: Optional[str] = None,
        category: Optional[str] = None,
        search: Optional[str] = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        results = []
        for entry in self._buffer:
            if service_name and entry["service_name"].lower() != service_name.lower():
                continue
            if level and entry["level"].lower() != level.lower():
                continue
            if category and entry["category"].lower() != category.lower():
                continue
            if search and search.lower() not in entry["message"].lower() and search.lower() not in (entry.get("path") or "").lower():
                continue
            results.append(entry)
            if len(results) >= limit:
                break
        return results

    def clear_logs(self) -> int:
        count = len(self._buffer)
        self._buffer.clear()
        return count


# Singleton instance
dynamic_logger_manager = DynamicLoggingManager()


class DynamicLoggingMiddleware(BaseHTTPMiddleware):
    """FastAPI Middleware to dynamically log requests, responses, and latency."""

    def __init__(self, app, service_name: str = "gateway"):
        super().__init__(app)
        self.service_name = service_name

    async def dispatch(self, request: Request, call_next) -> Response:
        start_time = time.time()
        correlation_id = request.headers.get("X-Correlation-ID", str(uuid.uuid4()))
        request.state.correlation_id = correlation_id

        response: Response = await call_next(request)
        duration_ms = round((time.time() - start_time) * 1000, 2)

        # Log request dynamically
        level = "ERROR" if response.status_code >= 500 else ("WARNING" if response.status_code >= 400 else "INFO")
        
        dynamic_logger_manager.push_log(
            service_name=self.service_name,
            level=level,
            category="HTTP_REQUEST",
            message=f"{request.method} {request.url.path} -> {response.status_code} ({duration_ms}ms)",
            method=request.method,
            path=request.url.path,
            status_code=response.status_code,
            duration_ms=duration_ms,
            correlation_id=correlation_id,
        )

        response.headers["X-Correlation-ID"] = correlation_id
        response.headers["X-Dynamic-Log-Level"] = dynamic_logger_manager.get_level(self.service_name)
        return response
