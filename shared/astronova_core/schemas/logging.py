from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]


class LogEntryCreate(BaseModel):
    service_name: str = Field(..., description="Service producing the log")
    level: LogLevel = Field("INFO", description="Log severity level")
    category: str = Field("SYSTEM", description="Category e.g. HTTP_REQUEST, DB_QUERY, SYSTEM")
    message: str = Field(..., description="Log message text")
    method: str | None = None
    path: str | None = None
    status_code: int | None = None
    duration_ms: float | None = None
    sql_query: str | None = None
    correlation_id: str | None = None
    extra_data: str | None = None


class LogEntryResponse(BaseModel):
    id: str
    timestamp: datetime
    service_name: str
    level: str
    category: str
    message: str
    method: str | None = None
    path: str | None = None
    status_code: int | None = None
    duration_ms: float | None = None
    sql_query: str | None = None
    correlation_id: str | None = None
    extra_data: str | None = None


class LogLevelUpdate(BaseModel):
    service_name: str = Field(..., description="Target service name or 'global'")
    level: LogLevel = Field(..., description="New dynamic log level")


class LogLevelResponse(BaseModel):
    service_name: str
    current_level: LogLevel
    all_service_levels: dict[str, str]


class DynamicQueryRequest(BaseModel):
    query: str = Field(..., description="Read-only SQL query (SELECT statements only)")
    limit: int = Field(100, ge=1, le=1000, description="Max rows to return")


class DynamicQueryResponse(BaseModel):
    query: str
    row_count: int
    columns: list[str]
    rows: list[dict[str, Any]]
    execution_time_ms: float


class DatabaseTableSummary(BaseModel):
    table_name: str
    column_count: int
    estimated_row_count: int
    columns: list[dict[str, str]]


class DatabaseStatsResponse(BaseModel):
    database_type: str
    total_tables: int
    tables: list[DatabaseTableSummary]
    total_dynamic_logs: int
    active_connections: int
