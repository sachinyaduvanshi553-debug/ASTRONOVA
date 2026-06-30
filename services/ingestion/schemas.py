from datetime import datetime

from pydantic import BaseModel


class IngestionJobResponse(BaseModel):
    job_id: str
    status: str
    source_file: str
    rows_ingested: int
    started_at: datetime
    completed_at: datetime | None = None

class BulkIngestionRequest(BaseModel):
    directory_path: str

class ScheduleRequest(BaseModel):
    cron_expression: str
