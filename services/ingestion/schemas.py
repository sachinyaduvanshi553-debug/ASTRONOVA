from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional, List

class IngestionJobResponse(BaseModel):
    job_id: str
    status: str
    source_file: str
    rows_ingested: int
    started_at: datetime
    completed_at: Optional[datetime] = None

class BulkIngestionRequest(BaseModel):
    directory_path: str

class ScheduleRequest(BaseModel):
    cron_expression: str
