import os
from unittest.mock import MagicMock

import pandas as pd
import pytest
from services.ingestion.services.ingestion_service import IngestionService


@pytest.mark.asyncio
async def test_ingest_file_csv_success():
    # Setup mock DB session
    db_session = MagicMock()

    # Create temporary CSV file for testing
    csv_path = "test_data.csv"
    df = pd.DataFrame({
        "time": ["2026-06-21T12:00:00", "2026-06-21T12:01:00"],
        "soft_xray_flux": [1.2e-8, 1.5e-8],
        "hard_xray_flux": [1.2e-9, 1.5e-9]
    })
    df.to_csv(csv_path, index=False)

    try:
        service = IngestionService()
        # Mock kafka producer within the service
        service.producer = MagicMock()

        job = await service.ingest_file(csv_path, "csv", db_session)

        assert job.status == "completed"
        assert job.rows_ingested == 2
        assert db_session.add.called
        assert db_session.commit.called
    finally:
        if os.path.exists(csv_path):
            os.remove(csv_path)

@pytest.mark.asyncio
async def test_ingest_file_missing_columns():
    db_session = MagicMock()
    csv_path = "test_invalid_data.csv"
    # missing soft_xray_flux
    df = pd.DataFrame({
        "time": ["2026-06-21T12:00:00"],
        "hard_xray_flux": [1.2e-9]
    })
    df.to_csv(csv_path, index=False)

    try:
        service = IngestionService()
        service.producer = MagicMock()

        job = await service.ingest_file(csv_path, "csv", db_session)

        assert job.status == "failed"
        assert len(job.errors) > 0
        assert "Missing required column" in job.errors[0]
    finally:
        if os.path.exists(csv_path):
            os.remove(csv_path)
