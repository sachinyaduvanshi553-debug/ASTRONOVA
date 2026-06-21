import os

def create_file(path, content):
    dirname = os.path.dirname(path)
    if dirname:
        os.makedirs(dirname, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"Created: {path}")

# --- 1. services/ingestion/tests/test_ingestion_service.py ---
create_file("services/ingestion/tests/test_ingestion_service.py", """import pytest
import os
import pandas as pd
from unittest.mock import MagicMock
from services.ingestion.services.ingestion_service import IngestionService
from astronova_core.models.timeseries import SolexsObservation

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
""")

# --- 2. services/forecasting/tests/test_forecasting.py ---
create_file("services/forecasting/tests/test_forecasting.py", """import pytest
from services.forecasting.services.nowcasting import NowcastingService
from services.forecasting.services.solar_hazard_index import SolarHazardIndexCalculator
from services.forecasting.services.inference_engine import InferenceEngine

def test_nowcast_detection():
    service = NowcastingService()
    # Below M-class threshold
    res = service.analyze_nowcast(1e-6)
    assert res["is_flare"] is False
    assert "C" in res["goes_class"]
    
    # Above M-class threshold
    res = service.analyze_nowcast(1.5e-5)
    assert res["is_flare"] is True
    assert "M" in res["goes_class"]

def test_solar_hazard_index():
    probabilities = {"A": 0.05, "B": 0.05, "C": 0.1, "M": 0.5, "X": 0.3}
    gradient = 1.2e-5
    
    shi = SolarHazardIndexCalculator.calculate_shi(probabilities, gradient)
    assert 0.0 <= shi["score"] <= 1.0
    assert shi["category"] in ["Safe", "Moderate", "High", "Extreme"]
    assert "x_flare_probability" in shi["components"]

def test_inference_engine_prediction():
    engine = InferenceEngine()
    res = engine.predict([])
    
    assert "prediction" in res
    assert "horizon_minutes" in res["prediction"]
    assert "probabilities" in res["prediction"]
    assert "predicted_class" in res["prediction"]
""")

print("TESTS GENERATED SUCCESSFULLY")
