import pytest
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
    assert "composite_flare_probability" in shi["components"]

def test_inference_engine_prediction():
    engine = InferenceEngine()
    res = engine.predict([])
    
    assert "prediction" in res
    assert "probabilities" in res["prediction"]
    assert "predicted_class" in res["prediction"]
