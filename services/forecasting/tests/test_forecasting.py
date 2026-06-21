import pytest
from services.forecasting.services.nowcasting import NowcastingService
from services.forecasting.services.solar_hazard_index import SolarHazardIndexCalculator
from services.forecasting.services.inference_engine import InferenceEngine
from astronova_core.utils.physics import apply_physics_constraints, track_lifecycle_phase, compute_shi

def test_nowcast_detection():
    service = NowcastingService()
    # Below M-class threshold
    res = service.analyze_nowcast(1e-6)
    assert res["is_flare"] is False
    assert "C" in res["goes_class"]
    assert "event_lifecycle" in res
    assert res["event_lifecycle"]["current_phase"] == "Quiescent"
    assert res["event_lifecycle"]["recommended_cadence_seconds"] == 300
    
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
    res = engine.predict([], current_flux=1e-7)
    
    assert "prediction" in res
    assert "horizons" in res
    assert "30" in res["horizons"]
    
    pred_30 = res["horizons"]["30"]
    assert "predicted_class" in pred_30
    assert "quantile_10" in pred_30
    assert "quantile_90" in pred_30
    assert "standard_deviation" in pred_30
    assert pred_30["quantile_10"] <= pred_30["peak_flux_estimate"] <= pred_30["quantile_90"]

def test_physics_constraints():
    # Test normal growth (under limit)
    # limit is 1e-3 W/m^2 per minute.
    # At 5 minute horizon, max allowed growth is 5 * 1e-3 = 5e-3.
    # predicted = 2e-7, current = 1e-7, growth = 1e-7 (well within limit)
    val = apply_physics_constraints(current_flux=1e-7, predicted_flux=2e-7, dt_minutes=5.0)
    assert val == 2e-7

    # Test extreme anomalous growth (exceeds limit)
    # predicted = 1.0, current = 1e-7, growth = 0.999 (violates limit)
    val = apply_physics_constraints(current_flux=1e-7, predicted_flux=1.0, dt_minutes=5.0)
    # should be clamped to current + max_growth = 1e-7 + 5e-3 = 0.0050001
    assert val == pytest.approx(1e-7 + 5e-3)

def test_lifecycle_tracking():
    # Quiescent
    series_q = [1e-8, 1.1e-8, 1.2e-8, 1.1e-8, 1e-8]
    assert track_lifecycle_phase(series_q) == "Quiescent"
    
    # Rise (grad[-1] > 1e-5)
    series_r = [1e-8, 1e-7, 1e-6, 1e-5, 3e-5]
    assert track_lifecycle_phase(series_r) == "Rise"
    
    # Decay (grad[-1] < -1e-6)
    series_d = [5e-5, 4e-5, 3e-5, 2e-5, 1e-5]
    assert track_lifecycle_phase(series_d) == "Decay"

def test_dynamic_lead_time_opt():
    service = NowcastingService()
    # Rising series
    flux_history = [1e-8, 1e-7, 1e-6, 1e-5, 3e-5]
    res = service.analyze_nowcast(current_flux=3e-5, flux_history=flux_history)
    
    assert res["event_lifecycle"]["current_phase"] == "Rise"
    assert res["event_lifecycle"]["recommended_cadence_seconds"] == 10
    assert res["event_lifecycle"]["dynamic_lead_time_minutes"] == 5
    assert "Immediate Warning Dispatch" in res["event_lifecycle"]["operational_directive"]
