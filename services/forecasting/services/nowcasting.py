from ml.models.nowcasting import ThresholdDetector
from astronova_core.utils.physics import classify_flare, track_lifecycle_phase
from typing import List, Optional

class NowcastingService:
    def __init__(self):
        self.detector = ThresholdDetector(threshold=1e-5) # M-class threshold

    def analyze_nowcast(self, current_flux: float, flux_history: Optional[List[float]] = None) -> dict:
        """
        Performs nowcasting analysis with event lifecycle tracking and dynamic lead-time optimization.
        """
        is_flare = self.detector.detect(current_flux)
        goes_class = classify_flare(current_flux)
        
        # Resolve flux history for lifecycle tracking (default to replicating current_flux if None)
        if not flux_history:
            flux_history = [current_flux] * 5
        elif len(flux_history) < 5:
            flux_history = [current_flux] * (5 - len(flux_history)) + list(flux_history)
            
        lifecycle_phase = track_lifecycle_phase(flux_history)
        
        # Dynamic Lead-Time and Cadence Optimization based on lifecycle phase
        if lifecycle_phase in ["Pre-flare", "Rise"]:
            recommended_polling_interval_seconds = 10
            optimal_lead_time_minutes = 5
            cadence_urgency = "High"
            action_required = "Immediate Warning Dispatch / Sensor Cadence Optimization Active"
        elif lifecycle_phase == "Peak":
            recommended_polling_interval_seconds = 30
            optimal_lead_time_minutes = 15
            cadence_urgency = "Medium-High"
            action_required = "Continuous Peak Flux Tracking / Secondary Impact Scans"
        elif lifecycle_phase == "Decay":
            recommended_polling_interval_seconds = 60
            optimal_lead_time_minutes = 30
            cadence_urgency = "Medium"
            action_required = "Cooling Phase Tracking"
        else:  # Quiescent
            recommended_polling_interval_seconds = 300  # 5 minutes
            optimal_lead_time_minutes = 60
            cadence_urgency = "Low"
            action_required = "Routine Telemetry Monitoring"

        return {
            "is_flare": is_flare,
            "goes_class": goes_class,
            "confidence": 0.95 if is_flare else 0.99,
            "detection_method": "threshold_detector",
            "peak_flux": current_flux,
            "event_lifecycle": {
                "current_phase": lifecycle_phase,
                "dynamic_lead_time_minutes": optimal_lead_time_minutes,
                "recommended_cadence_seconds": recommended_polling_interval_seconds,
                "cadence_urgency_level": cadence_urgency,
                "operational_directive": action_required
            }
        }
