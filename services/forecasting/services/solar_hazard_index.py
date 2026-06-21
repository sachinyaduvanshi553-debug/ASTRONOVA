from astronova_core.utils.physics import compute_shi
from typing import Dict

class SolarHazardIndexCalculator:
    @staticmethod
    def calculate_shi(
        probabilities: dict,
        gradient: float,
        similarity: float = 0.1,
        sat_risk: float = 0.1,
        impact_risk: float = 0.1
    ) -> dict:
        """
        Calculates Solar Hazard Index (SHI) using the standardized, multi-factor physics formula:
        SHI = 0.35 * Flare_Prob + 0.25 * Flux_Growth + 0.15 * Similarity + 0.15 * Sat_Risk + 0.10 * Earth_Impact
        """
        # Composite flare probability weight: 70% X-class, 30% M-class
        x_prob = probabilities.get("X", 0.0)
        m_prob = probabilities.get("M", 0.0)
        composite_prob = (x_prob * 0.7) + (m_prob * 0.3)
        
        # Calculate score using shared core physics
        score = compute_shi(
            prob=composite_prob,
            growth=gradient,
            similarity=similarity,
            sat_risk=sat_risk,
            impact_risk=impact_risk
        )
        
        if score < 0.2:
            category = "Safe"
        elif score < 0.5:
            category = "Moderate"
        elif score < 0.8:
            category = "High"
        else:
            category = "Extreme"
            
        return {
            "score": score,
            "category": category,
            "components": {
                "composite_flare_probability": composite_prob,
                "flux_gradient": gradient,
                "historical_similarity": similarity,
                "satellite_risk_factor": sat_risk,
                "earth_impact_risk_factor": impact_risk
            }
        }
