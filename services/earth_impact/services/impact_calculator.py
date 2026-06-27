from typing import Any


class EarthImpactCalculator:
    REGIONS = ["South-Asia", "Asia-Pacific", "Europe", "North-America", "South-America", "Africa"]

    def calculate_impact(self, goes_class: str) -> dict[str, Any]:
        # Parse flare class to estimate severity
        severity = "Low"
        base_score = 0.1
        if goes_class.startswith("M"):
            severity = "Medium"
            base_score = 0.5
        elif goes_class.startswith("X"):
            severity = "High"
            base_score = 0.9

        regional_risks = []
        for region in self.REGIONS:
            # day-side gets higher risk factor;South-Asia (ISRO operations focus) highlighted
            is_day = region in ["South-Asia", "Asia-Pacific"]
            factor = 1.2 if is_day else 0.4
            risk_score = min(base_score * factor, 1.0)
            regional_risks.append({
                "region": region,
                "risk_score": risk_score,
                "population_exposure": 0.8 if region == "South-Asia" else 0.5,
                "infrastructure_risk": 0.7 if region == "South-Asia" else 0.4
            })

        return {
            "overall_severity": severity,
            "geomagnetic_storm_probability": 0.85 if goes_class.startswith("X") else 0.3,
            "regional_risks": regional_risks
        }
