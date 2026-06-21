from typing import Dict, Any, List

class SatelliteRiskCalculator:
    ORBITS = ["LEO", "MEO", "GEO", "HEO"]
    SATELLITES = ["GSAT", "INSAT", "EOS", "Cartosat"]

    def calculate_satellite_risk(self, goes_class: str) -> Dict[str, Any]:
        severity = 0.2
        if goes_class.startswith("M"):
            severity = 0.5
        elif goes_class.startswith("X"):
            severity = 0.9

        satellite_risks = []
        for sat in self.SATELLITES:
            orbit = "GEO" if sat.startswith("INSAT") or sat.startswith("GSAT") else "LEO"
            risk = severity * (0.8 if orbit == "LEO" else 0.6)
            
            mitigations = []
            if risk > 0.4:
                mitigations.append("Prepare backup attitude control systems")
            if risk > 0.7:
                mitigations.append("Enter safe-hold mode")
                mitigations.append("Disable non-essential instruments")
                
            satellite_risks.append({
                "satellite_id": sat,
                "orbit_type": orbit,
                "risk_score": min(risk, 1.0),
                "mitigation_actions": mitigations if mitigations else ["Nominal Operations"]
            })

        # Comms disruption predictions
        comms_disruption = [
            {
                "system_type": "GPS/NavIC",
                "severity": "High" if severity > 0.7 else ("Medium" if severity > 0.4 else "Low"),
                "confidence": 0.85,
                "affected_regions": ["South-Asia", "Asia-Pacific"]
            },
            {
                "system_type": "HF Radio",
                "severity": "Critical" if severity > 0.7 else ("High" if severity > 0.4 else "Low"),
                "confidence": 0.9,
                "affected_regions": ["Global Day-side"]
            }
        ]

        return {
            "satellite_risks": satellite_risks,
            "communication_disruption": comms_disruption
        }
