from fastapi import APIRouter, Query
from typing import Dict, List

router = APIRouter(prefix="/api/v1/comms", tags=["comms-impact"])

@router.get("/assess")
async def assess_comms_impact(
    goes_class: str = Query(..., description="GOES Class of predicted flare"),
    lifecycle_phase: str = Query("Quiescent", description="Current solar flare lifecycle phase")
):
    """
    Performs comms impact assessment and generates operational recommendations 
    tailored to the flare severity and active lifecycle phase.
    """
    severity = "Low"
    absorption_db = 1.2
    scintillation_s4 = 0.15
    
    if goes_class.startswith("M"):
        severity = "Moderate"
        absorption_db = 8.5
        scintillation_s4 = 0.45
    elif goes_class.startswith("X"):
        severity = "Critical"
        absorption_db = 22.4
        scintillation_s4 = 0.85
        
    # Standard impact assessment data
    comms_assessment = {
        "gps_degradation": {
            "severity": severity,
            "position_error_increase_meters": 1.5 if severity == "Low" else (5.4 if severity == "Moderate" else 14.8),
            "confidence": 0.89
        },
        "navic_degradation": {
            "severity": severity,
            "s4_index": scintillation_s4,
            "scintillation_warning": bool(scintillation_s4 >= 0.4),
            "confidence": 0.92
        },
        "hf_radio": {
            "dellinger_absorption_db": absorption_db,
            "blackout_probability": 0.15 if severity == "Low" else (0.65 if severity == "Moderate" else 0.98),
            "affected_frequency_mhz_ceiling": 15 if severity == "Low" else (25 if severity == "Moderate" else 35)
        }
    }
    
    # Operational recommendation generation based on severity & lifecycle phase
    recommendations = []
    
    if severity == "Critical":
        if lifecycle_phase in ["Rise", "Pre-flare"]:
            recommendations.append({
                "system": "GSAT GEO Satellites",
                "status": "Red-Alert / Safing Mode",
                "mitigation": "Disable non-essential transponders immediately. Rotate solar panels to minimum proton collision cross-section. Engage backup star-trackers."
            })
            recommendations.append({
                "system": "NavIC Receivers",
                "status": "Red-Alert / Scintillation Imminent",
                "mitigation": "Expand receiver tracking loop bandwidth. Switch to dual-frequency (L5 + S-band) ionospheric delay mitigation algorithms."
            })
            recommendations.append({
                "system": "HF Radio Networks",
                "status": "Blackout Warning",
                "mitigation": "Suspend high-frequency communications. Reroute polar aviation routes to sub-auroral zones. Switch to SatCom links."
            })
            recommendations.append({
                "system": "Power Grids",
                "status": "GIC Vulnerability High",
                "mitigation": "Deploy series capacitor banks. Monitor transformer neutral currents for Geomagnetically Induced Currents (GIC) indicators."
            })
        elif lifecycle_phase == "Peak":
            recommendations.append({
                "system": "All Systems",
                "status": "Maximum Exposure Peak",
                "mitigation": "Maintain safing states. Prevent hot-swapping or re-engaging primary payloads until decay phase is established."
            })
        else:  # Decay / Quiescent
            recommendations.append({
                "system": "All Systems",
                "status": "Post-event Audit",
                "mitigation": "Perform electrostatic discharge (ESD) diagnosis on space assets. Review signal acquisition logs and restore normal operational states slowly."
            })
    elif severity == "Moderate":
        if lifecycle_phase in ["Rise", "Pre-flare", "Peak"]:
            recommendations.append({
                "system": "GSAT GEO Satellites",
                "status": "Amber Warning",
                "mitigation": "Monitor transponder temperature anomalies. Prepare to isolate charging-vulnerable subsystems."
            })
            recommendations.append({
                "system": "NavIC Receivers",
                "status": "Amber Warning",
                "mitigation": "Engage scintillation detection flags. Warn users of potential 5-10m GNSS positioning errors."
            })
            recommendations.append({
                "system": "HF Radio Networks",
                "status": "Degradation Active",
                "mitigation": "Utilize lower-frequency bands below Dellinger absorption ceiling."
            })
        else:
            recommendations.append({
                "system": "All Systems",
                "status": "Green / Monitoring",
                "mitigation": "No immediate actions required. Continue nominal operations."
            })
    else:  # Low
        recommendations.append({
            "system": "All Systems",
            "status": "Green / Nominal",
            "mitigation": "Routine space weather conditions. Standard telemetry polling active."
        })

    return {
        "assessment": comms_assessment,
        "lifecycle_phase": lifecycle_phase,
        "operational_recommendations": recommendations
    }

@router.get("/health")
def health():
    return {"status": "healthy"}
