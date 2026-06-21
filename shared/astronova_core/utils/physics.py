from typing import Dict, Tuple, List
import numpy as np

GOES_THRESHOLDS: Dict[str, Tuple[float, float]] = {
    'A': (1e-8, 1e-7),
    'B': (1e-7, 1e-6),
    'C': (1e-6, 1e-5),
    'M': (1e-5, 1e-4),
    'X': (1e-4, float('inf')),
}

def classify_flare(flux: float) -> str:
    for classification, (low, high) in GOES_THRESHOLDS.items():
        if low <= flux < high:
            multiplier = flux / low
            return f"{classification}{multiplier:.1f}"
    if flux < 1e-8:
        return "A0.0"
    return "X10.0"

def compute_xray_ratio(soft_flux: float, hard_flux: float) -> float:
    if soft_flux <= 0 or hard_flux <= 0:
        return 0.0
    return soft_flux / hard_flux

def compute_advanced_features(soft_flux_series: List[float], hard_flux_series: List[float]) -> dict:
    """
    Computes research-grade time-series features for solar flare forecasting:
    - Soft & Hard X-ray Gradients
    - Soft/Hard Ratio
    - Cumulative Energy Accumulation Rate
    - Rolling/Spectral Entropy (measures signal complexity)
    - Precursor Activity Score
    """
    soft = np.array(soft_flux_series)
    hard = np.array(hard_flux_series)
    
    soft_grad = np.gradient(soft) if len(soft) > 1 else np.zeros_like(soft)
    hard_grad = np.gradient(hard) if len(hard) > 1 else np.zeros_like(hard)
    
    # Thermodynamic acceleration (second derivative)
    soft_accel = np.gradient(soft_grad) if len(soft_grad) > 1 else np.zeros_like(soft_grad)
    
    # Spectral rolling entropy (Shannon entropy of normalized window)
    soft_norm = (soft - np.min(soft)) / (np.max(soft) - np.min(soft) + 1e-12)
    soft_prob = soft_norm / (np.sum(soft_norm) + 1e-12)
    entropy = -np.sum(soft_prob * np.log2(soft_prob + 1e-12))
    
    # Precursor Activity Score (combining gradient factor & ratio)
    precursor_score = float(max(soft_grad[-1] * 1e5, 0.0) + (soft[-1]/hard[-1] if hard[-1] > 0 else 0.0) * 0.05)
    
    return {
        "soft_xray_gradient": float(soft_grad[-1]),
        "hard_xray_gradient": float(hard_grad[-1]),
        "flux_acceleration": float(soft_accel[-1]),
        "xray_ratio": float(soft[-1] / hard[-1]) if hard[-1] > 0 else 0.0,
        "energy_accumulation_rate": float(np.trapz(soft)),
        "rolling_entropy": float(entropy),
        "precursor_activity_score": precursor_score
    }

def compute_shi(prob: float, growth: float, similarity: float, sat_risk: float, impact_risk: float) -> float:
    """
    Calculates Solar Hazard Index (SHI) based on the ISRO scientific upgrade formula:
    SHI = 0.35 * Flare_Prob + 0.25 * Flux_Growth + 0.15 * Similarity + 0.15 * Sat_Risk + 0.10 * Earth_Impact
    """
    growth_factor = min(max(growth * 1e5, 0.0), 1.0)
    score = (0.35 * prob) + (0.25 * growth_factor) + (0.15 * similarity) + (0.15 * sat_risk) + (0.10 * impact_risk)
    return min(max(score, 0.0), 1.0)

def track_lifecycle_phase(soft_flux_series: List[float]) -> str:
    """
    Tracks the active phase of the solar flare lifecycle:
    - Quiescent: low stable flux
    - Pre-flare: slow positive flux increase
    - Rise: rapid flux increase
    - Peak: maximum plateau
    - Decay: exponential cooling phase
    """
    if len(soft_flux_series) < 5:
        return "Quiescent"
        
    recent = soft_flux_series[-5:]
    grad = np.gradient(recent)
    mean_flux = np.mean(recent)
    
    if grad[-1] > 1e-5:
        return "Rise"
    elif grad[-1] > 1e-7:
        return "Pre-flare"
    elif grad[-1] < -1e-6:
        return "Decay"
    elif mean_flux > 1e-5:
        return "Peak"
    return "Quiescent"

# Thermodynamic limits for solar plasma reconnection flux growth rate
# Standard physical upper limit is 1e-4 to 1e-3 W/m^2 per minute for extreme X-class solar flares
THERMODYNAMIC_GROWTH_LIMIT = 1e-3  # W/m^2 per minute

def apply_physics_constraints(current_flux: float, predicted_flux: float, dt_minutes: float) -> float:
    """
    Enforces thermodynamic limits on flux growth rates.
    If the predicted flux growth rate exceeds the physical maximum reconnection growth rate,
    clamps/smooths the prediction to stay within physical solar physics constraints.
    """
    max_allowed_growth = THERMODYNAMIC_GROWTH_LIMIT * dt_minutes
    actual_growth = predicted_flux - current_flux
    
    if actual_growth > max_allowed_growth:
        # Clamped prediction
        return current_flux + max_allowed_growth
    return max(predicted_flux, 1e-9)  # Ensure non-negative/non-zero flux
