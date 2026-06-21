import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Union
import logging

logger = logging.getLogger("astronova.physics_validator")

class PhysicsValidator:
    """
    Validates space weather predictions and observations against physical constraints
    and thermodynamic limits (e.g. non-negativity, growth rates, energy conservation).
    """
    
    def __init__(self, max_growth_rate_per_min: float = 100.0, min_quality_score: float = 0.5):
        """
        Args:
            max_growth_rate_per_min: Maximum physically allowable multiplication factor of flux in 1 minute.
            min_quality_score: Minimum threshold for sensor data to be considered reliable.
        """
        self.max_growth_rate_per_min = max_growth_rate_per_min
        self.min_quality_score = min_quality_score
        
    def validate_flux_nonnegativity(self, flux_values: Union[np.ndarray, List[float]]) -> Tuple[bool, float]:
        """
        Ensures all flux predictions/observations are non-negative.
        Returns (is_valid, min_value).
        """
        flux_arr = np.array(flux_values)
        if len(flux_arr) == 0:
            return True, 0.0
            
        min_val = np.min(flux_arr)
        is_valid = bool(min_val >= 0.0)
        
        if not is_valid:
            logger.warning(f"Physical validation failed: Negative flux detected (min = {min_val:.2e} W/m^2)")
            
        return is_valid, float(min_val)
        
    def validate_energy_conservation(self, soft_flux: float, hard_flux: float) -> bool:
        """
        Checks if soft X-ray flux and hard X-ray flux are physically consistent.
        Physically, hard X-ray flux (non-thermal) should not dwarf soft X-ray flux (thermal/plasma emission)
        by orders of magnitude (usually, soft_flux >= hard_flux during quiescent and active periods).
        """
        if pd.isna(soft_flux) or pd.isna(hard_flux):
            return True
            
        if soft_flux <= 0 or hard_flux <= 0:
            return True # Non-negativity is handled separately
            
        # Hard X-ray should not exceed soft X-ray by more than a physical margin (e.g., F_hard / F_soft <= 2.0)
        # Even in hard flares, soft X-ray flux is much larger or comparable because it represents integrated energy.
        ratio = hard_flux / soft_flux
        is_valid = ratio <= 2.0
        
        if not is_valid:
            logger.warning(f"Physical validation failed: Hard X-ray flux exceeds soft X-ray flux unexpectedly (ratio = {ratio:.2f})")
            
        return is_valid
        
    def validate_temporal_continuity(
        self, 
        flux_values: Union[np.ndarray, List[float]], 
        dt_minutes: float = 1.0
    ) -> Tuple[bool, float]:
        """
        Ensures the rate of change of solar flux does not violate thermal limits.
        The Sun cannot heat up and increase X-ray output by 1000x in 1 minute.
        Returns (is_valid, max_ratio_change).
        """
        flux_arr = np.array(flux_values)
        if len(flux_arr) < 2:
            return True, 1.0
            
        # Calculate ratio of consecutive values
        # Avoid division by zero/near zero by clipping denominator
        denom = np.clip(flux_arr[:-1], 1e-9, None)
        ratios = flux_arr[1:] / denom
        
        max_ratio = np.max(ratios)
        
        # Adjust threshold based on time step
        threshold = self.max_growth_rate_per_min * dt_minutes
        is_valid = bool(max_ratio <= threshold)
        
        if not is_valid:
            logger.warning(f"Physical validation failed: Flux growth rate exceeds thermodynamic limit ({max_ratio:.1f}x in {dt_minutes}m)")
            
        return is_valid, float(max_ratio)
        
    def validate_sensor_reliability(self, quality_score: float) -> bool:
        """Checks if the computed observation quality score is above minimum threshold."""
        return bool(quality_score >= self.min_quality_score)
        
    def generate_validation_report(
        self, 
        soft_fluxes: List[float], 
        hard_fluxes: List[float] = None, 
        quality_score: float = 1.0,
        dt_minutes: float = 1.0
    ) -> Dict[str, Any]:
        """
        Generates a comprehensive physics validation report.
        """
        non_neg_valid, min_flux = self.validate_flux_nonnegativity(soft_fluxes)
        
        continuity_valid = True
        max_ratio = 1.0
        if len(soft_fluxes) >= 2:
            continuity_valid, max_ratio = self.validate_temporal_continuity(soft_fluxes, dt_minutes)
            
        energy_valid = True
        if hard_fluxes is not None and len(soft_fluxes) > 0 and len(hard_fluxes) > 0:
            # check the latest values
            energy_valid = self.validate_energy_conservation(soft_fluxes[-1], hard_fluxes[-1])
            
        sensor_valid = self.validate_sensor_reliability(quality_score)
        
        all_passed = bool(non_neg_valid and continuity_valid and energy_valid and sensor_valid)
        
        return {
            "all_passed": all_passed,
            "non_negativity": {
                "passed": non_neg_valid,
                "min_flux": min_flux
            },
            "temporal_continuity": {
                "passed": continuity_valid,
                "max_ratio_change": max_ratio
            },
            "energy_conservation": {
                "passed": energy_valid
            },
            "sensor_reliability": {
                "passed": sensor_valid,
                "quality_score": quality_score
            }
        }
