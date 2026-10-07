"""
Physical Feature Analysis and Categorization.
Analyzes the 16 selected photospheric magnetic parameters in physical groups
and provides specialized analysis for key drivers (TOTUSJH, MEANJZH, TOTPOT).
"""

import logging
from typing import Dict, List
import pandas as pd

logger = logging.getLogger(__name__)

# Physical groupings of the 16 features
PHYSICAL_FEATURE_GROUPS: Dict[str, List[str]] = {
    "helicity_current": [
        "ABSNJZH",  # Absolute value of the net current helicity
        "MEANJZH",  # Mean current helicity
        "TOTUSJH",  # Total unsigned current helicity
        "TOTUSJZ",  # Total unsigned vertical current
        "SAVNCPP",  # Sum of modulus of net current
    ],
    "lorentz_force": [
        "TOTBSQ",  # Total magnitude of Lorentz force
        "TOTFX",   # Sum of x-component of Lorentz force
        "TOTFY",   # Sum of y-component of Lorentz force
        "TOTFZ",   # Sum of z-component of Lorentz force
    ],
    "energy_free_energy": [
        "MEANPOT",  # Mean photospheric magnetic free energy
        "TOTPOT",   # Total magnetic energy density
    ],
    "flux_morphology": [
        "USFLUX",    # Total unsigned flux
        "AREA_ACR",  # Area of strong field pixels
        "R_VALUE",   # Sum of flux near polarity inversion line
    ],
    "twist_shear": [
        "MEANALP",  # Mean characteristic twist parameter alpha
        "MEANSHR",  # Mean shear angle
    ],
}

# The 3 crucial physical features emphasized in the paper
KEY_PHYSICAL_FEATURES: List[str] = ["TOTUSJH", "MEANJZH", "TOTPOT"]


class PhysicalFeatureAnalyzer:
    """
    Computes statistical and group distributions of photospheric magnetic parameters.
    """

    @staticmethod
    def compute_group_statistics(df: pd.DataFrame, label_col: str = "label") -> pd.DataFrame:
        """
        Computes mean, standard deviation, and median for positive vs negative classes
        across all physical feature groups.
        """
        records = []
        for group_name, feats in PHYSICAL_FEATURE_GROUPS.items():
            for feat in feats:
                if feat in df.columns:
                    pos_vals = df[df[label_col] == 1][feat]
                    neg_vals = df[df[label_col] == 0][feat]
                    records.append(
                        {
                            "group": group_name,
                            "feature": feat,
                            "pos_mean": float(pos_vals.mean()) if len(pos_vals) > 0 else 0.0,
                            "pos_std": float(pos_vals.std()) if len(pos_vals) > 0 else 0.0,
                            "neg_mean": float(neg_vals.mean()) if len(neg_vals) > 0 else 0.0,
                            "neg_std": float(neg_vals.std()) if len(neg_vals) > 0 else 0.0,
                            "ratio_pos_to_neg": float(pos_vals.mean() / (neg_vals.mean() + 1e-8)) if len(pos_vals) > 0 and len(neg_vals) > 0 else 1.0,
                            "is_key_feature": feat in KEY_PHYSICAL_FEATURES,
                        }
                    )
        return pd.DataFrame(records)
