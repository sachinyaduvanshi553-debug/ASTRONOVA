"""
Unified Feature Selection Pipeline.
Combines ANOVA and Mutual Information selections via union operation,
reproducing the base paper's 16 photospheric magnetic feature set.
"""

import json
import logging
import os
from typing import Dict, List, Optional, Tuple
import pandas as pd
from features.anova import ANOVASelector
from features.mutual_information import MutualInformationSelector
from preprocessing.cleaner import BASE_PAPER_16_FEATURES, BASE_PAPER_25_FEATURES

logger = logging.getLogger(__name__)


class SolarFeatureSelector:
    """
    Executes ANOVA and Mutual Information feature selection on training data only
    and computes their union to obtain the final optimal feature set.
    """

    def __init__(
        self,
        anova_threshold: float = 0.1,
        mi_threshold: float = 0.2,
        initial_features: Optional[List[str]] = None,
    ):
        self.anova_threshold = anova_threshold
        self.mi_threshold = mi_threshold
        self.initial_features = initial_features or BASE_PAPER_25_FEATURES
        self.anova_selector = ANOVASelector(threshold=anova_threshold)
        self.mi_selector = MutualInformationSelector(threshold=mi_threshold)
        self.selected_features_: List[str] = []
        self.feature_summary_df_: Optional[pd.DataFrame] = None

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> "SolarFeatureSelector":
        """
        Fits ANOVA and MI selectors strictly on training data.
        """
        # Ensure available features from initial list
        avail_features = [f for f in self.initial_features if f in X_train.columns]
        logger.info("Running feature selection on %d candidate features.", len(avail_features))

        # 1. Fit ANOVA
        self.anova_selector.fit(X_train, y_train, avail_features)
        anova_selected = set(self.anova_selector.selected_features_)

        # 2. Fit Mutual Information
        self.mi_selector.fit(X_train, y_train, avail_features)
        mi_selected = set(self.mi_selector.selected_features_)

        # 3. Union of both selection sets
        union_selected = sorted(list(anova_selected.union(mi_selected)))

        # Ensure base-paper canonical features are preserved if present in candidate columns
        if not union_selected:
            logger.warning("Union selection returned empty set; defaulting to base-paper 16 features.")
            union_selected = [f for f in BASE_PAPER_16_FEATURES if f in X_train.columns]

        self.selected_features_ = union_selected

        # 4. Build comprehensive summary dataframe
        anova_df = self.anova_selector.get_scores_dataframe().set_index("feature")
        mi_df = self.mi_selector.get_scores_dataframe().set_index("feature")

        summary_df = anova_df.join(mi_df, how="outer").reset_index()
        summary_df["selected_in_final_union"] = summary_df["feature"].isin(self.selected_features_)
        summary_df["in_base_paper_16"] = summary_df["feature"].isin(BASE_PAPER_16_FEATURES)
        self.feature_summary_df_ = summary_df.sort_values(
            by=["selected_in_final_union", "normalized_mi_score", "normalized_f_score"],
            ascending=[False, False, False],
        ).reset_index(drop=True)

        logger.info("Final selected feature count: %d. Features: %s", len(self.selected_features_), self.selected_features_)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms dataframe to retain only selected features.
        """
        return X[self.selected_features_]

    def save_feature_list(self, output_path: str = "checkpoints/feature_list.json") -> None:
        """
        Saves selected feature names and metadata.
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        payload = {
            "num_features": len(self.selected_features_),
            "selected_features": self.selected_features_,
            "anova_threshold": self.anova_threshold,
            "mi_threshold": self.mi_threshold,
            "base_paper_16_match": all(f in self.selected_features_ for f in BASE_PAPER_16_FEATURES if f in self.initial_features),
        }
        with open(output_path, "w") as f:
            json.dump(payload, f, indent=2)
        logger.info("Saved feature list to %s", output_path)
