"""
Unified Feature Importance Aggregator
======================================
Combines importance signals from multiple sources:
  - SHAP values (XGBoost + LightGBM)
  - Integrated Gradients (BiLSTM)
  - Model-native feature importances (gain, weight, split)

Produces a consensus ranking and cross-model comparison plots.
"""

import logging
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Project root guard ────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger("astronova.xai.feature_importance")

FEATURE_NAMES = [
    "log_soft_flux",
    "log_hard_flux",
    "xray_ratio",
    "soft_gradient",
    "hard_gradient",
    "soft_rolling_mean_15",
    "soft_rolling_std_15",
    "soft_rolling_max_30",
    "soft_rolling_min_30",
    "flux_acceleration",
    "time_since_prev_flare",
    "hour_sin",
    "doy_sin",
    "noaa_ar_count",
    "magnetic_complexity",
]
HORIZON_LABELS = ["15m", "30m", "1h", "6h"]

MODEL_COLOURS = {
    "XGBoost": "#F59E0B",
    "LightGBM": "#10B981",
    "BiLSTM": "#818CF8",
    "Consensus": "#F43F5E",
}


# ─────────────────────────────────────────────────────────────────────────────
class FeatureImportanceAggregator:
    """
    Aggregates feature importances across multiple XAI methods into a
    unified, consensus ranking with cross-model comparison plots.

    Usage
    -----
    >>> agg = FeatureImportanceAggregator(out_dir="reports/xai")
    >>> agg.add_shap_importance("XGBoost",  xgb_imp_array)   # [n_features]
    >>> agg.add_shap_importance("LightGBM", lgb_imp_array)
    >>> agg.add_ig_importance(lstm_imp_array)
    >>> agg.add_native_importance("XGBoost", xgb_native_dict)
    >>> df = agg.consensus_ranking()
    >>> agg.plot_cross_model_comparison()
    >>> agg.plot_radar_chart()
    """

    def __init__(self, out_dir: str = "reports/xai"):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

        # Storage: model_name → ndarray [n_features]
        self._importances: dict[str, np.ndarray] = {}

    # ─────────────────────────────────────────────────────────── add data ──
    def add_shap_importance(
        self,
        model_name: str,
        importance: np.ndarray,
        horizon_idx: int = 0,
    ):
        """
        Add flat feature importance from SHAP.
        importance : [n_features]  mean-absolute SHAP across all timesteps + samples
        """
        key = f"{model_name}(SHAP,{HORIZON_LABELS[horizon_idx]})"
        self._importances[key] = np.asarray(importance)
        logger.info("Added SHAP importance for %s [horizon=%s]", model_name, HORIZON_LABELS[horizon_idx])

    def add_ig_importance(
        self,
        importance: np.ndarray,
        horizon_idx: int = 0,
        class_idx: int = 3,
    ):
        """
        Add flat feature importance from Integrated Gradients.
        importance : [n_features]
        """
        key = f"BiLSTM(IG,{HORIZON_LABELS[horizon_idx]})"
        self._importances[key] = np.asarray(importance)
        logger.info("Added IG importance [horizon=%s]", HORIZON_LABELS[horizon_idx])

    def add_native_importance(
        self,
        model_name: str,
        importance: np.ndarray,
    ):
        """
        Add model-native feature importance (e.g. from XGBClassifier.feature_importances_).
        importance : [n_flat_features]  or  [n_features] (will be auto-detected)
        """
        imp = np.asarray(importance)
        # If flat (seq_len × n_features), average over timesteps
        if imp.ndim == 1 and imp.shape[0] > len(FEATURE_NAMES):
            n_feat = len(FEATURE_NAMES)
            seq_len = imp.shape[0] // n_feat
            imp = imp.reshape(seq_len, n_feat).mean(axis=0)
        key = f"{model_name}(native)"
        self._importances[key] = imp
        logger.info("Added native importance for %s", model_name)

    # ──────────────────────────────────────────────────────────── analysis ──
    def consensus_ranking(self, top_n: int = 15) -> pd.DataFrame:
        """
        Normalise each importance vector to [0, 1] and average across all sources
        to produce a consensus ranking.

        Returns a DataFrame sorted by consensus score (descending).
        """
        if not self._importances:
            raise RuntimeError("No importances added yet.")

        norm_dict = {}
        for key, imp in self._importances.items():
            # Truncate/extend to n_features if needed
            imp = self._align_to_base(imp)
            max_v = imp.max()
            norm_dict[key] = imp / (max_v + 1e-12)

        matrix = np.stack(list(norm_dict.values()), axis=0)  # [sources, F]
        consensus = matrix.mean(axis=0)

        df = pd.DataFrame(
            {
                "feature": FEATURE_NAMES,
                "consensus": consensus,
            }
        )
        for key, v in norm_dict.items():
            df[key] = v

        df = df.sort_values("consensus", ascending=False).head(top_n).reset_index(drop=True)
        logger.info("Consensus ranking computed for %d features.", len(df))
        return df

    def source_correlation(self) -> pd.DataFrame:
        """Pearson correlation matrix between importance sources."""
        aligned = {k: self._align_to_base(v) for k, v in self._importances.items()}
        df = pd.DataFrame(aligned, index=FEATURE_NAMES)
        return df.corr()

    # ──────────────────────────────────────────────────────────────── plots ──
    def plot_cross_model_comparison(self, top_n: int = 15, save: bool = True) -> Path:
        """
        Grouped bar chart comparing normalised importances across sources
        for the top-N consensus features.
        """
        df = self.consensus_ranking(top_n)
        source_cols = [c for c in df.columns if c not in ("feature", "consensus")]
        n_sources = len(source_cols)
        n_feats = len(df)

        fig, ax = plt.subplots(figsize=(14, 7), facecolor="#0D1117")
        ax.set_facecolor("#0D1117")

        x = np.arange(n_feats)
        width = 0.8 / max(n_sources, 1)
        cmap = matplotlib.colormaps.get_cmap("tab10")

        for si, col in enumerate(source_cols):
            vals = df[col].fillna(0).values
            offset = (si - n_sources / 2 + 0.5) * width
            ax.bar(x + offset, vals, width, label=col, color=cmap(si), alpha=0.88)

        ax.bar(x, df["consensus"].values, 0, alpha=0, label=None, color="none")

        ax.set_xticks(x)
        ax.set_xticklabels(df["feature"], rotation=45, ha="right", color="white", fontsize=8)
        ax.set_ylabel("Normalised Importance", color="white", fontsize=10)
        ax.set_title("Cross-Model Feature Importance Comparison", color="white", fontsize=13, pad=12)
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#333")

        ax.legend(
            loc="upper right",
            facecolor="#1C2333",
            edgecolor="#444",
            labelcolor="white",
            fontsize=8,
            framealpha=0.8,
        )
        plt.tight_layout()
        out = self.out_dir / "feature_importance_cross_model.png"
        if save:
            plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
            logger.info("Saved cross-model comparison → %s", out)
        plt.close()
        return out

    def plot_consensus_ranking(self, top_n: int = 15, save: bool = True) -> Path:
        """Horizontal bar chart of consensus feature importance."""
        df = self.consensus_ranking(top_n)

        fig, ax = plt.subplots(figsize=(10, 7), facecolor="#0D1117")
        ax.set_facecolor("#0D1117")

        colours = plt.cm.plasma(np.linspace(0.15, 0.9, len(df)))[::-1]
        ax.barh(df["feature"][::-1], df["consensus"][::-1], color=colours)
        ax.set_xlabel("Consensus Importance Score", color="white", fontsize=10)
        ax.set_title(
            "Unified Feature Importance — Consensus Ranking\n(Averaged across SHAP + IG + Native)",
            color="white",
            fontsize=12,
            pad=10,
        )
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#333")

        plt.tight_layout()
        out = self.out_dir / "feature_importance_consensus.png"
        if save:
            plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
            logger.info("Saved consensus ranking → %s", out)
        plt.close()
        return out

    def plot_radar_chart(self, top_n: int = 8, save: bool = True) -> Path:
        """
        Radar (spider) chart comparing importance profiles for top-N features
        across SHAP, IG, and native sources.
        """
        df = self.consensus_ranking(top_n)
        source_cols = [c for c in df.columns if c not in ("feature", "consensus")]
        features = df["feature"].tolist()
        n_feats = len(features)

        angles = np.linspace(0, 2 * np.pi, n_feats, endpoint=False).tolist()
        angles += angles[:1]  # close the polygon

        fig, ax = plt.subplots(figsize=(9, 9), subplot_kw={"polar": True}, facecolor="#0D1117")
        ax.set_facecolor("#0D1117")
        ax.spines["polar"].set_edgecolor("#333")
        ax.set_facecolor("#0D1117")

        cmap = matplotlib.colormaps.get_cmap("tab10")

        for si, col in enumerate(source_cols):
            vals = df[col].fillna(0).tolist()
            vals += vals[:1]
            ax.plot(angles, vals, linewidth=1.5, color=cmap(si), label=col, alpha=0.9)
            ax.fill(angles, vals, alpha=0.10, color=cmap(si))

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(features, color="white", size=8)
        ax.yaxis.set_tick_params(colors="white")
        ax.set_yticklabels([], color="white")
        ax.grid(color="#333", linewidth=0.5)

        ax.legend(
            loc="upper right",
            bbox_to_anchor=(1.3, 1.1),
            facecolor="#1C2333",
            edgecolor="#444",
            labelcolor="white",
            fontsize=8,
        )
        ax.set_title(
            "Feature Importance Radar\n(Normalised scores by source)",
            color="white",
            fontsize=12,
            pad=20,
        )

        plt.tight_layout()
        out = self.out_dir / "feature_importance_radar.png"
        if save:
            plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
            logger.info("Saved radar chart → %s", out)
        plt.close()
        return out

    def plot_source_correlation(self, save: bool = True) -> Path:
        """Heatmap of Pearson correlation between importance sources."""
        corr = self.source_correlation()

        fig, ax = plt.subplots(figsize=(max(6, len(corr)), max(5, len(corr))), facecolor="#0D1117")
        ax.set_facecolor("#0D1117")

        im = ax.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
        labels = corr.columns.tolist()
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right", color="white", fontsize=8)
        ax.set_yticklabels(labels, color="white", fontsize=8)

        for i in range(len(labels)):
            for j in range(len(labels)):
                ax.text(
                    j,
                    i,
                    f"{corr.values[i, j]:.2f}",
                    ha="center",
                    va="center",
                    color="white",
                    fontsize=7,
                )

        cbar = fig.colorbar(im, ax=ax, fraction=0.03)
        cbar.ax.yaxis.set_tick_params(color="white")
        plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")
        ax.set_title("Feature Importance Source Correlation", color="white", fontsize=12, pad=10)

        plt.tight_layout()
        out = self.out_dir / "feature_importance_correlation.png"
        if save:
            plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
            logger.info("Saved source correlation → %s", out)
        plt.close()
        return out

    # ──────────────────────────────────────────────────────────── helpers ──
    @staticmethod
    def _align_to_base(imp: np.ndarray) -> np.ndarray:
        """Pad or truncate to len(FEATURE_NAMES)."""
        n = len(FEATURE_NAMES)
        if len(imp) == n:
            return imp
        if len(imp) > n:
            return imp[:n]
        return np.pad(imp, (0, n - len(imp)))
