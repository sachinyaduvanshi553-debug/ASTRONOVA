"""
SHAP Explainer for Tree-Based Solar Flare Forecasting Models
============================================================
Provides TreeExplainer-based SHAP values for XGBoost and LightGBM forecasters.
Supports per-horizon explanations, global feature importance aggregation,
and waterfall / beeswarm plot generation.
"""

import logging
import sys
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Project root guard ────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger("astronova.xai.shap_explainer")

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
CLASS_NAMES = ["A/B", "C", "M", "X"]

# Colour palette for horizon bars
PALETTE = ["#4FC3F7", "#29B6F6", "#0288D1", "#01579B"]


# ─────────────────────────────────────────────────────────────────────────────
class SHAPExplainer:
    """
    Real SHAP-based explainability for XGBoost / LightGBM forecasters.

    Parameters
    ----------
    model       : fitted XGBoostForecaster or LightGBMForecaster object
    model_name  : human-readable name ("XGBoost" | "LightGBM")
    seq_len     : sequence length used during training (default 10)
    out_dir     : directory to save plots (default "reports/xai")
    """

    def __init__(
        self,
        model: Any,
        model_name: str = "XGBoost",
        seq_len: int = 10,
        out_dir: str = "reports/xai",
    ):
        try:
            import shap

            self._shap = shap
        except ImportError:
            raise ImportError("shap is required: pip install shap")

        self.model = model
        self.model_name = model_name
        self.seq_len = seq_len
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

        # Build flat feature names: timestep_t0_feat … timestep_t9_feat
        self.flat_feature_names: list[str] = []
        for t in range(seq_len):
            for f in FEATURE_NAMES:
                self.flat_feature_names.append(f"t{t}_{f}")

        # Cache of computed shap values keyed by horizon index
        self._shap_values: dict[int, Any] = {}
        self._explainers: dict[int, Any] = {}

    # ─────────────────────────────────────────────────────────── public API ──
    def compute_shap_values(
        self,
        X: np.ndarray,
        horizon_idx: int = 0,
        max_samples: int = 500,
    ) -> np.ndarray:
        """
        Compute SHAP values for a given forecast horizon.

        Parameters
        ----------
        X           : [N, seq_len, features]  or  [N, flat_features]
        horizon_idx : which horizon (0=15m, 1=30m, 2=1h, 3=6h)
        max_samples : cap to avoid slow computation

        Returns
        -------
        shap_values : ndarray [N, flat_features, num_classes]
        """
        X_flat = self._flatten(X)[:max_samples]
        clf = self.model.classifiers[horizon_idx]

        logger.info(
            "[%s] Computing SHAP (horizon=%s, samples=%d) …",
            self.model_name,
            HORIZON_LABELS[horizon_idx],
            len(X_flat),
        )

        explainer = self._shap.TreeExplainer(clf)
        shap_vals = explainer.shap_values(X_flat)  # list[num_classes] or ndarray

        # Normalise to [N, features, classes]
        if isinstance(shap_vals, list):
            # Legacy SHAP: list of [N, F] arrays, one per class
            shap_arr = np.stack(shap_vals, axis=-1)  # [N, F, C]
        else:
            shap_arr = np.asarray(shap_vals)
            # Newer SHAP may return (N, F, C, 1) – squeeze trailing 1-dims
            while shap_arr.ndim > 3 and shap_arr.shape[-1] == 1:
                shap_arr = shap_arr.squeeze(-1)
            if shap_arr.ndim == 2:  # [N, F] binary case
                shap_arr = shap_arr[..., np.newaxis]  # → [N, F, 1]

        self._shap_values[horizon_idx] = shap_arr
        self._explainers[horizon_idx] = explainer

        logger.info("[%s] SHAP values computed. shape=%s", self.model_name, shap_arr.shape)
        return shap_arr

    def global_feature_importance(
        self,
        horizon_idx: int = 0,
        top_n: int = 15,
    ) -> pd.DataFrame:
        """
        Aggregate mean-absolute SHAP across classes → global importance.

        Returns a DataFrame with columns: feature, importance, std.
        """
        if horizon_idx not in self._shap_values:
            raise RuntimeError(f"Run compute_shap_values(horizon_idx={horizon_idx}) first.")
        shap_arr = self._shap_values[horizon_idx]  # [N, F, C]
        mean_abs = np.abs(shap_arr).mean(axis=(0, 2))  # [F]
        std_abs = np.abs(shap_arr).std(axis=(0, 2))

        df = pd.DataFrame(
            {
                "feature": self.flat_feature_names,
                "importance": mean_abs,
                "std": std_abs,
            }
        )
        df = df.sort_values("importance", ascending=False).head(top_n).reset_index(drop=True)
        return df

    def local_explanation(
        self,
        X: np.ndarray,
        sample_idx: int = 0,
        horizon_idx: int = 0,
        target_class: int = 3,
        top_n: int = 10,
    ) -> dict[str, Any]:
        """
        Per-sample waterfall-style explanation for the target flare class.
        """
        if horizon_idx not in self._shap_values:
            self.compute_shap_values(X, horizon_idx)

        shap_arr = self._shap_values[horizon_idx]  # [N, F, C]
        sv_sample = shap_arr[sample_idx, :, target_class]  # [F]
        feat_vals = self._flatten(X)[sample_idx]

        order = np.argsort(np.abs(sv_sample))[::-1][:top_n]
        pos_feats = [
            {
                "feature": self.flat_feature_names[i],
                "shap_value": float(sv_sample[i]),
                "feature_value": float(feat_vals[i]),
            }
            for i in order
            if sv_sample[i] > 0
        ]
        neg_feats = [
            {
                "feature": self.flat_feature_names[i],
                "shap_value": float(sv_sample[i]),
                "feature_value": float(feat_vals[i]),
            }
            for i in order
            if sv_sample[i] <= 0
        ]

        return {
            "model_name": self.model_name,
            "horizon": HORIZON_LABELS[horizon_idx],
            "target_class": CLASS_NAMES[target_class],
            "positive_factors": pos_feats,
            "negative_factors": neg_feats,
        }

    # ──────────────────────────────────────────────────────────────── plots ──
    def plot_beeswarm(
        self,
        horizon_idx: int = 0,
        target_class: int = 3,
        top_n: int = 15,
        save: bool = True,
    ) -> Path:
        """
        Beeswarm-style summary plot for the given horizon and target class.
        """
        shap_arr = self._shap_values.get(horizon_idx)
        if shap_arr is None:
            raise RuntimeError("Call compute_shap_values() first.")

        sv = shap_arr[:, :, target_class]  # [N, F]
        order = np.argsort(np.abs(sv).mean(axis=0))[::-1][:top_n]
        sv_top = sv[:, order]
        fnames = [self.flat_feature_names[i] for i in order]

        fig, ax = plt.subplots(figsize=(10, 8), facecolor="#0D1117")
        ax.set_facecolor("#0D1117")

        # Scatter with colour = feature magnitude
        feat_vals = np.zeros_like(sv_top)
        for fi, _orig_i in enumerate(order):
            feat_vals[:, fi] = self._flatten(np.zeros((1, 1)))[0][0]  # placeholder

        plt.cm.RdBu_r(np.linspace(0, 1, sv_top.shape[0]))
        for fi in range(sv_top.shape[1] - 1, -1, -1):
            jitter = np.random.normal(0, 0.08, sv_top.shape[0])
            ax.scatter(
                sv_top[:, fi],
                np.full(sv_top.shape[0], fi) + jitter,
                c=sv_top[:, fi],
                cmap="RdBu_r",
                alpha=0.55,
                s=12,
                linewidths=0,
            )

        ax.axvline(0, color="#555", linewidth=0.8, linestyle="--")
        ax.set_yticks(range(len(fnames)))
        ax.set_yticklabels(fnames[::-1], color="white", fontsize=9)
        ax.set_xlabel("SHAP value", color="white", fontsize=10)
        ax.set_title(
            f"{self.model_name} — SHAP Beeswarm\n"
            f"Horizon: {HORIZON_LABELS[horizon_idx]} | Class: {CLASS_NAMES[target_class]}",
            color="white",
            fontsize=12,
            pad=10,
        )
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#333")

        sm = plt.cm.ScalarMappable(cmap="RdBu_r")
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, fraction=0.02, pad=0.02)
        cbar.ax.yaxis.set_tick_params(color="white")
        cbar.ax.set_ylabel("SHAP value direction", color="white", fontsize=8)
        plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")

        plt.tight_layout()
        out = self.out_dir / f"shap_beeswarm_{self.model_name.lower()}_{HORIZON_LABELS[horizon_idx]}.png"
        if save:
            plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
            logger.info("Saved beeswarm plot → %s", out)
        plt.close()
        return out

    def plot_global_importance(
        self,
        horizon_idx: int = 0,
        top_n: int = 15,
        save: bool = True,
    ) -> Path:
        """Horizontal bar chart of global SHAP feature importance."""
        df = self.global_feature_importance(horizon_idx, top_n)

        fig, ax = plt.subplots(figsize=(10, 7), facecolor="#0D1117")
        ax.set_facecolor("#0D1117")

        # Shorten flat name for display: keep only the base feature name
        display_names = [n.split("_", 1)[1] if "_" in n else n for n in df["feature"]]
        colours = plt.cm.plasma(np.linspace(0.2, 0.9, len(df)))[::-1]

        ax.barh(
            display_names[::-1],
            df["importance"][::-1],
            xerr=df["std"][::-1],
            color=colours,
            error_kw={"ecolor": "#888", "capsize": 3},
        )
        ax.set_xlabel("Mean |SHAP value|", color="white", fontsize=10)
        ax.set_title(
            f"{self.model_name} — Global SHAP Importance\nHorizon: {HORIZON_LABELS[horizon_idx]}",
            color="white",
            fontsize=12,
            pad=10,
        )
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#333")

        plt.tight_layout()
        out = self.out_dir / f"shap_importance_{self.model_name.lower()}_{HORIZON_LABELS[horizon_idx]}.png"
        if save:
            plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
            logger.info("Saved global importance plot → %s", out)
        plt.close()
        return out

    def plot_all_horizons_importance(self, top_n: int = 10, save: bool = True) -> Path:
        """2×2 subplot showing importance across all 4 horizons."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10), facecolor="#0D1117")
        axes = axes.flatten()

        for hi, ax in enumerate(axes):
            if hi not in self._shap_values:
                ax.axis("off")
                continue

            df = self.global_feature_importance(hi, top_n)
            display = [n.split("_", 1)[1] if "_" in n else n for n in df["feature"]]
            colours = plt.cm.viridis(np.linspace(0.3, 0.9, len(df)))[::-1]

            ax.set_facecolor("#0D1117")
            ax.barh(display[::-1], df["importance"][::-1], color=colours)
            ax.set_title(f"Horizon {HORIZON_LABELS[hi]}", color="white", fontsize=10)
            ax.tick_params(colors="white", labelsize=8)
            for spine in ax.spines.values():
                spine.set_edgecolor("#333")

        fig.suptitle(
            f"{self.model_name} — SHAP Importance Across Horizons",
            color="white",
            fontsize=14,
            y=1.01,
        )
        plt.tight_layout()
        out = self.out_dir / f"shap_horizons_{self.model_name.lower()}.png"
        if save:
            plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
            logger.info("Saved multi-horizon importance → %s", out)
        plt.close()
        return out

    # ───────────────────────────────────────────────────────── class-level ──
    def explain_prediction(self, X: np.ndarray = None) -> dict[str, Any]:
        """
        Backward-compatible API: returns a combined local+global explanation dict.
        """
        if X is None or 0 not in self._shap_values:
            return {
                "global_feature_importance": {},
                "local_explanation": {
                    "top_positive_features": [],
                    "top_negative_features": [],
                },
                "note": "Call compute_shap_values() with actual data first.",
            }
        df = self.global_feature_importance(0)
        return {
            "global_feature_importance": dict(zip(df["feature"], df["importance"].round(4), strict=False)),
            "local_explanation": self.local_explanation(X, 0, 0, 3),
        }

    # ──────────────────────────────────────────────────────────── helpers ──
    @staticmethod
    def _flatten(X: np.ndarray) -> np.ndarray:
        if X.ndim == 3:
            return X.reshape(X.shape[0], -1)
        return X
