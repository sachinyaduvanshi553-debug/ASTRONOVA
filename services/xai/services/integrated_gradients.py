"""
Integrated Gradients Explainer for BiLSTM Solar Flare Forecaster
=================================================================
Uses the Integrated Gradients attribution technique (Sundararajan et al., 2017)
implemented via Captum to explain BiLSTM predictions at the input feature level.

Falls back to a pure-PyTorch finite-difference IG implementation if Captum
is not installed, so the module always runs.
"""

import logging
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

# ── Project root guard ────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger("astronova.xai.integrated_gradients")

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


# ─────────────────────────────────────────────────────────────────────────────
class _LSTMWrapper(nn.Module):
    """
    Thin wrapper around BiLSTMForecaster so we can call Captum on
    a specific (horizon, class) scalar output.
    """

    def __init__(self, model: nn.Module, horizon_idx: int, class_idx: int):
        super().__init__()
        self.model = model
        self.horizon_idx = horizon_idx
        self.class_idx = class_idx

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        probs, _ = self.model(x, return_tuple=True)  # [B, H, C]
        return probs[:, self.horizon_idx, self.class_idx]  # [B]


# ─────────────────────────────────────────────────────────────────────────────
class IntegratedGradientsExplainer:
    """
    Integrated Gradients explainability for the BiLSTM forecaster.

    Uses Captum if available; otherwise falls back to a pure-PyTorch
    Riemann sum approximation.

    Parameters
    ----------
    model     : fitted BiLSTMForecaster (nn.Module)
    device    : "cpu" | "cuda"
    out_dir   : directory to save plots (default "reports/xai")
    n_steps   : number of IG interpolation steps (default 50)
    """

    def __init__(
        self,
        model: nn.Module,
        device: str = "cpu",
        out_dir: str = "reports/xai",
        n_steps: int = 50,
    ):
        self.model = model
        self.device = torch.device(device)
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.n_steps = n_steps

        self.model.to(self.device)
        self.model.eval()

        self._has_captum = False
        try:
            from captum.attr import IntegratedGradients

            self._ig_cls = IntegratedGradients
            self._has_captum = True
            logger.info("Captum detected — using IntegratedGradients.")
        except ImportError:
            logger.warning("Captum not installed — using pure-PyTorch IG fallback.")

        # Cache: {(horizon_idx, class_idx): ndarray [N, seq_len, n_features]}
        self._attributions: dict[tuple[int, int], np.ndarray] = {}

    # ─────────────────────────────────────────────────────────── public API ──
    def compute_attributions(
        self,
        X: torch.Tensor,
        horizon_idx: int = 0,
        class_idx: int = 3,
        baseline: torch.Tensor | None = None,
        max_samples: int = 200,
    ) -> np.ndarray:
        """
        Compute IG attributions for [N, seq_len, n_features] input.

        Returns
        -------
        attributions : ndarray [N, seq_len, n_features]
        """
        if isinstance(X, np.ndarray):
            X = torch.tensor(X, dtype=torch.float32)

        X = X[:max_samples].to(self.device)
        N, _T, _F = X.shape

        if baseline is None:
            baseline = torch.zeros_like(X)

        logger.info(
            "Computing IG attributions (horizon=%s, class=%s, samples=%d) …",
            HORIZON_LABELS[horizon_idx],
            CLASS_NAMES[class_idx],
            N,
        )

        if self._has_captum:
            attrs = self._captum_ig(X, baseline, horizon_idx, class_idx)
        else:
            attrs = self._pytorch_ig(X, baseline, horizon_idx, class_idx)

        self._attributions[(horizon_idx, class_idx)] = attrs
        logger.info("IG attributions computed. shape=%s", attrs.shape)
        return attrs

    def feature_importance_from_ig(
        self,
        horizon_idx: int = 0,
        class_idx: int = 3,
        aggregate: str = "mean",
    ) -> np.ndarray:
        """
        Collapse [N, T, F] attributions → [F] feature importance vector.
        aggregate: "mean" | "sum" | "max"
        """
        attrs = self._attributions.get((horizon_idx, class_idx))
        if attrs is None:
            raise RuntimeError("Call compute_attributions() first.")
        abs_attrs = np.abs(attrs)  # [N, T, F]
        time_agg = abs_attrs.mean(axis=1)  # [N, F]  – average over timesteps
        if aggregate == "sum":
            return time_agg.sum(axis=0)
        elif aggregate == "max":
            return time_agg.max(axis=0)
        return time_agg.mean(axis=0)  # [F]

    # ──────────────────────────────────────────────────────────────── plots ──
    def plot_temporal_heatmap(
        self,
        horizon_idx: int = 0,
        class_idx: int = 3,
        sample_idx: int = 0,
        save: bool = True,
    ) -> Path:
        """
        Heatmap of attribution magnitude over [seq_len × features] for a single sample.
        """
        attrs = self._attributions.get((horizon_idx, class_idx))
        if attrs is None:
            raise RuntimeError("Call compute_attributions() first.")

        sample_attr = attrs[sample_idx]  # [T, F]

        fig, ax = plt.subplots(figsize=(12, 5), facecolor="#0D1117")
        ax.set_facecolor("#0D1117")

        im = ax.imshow(
            sample_attr.T,
            aspect="auto",
            cmap="magma",
            interpolation="nearest",
        )
        ax.set_xlabel("Timestep", color="white", fontsize=10)
        ax.set_ylabel("Feature", color="white", fontsize=10)
        ax.set_yticks(range(len(FEATURE_NAMES)))
        ax.set_yticklabels(FEATURE_NAMES, color="white", fontsize=8)
        ax.set_xticks(range(sample_attr.shape[0]))
        ax.set_xticklabels([f"t{i}" for i in range(sample_attr.shape[0])], color="white", fontsize=8)
        ax.set_title(
            f"BiLSTM Integrated Gradients — Sample {sample_idx}\n"
            f"Horizon: {HORIZON_LABELS[horizon_idx]} | Class: {CLASS_NAMES[class_idx]}",
            color="white",
            fontsize=12,
            pad=10,
        )

        cbar = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.01)
        cbar.ax.yaxis.set_tick_params(color="white")
        cbar.ax.set_ylabel("Attribution", color="white", fontsize=8)
        plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")

        plt.tight_layout()
        out = self.out_dir / f"ig_temporal_heatmap_h{HORIZON_LABELS[horizon_idx]}_c{CLASS_NAMES[class_idx]}.png"
        if save:
            plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
            logger.info("Saved temporal heatmap → %s", out)
        plt.close()
        return out

    def plot_feature_importance(
        self,
        horizon_idx: int = 0,
        class_idx: int = 3,
        save: bool = True,
    ) -> Path:
        """Horizontal bar chart of IG-based feature importance."""
        imp = self.feature_importance_from_ig(horizon_idx, class_idx)
        order = np.argsort(imp)[::-1][:15]
        imp_top = imp[order]
        names_top = [FEATURE_NAMES[i] for i in order]

        fig, ax = plt.subplots(figsize=(10, 7), facecolor="#0D1117")
        ax.set_facecolor("#0D1117")

        colours = plt.cm.plasma(np.linspace(0.2, 0.9, len(imp_top)))[::-1]
        ax.barh(names_top[::-1], imp_top[::-1], color=colours)
        ax.set_xlabel("Mean |IG Attribution|", color="white", fontsize=10)
        ax.set_title(
            f"BiLSTM Integrated Gradients — Feature Importance\n"
            f"Horizon: {HORIZON_LABELS[horizon_idx]} | Class: {CLASS_NAMES[class_idx]}",
            color="white",
            fontsize=12,
            pad=10,
        )
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#333")

        plt.tight_layout()
        out = self.out_dir / f"ig_importance_h{HORIZON_LABELS[horizon_idx]}.png"
        if save:
            plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
            logger.info("Saved IG importance → %s", out)
        plt.close()
        return out

    def plot_all_horizons(self, class_idx: int = 3, save: bool = True) -> Path:
        """2×2 IG importance across all 4 horizons."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10), facecolor="#0D1117")
        axes = axes.flatten()

        for hi, ax in enumerate(axes):
            key = (hi, class_idx)
            ax.set_facecolor("#0D1117")
            if key not in self._attributions:
                ax.text(
                    0.5,
                    0.5,
                    "No data",
                    ha="center",
                    va="center",
                    color="white",
                    transform=ax.transAxes,
                )
                ax.set_title(HORIZON_LABELS[hi], color="white")
                continue

            imp = self.feature_importance_from_ig(hi, class_idx)
            order = np.argsort(imp)[::-1][:10]
            ax.barh(
                [FEATURE_NAMES[i] for i in order][::-1],
                imp[order][::-1],
                color=plt.cm.viridis(np.linspace(0.3, 0.9, len(order)))[::-1],
            )
            ax.set_title(f"Horizon {HORIZON_LABELS[hi]}", color="white", fontsize=10)
            ax.tick_params(colors="white", labelsize=8)
            for spine in ax.spines.values():
                spine.set_edgecolor("#333")

        fig.suptitle(
            f"BiLSTM IG Attributions — Class: {CLASS_NAMES[class_idx]}",
            color="white",
            fontsize=14,
            y=1.01,
        )
        plt.tight_layout()
        out = self.out_dir / f"ig_horizons_class{CLASS_NAMES[class_idx]}.png"
        if save:
            plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
            logger.info("Saved multi-horizon IG → %s", out)
        plt.close()
        return out

    # ──────────────────────────────────────────────────────────── internals ──
    def _captum_ig(
        self,
        X: torch.Tensor,
        baseline: torch.Tensor,
        horizon_idx: int,
        class_idx: int,
    ) -> np.ndarray:
        from captum.attr import IntegratedGradients

        wrapper = _LSTMWrapper(self.model, horizon_idx, class_idx).to(self.device)
        ig = IntegratedGradients(wrapper)
        attrs, delta = ig.attribute(
            X,
            baselines=baseline,
            n_steps=self.n_steps,
            return_convergence_delta=True,
        )
        logger.debug("IG convergence delta: %.6f", delta.abs().mean().item())
        return attrs.detach().cpu().numpy()

    def _pytorch_ig(
        self,
        X: torch.Tensor,
        baseline: torch.Tensor,
        horizon_idx: int,
        class_idx: int,
    ) -> np.ndarray:
        """Pure-PyTorch Riemann sum approximation of Integrated Gradients."""
        alphas = torch.linspace(0, 1, self.n_steps, device=self.device)
        grads = []

        for alpha in alphas:
            interp = baseline + alpha * (X - baseline)
            interp = interp.requires_grad_(True)

            wrapper = _LSTMWrapper(self.model, horizon_idx, class_idx).to(self.device)
            out = wrapper(interp)
            scalar = out.sum()
            scalar.backward()

            grads.append(interp.grad.detach().clone())

        integrated = torch.stack(grads, dim=0).mean(dim=0)  # [N, T, F]
        attrs = (X - baseline) * integrated
        return attrs.detach().cpu().numpy()
