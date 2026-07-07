"""
Unified metrics interface for the ASTRONOVA solar prediction pipeline.

Re-exports every metric from ``evaluate.py`` and adds flare-classification
metrics (precision / recall / F1 for binary and multi-class C/M/X).
"""

from typing import Dict, List, Optional, Sequence, Union

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Re-export all image-quality metrics from evaluate
# ---------------------------------------------------------------------------
from .evaluate import (
    calculate_psnr,
    calculate_mae,
    calculate_rmse,
    calculate_ssim,
    calculate_per_channel_metrics,
    calculate_fid_features,
    calculate_fid,
    Evaluator,
)

__all__ = [
    # pixel / structural metrics
    'calculate_psnr',
    'calculate_mae',
    'calculate_rmse',
    'calculate_ssim',
    'calculate_per_channel_metrics',
    # distribution metrics
    'calculate_fid_features',
    'calculate_fid',
    # evaluator
    'Evaluator',
    # flare classification
    'FlareClassificationMetrics',
    # convenience
    'compute_all_metrics',
]


# ---------------------------------------------------------------------------
# Flare classification metrics
# ---------------------------------------------------------------------------

class FlareClassificationMetrics:
    """Precision, Recall, and F1 for solar-flare prediction.

    Supports two modes controlled at construction time:

    *   **binary** (default ``mode='binary'``):  flare vs. no-flare.
    *   **multi-class** (``mode='multiclass'``): C / M / X flare classes,
        encoded as integer labels ``{0: C, 1: M, 2: X}``.

    All inputs are expected as 1-D integer tensors or Python lists.
    """

    CLASS_NAMES: Dict[int, str] = {0: 'C', 1: 'M', 2: 'X'}

    def __init__(self, mode: str = 'binary', num_classes: Optional[int] = None):
        if mode not in ('binary', 'multiclass'):
            raise ValueError(f"mode must be 'binary' or 'multiclass', got '{mode}'")
        self.mode = mode
        if mode == 'binary':
            self.num_classes = 2
        else:
            self.num_classes = num_classes or 3  # C / M / X

    # -- helpers -----------------------------------------------------------

    @staticmethod
    def _to_numpy(x) -> np.ndarray:
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy().astype(int)
        return np.asarray(x, dtype=int)

    @staticmethod
    def _safe_div(num: float, den: float) -> float:
        return num / den if den > 0 else 0.0

    # -- core computation --------------------------------------------------

    def compute(
        self,
        predictions: Union[torch.Tensor, Sequence[int]],
        targets: Union[torch.Tensor, Sequence[int]],
    ) -> Dict:
        """Return a dict of precision, recall, F1 (per-class and macro).

        Parameters
        ----------
        predictions, targets : array-like of int
            Predicted and ground-truth labels.

        Returns
        -------
        dict
            For **binary** mode the dict has keys ``precision``, ``recall``,
            ``f1``, ``support``, ``accuracy``.

            For **multiclass** mode each class gets its own sub-dict, plus
            ``macro_precision``, ``macro_recall``, ``macro_f1``, and
            ``accuracy``.
        """
        preds = self._to_numpy(predictions)
        tgts = self._to_numpy(targets)

        if self.mode == 'binary':
            return self._binary_metrics(preds, tgts)
        return self._multiclass_metrics(preds, tgts)

    # -- binary ------------------------------------------------------------

    def _binary_metrics(self, preds: np.ndarray, tgts: np.ndarray) -> Dict:
        tp = int(((preds == 1) & (tgts == 1)).sum())
        fp = int(((preds == 1) & (tgts == 0)).sum())
        fn = int(((preds == 0) & (tgts == 1)).sum())
        tn = int(((preds == 0) & (tgts == 0)).sum())

        precision = self._safe_div(tp, tp + fp)
        recall = self._safe_div(tp, tp + fn)
        f1 = self._safe_div(2.0 * precision * recall, precision + recall)
        accuracy = self._safe_div(tp + tn, tp + fp + fn + tn)

        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'accuracy': accuracy,
            'support': int(tgts.sum()),
            'confusion': {'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn},
        }

    # -- multi-class -------------------------------------------------------

    def _multiclass_metrics(self, preds: np.ndarray, tgts: np.ndarray) -> Dict:
        per_class: Dict[str, Dict[str, float]] = {}
        precisions, recalls, f1s = [], [], []

        for cls_id in range(self.num_classes):
            tp = int(((preds == cls_id) & (tgts == cls_id)).sum())
            fp = int(((preds == cls_id) & (tgts != cls_id)).sum())
            fn = int(((preds != cls_id) & (tgts == cls_id)).sum())

            p = self._safe_div(tp, tp + fp)
            r = self._safe_div(tp, tp + fn)
            f = self._safe_div(2.0 * p * r, p + r)

            cls_name = self.CLASS_NAMES.get(cls_id, str(cls_id))
            per_class[cls_name] = {
                'precision': p,
                'recall': r,
                'f1': f,
                'support': int((tgts == cls_id).sum()),
            }
            precisions.append(p)
            recalls.append(r)
            f1s.append(f)

        accuracy = self._safe_div(int((preds == tgts).sum()), len(tgts))

        return {
            'per_class': per_class,
            'macro_precision': float(np.mean(precisions)),
            'macro_recall': float(np.mean(recalls)),
            'macro_f1': float(np.mean(f1s)),
            'accuracy': accuracy,
        }


# ---------------------------------------------------------------------------
# Convenience: compute every metric in one call
# ---------------------------------------------------------------------------

def compute_all_metrics(
    pred_images: Optional[torch.Tensor] = None,
    target_images: Optional[torch.Tensor] = None,
    pred_labels: Optional[torch.Tensor] = None,
    target_labels: Optional[torch.Tensor] = None,
    data_range: float = 1.0,
    flare_mode: str = 'binary',
) -> Dict:
    """One-stop shop: compute all available metrics and return a unified dict.

    Parameters
    ----------
    pred_images, target_images : torch.Tensor, optional
        Predicted and ground-truth images ``(B, C, H, W)`` for image-quality
        metrics (PSNR, MAE, RMSE, SSIM, per-channel).
    pred_labels, target_labels : torch.Tensor, optional
        Predicted and ground-truth integer labels for flare classification
        metrics.
    data_range : float
        Dynamic range of pixel values for image metrics.
    flare_mode : str
        ``'binary'`` or ``'multiclass'`` — passed to
        ``FlareClassificationMetrics``.

    Returns
    -------
    dict
        Combined metrics dictionary.  Keys present depend on which inputs
        were supplied.
    """
    results: Dict = {}

    # --- image-quality metrics ---
    if pred_images is not None and target_images is not None:
        if pred_images.dim() == 3:
            pred_images = pred_images.unsqueeze(0)
            target_images = target_images.unsqueeze(0)

        psnr_vals, mae_vals, rmse_vals, ssim_vals = [], [], [], []
        per_channel_all: List[Dict] = []

        for i in range(pred_images.shape[0]):
            p = pred_images[i].unsqueeze(0)
            t = target_images[i].unsqueeze(0)
            psnr_vals.append(calculate_psnr(p, t, data_range).item())
            mae_vals.append(calculate_mae(p, t))
            rmse_vals.append(calculate_rmse(p, t))
            ssim_vals.append(calculate_ssim(p, t, data_range=data_range).item())
            per_channel_all.append(calculate_per_channel_metrics(p, t, data_range))

        results['image_metrics'] = {
            'psnr': float(np.mean(psnr_vals)),
            'mae': float(np.mean(mae_vals)),
            'rmse': float(np.mean(rmse_vals)),
            'ssim': float(np.mean(ssim_vals)),
            'per_channel': per_channel_all,
        }

    # --- flare classification metrics ---
    if pred_labels is not None and target_labels is not None:
        flare_metrics = FlareClassificationMetrics(mode=flare_mode)
        results['flare_metrics'] = flare_metrics.compute(pred_labels, target_labels)

    return results
