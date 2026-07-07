import torch
import numpy as np
import cv2
from typing import List, Union, Dict, Any, Optional

from .visualization import XAIVisualizer


# ---------------------------------------------------------------------------
# GOES X-ray flux class thresholds  (W / m²)
# ---------------------------------------------------------------------------
_FLARE_THRESHOLDS = {
    "X": 1e-4,   # X-class  >= 1×10⁻⁴ W/m²
    "M": 1e-5,   # M-class  >= 1×10⁻⁵ W/m²
    "C": 1e-6,   # C-class  >= 1×10⁻⁶ W/m²
    "B": 1e-7,   # B-class  >= 1×10⁻⁷ W/m²
}

_HIGH_INTENSITY_THRESHOLD = 0.8   # normalised pixel value


class VisionInferencePipeline:
    """
    Production inference pipeline for the ASTRONOVA Multimodal Solar Model.

    Accepts raw numpy arrays / lists of images, telemetry floats, and physics
    floats.  Returns predicted images together with a physics-based flare
    probability and an MC-Dropout confidence score.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        target_size: int = 512,
        mc_dropout_samples: int = 5,
    ):
        self.device = device
        self.target_size = target_size
        self.mc_dropout_samples = mc_dropout_samples

        # Load model architecture
        from .model import MultimodalSolarModel

        self.model = MultimodalSolarModel(pretrained_encoder=False).to(self.device)
        if model_path:
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

        # XAI visualiser (hooks into the loaded model)
        self.xai = XAIVisualizer(self.model)

    # ------------------------------------------------------------------
    # Input preprocessing helpers
    # ------------------------------------------------------------------

    def _preprocess_image(self, img: np.ndarray) -> np.ndarray:
        """Resize to (target_size, target_size), convert to float32 [0, 1],
        and re-order from HWC to CHW."""
        if img is None or img.size == 0:
            raise ValueError("Received an empty image array.")
        # Handle grayscale
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
        elif img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img = cv2.resize(img, (self.target_size, self.target_size),
                         interpolation=cv2.INTER_LINEAR)
        img = img.astype(np.float32) / 255.0          # scale to [0, 1]
        img = np.transpose(img, (2, 0, 1))             # HWC → CHW
        return img

    def _prepare_image_sequence(
        self, image_sequence: Union[np.ndarray, List[np.ndarray], torch.Tensor]
    ) -> torch.Tensor:
        """Convert a list/array of images into a (1, T, 3, H, W) tensor."""
        if isinstance(image_sequence, torch.Tensor):
            if image_sequence.dim() == 4:               # (T, C, H, W)
                image_sequence = image_sequence.unsqueeze(0)
            return image_sequence.to(self.device)

        if isinstance(image_sequence, np.ndarray) and image_sequence.ndim == 4:
            # Assume (T, H, W, C) or (T, C, H, W)
            frames = [image_sequence[i] for i in range(image_sequence.shape[0])]
        elif isinstance(image_sequence, (list, tuple)):
            frames = list(image_sequence)
        else:
            # Single image – wrap in a list so T=1
            frames = [image_sequence]

        processed = [self._preprocess_image(f) for f in frames]   # list of (C,H,W)
        stacked = np.stack(processed, axis=0)                     # (T, C, H, W)
        tensor = torch.from_numpy(stacked).unsqueeze(0)           # (1, T, C, H, W)
        return tensor.to(self.device)

    @staticmethod
    def _prepare_vector(data: Union[list, np.ndarray, torch.Tensor],
                        target_len: int) -> torch.Tensor:
        """Pad or truncate a 1-D sequence to *target_len* and return
        a (1, target_len) float tensor."""
        if isinstance(data, torch.Tensor):
            vec = data.detach().cpu().numpy().flatten()
        elif isinstance(data, np.ndarray):
            vec = data.flatten().astype(np.float32)
        elif isinstance(data, (list, tuple)):
            vec = np.array(data, dtype=np.float32)
        else:
            vec = np.zeros(target_len, dtype=np.float32)

        # Truncate
        vec = vec[:target_len]
        # Pad
        if len(vec) < target_len:
            vec = np.pad(vec, (0, target_len - len(vec)),
                         mode="constant", constant_values=0.0)
        return torch.from_numpy(vec).unsqueeze(0).float()        # (1, target_len)

    # ------------------------------------------------------------------
    # Confidence via MC Dropout
    # ------------------------------------------------------------------

    def _mc_dropout_confidence(
        self,
        images: torch.Tensor,
        telemetry: torch.Tensor,
        physics: torch.Tensor,
    ) -> float:
        """Run *mc_dropout_samples* stochastic forward passes (dropout
        enabled) and return 1 − normalised variance as a confidence score
        in [0, 1]."""
        self.model.train()   # enable dropout / stochastic layers
        preds: List[np.ndarray] = []
        with torch.no_grad():
            for _ in range(self.mc_dropout_samples):
                out = self.model(images, telemetry, physics)
                preds.append(out.cpu().numpy())
        self.model.eval()

        preds_np = np.stack(preds, axis=0)               # (S, B, C, H, W)
        variance = np.var(preds_np, axis=0).mean()        # scalar
        # Map variance to confidence: low variance → high confidence
        confidence = float(1.0 / (1.0 + variance))
        return confidence

    # ------------------------------------------------------------------
    # Flare probability using GOES thresholds
    # ------------------------------------------------------------------

    @staticmethod
    def _estimate_flare_probability(predicted: np.ndarray) -> float:
        """Estimate flare probability from a predicted image array (C, H, W)
        normalised to [0, 1].

        Strategy
        --------
        1. Identify *active region* pixels with intensity > 0.8.
        2. Compute *brightness ratio* = mean(active pixels) × fraction.
        3. Map that ratio to the GOES flux scale and derive a probability.
        """
        if predicted.ndim == 4:
            predicted = predicted[0]                      # drop batch dim
        # Grayscale proxy – average across channels
        gray = np.mean(predicted, axis=0)                 # (H, W)

        high_mask = gray > _HIGH_INTENSITY_THRESHOLD
        active_fraction = high_mask.sum() / gray.size
        if active_fraction == 0:
            return 0.0

        active_mean = gray[high_mask].mean()
        brightness_ratio = active_mean * active_fraction  # 0 … 1

        # Heuristic mapping to a simulated GOES X-ray flux (W/m²):
        #   brightness_ratio ~0     → ~1e-8  (A-class, no flare)
        #   brightness_ratio ~1     → ~2e-4  (X-class)
        estimated_flux = 1e-8 * (10 ** (brightness_ratio * 4.3))

        # Convert flux to cumulative probability
        if estimated_flux >= _FLARE_THRESHOLDS["X"]:
            prob = 0.90 + 0.10 * min(
                (estimated_flux - _FLARE_THRESHOLDS["X"]) / _FLARE_THRESHOLDS["X"], 1.0
            )
        elif estimated_flux >= _FLARE_THRESHOLDS["M"]:
            prob = 0.60 + 0.30 * (
                (estimated_flux - _FLARE_THRESHOLDS["M"])
                / (_FLARE_THRESHOLDS["X"] - _FLARE_THRESHOLDS["M"])
            )
        elif estimated_flux >= _FLARE_THRESHOLDS["C"]:
            prob = 0.25 + 0.35 * (
                (estimated_flux - _FLARE_THRESHOLDS["C"])
                / (_FLARE_THRESHOLDS["M"] - _FLARE_THRESHOLDS["C"])
            )
        elif estimated_flux >= _FLARE_THRESHOLDS["B"]:
            prob = 0.05 + 0.20 * (
                (estimated_flux - _FLARE_THRESHOLDS["B"])
                / (_FLARE_THRESHOLDS["C"] - _FLARE_THRESHOLDS["B"])
            )
        else:
            prob = estimated_flux / _FLARE_THRESHOLDS["B"] * 0.05

        return float(np.clip(prob, 0.0, 1.0))

    # ------------------------------------------------------------------
    # Main predict entry-point
    # ------------------------------------------------------------------

    def predict(
        self,
        image_sequence: Union[np.ndarray, List[np.ndarray], torch.Tensor],
        telemetry: Union[list, np.ndarray, torch.Tensor],
        physics: Union[list, np.ndarray, torch.Tensor],
    ) -> Dict[str, Any]:
        """
        Run inference on a sequence of solar images with associated
        telemetry and physics vectors.

        Parameters
        ----------
        image_sequence : array-like
            Raw images (numpy HWC uint8/float, or pre-built tensor).
        telemetry : array-like
            1-D vector of 10 telemetry scalars (padded / truncated
            automatically).
        physics : array-like
            1-D vector of 5 physics scalars.

        Returns
        -------
        dict with keys ``predicted_image``, ``confidence``,
        ``flare_probability``.
        """
        # 1. Preprocess ---------------------------------------------------
        images_t = self._prepare_image_sequence(image_sequence)
        telemetry_t = self._prepare_vector(telemetry, target_len=10).to(self.device)
        physics_t = self._prepare_vector(physics, target_len=5).to(self.device)

        # 2. Deterministic prediction --------------------------------------
        self.model.eval()
        with torch.no_grad():
            pred_image = self.model(images_t, telemetry_t, physics_t)

        pred_np = pred_image.cpu().numpy()                 # (1, C, H, W)

        # 3. Confidence (MC Dropout variance) ------------------------------
        confidence = self._mc_dropout_confidence(images_t, telemetry_t, physics_t)

        # 4. Flare probability (GOES-flux heuristic) -----------------------
        flare_prob = self._estimate_flare_probability(pred_np)

        return {
            "predicted_image": pred_np,
            "confidence": confidence,
            "flare_probability": flare_prob,
        }

    # ------------------------------------------------------------------
    # XAI / explain entry-point
    # ------------------------------------------------------------------

    def explain(
        self,
        image_sequence: Union[np.ndarray, List[np.ndarray], torch.Tensor],
        telemetry: Union[list, np.ndarray, torch.Tensor],
        physics: Union[list, np.ndarray, torch.Tensor],
    ) -> Dict[str, np.ndarray]:
        """Return GradCAM, attention, and uncertainty maps for the given
        inputs.  All maps are 2-D numpy arrays normalised to [0, 1] and
        sized (512, 512)."""
        images_t = self._prepare_image_sequence(image_sequence)
        telemetry_t = self._prepare_vector(telemetry, target_len=10).to(self.device)
        physics_t = self._prepare_vector(physics, target_len=5).to(self.device)

        gradcam = self.xai.generate_gradcam(images_t, telemetry_t, physics_t)
        attention = self.xai.generate_attention_map(images_t, telemetry_t, physics_t)
        uncertainty = self.xai.generate_uncertainty_map(images_t, telemetry_t, physics_t)

        return {
            "gradcam": gradcam,
            "attention_map": attention,
            "uncertainty_map": uncertainty,
        }
