import torch
import numpy as np
import cv2
from typing import List, Union, Dict, Any, Optional

from .visualization import XAIVisualizer
from .model import SolarVisionPredictor
from .uncertainty import UncertaintyEngine


class VisionInferencePipeline:
    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        target_size: int = 512,
        mc_dropout_samples: int = 20,
    ):
        self.device = device
        self.target_size = target_size
        self.mc_dropout_samples = mc_dropout_samples

        self.model = SolarVisionPredictor(pretrained_encoder=False).to(self.device)
        if model_path:
            try:
                checkpoint = torch.load(model_path, map_location=self.device)
                self.model.load_state_dict(checkpoint["model_state_dict"])
            except Exception as e:
                print(f"Failed to load checkpoint strictly: {e}. Trying backward compat load.")
                
        self.model.eval()

        self.xai = XAIVisualizer(self.model)
        self.uncertainty_engine = UncertaintyEngine(self.model, n_samples=self.mc_dropout_samples, device=self.device)

        self.class_names = ["A", "B", "C", "M", "X"]

    def _preprocess_image(self, img: np.ndarray) -> np.ndarray:
        if img is None or img.size == 0:
            raise ValueError("Received an empty image array.")
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
        elif img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img = cv2.resize(img, (self.target_size, self.target_size), interpolation=cv2.INTER_LINEAR)
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        return img

    def _prepare_image_sequence(self, image_sequence: Union[np.ndarray, List[np.ndarray], torch.Tensor]) -> torch.Tensor:
        if isinstance(image_sequence, torch.Tensor):
            if image_sequence.dim() == 4:
                image_sequence = image_sequence.unsqueeze(0)
            return image_sequence.to(self.device)

        if isinstance(image_sequence, np.ndarray) and image_sequence.ndim == 4:
            frames = [image_sequence[i] for i in range(image_sequence.shape[0])]
        elif isinstance(image_sequence, (list, tuple)):
            frames = list(image_sequence)
        else:
            frames = [image_sequence]

        processed = [self._preprocess_image(f) for f in frames]
        stacked = np.stack(processed, axis=0)
        tensor = torch.from_numpy(stacked).unsqueeze(0)
        return tensor.to(self.device)

    @staticmethod
    def _prepare_vector(data: Union[list, np.ndarray, torch.Tensor], target_len: int) -> torch.Tensor:
        if isinstance(data, torch.Tensor):
            vec = data.detach().cpu().numpy().flatten()
        elif isinstance(data, np.ndarray):
            vec = data.flatten().astype(np.float32)
        elif isinstance(data, (list, tuple)):
            vec = np.array(data, dtype=np.float32)
        else:
            vec = np.zeros(target_len, dtype=np.float32)

        vec = vec[:target_len]
        if len(vec) < target_len:
            vec = np.pad(vec, (0, target_len - len(vec)), mode="constant", constant_values=0.0)
        return torch.from_numpy(vec).unsqueeze(0).float()

    def predict(
        self,
        image_sequence: Union[np.ndarray, List[np.ndarray], torch.Tensor],
        telemetry: Union[list, np.ndarray, torch.Tensor],
        physics: Union[list, np.ndarray, torch.Tensor],
    ) -> Dict[str, Any]:
        images_t = self._prepare_image_sequence(image_sequence)
        telemetry_t = self._prepare_vector(telemetry, target_len=10).to(self.device)
        physics_t = self._prepare_vector(physics, target_len=5).to(self.device)

        self.model.eval()
        with torch.no_grad():
            out = self.model(images_t, telemetry_t, physics_t)

        pred_np = out["predicted_image"].cpu().numpy()
        
        class_probs = out["class_probs"][0].cpu().numpy()
        class_idx = int(np.argmax(class_probs))
        flare_class = self.class_names[class_idx]
        
        flare_prob = float(class_probs[3] + class_probs[4]) # M + X class
        
        log_flux = out["reg_output"].item()
        flux_w_m2 = 10 ** log_flux
        
        latent = out["latent_embedding"][0].cpu().numpy()

        class_probs_dict = {name: float(prob) for name, prob in zip(self.class_names, class_probs)}

        return {
            "predicted_image": pred_np,
            "flare_class": flare_class,
            "flare_class_index": class_idx,
            "class_probabilities": class_probs_dict,
            "flare_probability": flare_prob,
            "predicted_log_flux": log_flux,
            "predicted_flux": flux_w_m2,
            "latent_embedding": latent,
        }

    def predict_with_uncertainty(
        self,
        image_sequence: Union[np.ndarray, List[np.ndarray], torch.Tensor],
        telemetry: Union[list, np.ndarray, torch.Tensor],
        physics: Union[list, np.ndarray, torch.Tensor],
    ) -> Dict[str, Any]:
        images_t = self._prepare_image_sequence(image_sequence)
        telemetry_t = self._prepare_vector(telemetry, target_len=10).to(self.device)
        physics_t = self._prepare_vector(physics, target_len=5).to(self.device)

        # Get deterministic baseline
        result = self.predict(image_sequence, telemetry, physics)
        
        # Add uncertainty quantification
        unc_stats = self.uncertainty_engine.compute_pixel_uncertainty(images_t, telemetry_t, physics_t)
        
        result["confidence"] = unc_stats["confidence"]
        result["pixel_variance_map"] = unc_stats["pixel_variance"]
        result["class_uncertainty"] = unc_stats["class_uncertainty"]
        result["flux_uncertainty"] = unc_stats["flux_uncertainty"]
        
        return result

    def explain(
        self,
        image_sequence: Union[np.ndarray, List[np.ndarray], torch.Tensor],
        telemetry: Union[list, np.ndarray, torch.Tensor],
        physics: Union[list, np.ndarray, torch.Tensor],
    ) -> Dict[str, np.ndarray]:
        images_t = self._prepare_image_sequence(image_sequence)
        telemetry_t = self._prepare_vector(telemetry, target_len=10).to(self.device)
        physics_t = self._prepare_vector(physics, target_len=5).to(self.device)

        gradcam = self.xai.generate_gradcam(images_t, telemetry_t, physics_t)
        attention = self.xai.generate_attention_map(images_t, telemetry_t, physics_t)
        ig = self.xai.integrated_gradients(images_t, telemetry_t, physics_t)

        return {
            "gradcam": gradcam,
            "attention_map": attention,
            "integrated_gradients": ig,
        }
