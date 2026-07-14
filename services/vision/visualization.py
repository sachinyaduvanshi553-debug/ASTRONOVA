import torch
import numpy as np
import cv2
from typing import Optional

class XAIVisualizer:
    """
    Explainable AI (XAI) Visualizer for the Multimodal Solar Model.
    Provides authentic implementation of Attention Maps, Uncertainty Estimation,
    and GradCAM for interpreting the model's predictions.
    """
    def __init__(self, model):
        self.model = model
        
        # For GradCAM
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        """
        Registers forward and backward hooks on the last convolutional layer
        of the image encoder (ResNet50 backbone) to capture gradients and activations
        for GradCAM.
        """
        target_layer = self.model.image_encoder.proj
        
        def forward_hook(module, input, output):
            self.activations = output
            
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]
            
        target_layer.register_forward_hook(forward_hook)
        target_layer.register_full_backward_hook(backward_hook)

    def generate_attention_map(self, image_sequence, telemetry, physics):
        self.model.eval()
        with torch.no_grad():
            _ = self.model(image_sequence, telemetry, physics)
            
        attn_weights = self.model.fusion.attention_weights
        attn_weights = attn_weights[0] 
        
        if len(attn_weights.shape) == 3:
            attn_weights = torch.mean(attn_weights, dim=0)
            
        H = W = int(np.sqrt(attn_weights.shape[0]))
        attention_map = attn_weights.view(H, W).cpu().numpy()
        attention_map = (attention_map - np.min(attention_map)) / (np.max(attention_map) - np.min(attention_map) + 1e-8)
        attention_map_resized = cv2.resize(attention_map, (512, 512), interpolation=cv2.INTER_CUBIC)
        return attention_map_resized

    def generate_uncertainty_map(self, image_sequence, telemetry, physics, num_samples=10):
        self.model.train() 
        predictions = []
        with torch.no_grad():
            for _ in range(num_samples):
                preds = self.model(image_sequence, telemetry, physics)
                if isinstance(preds, dict):
                    predictions.append(preds['predicted_image'].cpu().numpy())
                else:
                    predictions.append(preds.cpu().numpy())
                
        predictions = np.stack(predictions)
        variance = np.var(predictions, axis=0) 
        spatial_uncertainty = np.mean(variance[0], axis=0) 
        spatial_uncertainty = (spatial_uncertainty - np.min(spatial_uncertainty)) / (np.max(spatial_uncertainty) - np.min(spatial_uncertainty) + 1e-8)
        return spatial_uncertainty

    def generate_gradcam(self, image_sequence, telemetry, physics, target_channel=0):
        self.model.eval()
        self.model.zero_grad()
        
        image_sequence.requires_grad_(True)
        preds = self.model(image_sequence, telemetry, physics)
        
        if isinstance(preds, dict):
            score = preds['predicted_image'][0, target_channel, :, :].mean()
        else:
            score = preds[0, target_channel, :, :].mean()
        
        score.backward()
        
        gradients = self.gradients[0].cpu().data.numpy() 
        activations = self.activations[0].cpu().data.numpy() 
        
        weights = np.mean(gradients, axis=(1, 2)) 
        
        cam = np.zeros(activations.shape[1:], dtype=np.float32) 
        for i, w in enumerate(weights):
            cam += w * activations[i, :, :]
            
        cam = np.maximum(cam, 0)
        cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam) + 1e-8)
        cam_resized = cv2.resize(cam, (512, 512), interpolation=cv2.INTER_CUBIC)
        
        return cam_resized

    def integrated_gradients(
        self, 
        image_sequence: torch.Tensor, 
        telemetry: torch.Tensor, 
        physics: torch.Tensor,
        target_output: str = 'image',  # 'image', 'class', 'flux'
        n_steps: int = 50,
        baseline: Optional[torch.Tensor] = None,
    ) -> np.ndarray:
        """Integrated Gradients attribution."""
        self.model.eval()
        if baseline is None:
            baseline = torch.zeros_like(image_sequence)
            
        baseline.requires_grad_(True)
        image_sequence.requires_grad_(True)
        
        alphas = torch.linspace(0, 1, steps=n_steps, device=image_sequence.device)
        
        integral = torch.zeros_like(image_sequence)
        
        for alpha in alphas:
            interpolated = baseline + alpha * (image_sequence - baseline)
            interpolated.requires_grad_(True)
            interpolated.retain_grad()
            
            out = self.model(interpolated, telemetry, physics)
            
            if target_output == 'image':
                score = out['predicted_image'].mean() if isinstance(out, dict) else out.mean()
            elif target_output == 'class':
                score = out['class_probs'][0, out['class_probs'].argmax()].mean()
            elif target_output == 'flux':
                score = out['reg_output'].mean()
                
            score.backward()
            integral += interpolated.grad / n_steps
            
        attribution = (image_sequence - baseline) * integral
        attr_np = attribution[0].detach().cpu().numpy()
        
        # Spatial importance across time and channels
        spatial_attr = np.mean(np.abs(attr_np), axis=(0, 1)) # (H, W)
        spatial_attr = (spatial_attr - np.min(spatial_attr)) / (np.max(spatial_attr) - np.min(spatial_attr) + 1e-8)
        
        return spatial_attr

    def temporal_importance(
        self,
        image_sequence: torch.Tensor,
        telemetry: torch.Tensor,
        physics: torch.Tensor,
    ) -> np.ndarray:
        """Compute importance of each frame via occlusion."""
        self.model.eval()
        with torch.no_grad():
            base_out = self.model(image_sequence, telemetry, physics)
            if isinstance(base_out, dict):
                base_pred = base_out['predicted_image']
            else:
                base_pred = base_out
                
        T = image_sequence.shape[1]
        importances = np.zeros(T)
        
        for t in range(T):
            occluded_seq = image_sequence.clone()
            occluded_seq[:, t, :, :, :] = 0.0
            
            with torch.no_grad():
                occ_out = self.model(occluded_seq, telemetry, physics)
                if isinstance(occ_out, dict):
                    occ_pred = occ_out['predicted_image']
                else:
                    occ_pred = occ_out
                    
            diff = torch.nn.functional.mse_loss(base_pred, occ_pred).item()
            importances[t] = diff
            
        # Normalize to sum to 1
        if np.sum(importances) > 0:
            importances = importances / np.sum(importances)
            
        return importances

    def physics_feature_importance(
        self,
        image_sequence: torch.Tensor,
        telemetry: torch.Tensor,
        physics: torch.Tensor,
        feature_names: Optional[list] = None,
    ) -> dict:
        """Compute importance of each telemetry/physics feature via ablation."""
        self.model.eval()
        with torch.no_grad():
            base_out = self.model(image_sequence, telemetry, physics)
            base_score = base_out['reg_output'].item() if isinstance(base_out, dict) else base_out.mean().item()
            
        telemetry_dim = telemetry.shape[1]
        physics_dim = physics.shape[1]
        
        if feature_names is None:
            feature_names = [f"t_{i}" for i in range(telemetry_dim)] + [f"p_{i}" for i in range(physics_dim)]
            
        importances = {}
        idx = 0
        
        # Telemetry ablation
        for i in range(telemetry_dim):
            occ_tel = telemetry.clone()
            occ_tel[:, i] = 0.0
            with torch.no_grad():
                occ_out = self.model(image_sequence, occ_tel, physics)
                occ_score = occ_out['reg_output'].item() if isinstance(occ_out, dict) else occ_out.mean().item()
            importances[feature_names[idx]] = abs(base_score - occ_score)
            idx += 1
            
        # Physics ablation
        for i in range(physics_dim):
            occ_phys = physics.clone()
            occ_phys[:, i] = 0.0
            with torch.no_grad():
                occ_out = self.model(image_sequence, telemetry, occ_phys)
                occ_score = occ_out['reg_output'].item() if isinstance(occ_out, dict) else occ_out.mean().item()
            importances[feature_names[idx]] = abs(base_score - occ_score)
            idx += 1
            
        # Normalize
        total = sum(importances.values())
        if total > 0:
            for k in importances:
                importances[k] /= total
                
        return importances

    def generate_overlay_heatmap(
        self,
        original_image: np.ndarray,
        heatmap: np.ndarray,
        colormap: int = cv2.COLORMAP_JET,
        alpha: float = 0.4,
    ) -> np.ndarray:
        """Overlay a heatmap on the original image with alpha blending."""
        if len(original_image.shape) == 2:
            original_image = cv2.cvtColor(original_image, cv2.COLOR_GRAY2BGR)
        elif original_image.shape[2] == 4:
            original_image = cv2.cvtColor(original_image, cv2.COLOR_BGRA2BGR)
            
        orig_resized = cv2.resize(original_image, (heatmap.shape[1], heatmap.shape[0]))
        
        heat_norm = np.uint8(heatmap * 255)
        color_heatmap = cv2.applyColorMap(heat_norm, colormap)
        
        overlay = cv2.addWeighted(color_heatmap, alpha, orig_resized, 1 - alpha, 0)
        return overlay
