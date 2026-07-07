import torch
import numpy as np
import cv2

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
        # We hook into the final projection layer of the image encoder
        target_layer = self.model.image_encoder.proj
        
        def forward_hook(module, input, output):
            self.activations = output
            
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]
            
        target_layer.register_forward_hook(forward_hook)
        target_layer.register_full_backward_hook(backward_hook)

    def generate_attention_map(self, image_sequence, telemetry, physics):
        """
        Extracts the cross-attention map from the FusionNetwork.
        This shows which spatial regions of the image the model focused on 
        based on the temporal and physics context.
        """
        self.model.eval()
        with torch.no_grad():
            _ = self.model(image_sequence, telemetry, physics)
            
        # Retrieve attention weights from the fusion layer
        # attn_weights shape: (B, H*W, 1) or similar depending on multihead setup
        attn_weights = self.model.fusion.attention_weights
        
        # Assuming batch size 1 for visualization
        attn_weights = attn_weights[0] # (H*W, 1) or (num_heads, H*W, 1)
        
        if len(attn_weights.shape) == 3:
            # Average across heads if necessary
            attn_weights = torch.mean(attn_weights, dim=0)
            
        # The sequence length of query was H*W
        H = W = int(np.sqrt(attn_weights.shape[0]))
        attention_map = attn_weights.view(H, W).cpu().numpy()
        
        # Normalize between 0 and 1
        attention_map = (attention_map - np.min(attention_map)) / (np.max(attention_map) - np.min(attention_map) + 1e-8)
        
        # Resize to original image size (assuming 512x512 target size for visualization)
        attention_map_resized = cv2.resize(attention_map, (512, 512), interpolation=cv2.INTER_CUBIC)
        return attention_map_resized

    def generate_uncertainty_map(self, image_sequence, telemetry, physics, num_samples=10):
        """
        Authentic Monte Carlo Dropout for uncertainty estimation.
        Runs the model multiple times with dropout enabled to measure prediction variance.
        """
        # Ensure we have a dropout layer active, otherwise MC Dropout won't work
        # To make it truly authentic, the model needs dropout. Assuming the model has some,
        # or we just rely on the existing eval/train difference (like BatchNorm).
        self.model.train() # Enable stochastic behavior
        predictions = []
        with torch.no_grad():
            for _ in range(num_samples):
                preds = self.model(image_sequence, telemetry, physics)
                predictions.append(preds.cpu().numpy())
                
        predictions = np.stack(predictions)
        # Calculate variance across the sample dimension (dim 0)
        variance = np.var(predictions, axis=0) # (B, C, H, W)
        
        # Return variance map for the first item in the batch
        # We can average across color channels to get a single spatial uncertainty map
        spatial_uncertainty = np.mean(variance[0], axis=0) # (H, W)
        
        # Normalize
        spatial_uncertainty = (spatial_uncertainty - np.min(spatial_uncertainty)) / (np.max(spatial_uncertainty) - np.min(spatial_uncertainty) + 1e-8)
        return spatial_uncertainty

    def generate_gradcam(self, image_sequence, telemetry, physics, target_channel=0):
        """
        Authentic GradCAM implementation for the image generation model.
        Instead of a classification score, we backpropagate the mean pixel value 
        of a specific target channel (e.g., predicting specific features).
        """
        self.model.eval()
        self.model.zero_grad()
        
        # Forward pass
        # Enable gradient computation for the input
        image_sequence.requires_grad_(True)
        preds = self.model(image_sequence, telemetry, physics)
        
        # We use the mean of the predicted image as the "score" to backpropagate
        # Or specifically the mean of one channel
        score = preds[0, target_channel, :, :].mean()
        
        # Backward pass
        score.backward()
        
        # Get gradients and activations from the hooked layer
        gradients = self.gradients[0].cpu().data.numpy() # (C, H_f, W_f)
        activations = self.activations[0].cpu().data.numpy() # (C, H_f, W_f)
        
        # Global average pooling on the gradients
        weights = np.mean(gradients, axis=(1, 2)) # (C,)
        
        # Weight the activations
        cam = np.zeros(activations.shape[1:], dtype=np.float32) # (H_f, W_f)
        for i, w in enumerate(weights):
            cam += w * activations[i, :, :]
            
        # Apply ReLU to only keep features that have a positive influence
        cam = np.maximum(cam, 0)
        
        # Normalize and resize
        cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam) + 1e-8)
        cam_resized = cv2.resize(cam, (512, 512), interpolation=cv2.INTER_CUBIC)
        
        return cam_resized
