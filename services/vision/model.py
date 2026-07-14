import torch
import torch.nn as nn
import torch.nn.functional as F
from .encoder import ImageEncoder, TemporalEncoder, PhysicsEncoder
from .fusion import FusionNetwork
from .decoder import ImageDecoder

class TransformerRefiner(nn.Module):
    """Spatial refinement Transformer applied after cross-attention fusion."""
    def __init__(self, d_model: int, nhead: int = 4, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W) -> flatten spatial -> Transformer -> reshape back
        B, C, H, W = x.shape
        # Flatten spatial dimensions: (B, C, H*W) -> (B, H*W, C)
        x_flat = x.view(B, C, -1).permute(0, 2, 1)
        # Apply transformer
        refined_flat = self.transformer(x_flat) # (B, H*W, C)
        # Reshape back to spatial
        refined = refined_flat.permute(0, 2, 1).view(B, C, H, W)
        return refined

class SolarVisionPredictor(nn.Module):
    """
    Multimodal solar flare image prediction model.
    
    Inputs:
        images: (B, T, 3, H, W) - sequence of T historical solar images
        telemetry: (B, 10) - GOES/HEL1OS/SoLEXS telemetry features
        physics: (B, 5) - physics features from NOAA/CME/SEP catalogs
    
    Outputs (dict):
        predicted_image: (B, 3, H, W) - predicted future solar image [0,1]
        class_logits: (B, 5) - GOES class logits [A,B,C,M,X]
        class_probs: (B, 5) - softmax class probabilities
        reg_output: (B, 1) - log10(flux) regression value
        latent_embedding: (B, D) - latent representation for downstream tasks
    """
    def __init__(
        self,
        spatial_dim: int = 256,
        temporal_dim: int = 256, 
        physics_dim: int = 128,
        n_classes: int = 5,
        pretrained_encoder: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        # Encoders
        self.image_encoder = ImageEncoder(output_dim=spatial_dim, pretrained=pretrained_encoder)
        self.temporal_encoder = TemporalEncoder(self.image_encoder, hidden_dim=temporal_dim)
        self.physics_encoder = PhysicsEncoder(telemetry_dim=10, physics_dim=5, output_dim=physics_dim)
        
        # Fusion + Refinement
        self.fusion = FusionNetwork(spatial_dim, temporal_dim, physics_dim)
        self.transformer_refiner = TransformerRefiner(d_model=spatial_dim, nhead=4, num_layers=2, dropout=dropout)
        
        # Decoder — future image prediction
        self.decoder = ImageDecoder(in_channels=spatial_dim, out_channels=3)
        
        # Global pooling for 1D heads
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=dropout)
        
        # Classification head — GOES class
        self.classifier = nn.Sequential(
            nn.Linear(spatial_dim, 128),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(128, n_classes)
        )
        
        # Regression head — log10(flux)
        self.regressor = nn.Sequential(
            nn.Linear(spatial_dim, 64),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(64, 1)
        )
    
    def forward(self, images, telemetry, physics) -> dict:
        # Handle 4D single image input for backward compat with some tests
        if images.dim() == 4:
            images = images.unsqueeze(1) # Add sequence dim
            
        # 1. Encode
        last_frame_spatial, temporal_emb = self.temporal_encoder(images)
        physics_emb = self.physics_encoder(telemetry, physics)
        
        # 2. Cross-attention fusion
        fused = self.fusion(last_frame_spatial, temporal_emb, physics_emb)  # (B, C, H, W)
        
        # 3. Transformer spatial refinement
        refined = self.transformer_refiner(fused)  # (B, C, H, W)
        
        # 4. Decode to future image
        predicted_image = self.decoder(refined)  # (B, 3, H, W)
        predicted_image = torch.sigmoid(predicted_image)  # Normalize to [0, 1]
        
        # 5. Global pool for 1D heads
        pooled = self.global_pool(refined).flatten(1)  # (B, C)
        pooled_dropped = self.dropout(pooled)
        
        # 6. Classification
        class_logits = self.classifier(pooled_dropped)  # (B, 5)
        class_probs = F.softmax(class_logits, dim=-1)
        
        # 7. Regression
        reg_output = self.regressor(pooled_dropped)  # (B, 1)
        
        return {
            'predicted_image': predicted_image,
            'class_logits': class_logits,
            'class_probs': class_probs,
            'reg_output': reg_output,
            'latent_embedding': pooled,
        }

    def get_num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
        
    def freeze_encoder(self):
        """Freezes the ResNet backbone for fine-tuning heads."""
        for param in self.image_encoder.parameters():
            param.requires_grad = False
            
    def unfreeze_encoder(self):
        """Unfreezes the ResNet backbone."""
        for param in self.image_encoder.parameters():
            param.requires_grad = True


# Deprecated alias for backward compatibility
MultimodalSolarModel = SolarVisionPredictor
