import torch
import torch.nn as nn
from .encoder import ImageEncoder, TemporalEncoder, PhysicsEncoder
from .fusion import FusionNetwork
from .decoder import ImageDecoder

class MultimodalSolarModel(nn.Module):
    """
    Overarching multimodal model integrating encoders, fusion layer, and decoder.
    Predicts future solar images based on sequences of past images, telemetry, and physics.
    """
    def __init__(self, spatial_dim=256, temporal_dim=256, physics_dim=128, pretrained_encoder=True):
        super(MultimodalSolarModel, self).__init__()
        
        self.image_encoder = ImageEncoder(output_dim=spatial_dim, pretrained=pretrained_encoder)
        self.temporal_encoder = TemporalEncoder(self.image_encoder, hidden_dim=temporal_dim)
        self.physics_encoder = PhysicsEncoder(telemetry_dim=10, physics_dim=5, output_dim=physics_dim)
        
        self.fusion = FusionNetwork(spatial_dim=spatial_dim, temporal_dim=temporal_dim, physics_dim=physics_dim)
        
        self.decoder = ImageDecoder(in_channels=spatial_dim, out_channels=3)

    def forward(self, images, telemetry, physics):
        # images: (B, T, C, H, W)
        # telemetry: (B, 10)
        # physics: (B, 5)
        
        # 1. Temporal & Spatial Encoding
        last_frame_spatial, temporal_embedding = self.temporal_encoder(images)
        
        # 2. Physics Encoding
        physics_embedding = self.physics_encoder(telemetry, physics)
        
        # 3. Fusion
        fused_features = self.fusion(last_frame_spatial, temporal_embedding, physics_embedding)
        
        # 4. Decoding
        predicted_image = self.decoder(fused_features)
        
        return predicted_image
