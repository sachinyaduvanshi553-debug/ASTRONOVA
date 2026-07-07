import torch
import torch.nn as nn
import torchvision.models as models

class ImageEncoder(nn.Module):
    """
    Encodes spatial features from solar imagery.
    Using ResNet50 as the backbone.
    """
    def __init__(self, output_dim=256, pretrained=True):
        super(ImageEncoder, self).__init__()
        # Use ResNet50
        resnet = models.resnet50(pretrained=pretrained)
        # Remove the classification head
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        # ResNet50 outputs (2048, H/32, W/32) before pool
        # We want spatial features, so we keep them, or we project them
        # Let's project to output_dim channels
        self.proj = nn.Conv2d(2048, output_dim, kernel_size=1)
        
    def forward(self, x):
        # x is (B, C, H, W)
        features = self.backbone(x)
        features = self.proj(features) # (B, output_dim, H/32, W/32)
        return features

class TemporalEncoder(nn.Module):
    """
    Encodes temporal sequences of images using ConvLSTM or 3D Convs.
    For simplicity, we can use 3D Conv or just process each frame with ImageEncoder
    and then use an RNN/LSTM over the spatial features.
    Here we process sequence using the ImageEncoder, then pool and LSTM.
    """
    def __init__(self, image_encoder, hidden_dim=256):
        super(TemporalEncoder, self).__init__()
        self.image_encoder = image_encoder
        self.hidden_dim = hidden_dim
        # We will flatten spatial features and pass through LSTM for a global temporal embedding
        # Or keep spatial features. Let's keep spatial features using ConvLSTM if needed.
        # But a simple approach: process each frame, pool, then standard LSTM.
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.lstm = nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim, batch_first=True)

    def forward(self, x):
        # x is (B, T, C, H, W)
        B, T, C, H, W = x.shape
        # Process each frame
        # Flatten B and T
        x = x.view(B * T, C, H, W)
        spatial_features = self.image_encoder(x) # (B*T, output_dim, H/32, W/32)
        
        # We also want the last frame's spatial features to pass to the decoder
        # Let's extract the last frame spatial features
        _, C_f, H_f, W_f = spatial_features.shape
        spatial_features = spatial_features.view(B, T, C_f, H_f, W_f)
        last_frame_spatial = spatial_features[:, -1, :, :, :] # (B, output_dim, H/32, W/32)
        
        # Global temporal embedding
        pooled = self.pool(spatial_features.view(B * T, C_f, H_f, W_f)) # (B*T, C_f, 1, 1)
        pooled = pooled.view(B, T, C_f)
        lstm_out, (hn, cn) = self.lstm(pooled)
        temporal_embedding = lstm_out[:, -1, :] # (B, hidden_dim)
        
        return last_frame_spatial, temporal_embedding

class PhysicsEncoder(nn.Module):
    """
    Encodes 1D telemetry and physics features.
    """
    def __init__(self, telemetry_dim=10, physics_dim=5, output_dim=128):
        super(PhysicsEncoder, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(telemetry_dim + physics_dim, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim),
            nn.ReLU()
        )

    def forward(self, telemetry, physics):
        x = torch.cat([telemetry, physics], dim=1)
        return self.fc(x)
