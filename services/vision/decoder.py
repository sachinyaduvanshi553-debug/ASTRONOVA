import torch
import torch.nn as nn
import torch.nn.functional as F

class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpBlock, self).__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        x = self.conv(x)
        return x

class ImageDecoder(nn.Module):
    """
    Decodes the fused latent embeddings back into an image space.
    Simple UNet style decoder.
    """
    def __init__(self, in_channels=256, out_channels=3):
        super(ImageDecoder, self).__init__()
        # If input is (256, 16, 16) for a 512x512 image processed by ResNet50
        # We need to upscale 16 -> 32 -> 64 -> 128 -> 256 -> 512 (5 upsampling steps)
        
        self.up1 = UpBlock(in_channels, 128)      # 16 -> 32
        self.up2 = UpBlock(128, 64)               # 32 -> 64
        self.up3 = UpBlock(64, 32)                # 64 -> 128
        self.up4 = UpBlock(32, 16)                # 128 -> 256
        self.up5 = UpBlock(16, 16)                # 256 -> 512
        
        self.final_conv = nn.Conv2d(16, out_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid() # Normalize output to [0, 1]

    def forward(self, x):
        # x is (B, in_channels, H/32, W/32)
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        x = self.up5(x)
        x = self.final_conv(x)
        # Using sigmoid because images are typically normalized to [0, 1] for these tasks
        # If standardized with mean/std, remove sigmoid or replace with suitable activation.
        # Let's keep it without sigmoid if we expect standardized outputs, but for images 
        # usually we want to predict raw pixels or standardized pixels.
        # We will output logits and handle loss accordingly, or just output raw values.
        return x
