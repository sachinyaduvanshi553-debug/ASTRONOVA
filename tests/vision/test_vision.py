import pytest
import torch
import numpy as np
from services.vision.dataset import SolarImageDataset
from services.vision.encoder import ImageEncoder, TemporalEncoder, PhysicsEncoder
from services.vision.preprocessing import SolarImagePreprocessor
import tempfile
import cv2
import os

def test_dataset_outputs():
    # Create dummy images
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create 3 dummy images
        for i in range(3):
            img = np.zeros((256, 256, 3), dtype=np.uint8)
            cv2.imwrite(os.path.join(tmpdir, f"img_{i}.png"), img)
            
        dataset = SolarImageDataset(image_dir=tmpdir, sequence_length=2)
        assert len(dataset) == 1
        
        item = dataset[0]
        assert 'image' in item
        assert 'telemetry' in item
        assert 'physics' in item
        assert 'target' in item
        
        assert item['image'].shape == (2, 3, 512, 512)
        assert item['target'].shape == (3, 512, 512)
        assert item['telemetry'].shape == (10,)
        assert item['physics'].shape == (5,)

def test_preprocessing():
    preprocessor = SolarImagePreprocessor(target_size=256, augment=False)
    dummy_img = np.zeros((512, 512, 3), dtype=np.uint8)
    
    tensor = preprocessor.preprocess_image(dummy_img, is_training=False)
    assert tensor.shape == (3, 256, 256)

def test_encoders():
    # Image Encoder
    img_enc = ImageEncoder(output_dim=128, pretrained=False)
    dummy_img = torch.randn(2, 3, 256, 256)
    out = img_enc(dummy_img)
    assert out.shape == (2, 128, 8, 8) # ResNet50 reduces spatial by 32 (256/32 = 8)
    
    # Temporal Encoder
    temp_enc = TemporalEncoder(img_enc, hidden_dim=64)
    dummy_seq = torch.randn(2, 3, 3, 256, 256) # B=2, T=3, C=3, H=256, W=256
    last_spatial, temporal = temp_enc(dummy_seq)
    assert last_spatial.shape == (2, 128, 8, 8)
    assert temporal.shape == (2, 64)
    
    # Physics Encoder
    phys_enc = PhysicsEncoder(telemetry_dim=10, physics_dim=5, output_dim=32)
    dummy_tel = torch.randn(2, 10)
    dummy_phys = torch.randn(2, 5)
    phys_out = phys_enc(dummy_tel, dummy_phys)
    assert phys_out.shape == (2, 32)
