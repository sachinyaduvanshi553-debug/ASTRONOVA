"""
tests/vision/test_solar_image_forecaster.py — Unit tests for the multimodal pipeline.

Tests:
    1. Model instantiation (CPU)
    2. Forward pass shape validation
    3. Per-horizon classification output shapes
    4. Heatmap output shape
    5. Future image output shape
    6. MC-Dropout uncertainty
    7. Focal loss computation
    8. SSIM loss computation
    9. Chronological sequence builder
    10. Image normalizer fit/transform
    11. Dataset loading
    12. Grad-CAM generation

ALL tests use SYNTHETIC data — smoke test mode.
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

# PYTHONPATH
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Model tests
# ---------------------------------------------------------------------------
class TestSolarImageForecaster:
    """Test the main SolarImageForecaster model."""

    @pytest.fixture
    def model(self):
        from models.vision.solar_image_forecaster import SolarImageForecaster
        return SolarImageForecaster(
            backbone="resnet18",
            image_size=64,  # Small for fast tests
            channels=3,
            pretrained_encoder=False,
            temporal_model="transformer",
            spatial_dim=32,
            temporal_dim=32,
            n_heads=2,
            n_temporal_layers=1,
            tel_embed_dim=16,
            mag_embed_dim=16,
            fusion_output_dim=32,
            classifier_hidden=16,
            dropout=0.1,
            max_seq_len=4,
        )

    @pytest.fixture
    def dummy_input(self):
        B, T, C, H, W = 2, 4, 3, 64, 64
        return {
            "image_seq": torch.randn(B, T, C, H, W),
            "telemetry": torch.randn(B, T, 2),
        }

    def test_model_creation_cpu(self, model):
        """Model can be created on CPU."""
        assert model is not None
        param_count = sum(p.numel() for p in model.parameters())
        assert param_count > 0

    def test_forward_pass(self, model, dummy_input):
        """Forward pass returns all expected keys."""
        model.eval()
        with torch.no_grad():
            output = model(**dummy_input)

        assert "class_logits" in output
        assert "class_probs" in output
        assert "flare_heatmap" in output
        assert "predicted_image" in output
        assert output["image_type"] == "AI_FORECAST"
        assert "latent" in output

    def test_classification_shapes(self, model, dummy_input):
        """Classification outputs have correct shapes for all horizons."""
        model.eval()
        with torch.no_grad():
            output = model(**dummy_input)

        from models.vision.solar_image_forecaster import HORIZONS
        B = dummy_input["image_seq"].shape[0]

        for h in HORIZONS:
            assert h in output["class_logits"], f"Missing horizon: {h}"
            assert output["class_logits"][h].shape == (B, 3)
            assert output["class_probs"][h].shape == (B, 3)
            # Probabilities should be in [0, 1]
            assert output["class_probs"][h].min() >= 0.0
            assert output["class_probs"][h].max() <= 1.0

    def test_heatmap_shape(self, model, dummy_input):
        """Heatmap output has correct shape."""
        model.eval()
        with torch.no_grad():
            output = model(**dummy_input)

        B = dummy_input["image_seq"].shape[0]
        H = W = 64
        assert output["flare_heatmap"].shape == (B, 1, H, W)
        assert output["flare_heatmap"].min() >= 0.0
        assert output["flare_heatmap"].max() <= 1.0

    def test_predicted_image_shape(self, model, dummy_input):
        """Predicted image has correct shape and range."""
        model.eval()
        with torch.no_grad():
            output = model(**dummy_input)

        B, C, H, W = 2, 3, 64, 64
        assert output["predicted_image"].shape == (B, C, H, W)
        assert output["predicted_image"].min() >= 0.0
        assert output["predicted_image"].max() <= 1.0

    def test_mc_dropout(self, model, dummy_input):
        """MC-Dropout produces uncertainty estimates."""
        output = model.predict_with_uncertainty(
            **dummy_input, n_passes=3,
        )
        assert "class_std" in output
        assert "uncertainty_image" in output
        assert output["n_mc_passes"] == 3

    def test_with_magnetic_features(self, model, dummy_input):
        """Forward pass works with magnetic features provided."""
        B, T = 2, 4
        magnetic = torch.randn(B, T, 16)
        mag_missing = torch.zeros(B, T, dtype=torch.bool)

        model.eval()
        with torch.no_grad():
            output = model(
                **dummy_input,
                magnetic=magnetic,
                mag_missing=mag_missing,
            )
        assert "class_probs" in output

    def test_missing_magnetic_features(self, model, dummy_input):
        """Forward pass works when magnetic features are missing."""
        model.eval()
        with torch.no_grad():
            output = model(
                **dummy_input,
                magnetic=None,
                mag_missing=None,
            )
        assert "class_probs" in output


# ---------------------------------------------------------------------------
# Loss tests
# ---------------------------------------------------------------------------
class TestLosses:
    def test_focal_loss(self):
        sys.path.insert(0, str(PROJECT_ROOT))
        # Import from the training script's FocalLoss
        from scripts.train_image_forecaster import FocalLoss
        loss_fn = FocalLoss(gamma=2.0, alpha=0.25)
        logits = torch.randn(4, 3)
        targets = torch.tensor([[1, 0, 0], [0, 1, 0], [1, 1, 0], [0, 0, 0]], dtype=torch.float32)
        loss = loss_fn(logits, targets)
        assert loss.item() >= 0
        assert not torch.isnan(loss)

    def test_ssim_loss(self):
        from models.vision.image_forecaster import SSIMLoss
        ssim = SSIMLoss(window_size=7, channels=3)
        pred = torch.rand(2, 3, 32, 32)
        target = torch.rand(2, 3, 32, 32)
        loss = ssim(pred, target)
        assert 0 <= loss.item() <= 2.0

    def test_image_forecast_loss(self):
        from models.vision.image_forecaster import ImageForecastLoss
        loss_fn = ImageForecastLoss(ssim_weight=0.5, channels=3)
        pred = torch.rand(2, 3, 32, 32)
        target = torch.rand(2, 3, 32, 32)
        total, components = loss_fn(pred, target)
        assert total.item() >= 0
        assert "l1" in components
        assert "ssim" in components


# ---------------------------------------------------------------------------
# Preprocessing tests
# ---------------------------------------------------------------------------
class TestPreprocessing:
    def test_image_normalizer_per_image(self):
        from services.vision.preprocessing import ImageNormalizer
        norm = ImageNormalizer(mode="per_image")
        img = torch.rand(3, 64, 64) * 255
        normed = norm.transform(img)
        assert normed.min() >= 0.0
        assert normed.max() <= 1.0 + 1e-6

    def test_image_normalizer_robust(self):
        from services.vision.preprocessing import ImageNormalizer
        norm = ImageNormalizer(mode="robust")
        images = [torch.rand(3, 32, 32) for _ in range(10)]
        norm.fit(images)

        img = torch.rand(3, 32, 32)
        normed = norm.transform(img)
        assert normed is not None

    def test_normalizer_save_load(self):
        from services.vision.preprocessing import ImageNormalizer
        norm = ImageNormalizer(mode="robust")
        images = [torch.rand(3, 16, 16) for _ in range(5)]
        norm.fit(images)

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            norm.save(f.name)
            loaded = ImageNormalizer.load(f.name)

        img = torch.rand(3, 16, 16)
        a = norm.transform(img)
        b = loaded.transform(img)
        assert torch.allclose(a, b, atol=1e-6)

    def test_solar_augmentation(self):
        from services.vision.preprocessing import SolarAugmentation
        aug = SolarAugmentation(enabled=True)
        img = torch.rand(3, 64, 64)
        result = aug(img)
        assert result.shape == (3, 64, 64)
        assert result.min() >= 0.0
        assert result.max() <= 1.0


# ---------------------------------------------------------------------------
# Sequence builder test
# ---------------------------------------------------------------------------
class TestSequenceBuilder:
    def test_chronological_split(self):
        from scripts.build_image_dataset import chronological_split

        seqs = [{"anchor_timestamp": f"2024-10-{i+1:02d}T00:00:00+00:00"} for i in range(20)]
        train, val, test = chronological_split(seqs, train_frac=0.7, val_frac=0.15)

        assert len(train) + len(val) + len(test) == 20
        assert len(train) >= len(val)
        assert len(train) >= len(test)

        # Verify chronological order maintained
        all_ts = [s["anchor_timestamp"] for s in train + val + test]
        assert all_ts == sorted(all_ts)

    def test_flux_to_goes_class(self):
        from scripts.build_image_dataset import flux_to_goes_class
        assert flux_to_goes_class(1e-9) == "A"
        assert flux_to_goes_class(5e-8) == "B"
        assert flux_to_goes_class(5e-7) == "C"
        assert flux_to_goes_class(5e-6) == "M"
        assert flux_to_goes_class(5e-5) == "X"


# ---------------------------------------------------------------------------
# Dataset test
# ---------------------------------------------------------------------------
class TestDataset:
    def test_dataset_loading(self):
        """Test SolarSequenceDataset with synthetic data."""
        from services.vision.preprocessing import SolarSequenceDataset, ImageNormalizer

        # Create minimal synthetic sequence JSON
        sequences = [
            {
                "anchor_timestamp": "2024-10-01T12:00:00+00:00",
                "anchor_idx": 0,
                "window_start": "2024-10-01T08:00:00+00:00",
                "image_paths": [],  # Empty — will get zero images
                "timestamps": ["2024-10-01T08:00:00+00:00"],
                "soft_xray_flux": [1e-7],
                "log_soft_flux": [-7.0],
                "n_images": 1,
                "labels": {
                    "h15m": {"C_plus": True, "M_plus": False, "X_plus": False},
                    "h30m": {"C_plus": True, "M_plus": False, "X_plus": False},
                    "h60m": {"C_plus": True, "M_plus": True, "X_plus": False},
                    "h360m": {"C_plus": True, "M_plus": True, "X_plus": False},
                    "h720m": {"C_plus": True, "M_plus": True, "X_plus": True},
                    "h1440m": {"C_plus": True, "M_plus": True, "X_plus": True},
                },
                "synthetic": True,
            }
        ]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(sequences, f)
            json_path = f.name

        dataset = SolarSequenceDataset(
            json_path,
            image_size=32,
            normalizer=ImageNormalizer(mode="per_image"),
            max_seq_len=4,
        )

        assert len(dataset) == 1
        sample = dataset[0]
        assert "image_seq" in sample
        assert "telemetry" in sample
        assert "labels" in sample
        assert sample["is_synthetic"] is True

        Path(json_path).unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Grad-CAM test
# ---------------------------------------------------------------------------
class TestGradCAM:
    def test_grad_cam_generation(self):
        from models.vision.solar_image_forecaster import SolarImageForecaster
        from models.vision.grad_cam import GradCAM

        model = SolarImageForecaster(
            backbone="resnet18",
            image_size=64,
            spatial_dim=32,
            temporal_dim=32,
            n_heads=2,
            n_temporal_layers=1,
            tel_embed_dim=16,
            mag_embed_dim=16,
            fusion_output_dim=32,
            classifier_hidden=16,
            max_seq_len=4,
        )

        cam = GradCAM(model, target_layer="cnn_encoder.feature_extractor")

        image_seq = torch.randn(1, 4, 3, 64, 64)
        telemetry = torch.randn(1, 4, 2)

        heatmap = cam.generate(image_seq, telemetry, horizon="h60m", class_idx=1)
        assert heatmap.shape == (1, 64, 64)
        assert heatmap.min() >= 0.0
        assert heatmap.max() <= 1.0

        cam.remove_hooks()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
