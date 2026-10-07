"""
tests/vision/test_image_forecaster_compatibility.py — Compatibility and Contract Test Suite.

Verifies:
  1. Imports and exports contract.
  2. All six standard forecasting horizons exist.
  3. Six classification heads exist and output [B, 3] logits.
  4. Sigmoid probabilities in [0, 1].
  5. Image input shape handling [B, T, 3, H, W].
  6. Telemetry feature input [B, T, 2].
  7. Magnetic feature input [B, 16] with missing mask.
  8. Physics feature input [B, 5].
  9. Missing/None optional features work gracefully.
  10. Spatial heatmap output exists [B, 1, H, W].
  11. Predicted future image exists [B, 3, H, W] in [0, 1].
  12. ImageForecastLoss executes and logs components.
  13. Loss ignores missing future image targets when target_available is False.
  14. FutureTargetResolver strictly rejects T <= T_anchor and prevents leakage.
  15. Checkpoint saving and loading works seamlessly.
  16. CPU and CUDA forward passes work.
  17. Batch size 2 and 128x128 image size pass.
"""
from __future__ import annotations

import io
import sys
import tempfile
from pathlib import Path

import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class TestSolarImageForecasterCompatibility:
    """Rigorous compatibility testing suite for SolarImageForecaster."""

    @pytest.fixture
    def model(self):
        from models.vision.solar_image_forecaster import SolarImageForecaster
        return SolarImageForecaster(
            backbone="resnet18",
            image_size=128,
            channels=3,
            pretrained_encoder=False,
            temporal_model="transformer",
            spatial_dim=64,
            temporal_dim=64,
            n_heads=2,
            n_temporal_layers=1,
            telemetry_dim=2,
            tel_embed_dim=32,
            magnetic_dim=16,
            mag_embed_dim=32,
            physics_dim=5,
            physics_embed_dim=32,
            fusion_output_dim=64,
            classifier_hidden=32,
            dropout=0.1,
            max_seq_len=4,
        )

    def test_01_imports(self):
        from models.vision.solar_image_forecaster import SolarImageForecaster, HORIZONS
        assert SolarImageForecaster is not None
        assert isinstance(HORIZONS, list)

    def test_02_six_horizons_exist(self):
        from models.vision.solar_image_forecaster import HORIZONS
        expected = ["h15m", "h30m", "h60m", "h360m", "h720m", "h1440m"]
        assert HORIZONS == expected

    def test_03_04_05_six_heads_logits_and_probs(self, model):
        from models.vision.solar_image_forecaster import HORIZONS
        model.eval()
        B, T, C, H, W = 2, 4, 3, 128, 128
        img_seq = torch.randn(B, T, C, H, W)
        tel = torch.randn(B, T, 2)

        with torch.no_grad():
            out = model(image_seq=img_seq, telemetry=tel)

        assert "class_logits" in out
        assert "class_probs" in out
        assert "m_plus_probability" in out

        for h in HORIZONS:
            assert h in out["class_logits"]
            assert out["class_logits"][h].shape == (B, 3)
            assert out["class_probs"][h].shape == (B, 3)
            assert (out["class_probs"][h] >= 0.0).all()
            assert (out["class_probs"][h] <= 1.0).all()
            assert out["m_plus_probability"][h].shape == (B,)

    def test_06_to_10_multimodal_inputs_and_missing_features(self, model):
        model.eval()
        B, T, C, H, W = 2, 4, 3, 128, 128
        img_seq = torch.randn(B, T, C, H, W)
        tel = torch.randn(B, 2)
        mag = torch.randn(B, 16)
        phys = torch.randn(B, 5)

        with torch.no_grad():
            # Full multimodal inputs
            out_full = model(img_seq, telemetry=tel, magnetic=mag, physics=phys)
            assert out_full["latent"].shape == (B, 64)

            # Missing optional inputs (telemetry=None, magnetic=None, physics=None)
            out_missing = model(img_seq, telemetry=None, magnetic=None, physics=None)
            assert out_missing["latent"].shape == (B, 64)

    def test_11_12_13_spatial_and_image_outputs(self, model):
        model.eval()
        B, T, C, H, W = 2, 4, 3, 128, 128
        img_seq = torch.randn(B, T, C, H, W)

        with torch.no_grad():
            out = model(img_seq)

        assert "location_heatmap" in out
        assert out["location_heatmap"].shape == (B, 1, 128, 128)
        assert (out["location_heatmap"] >= 0.0).all()
        assert (out["location_heatmap"] <= 1.0).all()

        assert "predicted_future_image" in out
        assert out["predicted_future_image"].shape == (B, 3, 128, 128)
        assert (out["predicted_future_image"] >= 0.0).all()
        assert (out["predicted_future_image"] <= 1.0).all()
        assert out["image_type"] == "AI_FORECAST"

    def test_14_15_loss_and_missing_target_handling(self, model):
        from models.vision.image_forecaster import ImageForecastLoss
        from models.vision.solar_image_forecaster import HORIZONS

        loss_fn = ImageForecastLoss(channels=3)
        B, T, C, H, W = 2, 4, 3, 128, 128
        img_seq = torch.randn(B, T, C, H, W)

        out = model(img_seq)

        # Targets with no future images (target_available = False)
        labels = {h: torch.zeros(B, 3) for h in HORIZONS}
        future_images = {h: None for h in HORIZONS}
        target_avail = {h: torch.zeros(B, dtype=torch.bool) for h in HORIZONS}

        targets = {
            "labels": labels,
            "future_images": future_images,
            "target_available": target_avail,
        }

        total_loss, details = loss_fn(out, targets)
        assert total_loss.item() >= 0.0
        assert not torch.isnan(total_loss)
        assert details["img_total"] == 0.0  # Ignored when targets unavailable!

    def test_16_future_target_resolver_leakage_safety(self):
        from models.vision.future_target_resolver import FutureTargetResolver
        from datetime import datetime, timezone
        from PIL import Image

        with tempfile.TemporaryDirectory() as tmpdir:
            past_p = str(Path(tmpdir) / "past.png")
            anchor_p = str(Path(tmpdir) / "anchor.png")
            future_p = str(Path(tmpdir) / "future_1h.png")

            Image.new("RGB", (64, 64), color="red").save(past_p)
            Image.new("RGB", (64, 64), color="green").save(anchor_p)
            Image.new("RGB", (64, 64), color="blue").save(future_p)

            resolver = FutureTargetResolver(image_size=64, channels=3)
            resolver.image_index = [
                {"path": past_p, "timestamp": datetime(2024, 10, 1, 12, 0, tzinfo=timezone.utc)},
                {"path": anchor_p, "timestamp": datetime(2024, 10, 1, 13, 0, tzinfo=timezone.utc)},
                {"path": future_p, "timestamp": datetime(2024, 10, 1, 14, 0, tzinfo=timezone.utc)},
            ]

            # Anchor is 13:00
            future_imgs, target_avail = resolver.resolve_targets(
                anchor_timestamp="2024-10-01T13:00:00+00:00",
                input_image_paths=[past_p, anchor_p],
            )

            # 15m, 30m should not match past or anchor
            assert target_avail["h15m"] is False
            assert target_avail["h30m"] is False
            # 60m matches 14:00 (strictly > 13:00 and not in input sequence)
            assert target_avail["h60m"] is True
            assert future_imgs["h60m"] is not None

    def test_17_checkpoint_save_and_reload(self, model):
        B, T, C, H, W = 2, 4, 3, 128, 128
        img_seq = torch.randn(B, T, C, H, W)

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            tmp_path = f.name

        buf = io.BytesIO()
        torch.save({"model_state_dict": model.state_dict(), "model_config": model.get_config()}, buf)
        with open(tmp_path, "wb") as f:
            f.write(buf.getvalue())

        loaded_data = torch.load(tmp_path, map_location="cpu", weights_only=False)
        assert "model_state_dict" in loaded_data
        model.load_state_dict(loaded_data["model_state_dict"])
        Path(tmp_path).unlink(missing_ok=True)

    def test_18_19_cpu_cuda_execution(self, model):
        # CPU
        model.cpu()
        img_seq = torch.randn(2, 4, 3, 128, 128)
        out = model(img_seq)
        assert out["m_plus_probability"]["h60m"].shape == (2,)

        # CUDA if available
        if torch.cuda.is_available():
            model.cuda()
            img_seq_cuda = img_seq.cuda()
            out_cuda = model(img_seq_cuda)
            assert out_cuda["m_plus_probability"]["h60m"].shape == (2,)

    def test_20_21_batch2_and_128x128(self, model):
        B, T, C, H, W = 2, 4, 3, 128, 128
        img_seq = torch.randn(B, T, C, H, W)
        out = model(img_seq)
        assert out["predicted_future_image"].shape == (2, 3, 128, 128)
        assert out["location_heatmap"].shape == (2, 1, 128, 128)
