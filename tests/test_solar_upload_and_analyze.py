import io
import pytest
from PIL import Image
from fastapi.testclient import TestClient
from services.api.main import app

client = TestClient(app)

def test_upload_solar_image():
    # Create synthetic solar image payload
    img = Image.new("RGB", (256, 256), color=(255, 128, 0))
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format="JPEG")
    img_bytes = img_byte_arr.getvalue()

    response = client.post(
        "/vision/upload-and-analyze",
        files={"file": ("test_sdo_image.jpg", img_bytes, "image/jpeg")}
    )

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["filename"] == "test_sdo_image.jpg"
    assert data["file_type"] == "Solar Disc Image"
    assert "next_flare_origination" in data
    assert "estimated_window" in data["next_flare_origination"]
    assert "predicted_flare" in data
    assert "goes_class" in data["predicted_flare"]
    assert "active_region" in data
    assert "earth_impact" in data
    assert "xai" in data
    assert "gradcam_heatmap_base64" in data["xai"]

def test_upload_solar_csv_telemetry():
    csv_bytes = b"timestamp,soft_flux,hard_flux\n2026-07-24T12:00:00,1.2e-5,2.4e-6\n2026-07-24T12:01:00,1.5e-5,3.1e-6\n"
    response = client.post(
        "/vision/upload-and-analyze",
        files={"file": ("goes_flux_telemetry.csv", csv_bytes, "text/csv")}
    )

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["file_type"] == "Solar Telemetry Dataset"
    assert "summary_advisory" in data
