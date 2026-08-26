import os
import sys
import time
import unittest

import psutil
from fastapi.testclient import TestClient

sys.path.append(os.path.abspath("."))
try:
    from services.forecasting.main import app
except ImportError:
    # Fallback if main.py is missing or fails to import
    from fastapi import FastAPI

    from services.forecasting.routers import forecast

    app = FastAPI()
    app.include_router(forecast.router)

client = TestClient(app)


class TestAPIEndToEnd(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.process = psutil.Process(os.getpid())

    def measure_resources(self):
        mem_mb = self.process.memory_info().rss / (1024 * 1024)
        cpu_pct = self.process.cpu_percent(interval=None)
        return mem_mb, cpu_pct

    def test_01_health(self):
        start = time.time()
        response = client.get("/api/v1/forecast/health")
        latency_ms = (time.time() - start) * 1000

        self.assertEqual(response.status_code, 200)
        print(f"Health Check Latency: {latency_ms:.2f} ms")
        self.assertLess(latency_ms, 100, "Latency exceeded 100ms")

    def test_02_predict_multi_horizon(self):
        payload = {
            "satellite_id": "GOES-16",
            "features": [[1.2e-5, 2.5e-6, 1.4e-6, 0.9e-6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]] * 10,
        }

        start = time.time()
        response = client.post("/api/v1/forecast/predict?current_flux=1e-7", json=payload)
        latency_ms = (time.time() - start) * 1000

        mem_mb, cpu_pct = self.measure_resources()

        print(f"Predict Latency: {latency_ms:.2f} ms | Mem: {mem_mb:.2f} MB | CPU: {cpu_pct:.1f}%")

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("prediction", data)
        self.assertIn("horizons", data["prediction"])

        # Check constraints
        self.assertLess(latency_ms, 100, "Latency exceeded 100ms")
        self.assertLess(mem_mb, 2048, "Memory exceeded 2GB")
        self.assertLess(cpu_pct, 70, "CPU exceeded 70%")

    def test_03_nowcast(self):
        start = time.time()
        response = client.get("/api/v1/forecast/nowcast?current_flux=2e-6")
        latency_ms = (time.time() - start) * 1000

        self.assertEqual(response.status_code, 200)
        print(f"Nowcast Latency: {latency_ms:.2f} ms")

    def test_04_shi(self):
        start = time.time()
        response = client.get("/api/v1/forecast/shi?current_flux=5e-5&similarity=0.8&sat_risk=0.5&impact_risk=0.4")
        latency_ms = (time.time() - start) * 1000

        self.assertEqual(response.status_code, 200)
        print(f"SHI Latency: {latency_ms:.2f} ms")

    def test_05_evaluate(self):
        start = time.time()
        response = client.get("/api/v1/forecast/evaluate")
        latency_ms = (time.time() - start) * 1000

        self.assertEqual(response.status_code, 200)
        print(f"Evaluate Latency: {latency_ms:.2f} ms")


if __name__ == "__main__":
    unittest.main(verbosity=2)
