import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd


def generate_data(num_points: int = 1440):
    start_time = datetime.utcnow() - timedelta(days=1)
    times = [start_time + timedelta(minutes=i) for i in range(num_points)]

    # Simulate quiescent solar background flux with periodic fluctuations
    base_flux = 1e-8
    noise = np.random.normal(0, 0.2 * base_flux, num_points)
    soft_flux = base_flux + noise + (np.sin(np.linspace(0, 4*np.pi, num_points)) * 0.1 * base_flux)

    # Inject a simulated M-class solar flare
    flare_start = int(num_points * 0.4)
    flare_peak = int(num_points * 0.45)
    flare_end = int(num_points * 0.6)

    for i in range(flare_start, flare_peak):
        factor = (i - flare_start) / (flare_peak - flare_start)
        soft_flux[i] += 1.5e-5 * (factor ** 2) # Rise phase

    for i in range(flare_peak, flare_end):
        factor = (flare_end - i) / (flare_end - flare_peak)
        soft_flux[i] += 1.5e-5 * (factor ** 3) # Decay phase

    hard_flux = soft_flux * 0.1 + np.random.normal(0, 0.01 * base_flux, num_points)

    df = pd.DataFrame({
        "time": times,
        "soft_xray_flux": np.clip(soft_flux, 1e-9, None),
        "hard_xray_flux": np.clip(hard_flux, 1e-10, None),
        "quality_flag": [0] * num_points
    })

    os.makedirs("data/sample", exist_ok=True)
    df.to_csv("data/sample/synthetic_solexs.csv", index=False)
    print("Generated synthetic data in data/sample/synthetic_solexs.csv")

if __name__ == "__main__":
    generate_data()
