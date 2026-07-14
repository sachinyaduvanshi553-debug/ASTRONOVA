#!/usr/bin/env python3
import os
import sys
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
import traceback

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.vision.data.alignment import MultiSourceAligner
from services.vision.data.sequence_builder import SequenceBuilder
from services.vision.model import SolarVisionPredictor
from services.vision.trainer import VisionTrainer
from services.vision.losses import SolarVisionLoss
from services.vision.inference import VisionInferencePipeline
from services.vision.visualization import XAIVisualizer
from services.vision.uncertainty import UncertaintyEngine
from services.vision.reports import ReportGenerator
from services.vision.export import ModelExporter


def verify_pipeline():
    print("=" * 60)
    print("ASTRONOVA VISION PIPELINE VERIFICATION")
    print("=" * 60)
    print(f"Time: {datetime.utcnow().isoformat()}")
    print("-" * 60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    results = {}
    
    # Generate dummy data
    def make_dummy_data():
        import pandas as pd
        
        os.makedirs('data/vision_dummy', exist_ok=True)
        # Create a dummy image
        img = np.zeros((512, 512, 3), dtype=np.uint8)
        img[200:300, 200:300] = 255 # White square
        
        times = pd.date_range("2024-01-01 00:00:00", periods=10, freq="15min", tz="UTC")
        img_paths = []
        for t in times:
            filename = t.strftime("%Y%m%d_%H%M%S") + "_dummy.jpg"
            path = os.path.join('data/vision_dummy', filename)
            import cv2
            cv2.imwrite(path, img)
            img_paths.append(path)
            
        goes_df = pd.DataFrame({
            'time': times,
            'xrsa_flux': np.linspace(1e-8, 1e-4, 10),
            'xrsb_flux': np.linspace(1e-7, 1e-3, 10)
        })
        goes_csv = 'data/vision_dummy/goes.csv'
        goes_df.to_csv(goes_csv, index=False)
        return 'data/vision_dummy', goes_csv
        
    try:
        data_dir, goes_csv = make_dummy_data()
        results["Data Preparation"] = "PASS"
    except Exception as e:
        results["Data Preparation"] = f"FAIL: {e}"
        return results

    # 1. Dataset loading & alignment
    try:
        print("[1] Testing Dataset Alignment...")
        aligner = MultiSourceAligner(image_dir=data_dir, goes_csv=goes_csv)
        df = aligner.align()
        assert len(df) == 10
        results["Dataset Alignment"] = "PASS"
    except Exception as e:
        results["Dataset Alignment"] = f"FAIL: {e}"
        traceback.print_exc()

    # 2. Sequence Builder
    try:
        print("[2] Testing Sequence Builder...")
        builder = SequenceBuilder(seq_len=2, prediction_horizon_minutes=15, cadence_minutes=15)
        seqs = builder.build_sequences(df)
        assert len(seqs) == 8 # 10 - 2 (seq_len)
        results["Sequence Builder"] = "PASS"
    except Exception as e:
        results["Sequence Builder"] = f"FAIL: {e}"
        traceback.print_exc()

    # 3. Model Forward/Backward Pass
    try:
        print("[3] Testing Model Forward/Backward...")
        model = SolarVisionPredictor(pretrained_encoder=False).to(device)
        images = torch.rand(2, 2, 3, 512, 512).to(device)
        telemetry = torch.rand(2, 10).to(device)
        physics = torch.rand(2, 5).to(device)
        
        # Forward
        out = model(images, telemetry, physics)
        assert 'predicted_image' in out
        assert out['predicted_image'].shape == (2, 3, 512, 512)
        assert out['class_logits'].shape == (2, 5)
        assert out['reg_output'].shape == (2, 1)
        
        # Backward
        loss = out['predicted_image'].sum() + out['reg_output'].sum()
        loss.backward()
        
        results["Model Forward/Backward"] = "PASS"
    except Exception as e:
        results["Model Forward/Backward"] = f"FAIL: {e}"
        traceback.print_exc()

    # 4. Uncertainty & XAI
    try:
        print("[4] Testing XAI & Uncertainty...")
        ue = UncertaintyEngine(model, n_samples=2, device=device)
        u_res = ue.compute_pixel_uncertainty(images, telemetry, physics)
        assert 'confidence' in u_res
        
        xai = XAIVisualizer(model)
        gradcam = xai.generate_gradcam(images, telemetry, physics)
        assert gradcam.shape == (512, 512)
        
        ig = xai.integrated_gradients(images, telemetry, physics, n_steps=2)
        assert ig.shape == (512, 512)
        
        results["XAI & Uncertainty"] = "PASS"
    except Exception as e:
        results["XAI & Uncertainty"] = f"FAIL: {e}"
        traceback.print_exc()

    # 5. Export
    try:
        print("[5] Testing Model Export...")
        exporter = ModelExporter(model, output_dir='models/vision_test')
        onnx_path = exporter.export_onnx(seq_len=2)
        assert os.path.exists(onnx_path)
        results["Model Export"] = "PASS"
    except Exception as e:
        results["Model Export"] = f"FAIL: {e}"
        traceback.print_exc()
        
    # Generate final report
    report_md = "# 🛡️ AstroNova Final Verification Report\n\n"
    for k, v in results.items():
        icon = "✅" if v == "PASS" else "❌"
        report_md += f"**{k}**: {icon} {v}\n\n"
        
    os.makedirs('reports', exist_ok=True)
    with open('reports/final_verification.md', 'w', encoding='utf-8') as f:
        f.write(report_md)
        
    print("\n--- Verification Complete ---")
    for k, v in results.items():
        print(f"{k}: {v}")

if __name__ == "__main__":
    verify_pipeline()
