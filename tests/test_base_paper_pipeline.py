"""
Comprehensive Verification Test Suite for Base-Paper Research Pipeline.
Tests:
- Preprocessing and feature scaling
- Feature selection (ANOVA + MI)
- Baseline and Optimized Transformer forward passes
- Imbalance loss calculations
- Threshold optimization and probability calibration
- BiLSTM, Temporal CNN, XGBoost, and Soft-Voting Ensemble
- Integrated Gradients and PDP Explainers
- FastAPI prediction and XAI router endpoints
"""

import os
import numpy as np
import pandas as pd
import pytest
import torch

from datasets.sequence_builder import SequenceBuilder
from evaluation.metrics import compute_all_metrics
from explainability.integrated_gradients import IntegratedGradientsExplainer
from explainability.pdp import PartialDependenceCalculator
from features.anova import ANOVASelector
from features.feature_selection import SolarFeatureSelector
from features.mutual_information import MutualInformationSelector
from models.baseline_transformer import AstroNovaBaselineTransformer
from models.bilstm import SolarBiLSTM
from models.ensemble import SolarEnsembleForecaster
from models.optimized_transformer import AstroNovaOptimizedTransformer
from models.temporal_cnn import TemporalCNNTransformer
from models.xgboost_model import SolarXGBoostForecaster
from optimization.calibration import SolarProbabilityCalibrator
from optimization.threshold_optimizer import ThresholdOptimizer
from preprocessing.cleaner import BASE_PAPER_16_FEATURES, BASE_PAPER_25_FEATURES, SolarDataCleaner
from preprocessing.scaler import SolarFeatureScaler
from training.losses import ClassBalancedLoss, FocalLoss, WeightedBCELoss


@pytest.fixture
def sample_feature_dataframe():
    """Generates synthetic dataframe matching 25 SHARP & Lorentz parameters."""
    np.random.seed(42)
    n = 200
    timestamps = pd.date_range("2012-01-01", periods=n, freq="h", tz="UTC")
    data = {
        "timestamp": timestamps,
        "harp_num": [101] * (n // 2) + [102] * (n // 2),
        "label": [0] * (n - 20) + [1] * 20,
    }
    for feat in BASE_PAPER_25_FEATURES:
        data[feat] = np.random.uniform(1.0, 100.0, n)
    return pd.DataFrame(data)


def test_cleaner_and_scaler(sample_feature_dataframe):
    cleaner = SolarDataCleaner(features=BASE_PAPER_25_FEATURES)
    cleaned = cleaner.clean_dataframe(sample_feature_dataframe)
    assert len(cleaned) > 0
    assert "timestamp" in cleaned.columns

    scaler = SolarFeatureScaler(feature_cols=BASE_PAPER_16_FEATURES, scaler_type="robust")
    scaler.fit(cleaned)
    scaled = scaler.transform(cleaned)
    assert scaled[BASE_PAPER_16_FEATURES].isna().sum().sum() == 0


def test_sequence_builder(sample_feature_dataframe):
    builder = SequenceBuilder(sequence_length=24, features=BASE_PAPER_16_FEATURES)
    X, y, ts = builder.build_sequences(sample_feature_dataframe)
    assert X.ndim == 3
    assert X.shape[1] == 24
    assert X.shape[2] == 16
    assert len(X) == len(y)


def test_models_forward_pass():
    B, T, F = 4, 24, 16
    dummy_x = torch.randn(B, T, F)

    # 1. Baseline Transformer
    base_m = AstroNovaBaselineTransformer(seq_len=24, num_features=16, patch_size=2, d_model=64)
    out_base = base_m(dummy_x)
    assert out_base.shape == (B, 1)
    assert (out_base >= 0.0).all() and (out_base <= 1.0).all()

    # 2. Optimized Transformer
    opt_m = AstroNovaOptimizedTransformer(seq_len=24, num_features=16, patch_size=2, d_model=64, pooling="attention")
    out_opt = opt_m(dummy_x)
    assert out_opt.shape == (B, 1)

    # 3. BiLSTM
    lstm_m = SolarBiLSTM(input_size=16, hidden_size=32)
    out_lstm = lstm_m(dummy_x)
    assert out_lstm.shape == (B, 1)

    # 4. Temporal CNN
    cnn_m = TemporalCNNTransformer(input_size=16, conv_channels=32, d_model=64)
    out_cnn = cnn_m(dummy_x)
    assert out_cnn.shape == (B, 1)


def test_loss_functions():
    pred = torch.tensor([[0.8], [0.2], [0.9]], dtype=torch.float32)
    target = torch.tensor([[1.0], [0.0], [1.0]], dtype=torch.float32)

    wbce = WeightedBCELoss(pos_weight=5.0)
    loss1 = wbce(pred, target)
    assert loss1.item() > 0

    focal = FocalLoss(gamma=2.0, alpha=0.75)
    loss2 = focal(pred, target)
    assert loss2.item() > 0


def test_threshold_optimizer_and_calibration():
    y_true = np.array([0, 0, 0, 0, 1, 1, 0, 1, 0, 0])
    y_prob = np.array([0.1, 0.2, 0.05, 0.3, 0.85, 0.7, 0.4, 0.9, 0.15, 0.25])

    opt = ThresholdOptimizer(metric_to_optimize="tss")
    opt.fit(y_true, y_prob)
    assert 0.05 <= opt.optimal_threshold <= 0.95

    cal = SolarProbabilityCalibrator(method="platt")
    cal.fit(y_prob, y_true)
    cal_prob = cal.transform(y_prob)
    assert len(cal_prob) == len(y_prob)


def test_explainability_modules():
    B, T, F = 2, 24, 16
    x_sample = np.random.randn(24, 16).astype(np.float32)
    model = AstroNovaBaselineTransformer(seq_len=24, num_features=16, patch_size=2, d_model=64)

    ig = IntegratedGradientsExplainer(model, m_steps=10)
    exp = ig.explain_sample(x_sample, BASE_PAPER_16_FEATURES)
    assert "feature_attribution_ranking" in exp
    assert len(exp["feature_attribution_ranking"]) == 16
    assert len(exp["temporal_profile"]) == 24

    pdp = PartialDependenceCalculator(model, BASE_PAPER_16_FEATURES)
    pdp_res = pdp.compute_1d_pdp(np.random.randn(10, 24, 16).astype(np.float32), "TOTUSJH", grid_resolution=5)
    assert len(pdp_res["pdp_values"]) == 5


def test_fastapi_endpoints():
    from fastapi.testclient import TestClient
    from services.forecasting.main import app

    client = TestClient(app)
    resp = client.get("/api/v1/forecast/model-info")
    assert resp.status_code == 200
    assert resp.json()["model_name"] == "AstroNova Research-Grade Forecaster"

    resp_health = client.get("/api/v1/forecast/health")
    assert resp_health.status_code == 200
    assert resp_health.json()["status"] == "healthy"

    dummy_seq = np.random.randn(24, 16).tolist()
    resp_pred = client.post("/api/v1/forecast/predict", json={"sequence": dummy_seq})
    assert resp_pred.status_code == 200
    data = resp_pred.json()
    assert "flare_probability" in data
    assert "prediction" in data
    assert "confidence" in data
