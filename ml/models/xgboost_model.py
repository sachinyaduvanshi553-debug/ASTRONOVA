import logging
import os
import pickle

import numpy as np

logger = logging.getLogger("astronova.xgboost_model")

try:
    import xgboost as xgb
    HAS_XGB = True
    logger.info("XGBoost library detected and loaded successfully.")
except ImportError:
    HAS_XGB = False
    from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
    logger.warning("XGBoost not available. Falling back to Scikit-Learn Gradient Boosting.")

class XGBoostForecaster:
    """
    XGBoost / Gradient Boosting solar flare forecaster.
    Implements a scikit-learn compatible interface for multi-horizon forecasting
    by training separate classifiers and regressors for each target horizon.
    """
    def __init__(
        self, 
        input_size: int = 15, 
        seq_len: int = 10,
        num_classes: int = 5,
        num_horizons: int = 4,
        max_depth: int = 5,
        n_estimators: int = 100,
        learning_rate: float = 0.05
    ):
        self.input_size = input_size
        self.seq_len = seq_len
        self.num_classes = num_classes
        self.num_horizons = num_horizons

        self.classifiers = []
        self.regressors = []

        for _i in range(num_horizons):
            if HAS_XGB:
                clf = xgb.XGBClassifier(
                    max_depth=max_depth,
                    n_estimators=n_estimators,
                    learning_rate=learning_rate,
                    objective='multi:softprob',
                    eval_metric='mlogloss',
                    random_state=42
                )
                reg = xgb.XGBRegressor(
                    max_depth=max_depth,
                    n_estimators=n_estimators,
                    learning_rate=learning_rate,
                    objective='reg:squarederror',
                    random_state=42
                )
            else:
                clf = GradientBoostingClassifier(
                    max_depth=max_depth,
                    n_estimators=n_estimators,
                    learning_rate=learning_rate,
                    random_state=42
                )
                reg = GradientBoostingRegressor(
                    max_depth=max_depth,
                    n_estimators=n_estimators,
                    learning_rate=learning_rate,
                    random_state=42
                )
            self.classifiers.append(clf)
            self.regressors.append(reg)

    def _flatten_features(self, X: np.ndarray) -> np.ndarray:
        """
        Flattens 3D sequence features [batch, seq_len, features] into 2D [batch, seq_len * features]
        for tree-based models.
        """
        if len(X.shape) == 3:
            return X.reshape(X.shape[0], -1)
        return X

    def fit(self, X: np.ndarray, y_class: np.ndarray, y_reg: np.ndarray):
        """
        Fits classifiers and regressors for all horizons.

        Args:
            X: Input sequences of shape [batch_size, seq_len, features] or [batch_size, features_flat]
            y_class: Multi-horizon classification labels of shape [batch_size, num_horizons]
            y_reg: Multi-horizon regression targets of shape [batch_size, num_horizons]
        """
        X_flat = self._flatten_features(X)
        logger.info(f"Training XGBoost/GBDT models on feature shape {X_flat.shape}...")

        for i in range(self.num_horizons):
            logger.info(f"Fitting models for Horizon {i + 1}/{self.num_horizons}...")
            # Classification
            self.classifiers[i].fit(X_flat, y_class[:, i])
            # Regression
            self.regressors[i].fit(X_flat, y_reg[:, i])

        logger.info("Model training completed successfully.")

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Predicts multi-horizon probabilities and regression values.

        Returns:
            class_probs: [batch_size, num_horizons, num_classes]
            reg_vals: [batch_size, num_horizons, 1]
        """
        X_flat = self._flatten_features(X)
        batch_size = X_flat.shape[0]

        probs_out = np.zeros((batch_size, self.num_horizons, self.num_classes))
        regs_out = np.zeros((batch_size, self.num_horizons, 1))

        for i in range(self.num_horizons):
            # Predict probabilities and map to 5 classes based on .classes_
            probs = self.classifiers[i].predict_proba(X_flat)
            if hasattr(self.classifiers[i], 'classes_'):
                classes = self.classifiers[i].classes_
                for c_idx, c in enumerate(classes):
                    probs_out[:, i, int(c)] = probs[:, c_idx]
            else:
                probs_out[:, i, :probs.shape[1]] = probs
                
            # Predict flux values
            regs_out[:, i, 0] = self.regressors[i].predict(X_flat)

        return probs_out, regs_out

    def save(self, filepath: str):
        """Saves the forecaster bundle using pickle."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        data = {
            "input_size": self.input_size,
            "seq_len": self.seq_len,
            "num_classes": self.num_classes,
            "num_horizons": self.num_horizons,
            "classifiers": self.classifiers,
            "regressors": self.regressors
        }
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        logger.info(f"Saved XGBoost forecaster bundle to {filepath}")

    @classmethod
    def load(cls, filepath: str) -> 'XGBoostForecaster':
        """Loads a forecaster bundle from file."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)

        forecaster = cls(
            input_size=data["input_size"],
            seq_len=data["seq_len"],
            num_classes=data["num_classes"],
            num_horizons=data["num_horizons"]
        )
        forecaster.classifiers = data["classifiers"]
        forecaster.regressors = data["regressors"]
        logger.info(f"Loaded XGBoost forecaster bundle from {filepath}")
        return forecaster
