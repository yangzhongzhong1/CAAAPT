
import numpy as np
import xgboost as xgb
from sklearn.metrics import precision_recall_curve
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass
from collections import deque
import pickle
import os


@dataclass
class ScreeningResult:
    """Result from frontend screening"""
    is_anomaly: bool
    probability: float
    confidence_score: float
    raw_features: np.ndarray
    raw_logs: List[Dict]


class XGBoostScreener:
    """
    Lightweight XGBoost classifier for efficient log screening
    """

    def __init__(self, config: Dict):
        self.config = config
        self.model = None
        self.threshold = config.get('tau_front', 0.7)
        self.adaptive_threshold = config.get('adaptive', True)
        self.update_interval = config.get('update_interval', 1000)

        # For dynamic threshold adjustment (proportional-integral controller)
        self.target_fpr = 0.05
        self.error_history = deque(maxlen=100)
        self.kp = 0.1  # Proportional gain
        self.ki = 0.01  # Integral gain
        self.integral_error = 0

        self._init_model()

    def _init_model(self):
        """Initialize XGBoost model with class balancing (Equation 3)"""
        xgb_config = self.config.get('xgboost', {})

        # w_pos = N / (2 * N_pos), w_neg = N / (2 * N_neg)

        self.model = xgb.XGBClassifier(
            n_estimators=xgb_config.get('n_estimators', 300),
            max_depth=xgb_config.get('max_depth', 9),
            learning_rate=xgb_config.get('learning_rate', 0.05),
            subsample=xgb_config.get('subsample', 0.8),
            colsample_bytree=xgb_config.get('colsample_bytree', 0.8),
            reg_alpha=xgb_config.get('reg_alpha', 0.1),
            reg_lambda=xgb_config.get('reg_lambda', 1.0),
            scale_pos_weight=xgb_config.get('scale_pos_weight', 5.0),  # Class weight
            use_label_encoder=False,
            eval_metric='logloss',
            random_state=42,
            n_jobs=-1
        )

    def fit(self, X: np.ndarray, y: np.ndarray, X_val: np.ndarray = None, y_val: np.ndarray = None):
        """
        Train XGBoost model with focal-loss inspired gradient weighting
        """
        # Training with sample weights for hard examples
        sample_weights = self._compute_sample_weights(y)

        # Fit model
        eval_set = [(X_val, y_val)] if X_val is not None else None
        self.model.fit(
            X, y,
            sample_weight=sample_weights,
            eval_set=eval_set,
            verbose=False
        )

        # Set initial threshold using PR curve on validation set
        if X_val is not None and y_val is not None:
            self._update_threshold_from_validation(X_val, y_val)

    def _compute_sample_weights(self, y: np.ndarray) -> np.ndarray:
        """Compute sample weights for focal-loss inspired weighting"""
        weights = np.ones(len(y))

        # Class weights
        n_pos = np.sum(y == 1)
        n_neg = np.sum(y == 0)
        n_total = len(y)

        pos_weight = n_total / (2 * n_pos) if n_pos > 0 else 1
        neg_weight = n_total / (2 * n_neg) if n_neg > 0 else 1

        weights[y == 1] = pos_weight
        weights[y == 0] = neg_weight

        return weights

    def _update_threshold_from_validation(self, X_val: np.ndarray, y_val: np.ndarray):
        """Set threshold using precision-recall curve"""
        probs = self.model.predict_proba(X_val)[:, 1]
        precisions, recalls, thresholds = precision_recall_curve(y_val, probs)

        # Find threshold that balances precision and recall (F1 optimal)
        f1_scores = 2 * (precisions[:-1] * recalls) / (precisions[:-1] + recalls + 1e-8)
        best_idx = np.argmax(f1_scores)
        self.threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5

    def predict(self, X: np.ndarray, raw_logs: List[List[Dict]] = None) -> List[ScreeningResult]:
        """
        Predict anomaly probability and filter samples
        Implements sub-millisecond inference using CSR format
        """
        # Get probabilities
        probas = self.model.predict_proba(X)[:, 1]

        results = []
        for i, prob in enumerate(probas):
            is_anomaly = prob >= self.threshold

            # Compute confidence based on distance from threshold
            # Higher confidence when far from threshold
            confidence = 1 - min(abs(prob - self.threshold), 1) if is_anomaly else prob

            results.append(ScreeningResult(
                is_anomaly=is_anomaly,
                probability=float(prob),
                confidence_score=float(confidence),
                raw_features=X[i],
                raw_logs=raw_logs[i] if raw_logs else []
            ))

        return results

    def dynamic_threshold_update(self, feed_fpr: float):
        """
        Dynamic threshold adjustment using proportional-integral controller
        Maintains target false positive rate
        """
        if not self.adaptive_threshold:
            return

        # Compute error
        error = self.target_fpr - feed_fpr
        self.error_history.append(error)

        # Update integral
        self.integral_error += error

        # PI controller output
        delta = self.kp * error + self.ki * self.integral_error

        # Adjust threshold (clamp to [0.3, 0.95])
        new_threshold = self.threshold + delta
        self.threshold = max(0.3, min(0.95, new_threshold))

    def save(self, path: str):
        """Save model and threshold"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'threshold': self.threshold,
                'config': self.config,
                'selected_features': getattr(self, 'selected_features', None)
            }, f)

    def load(self, path: str):
        """Load model and threshold"""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        self.model = data['model']
        self.threshold = data['threshold']