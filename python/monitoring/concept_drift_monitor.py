"""
Concept drift monitor detecting changes in the relationship between
features and target variables (e.g., volatility regime shifts).
Triggers automated retraining flags when model predictive power decays.
"""

import numpy as np
import polars as pl
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from collections import deque
import threading
from scipy import stats
from sklearn.linear_model import SGDClassifier, SGDRegressor
from sklearn.metrics import roc_auc_score, mean_squared_error

# Memory bounds
MAX_WINDOW_SIZE = 5000
MIN_SAMPLES_FOR_TEST = 500


@dataclass
class ConceptDriftResult:
    """Result of concept drift analysis."""
    timestamp_ns: int
    is_drifting: bool
    severity: str
    auc_decay: float
    error_increase: float
    recommended_action: str  # 'none', 'retrain_soon', 'retrain_now'


class ConceptDriftMonitor:
    """
    Monitors concept drift by tracking the stability of feature-target
    relationships using online learning and statistical tests.
    """

    def __init__(
        self,
        feature_names: List[str],
        task_type: str = 'classification',
        auc_threshold: float = 0.15,
        error_threshold: float = 0.25,
    ):
        self.feature_names = feature_names
        self.task_type = task_type
        self.auc_threshold = auc_threshold
        self.error_threshold = error_threshold
        
        # Rolling windows for recent and historical data
        self._recent_X: deque = deque(maxlen=MIN_SAMPLES_FOR_TEST)
        self._recent_y: deque = deque(maxlen=MIN_SAMPLES_FOR_TEST)
        self._historical_X: deque = deque(maxlen=MAX_WINDOW_SIZE)
        self._historical_y: deque = deque(maxlen=MAX_WINDOW_SIZE)
        
        # Online models for tracking
        if task_type == 'classification':
            self._recent_model = SGDClassifier(
                loss='log_loss',
                penalty='l2',
                alpha=0.0001,
                random_state=42,
                warm_start=True,
            )
            self._reference_model = SGDClassifier(
                loss='log_loss',
                penalty='l2',
                alpha=0.0001,
                random_state=42,
            )
        else:
            self._recent_model = SGDRegressor(
                penalty='l2',
                alpha=0.0001,
                random_state=42,
                warm_start=True,
            )
            self._reference_model = SGDRegressor(
                penalty='l2',
                alpha=0.0001,
                random_state=42,
            )
        
        self._model_fitted = False
        self._baseline_auc: float = 0.0
        self._baseline_error: float = 0.0
        
        # Tracking metrics
        self._auc_history: deque = deque(maxlen=100)
        self._error_history: deque = deque(maxlen=100)
        self._drift_flags: deque = deque(maxlen=1000)
        
        self._lock = threading.RLock()

    def fit_initial_model(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit reference model on initial training data."""
        with self._lock:
            # Subsample if too large
            if len(X) > MAX_WINDOW_SIZE:
                indices = np.random.choice(len(X), MAX_WINDOW_SIZE, replace=False)
                X = X[indices]
                y = y[indices]
            
            self._reference_model.fit(X, y)
            self._recent_model.fit(X, y)
            self._model_fitted = True
            
            # Store baseline performance
            if self.task_type == 'classification':
                y_pred = self._reference_model.predict_proba(X)[:, 1]
                self._baseline_auc = roc_auc_score(y, y_pred)
            else:
                y_pred = self._reference_model.predict(X)
                self._baseline_error = mean_squared_error(y, y_pred)
            
            # Populate historical window
            for xi, yi in zip(X, y):
                self._historical_X.append(xi)
                self._historical_y.append(yi)

    def add_sample(self, features: np.ndarray, target: float) -> None:
        """Add a new sample and update recent window."""
        with self._lock:
            self._recent_X.append(features)
            self._recent_y.append(target)
            self._historical_X.append(features)
            self._historical_y.append(target)
            
            # Partial fit on recent model (online learning)
            if self._model_fitted and len(self._recent_X) >= 10:
                X_batch = np.array(list(self._recent_X)[-10:])
                y_batch = np.array(list(self._recent_y)[-10:])
                try:
                    self._recent_model.partial_fit(X_batch, y_batch)
                except ValueError:
                    pass  # Ignore class imbalance errors in early stage

    def add_batch(self, X: np.ndarray, y: np.ndarray) -> None:
        """Add a batch of samples."""
        for xi, yi in zip(X, y):
            self.add_sample(xi, yi)

    def _compute_auc_decay(self) -> float:
        """Compute decay in AUC between reference and recent model."""
        if len(self._recent_X) < MIN_SAMPLES_FOR_TEST:
            return 0.0
        
        X_recent = np.array(list(self._recent_X))
        y_recent = np.array(list(self._recent_y))
        
        if self.task_type == 'classification':
            try:
                # Reference model performance on recent data
                y_pred_ref = self._reference_model.predict_proba(X_recent)[:, 1]
                auc_ref = roc_auc_score(y_recent, y_pred_ref)
                
                # Recent model performance
                y_pred_recent = self._recent_model.predict_proba(X_recent)[:, 1]
                auc_recent = roc_auc_score(y_recent, y_pred_recent)
                
                # Track history
                self._auc_history.append(auc_recent)
                
                # Decay from baseline
                if self._baseline_auc > 0:
                    return max(0, (self._baseline_auc - auc_recent) / self._baseline_auc)
                return 0.0
            except ValueError:
                return 0.0
        else:
            # Regression: use MSE increase instead
            y_pred_ref = self._reference_model.predict(X_recent)
            mse_ref = mean_squared_error(y_recent, y_pred_ref)
            
            y_pred_recent = self._recent_model.predict(X_recent)
            mse_recent = mean_squared_error(y_recent, y_pred_recent)
            
            self._error_history.append(mse_recent)
            
            if self._baseline_error > 0:
                return max(0, (mse_recent - mse_ref) / mse_ref)
            return 0.0

    def _compute_prediction_distribution_shift(self) -> float:
        """
        Use KS test to detect shift in prediction distributions.
        High values indicate concept drift.
        """
        if len(self._historical_X) < MIN_SAMPLES_FOR_TEST * 2:
            return 0.0
        
        X_hist = np.array(list(self._historical_X)[:MIN_SAMPLES_FOR_TEST])
        X_recent = np.array(list(self._recent_X))
        
        # Get predictions from reference model
        if self.task_type == 'classification':
            pred_hist = self._reference_model.predict_proba(X_hist)[:, 1]
            pred_recent = self._reference_model.predict_proba(X_recent)[:, 1]
        else:
            pred_hist = self._reference_model.predict(X_hist)
            pred_recent = self._reference_model.predict(X_recent)
        
        # KS test on prediction distributions
        ks_stat, _ = stats.ks_2samp(pred_hist, pred_recent)
        return ks_stat

    def check_concept_drift(self) -> Optional[ConceptDriftResult]:
        """
        Perform comprehensive concept drift check.
        Returns result or None if insufficient data.
        """
        with self._lock:
            if not self._model_fitted:
                return None
            
            if len(self._recent_X) < MIN_SAMPLES_FOR_TEST:
                return None
            
            # Calculate metrics
            auc_decay = self._compute_auc_decay()
            ks_shift = self._compute_prediction_distribution_shift()
            
            # Combined drift score
            drift_score = auc_decay + ks_shift * 0.5
            
            # Determine severity and action
            if drift_score < self.auc_threshold * 0.5:
                severity = 'none'
                action = 'none'
                is_drifting = False
            elif drift_score < self.auc_threshold:
                severity = 'low'
                action = 'none'
                is_drifting = False
            elif drift_score < self.auc_threshold * 1.5:
                severity = 'medium'
                action = 'retrain_soon'
                is_drifting = True
            else:
                severity = 'high'
                action = 'retrain_now'
                is_drifting = True
            
            result = ConceptDriftResult(
                timestamp_ns=int(np.datetime64('now', 'ns').astype(int)),
                is_drifting=is_drifting,
                severity=severity,
                auc_decay=auc_decay,
                error_increase=drift_score,
                recommended_action=action,
            )
            
            self._drift_flags.append(result)
            return result

    def get_retrain_recommendation(self) -> Tuple[bool, str]:
        """
        Get consolidated retrain recommendation.
        Returns (should_retrain, reason).
        """
        if not self._drift_flags:
            return False, 'insufficient_data'
        
        recent_flags = list(self._drift_flags)[-10:]
        high_severity_count = sum(1 for f in recent_flags if f.severity == 'high')
        medium_severity_count = sum(1 for f in recent_flags if f.severity == 'medium')
        
        if high_severity_count >= 3:
            return True, 'sustained_high_drift'
        
        if medium_severity_count >= 5:
            return True, 'sustained_medium_drift'
        
        if recent_flags and recent_flags[-1].recommended_action == 'retrain_now':
            return True, 'immediate_drift_detected'
        
        return False, 'no_significant_drift'

    def get_performance_trend(self) -> Dict:
        """Get trend of model performance over time."""
        if not self._auc_history and not self._error_history:
            return {'trend': 'unknown'}
        
        if self._auc_history:
            auc_list = list(self._auc_history)
            if len(auc_list) < 5:
                return {'trend': 'insufficient_data'}
            
            recent_avg = np.mean(auc_list[-5:])
            older_avg = np.mean(auc_list[-20:-5]) if len(auc_list) >= 20 else np.mean(auc_list[:-5])
            
            if recent_avg < older_avg * 0.95:
                trend = 'degrading'
            elif recent_avg > older_avg * 1.05:
                trend = 'improving'
            else:
                trend = 'stable'
            
            return {
                'trend': trend,
                'current_auc': recent_avg,
                'baseline_auc': self._baseline_auc,
                'decay_pct': (self._baseline_auc - recent_avg) / self._baseline_auc * 100,
            }
        else:
            # Regression case
            error_list = list(self._error_history)
            if len(error_list) < 5:
                return {'trend': 'insufficient_data'}
            
            recent_avg = np.mean(error_list[-5:])
            older_avg = np.mean(error_list[-20:-5]) if len(error_list) >= 20 else np.mean(error_list[:-5])
            
            if recent_avg > older_avg * 1.1:
                trend = 'degrading'
            elif recent_avg < older_avg * 0.9:
                trend = 'improving'
            else:
                trend = 'stable'
            
            return {'trend': trend, 'current_mse': recent_avg}

    def reset_reference_model(self) -> None:
        """Reset reference model to current recent model (post-retrain)."""
        with self._lock:
            # Copy recent model weights to reference
            self._reference_model.coef_ = self._recent_model.coef_.copy()
            if hasattr(self._recent_model, 'intercept_'):
                self._reference_model.intercept_ = self._recent_model.intercept_.copy()
            
            # Update baseline
            if self.task_type == 'classification':
                self._baseline_auc = 1.0 - self._compute_auc_decay()
            else:
                self._baseline_error = 0.0
            
            # Clear drift history
            self._drift_flags.clear()


if __name__ == "__main__":
    # Example usage
    np.random.seed(42)
    
    monitor = ConceptDriftMonitor(['f1', 'f2', 'f3'], task_type='classification')
    
    # Generate initial training data
    X_train = np.random.randn(1000, 3)
    y_train = (X_train[:, 0] + X_train[:, 1] > 0).astype(int)
    
    # Fit initial model
    monitor.fit_initial_model(X_train, y_train)
    print(f"Baseline AUC: {monitor._baseline_auc:.4f}")
    
    # Add samples with same concept
    for _ in range(300):
        x = np.random.randn(3)
        y = int(x[0] + x[1] > 0)
        monitor.add_sample(x, y)
    
    # Check drift (should be none)
    result = monitor.check_concept_drift()
    print(f"Initial check: {result.severity}, action={result.recommended_action}")
    
    # Add samples with shifted concept
    for _ in range(500):
        x = np.random.randn(3)
        y = int(x[0] - x[1] > 0)  # Different relationship
        monitor.add_sample(x, y)
    
    # Check drift (should detect)
    result = monitor.check_concept_drift()
    print(f"After concept shift: {result.severity}, action={result.recommended_action}")
    
    should_retrain, reason = monitor.get_retrain_recommendation()
    print(f"Retrain recommended: {should_retrain}, reason={reason}")
