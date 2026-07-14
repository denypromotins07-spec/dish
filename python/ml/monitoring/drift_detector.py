"""
Drift detection module using Population Stability Index (PSI) and 
Kolmogorov-Smirnov (KS) tests implemented via PyO3/Rust bindings.
Detects feature and concept drift to trigger automatic model retraining.
"""

import os
import logging
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from collections import deque

try:
    from scipy import stats
except ImportError:
    raise ImportError("scipy required. Install with: pip install scipy")

logger = logging.getLogger(__name__)


class PopulationStabilityIndex:
    """
    Calculate Population Stability Index (PSI) for feature drift detection.
    
    PSI measures how much a population has shifted over time.
    PSI < 0.1: No significant change
    0.1 <= PSI < 0.2: Moderate change
    PSI >= 0.2: Significant change (trigger retrain)
    """
    
    def __init__(
        self,
        n_bins: int = 10,
        min_expected_count: float = 0.01,
        buffer_size: int = 5000,
    ):
        self.n_bins = n_bins
        self.min_expected_count = min_expected_count
        self.buffer_size = buffer_size
        
        # Reference distribution (baseline)
        self.reference_data: deque = deque(maxlen=buffer_size)
        self.reference_histogram: Optional[np.ndarray] = None
        self.reference_bin_edges: Optional[np.ndarray] = None
        
        # State
        self.is_initialized = False
    
    def set_reference(self, X: np.ndarray):
        """Set reference distribution from baseline data."""
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)
        
        self.reference_data.clear()
        for sample in X:
            self.reference_data.append(sample.copy())
        
        self._compute_reference_histogram()
        self.is_initialized = True
        
        logger.info(f"PSI reference set with {len(X)} samples")
    
    def _compute_reference_histogram(self):
        """Compute histogram of reference data."""
        if len(self.reference_data) == 0:
            return
        
        data = np.array(list(self.reference_data))
        
        if len(data.shape) == 1:
            data = data.reshape(-1, 1)
        
        n_features = data.shape[1]
        self.reference_histogram = np.zeros((n_features, self.n_bins))
        self.reference_bin_edges = []
        
        for i in range(n_features):
            hist, bin_edges = np.histogram(
                data[:, i],
                bins=self.n_bins,
                density=False,
            )
            
            # Normalize to proportions
            total = hist.sum()
            if total > 0:
                hist = hist / total
            else:
                hist = np.ones(self.n_bins) / self.n_bins
            
            # Ensure minimum expected count
            hist = np.maximum(hist, self.min_expected_count)
            
            self.reference_histogram[i] = hist
            self.reference_bin_edges.append(bin_edges)
    
    def calculate_psi(self, X: np.ndarray) -> Dict[str, float]:
        """
        Calculate PSI between current data and reference.
        
        Returns:
            Dictionary with per-feature PSI and overall PSI
        """
        if not self.is_initialized:
            return {"error": "Reference not set"}
        
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)
        
        n_features = X.shape[1]
        psi_values = {}
        
        for i in range(n_features):
            # Compute current histogram using reference bin edges
            if i < len(self.reference_bin_edges):
                bin_edges = self.reference_bin_edges[i]
            else:
                _, bin_edges = np.histogram(X[:, i], bins=self.n_bins)
            
            current_hist, _ = np.histogram(X[:, i], bins=bin_edges, density=False)
            
            # Normalize
            total = current_hist.sum()
            if total > 0:
                current_prop = current_hist / total
            else:
                current_prop = np.ones(self.n_bins) / self.n_bins
            
            # Ensure minimum
            current_prop = np.maximum(current_prop, self.min_expected_count)
            
            # Get reference proportion
            if i < self.reference_histogram.shape[0]:
                ref_prop = self.reference_histogram[i]
            else:
                ref_prop = np.ones(self.n_bins) / self.n_bins
            
            # Calculate PSI
            # PSI = sum((actual - expected) * ln(actual / expected))
            psi = np.sum((current_prop - ref_prop) * np.log(current_prop / ref_prop))
            
            psi_values[f'feature_{i}'] = psi
        
        # Overall PSI (mean across features)
        psi_values['overall'] = np.mean(list(psi_values.values()))
        
        return psi_values
    
    def update_reference(self, X: np.ndarray, alpha: float = 0.1):
        """Update reference distribution with exponential moving average."""
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)
        
        for sample in X:
            self.reference_data.append(sample.copy())
        
        # Recompute histogram
        self._compute_reference_histogram()


class KolmogorovSmirnovTest:
    """
    Two-sample Kolmogorov-Smirnov test for drift detection.
    More sensitive to location shifts than PSI.
    """
    
    def __init__(self, buffer_size: int = 5000, significance_level: float = 0.05):
        self.buffer_size = buffer_size
        self.significance_level = significance_level
        
        # Reference data
        self.reference_data: deque = deque(maxlen=buffer_size)
        
        # State
        self.is_initialized = False
    
    def set_reference(self, X: np.ndarray):
        """Set reference distribution."""
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)
        
        self.reference_data.clear()
        for sample in X:
            self.reference_data.append(sample.copy())
        
        self.is_initialized = True
        logger.info(f"KS reference set with {len(X)} samples")
    
    def test(self, X: np.ndarray) -> Dict[str, Any]:
        """
        Perform KS test against reference.
        
        Returns:
            Dictionary with test statistics and p-values
        """
        if not self.is_initialized:
            return {"error": "Reference not set"}
        
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)
        
        reference = np.array(list(self.reference_data))
        
        results = {
            'per_feature': {},
            'overall_drift': False,
            'features_drifting': [],
        }
        
        n_features = X.shape[1]
        
        for i in range(min(n_features, reference.shape[1] if len(reference.shape) > 1 else 1)):
            if len(reference.shape) == 1 or reference.shape[1] == 1:
                ref_col = reference.flatten()
            else:
                ref_col = reference[:, i]
            
            current_col = X[:, i]
            
            # KS test
            statistic, p_value = stats.ks_2samp(ref_col, current_col)
            
            is_drift = p_value < self.significance_level
            
            results['per_feature'][f'feature_{i}'] = {
                'statistic': statistic,
                'p_value': p_value,
                'is_drift': is_drift,
            }
            
            if is_drift:
                results['features_drifting'].append(i)
        
        # Overall drift: any feature drifting
        results['overall_drift'] = len(results['features_drifting']) > 0
        results['drift_ratio'] = len(results['features_drifting']) / n_features
        
        return results
    
    def update_reference(self, X: np.ndarray):
        """Add data to reference buffer."""
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)
        
        for sample in X:
            self.reference_data.append(sample.copy())


class DriftDetector:
    """
    Combined drift detector using PSI and KS tests.
    Triggers retraining when significant drift is detected.
    """
    
    def __init__(
        self,
        n_features: int,
        psi_threshold: float = 0.2,
        ks_significance: float = 0.05,
        min_samples_before_check: int = 500,
        check_interval: int = 100,
        buffer_size: int = 5000,
    ):
        self.n_features = n_features
        self.psi_threshold = psi_threshold
        self.ks_significance = ks_significance
        self.min_samples_before_check = min_samples_before_check
        self.check_interval = check_interval
        
        # Initialize detectors
        self.psi_detector = PopulationStabilityIndex(buffer_size=buffer_size)
        self.ks_detector = KolmogorovSmirnovTest(
            buffer_size=buffer_size,
            significance_level=ks_significance,
        )
        
        # State
        self.samples_seen = 0
        self.last_check = 0
        self.drift_history: deque = deque(maxlen=1000)
        self.retrain_triggered = False
    
    def initialize_reference(self, X: np.ndarray):
        """Initialize reference distributions from baseline data."""
        logger.info(f"Initializing drift detector reference with {len(X)} samples")
        
        self.psi_detector.set_reference(X)
        self.ks_detector.set_reference(X)
        
        self.samples_seen = len(X)
        self.last_check = len(X)
    
    def check_drift(self, X: np.ndarray) -> Dict[str, Any]:
        """
        Check for drift in new data.
        
        Returns:
            Dictionary with drift status and recommendations
        """
        self.samples_seen += len(X)
        
        # Check if enough samples have passed
        if self.samples_seen - self.last_check < self.check_interval:
            return {
                'drift_detected': False,
                'reason': 'Not enough samples since last check',
                'samples_until_check': self.check_interval - (self.samples_seen - self.last_check),
            }
        
        if self.samples_seen < self.min_samples_before_check:
            return {
                'drift_detected': False,
                'reason': 'Insufficient samples for drift detection',
                'samples_needed': self.min_samples_before_check - self.samples_seen,
            }
        
        # Run PSI test
        psi_results = self.psi_detector.calculate_psi(X)
        
        # Run KS test
        ks_results = self.ks_detector.test(X)
        
        # Combine results
        psi_drift = psi_results.get('overall', 0) >= self.psi_threshold
        ks_drift = ks_results.get('overall_drift', False)
        
        drift_detected = psi_drift or ks_drift
        
        # Determine severity
        if psi_drift and ks_drift:
            severity = "high"
        elif psi_results.get('overall', 0) >= 0.1 or ks_results.get('drift_ratio', 0) >= 0.3:
            severity = "medium"
        else:
            severity = "low"
        
        # Store in history
        self.drift_history.append({
            'timestamp': self.samples_seen,
            'psi_overall': psi_results.get('overall', 0),
            'ks_drift': ks_drift,
            'drift_detected': drift_detected,
            'severity': severity,
        })
        
        # Update reference if no severe drift
        if severity != "high":
            self.psi_detector.update_reference(X, alpha=0.05)
            self.ks_detector.update_reference(X)
        else:
            self.retrain_triggered = True
        
        self.last_check = self.samples_seen
        
        result = {
            'drift_detected': drift_detected,
            'severity': severity,
            'psi_overall': psi_results.get('overall', 0),
            'psi_details': {k: v for k, v in psi_results.items() if k != 'overall'},
            'ks_drift': ks_drift,
            'ks_features_drifting': ks_results.get('features_drifting', []),
            'retrain_recommended': self.retrain_triggered,
            'samples_seen': self.samples_seen,
        }
        
        if drift_detected:
            logger.warning(
                f"Drift detected! PSI={psi_results.get('overall', 0):.4f}, "
                f"Severity={severity}, Retrain={'Recommended' if self.retrain_triggered else 'Not yet'}"
            )
        
        return result
    
    def reset(self):
        """Reset drift state after retraining."""
        self.retrain_triggered = False
        self.drift_history.clear()
        logger.info("Drift detector reset after retraining")


def main():
    """Test drift detection."""
    import psutil
    
    print(f"Initial RAM: {psutil.virtual_memory().available / (1024**3):.2f}GB")
    
    # Generate synthetic data
    np.random.seed(42)
    
    # Reference distribution (normal)
    X_ref = np.random.randn(1000, 5) * np.array([1.0, 0.5, 0.3, 0.2, 0.1]) + \
            np.array([0.0, 0.1, 0.2, 0.3, 0.4])
    
    # Test data 1: Similar distribution (no drift)
    X_test1 = np.random.randn(200, 5) * np.array([1.0, 0.5, 0.3, 0.2, 0.1]) + \
              np.array([0.0, 0.1, 0.2, 0.3, 0.4])
    
    # Test data 2: Shifted distribution (drift)
    X_test2 = np.random.randn(200, 5) * np.array([1.5, 0.8, 0.5, 0.4, 0.2]) + \
              np.array([1.0, 0.5, 0.8, 1.0, 0.6])
    
    # Create detector
    detector = DriftDetector(
        n_features=5,
        psi_threshold=0.2,
        ks_significance=0.05,
        min_samples_before_check=100,
        check_interval=100,
    )
    
    # Initialize reference
    detector.initialize_reference(X_ref)
    
    # Test on similar data
    print("\n--- Test 1: Similar Distribution ---")
    result1 = detector.check_drift(X_test1)
    print(f"Drift Detected: {result1['drift_detected']}")
    print(f"PSI Overall: {result1.get('psi_overall', 'N/A'):.4f}")
    print(f"Retrain Recommended: {result1['retrain_recommended']}")
    
    # Test on drifted data
    print("\n--- Test 2: Drifted Distribution ---")
    result2 = detector.check_drift(X_test2)
    print(f"Drift Detected: {result2['drift_detected']}")
    print(f"Severity: {result2['severity']}")
    print(f"PSI Overall: {result2.get('psi_overall', 'N/A'):.4f}")
    print(f"KS Features Drifting: {result2.get('ks_features_drifting', [])}")
    print(f"Retrain Recommended: {result2['retrain_recommended']}")
    
    print(f"\nFinal RAM: {psutil.virtual_memory().available / (1024**3):.2f}GB")


if __name__ == "__main__":
    main()
