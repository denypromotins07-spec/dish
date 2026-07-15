"""
Statistical data drift detection using Population Stability Index (PSI)
and Kolmogorov-Smirnov tests. Compares live feature distributions to
training baselines on a background thread. Strictly bounded RAM usage.
"""

import numpy as np
import polars as pl
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from scipy import stats
from collections import deque
import threading

# Memory bounds
MAX_HISTORY_BINS = 100
MAX_FEATURES_TRACKED = 50


@dataclass
class DriftResult:
    """Result of drift detection analysis."""
    feature_name: str
    psi_score: float
    ks_statistic: float
    ks_pvalue: float
    is_drifting: bool
    severity: str  # 'none', 'low', 'medium', 'high'


class DataDriftMonitor:
    """
    Monitors statistical data drift in real-time feature streams.
    Uses PSI and KS tests with memory-efficient rolling histograms.
    """

    def __init__(
        self,
        feature_names: List[str],
        n_bins: int = 20,
        psi_threshold: float = 0.25,
        ks_alpha: float = 0.05,
    ):
        self.feature_names = feature_names[:MAX_FEATURES_TRACKED]
        self.n_bins = n_bins
        self.psi_threshold = psi_threshold
        self.ks_alpha = ks_alpha
        
        # Baseline distributions (from training data)
        self._baseline_histograms: Dict[str, np.ndarray] = {}
        self._baseline_bin_edges: Dict[str, np.ndarray] = {}
        self._baseline_samples: Dict[str, deque] = {}
        
        # Live distributions (rolling window)
        self._live_samples: Dict[str, deque] = {
            f: deque(maxlen=MAX_HISTORY_BINS * 10) for f in self.feature_names
        }
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Drift history
        self._drift_history: deque = deque(maxlen=1000)
        self._alerts_triggered: Dict[str, int] = {f: 0 for f in self.feature_names}

    def set_baseline(
        self,
        feature_name: str,
        baseline_data: np.ndarray,
        sample_size: int = 10000,
    ) -> None:
        """
        Set baseline distribution from training data.
        Subsamples if data is too large to fit in memory.
        """
        if len(baseline_data) > sample_size:
            indices = np.random.choice(len(baseline_data), sample_size, replace=False)
            baseline_data = baseline_data[indices]
        
        with self._lock:
            # Store subsampled baseline
            self._baseline_samples[feature_name] = deque(baseline_data.tolist(), maxlen=sample_size)
            
            # Compute histogram
            hist, bin_edges = np.histogram(
                baseline_data,
                bins=self.n_bins,
                density=True,
            )
            
            # Normalize to probabilities
            self._baseline_histograms[feature_name] = hist / hist.sum()
            self._baseline_bin_edges[feature_name] = bin_edges

    def add_sample(self, feature_name: str, value: float) -> None:
        """Add a new live sample for a feature."""
        if feature_name not in self._live_samples:
            return
        
        with self._lock:
            self._live_samples[feature_name].append(value)

    def add_batch(self, feature_name: str, values: np.ndarray) -> None:
        """Add a batch of live samples."""
        if feature_name not in self._live_samples:
            return
        
        with self._lock:
            for v in values:
                self._live_samples[feature_name].append(v)

    def compute_psi(
        self,
        expected: np.ndarray,
        actual: np.ndarray,
        epsilon: float = 1e-4,
    ) -> float:
        """
        Calculate Population Stability Index.
        PSI < 0.1: No significant change
        0.1 <= PSI < 0.25: Moderate change
        PSI >= 0.25: Significant change
        """
        # Avoid division by zero
        expected = np.clip(expected, epsilon, 1.0)
        actual = np.clip(actual, epsilon, 1.0)
        
        # Normalize
        expected = expected / expected.sum()
        actual = actual / actual.sum()
        
        psi = np.sum((actual - expected) * np.log(actual / expected))
        return abs(psi)

    def compute_ks_test(
        self,
        baseline: np.ndarray,
        live: np.ndarray,
    ) -> Tuple[float, float]:
        """
        Perform Kolmogorov-Smirnov two-sample test.
        Returns (statistic, p-value).
        """
        return stats.ks_2samp(baseline, live)

    def check_drift(self, feature_name: str) -> Optional[DriftResult]:
        """
        Check for drift in a specific feature.
        Returns DriftResult or None if insufficient data.
        """
        with self._lock:
            if feature_name not in self._baseline_histograms:
                return None
            
            live_samples = list(self._live_samples[feature_name])
            if len(live_samples) < MAX_HISTORY_BINS:
                return None  # Insufficient live data
            
            baseline_samples = list(self._baseline_samples[feature_name])
            bin_edges = self._baseline_bin_edges[feature_name]
            
            # Compute live histogram using baseline bin edges
            live_hist, _ = np.histogram(
                live_samples,
                bins=bin_edges,
                density=True,
            )
            
            baseline_hist = self._baseline_histograms[feature_name]
            
            # Calculate PSI
            psi = self.compute_psi(baseline_hist, live_hist)
            
            # Calculate KS statistic
            ks_stat, ks_pval = self.compute_ks_test(
                np.array(baseline_samples),
                np.array(live_samples),
            )
            
            # Determine if drifting
            is_drifting = psi > self.psi_threshold or ks_pval < self.ks_alpha
            
            # Determine severity
            if psi < 0.1 and ks_pval >= 0.05:
                severity = 'none'
            elif psi < 0.2 and ks_pval >= 0.01:
                severity = 'low'
            elif psi < 0.3:
                severity = 'medium'
            else:
                severity = 'high'
            
            result = DriftResult(
                feature_name=feature_name,
                psi_score=psi,
                ks_statistic=ks_stat,
                ks_pvalue=ks_pval,
                is_drifting=is_drifting,
                severity=severity,
            )
            
            # Record in history
            self._drift_history.append(result)
            
            if is_drifting:
                self._alerts_triggered[feature_name] += 1
            
            return result

    def check_all_features(self) -> Dict[str, DriftResult]:
        """Check drift for all monitored features."""
        results = {}
        for feature in self.feature_names:
            result = self.check_drift(feature)
            if result is not None:
                results[feature] = result
        return results

    def get_drift_summary(self) -> Dict:
        """Get summary statistics of drift across all features."""
        if not self._drift_history:
            return {'total_checks': 0, 'drift_detected': 0}
        
        drift_count = sum(1 for r in self._drift_history if r.is_drifting)
        
        return {
            'total_checks': len(self._drift_history),
            'drift_detected': drift_count,
            'drift_rate': drift_count / len(self._drift_history),
            'avg_psi': np.mean([r.psi_score for r in self._drift_history]),
            'features_with_drift': {
                f: count for f, count in self._alerts_triggered.items() if count > 0
            },
        }

    def reset_live_data(self, feature_name: Optional[str] = None) -> None:
        """Reset live samples for re-baselining."""
        with self._lock:
            if feature_name:
                self._live_samples[feature_name].clear()
            else:
                for f in self._live_samples:
                    self._live_samples[f].clear()

    def export_baseline(self, feature_name: str) -> Optional[np.ndarray]:
        """Export baseline samples for external analysis."""
        with self._lock:
            if feature_name in self._baseline_samples:
                return np.array(list(self._baseline_samples[feature_name]))
        return None


if __name__ == "__main__":
    # Example usage
    np.random.seed(42)
    
    monitor = DataDriftMonitor(['feature_1', 'feature_2'], n_bins=20)
    
    # Set baseline from training data
    baseline_data = np.random.normal(0, 1, 10000)
    monitor.set_baseline('feature_1', baseline_data)
    
    # Add live samples (same distribution initially)
    for _ in range(500):
        monitor.add_sample('feature_1', np.random.normal(0, 1))
    
    # Check drift (should be none)
    result = monitor.check_drift('feature_1')
    print(f"Initial drift check: {result.severity}, PSI={result.psi_score:.4f}")
    
    # Add drifted samples (shifted mean)
    for _ in range(500):
        monitor.add_sample('feature_1', np.random.normal(2, 1))
    
    # Check drift (should detect)
    result = monitor.check_drift('feature_1')
    print(f"After shift: {result.severity}, PSI={result.psi_score:.4f}")
