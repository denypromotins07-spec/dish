"""
Real-time tracker of factor predictability and alpha signal half-life.
Uses rolling auto-correlation and exponential decay metrics.
Memory-efficient implementation for strict RAM constraints.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import deque


@dataclass
class DecayMetrics:
    """Metrics tracking alpha decay for a single factor."""
    factor_name: str
    current_ic: float = 0.0
    ic_half_life: float = 0.0  # In periods
    autocorr_decay_rate: float = 0.0
    predictability_score: float = 1.0
    n_observations: int = 0
    last_updated: int = 0


class AlphaDecayMonitor:
    """
    Monitors decay of alpha signals in real-time.
    Tracks Information Coefficient (IC) degradation over time.
    """
    
    def __init__(
        self,
        factor_names: List[str],
        ic_window: int = 252,
        max_lag: int = 60,
        min_samples: int = 50,
    ):
        self.factor_names = factor_names
        self.ic_window = ic_window
        self.max_lag = max_lag
        self.min_samples = min_samples
        
        # IC history per factor per lag
        self.ic_history: Dict[str, deque] = {
            name: deque(maxlen=ic_window) for name in factor_names
        }
        
        # Signal history for autocorrelation
        self.signal_history: Dict[str, deque] = {
            name: deque(maxlen=max_lag + 10) for name in factor_names
        }
        
        # Current metrics
        self.metrics: Dict[str, DecayMetrics] = {
            name: DecayMetrics(factor_name=name) for name in factor_names
        }
        
        # Forward return buffer for IC calculation
        self.forward_returns: deque = deque(maxlen=ic_window)
    
    def update(
        self,
        factor_signals: Dict[str, np.ndarray],
        forward_return: np.ndarray,
        timestamp: int,
    ) -> Dict[str, DecayMetrics]:
        """
        Update decay metrics with new observations.
        
        Args:
            factor_signals: Current factor signals (cross-sectional)
            forward_return: Realized forward returns
            timestamp: Current timestamp
        """
        # Store forward return
        self.forward_returns.append(forward_return.copy())
        
        for name in self.factor_names:
            if name not in factor_signals:
                continue
            
            signals = factor_signals[name]
            
            # Store signal history
            self.signal_history[name].append(signals.copy())
            
            # Calculate current IC
            if len(self.forward_returns) >= self.min_samples:
                ic = self._calculate_ic(signals, forward_return)
                self.ic_history[name].append(ic)
                
                # Update metrics
                self._update_factor_metrics(name, timestamp)
        
        return self.metrics.copy()
    
    def _calculate_ic(self, signals: np.ndarray, returns: np.ndarray) -> float:
        """Calculate rank Information Coefficient."""
        if len(signals) != len(returns) or len(signals) < 5:
            return 0.0
        
        try:
            from scipy.stats import spearmanr
            ic, _ = spearmanr(signals, returns)
            return ic if not np.isnan(ic) else 0.0
        except ImportError:
            # Fallback to Pearson correlation
            return np.corrcoef(signals, returns)[0, 1] if len(signals) > 1 else 0.0
    
    def _update_factor_metrics(self, factor_name: str, timestamp: int):
        """Update all decay metrics for a factor."""
        metrics = self.metrics[factor_name]
        ic_values = list(self.ic_history[factor_name])
        
        if len(ic_values) < self.min_samples // 2:
            return
        
        metrics.n_observations = len(ic_values)
        metrics.last_updated = timestamp
        
        # Current IC (most recent)
        metrics.current_ic = ic_values[-1] if ic_values else 0.0
        
        # Calculate IC autocorrelation decay
        ic_array = np.array(ic_values)
        autocorr = self._calculate_autocorrelation(ic_array)
        
        # Fit exponential decay to autocorrelation
        decay_rate, half_life = self._fit_exponential_decay(autocorr)
        
        metrics.autocorr_decay_rate = decay_rate
        metrics.ic_half_life = half_life
        
        # Predictability score (0-1, higher = more predictable/less decayed)
        metrics.predictability_score = self._calculate_predictability_score(
            ic_array, decay_rate, half_life
        )
    
    def _calculate_autocorrelation(self, series: np.ndarray) -> np.ndarray:
        """Calculate autocorrelation function up to max_lag."""
        n = len(series)
        mean = np.mean(series)
        var = np.var(series)
        
        if var < 1e-10:
            return np.zeros(self.max_lag)
        
        autocorr = np.zeros(self.max_lag)
        
        for lag in range(self.max_lag):
            if lag >= n:
                break
            
            cov = np.mean((series[:n-lag] - mean) * (series[lag:] - mean))
            autocorr[lag] = cov / var
        
        return autocorr
    
    def _fit_exponential_decay(
        self,
        autocorr: np.ndarray,
    ) -> Tuple[float, float]:
        """
        Fit exponential decay model: AC(lag) = exp(-decay_rate * lag)
        Returns decay rate and half-life.
        """
        # Use first portion of autocorrelation (where it's positive)
        valid_mask = autocorr > 0.05
        lags = np.arange(len(autocorr))[valid_mask]
        acf_vals = autocorr[valid_mask]
        
        if len(lags) < 3:
            return 0.0, float('inf')
        
        # Linear regression on log(ACF) vs lag
        with np.errstate(divide='ignore'):
            log_acf = np.log(acf_vals)
        
        # Simple OLS: log(ACF) = -decay_rate * lag
        slope = np.sum(lags * log_acf) / np.sum(lags ** 2)
        decay_rate = -slope
        
        if decay_rate <= 0:
            return 0.0, float('inf')
        
        # Half-life: time for ACF to drop to 0.5
        half_life = np.log(2) / decay_rate
        
        return decay_rate, half_life
    
    def _calculate_predictability_score(
        self,
        ic_values: np.ndarray,
        decay_rate: float,
        half_life: float,
    ) -> float:
        """
        Calculate overall predictability score (0-1).
        Combines IC magnitude, stability, and decay rate.
        """
        if len(ic_values) < 10:
            return 0.5
        
        # Component 1: Mean IC magnitude
        mean_ic = abs(np.mean(ic_values))
        ic_score = min(mean_ic * 10, 1.0)  # Scale: 0.1 IC = perfect score
        
        # Component 2: IC stability (inverse of variance)
        ic_std = np.std(ic_values)
        stability_score = 1.0 / (1.0 + ic_std * 5)
        
        # Component 3: Decay score (based on half-life)
        if half_life == float('inf') or half_life <= 0:
            decay_score = 0.0
        else:
            # Normalize: half-life of 20+ periods = good
            decay_score = min(half_life / 20, 1.0)
        
        # Weighted combination
        score = 0.4 * ic_score + 0.3 * stability_score + 0.3 * decay_score
        
        return min(max(score, 0.0), 1.0)
    
    def get_decayed_factors(self, threshold: float = 0.3) -> List[str]:
        """Get list of factors with predictability below threshold."""
        return [
            name for name, metrics in self.metrics.items()
            if metrics.predictability_score < threshold
        ]
    
    def get_fresh_factors(self, threshold: float = 0.7) -> List[str]:
        """Get list of factors with high predictability."""
        return [
            name for name, metrics in self.metrics.items()
            if metrics.predictability_score > threshold
        ]
    
    def get_summary(self) -> Dict[str, Dict]:
        """Get summary of all factor decay metrics."""
        summary = {}
        for name, metrics in self.metrics.items():
            summary[name] = {
                'current_ic': metrics.current_ic,
                'ic_half_life': metrics.ic_half_life,
                'decay_rate': metrics.autocorr_decay_rate,
                'predictability_score': metrics.predictability_score,
                'n_observations': metrics.n_observations,
                'status': self._get_status(metrics.predictability_score),
            }
        return summary
    
    def _get_status(self, score: float) -> str:
        if score > 0.7:
            return 'healthy'
        elif score > 0.4:
            return 'degrading'
        else:
            return 'decayed'


class SignalHalfLifeEstimator:
    """
    Estimates half-life of alpha signals using various methods.
    """
    
    def __init__(self, window_size: int = 252):
        self.window_size = window_size
        self.history: deque = deque(maxlen=window_size)
    
    def update(self, signal_value: float) -> Optional[float]:
        """Add observation and estimate half-life."""
        self.history.append(signal_value)
        
        if len(self.history) < 50:
            return None
        
        return self.estimate_half_life()
    
    def estimate_half_life(self) -> float:
        """Estimate half-life using Ornstein-Uhlenbeck process fitting."""
        values = np.array(self.history)
        
        if len(values) < 50:
            return float('inf')
        
        # Calculate returns/differences
        diffs = np.diff(values)
        lagged = values[:-1]
        
        if np.var(lagged) < 1e-10:
            return float('inf')
        
        # Fit AR(1): x_t = alpha + beta * x_{t-1} + epsilon
        # Half-life = -ln(2) / ln(beta)
        beta = np.cov(diffs, lagged)[0, 1] / np.var(lagged)
        
        if beta >= 1 or beta <= 0:
            return float('inf')
        
        half_life = -np.log(2) / np.log(1 + beta)
        
        return half_life if half_life > 0 else float('inf')
    
    def estimate_mean_reversion_speed(self) -> float:
        """Estimate speed of mean reversion (theta in OU process)."""
        half_life = self.estimate_half_life()
        
        if half_life == float('inf') or half_life <= 0:
            return 0.0
        
        return np.log(2) / half_life


if __name__ == "__main__":
    # Example usage
    np.random.seed(42)
    
    factor_names = ['momentum', 'value', 'volatility']
    monitor = AlphaDecayMonitor(factor_names, ic_window=100)
    
    n_assets = 50
    
    print("Simulating factor decay monitoring...\n")
    
    for t in range(150):
        # Generate factor signals with varying quality
        factor_signals = {
            'momentum': np.random.randn(n_assets) * max(0.1, 1 - t / 200),  # Decaying
            'value': np.random.randn(n_assets) * 0.5,  # Stable
            'volatility': np.random.randn(n_assets) * (0.3 + t / 300),  # Improving
        }
        
        # Generate forward returns partially correlated with signals
        forward_return = (
            0.1 * factor_signals['momentum'] +
            0.2 * factor_signals['value'] -
            0.1 * factor_signals['volatility'] +
            np.random.randn(n_assets) * 0.5
        )
        
        monitor.update(factor_signals, forward_return, t)
        
        if t % 25 == 0:
            print(f"t={t}:")
            summary = monitor.get_summary()
            for name, stats in summary.items():
                print(f"  {name}: IC={stats['current_ic']:.3f}, "
                      f"half_life={stats['ic_half_life']:.1f}, "
                      f"score={stats['predictability_score']:.3f} ({stats['status']})")
            print()
    
    print("\nFinal Summary:")
    print(f"  Decayed factors: {monitor.get_decayed_factors()}")
    print(f"  Fresh factors: {monitor.get_fresh_factors()}")
