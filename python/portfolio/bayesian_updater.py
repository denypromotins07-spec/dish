"""
Bayesian inference engine that continuously updates confidence intervals of strategy alpha signals
based on recent out-of-sample performance. Dynamically adjusts Omega matrix uncertainty.

Memory-efficient implementation bounded for <14GB total system RAM.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from collections import deque
import math


@dataclass(slots=True)
class SignalPerformance:
    """Track performance of a single alpha signal."""
    signal_id: str
    predictions: deque = field(default_factory=lambda: deque(maxlen=252))  # 1 year daily
    actuals: deque = field(default_factory=lambda: deque(maxlen=252))
    
    # Bayesian parameters
    prior_mean: float = 0.0
    prior_variance: float = 0.01
    posterior_mean: float = 0.0
    posterior_variance: float = 0.01
    
    # Confidence tracking
    hit_count: int = 0
    total_count: int = 0
    
    def add_observation(self, prediction: float, actual: float) -> None:
        """Add a new observation and update Bayesian posterior."""
        self.predictions.append(prediction)
        self.actuals.append(actual)
        
        # Track directional hits
        if (prediction > 0 and actual > 0) or (prediction < 0 and actual < 0):
            self.hit_count += 1
        self.total_count += 1
        
        # Update posterior using conjugate normal-normal model
        self._update_posterior()
    
    def _update_posterior(self) -> None:
        """Update posterior distribution using Bayesian updating."""
        if len(self.predictions) < 10:
            return
        
        preds = np.array(self.predictions)
        acts = np.array(self.actuals)
        
        # Calculate likelihood parameters from recent data
        residuals = acts - preds
        sample_variance = np.var(residuals) if len(residuals) > 1 else self.prior_variance
        
        # Bayesian update (conjugate normal)
        # Posterior precision = Prior precision + Data precision
        prior_precision = 1.0 / max(self.prior_variance, 1e-8)
        data_precision = len(self.predictions) / max(sample_variance, 1e-8)
        
        posterior_precision = prior_precision + data_precision
        posterior_variance = 1.0 / posterior_precision
        
        # Posterior mean is precision-weighted average
        weighted_sum = prior_precision * self.prior_mean + data_precision * np.mean(acts - preds + preds)
        posterior_mean = weighted_sum * posterior_variance
        
        self.posterior_mean = posterior_mean
        self.posterior_variance = posterior_variance
    
    @property
    def confidence(self) -> float:
        """Calculate confidence level from posterior (0 to 1)."""
        if self.total_count < 10:
            return 0.1
        
        # Base confidence from hit rate
        hit_rate = self.hit_count / self.total_count
        
        # Adjust by posterior variance (lower variance = higher confidence)
        variance_factor = 1.0 / (1.0 + math.sqrt(self.posterior_variance) * 10)
        
        # Combine factors
        confidence = hit_rate * variance_factor
        return np.clip(confidence, 0.0, 1.0)
    
    @property
    def sharpe_ratio(self) -> float:
        """Calculate information ratio of the signal."""
        if len(self.predictions) < 20:
            return 0.0
        
        preds = np.array(self.predictions)
        acts = np.array(self.actuals)
        
        returns = preds * acts  # P&L from following signal
        mean_ret = np.mean(returns)
        std_ret = np.std(returns)
        
        if std_ret < 1e-8:
            return 0.0
        
        return mean_ret / std_ret * np.sqrt(252)


class BayesianUpdater:
    """
    Manages Bayesian updating for multiple alpha signals.
    Outputs Omega matrix diagonal for Black-Litterman model.
    """
    
    __slots__ = ('signals', 'decay_factor', 'min_samples', 'omega_scale')
    
    def __init__(
        self,
        decay_factor: float = 0.95,
        min_samples: int = 20,
        omega_scale: float = 0.05
    ):
        self.signals: Dict[str, SignalPerformance] = {}
        self.decay_factor = decay_factor
        self.min_samples = min_samples
        self.omega_scale = omega_scale
    
    def register_signal(self, signal_id: str, prior_mean: float = 0.0, prior_variance: float = 0.01) -> None:
        """Register a new alpha signal for tracking."""
        self.signals[signal_id] = SignalPerformance(
            signal_id=signal_id,
            prior_mean=prior_mean,
            prior_variance=prior_variance
        )
    
    def update_signal(self, signal_id: str, prediction: float, actual: float) -> Optional[float]:
        """
        Update a signal with new prediction/actual pair.
        Returns updated confidence level.
        """
        if signal_id not in self.signals:
            return None
        
        signal = self.signals[signal_id]
        signal.add_observation(prediction, actual)
        
        return signal.confidence
    
    def get_omega_diagonal(self, signal_ids: List[str]) -> np.ndarray:
        """
        Compute Omega matrix diagonal for Black-Litterman.
        Lower confidence = higher uncertainty (diagonal element).
        """
        omega = np.zeros(len(signal_ids))
        
        for i, sig_id in enumerate(signal_ids):
            if sig_id in self.signals:
                signal = self.signals[sig_id]
                confidence = signal.confidence
                
                # Omega inversely proportional to confidence
                # Scale by signal variance for proper units
                if confidence > 1e-6:
                    omega[i] = self.omega_scale * signal.posterior_variance / confidence
                else:
                    omega[i] = self.omega_scale * signal.posterior_variance * 100  # High uncertainty
            
            else:
                # Unknown signal gets maximum uncertainty
                omega[i] = 1.0
        
        return omega
    
    def get_confidence_weights(self, signal_ids: List[str]) -> np.ndarray:
        """Get normalized confidence weights for signal blending."""
        confidences = []
        
        for sig_id in signal_ids:
            if sig_id in self.signals:
                confidences.append(self.signals[sig_id].confidence)
            else:
                confidences.append(0.1)  # Default low confidence
        
        weights = np.array(confidences)
        weight_sum = np.sum(weights)
        
        if weight_sum > 1e-8:
            weights /= weight_sum
        
        return weights
    
    def prune_underperforming_signals(
        self,
        min_sharpe: float = -1.0,
        min_confidence: float = 0.2
    ) -> List[str]:
        """Remove signals that don't meet performance thresholds."""
        to_remove = []
        
        for sig_id, signal in self.signals.items():
            if signal.sharpe_ratio < min_sharpe or signal.confidence < min_confidence:
                to_remove.append(sig_id)
        
        for sig_id in to_remove:
            del self.signals[sig_id]
        
        return to_remove
    
    def get_summary_stats(self) -> Dict[str, any]:
        """Get summary statistics for all tracked signals."""
        if not self.signals:
            return {}
        
        confidences = [s.confidence for s in self.signals.values()]
        sharpes = [s.sharpe_ratio for s in self.signals.values()]
        
        return {
            'num_signals': len(self.signals),
            'avg_confidence': float(np.mean(confidences)),
            'max_confidence': float(np.max(confidences)),
            'min_confidence': float(np.min(confidences)),
            'avg_sharpe': float(np.mean(sharpes)),
            'max_sharpe': float(np.max(sharpes)),
            'total_observations': sum(s.total_count for s in self.signals.values()),
        }


class DynamicOmegaAdapter:
    """
    Adapts Bayesian confidence to Black-Litterman Omega matrix.
    Provides real-time uncertainty scaling based on signal performance.
    """
    
    __slots__ = ('updater', 'tau', 'scaling_method')
    
    def __init__(
        self,
        updater: BayesianUpdater,
        tau: float = 0.05,
        scaling_method: str = 'variance_weighted'
    ):
        self.updater = updater
        self.tau = tau
        self.scaling_method = scaling_method
    
    def compute_omega(
        self,
        signal_ids: List[str],
        covariance_diag: np.ndarray
    ) -> np.ndarray:
        """
        Compute full Omega matrix diagonal with adaptive scaling.
        
        Args:
            signal_ids: List of signal IDs
            covariance_diag: Diagonal of asset covariance matrix
        
        Returns:
            Omega diagonal elements
        """
        n_signals = len(signal_ids)
        omega = np.zeros(n_signals)
        
        for i, sig_id in enumerate(signal_ids):
            if sig_id not in self.updater.signals:
                omega[i] = self.tau * np.max(covariance_diag) * 10
                continue
            
            signal = self.updater.signals[sig_id]
            confidence = signal.confidence
            
            if self.scaling_method == 'variance_weighted':
                # Scale by underlying asset variance
                base_variance = np.mean(covariance_diag)
                omega[i] = self.tau * base_variance / max(confidence, 0.01)
            
            elif self.scaling_method == 'sharpe_weighted':
                # Scale by signal Sharpe ratio
                sharpe = max(signal.sharpe_ratio, 0.01)
                omega[i] = self.tau / (confidence * sharpe)
            
            else:  # default
                omega[i] = self.tau / max(confidence, 0.01)
        
        return omega
    
    def get_bl_view_confidence(self, signal_id: str) -> float:
        """Get effective confidence for a single view."""
        if signal_id not in self.updater.signals:
            return 0.1
        
        signal = self.updater.signals[signal_id]
        
        # Blend hit rate with posterior certainty
        hit_rate = signal.hit_count / max(signal.total_count, 1)
        posterior_certainty = 1.0 - math.sqrt(signal.posterior_variance)
        
        return np.clip(0.5 * hit_rate + 0.5 * posterior_certainty, 0.0, 1.0)


# Example usage and testing
if __name__ == '__main__':
    # Test Bayesian updater
    updater = BayesianUpdater()
    updater.register_signal('btc_momentum', prior_mean=0.05, prior_variance=0.02)
    updater.register_signal('eth_mean_reversion', prior_mean=0.03, prior_variance=0.015)
    
    # Simulate observations
    np.random.seed(42)
    for _ in range(50):
        pred_btc = np.random.randn() * 0.02
        actual_btc = pred_btc + np.random.randn() * 0.01
        updater.update_signal('btc_momentum', pred_btc, actual_btc)
        
        pred_eth = np.random.randn() * 0.015
        actual_eth = pred_eth + np.random.randn() * 0.012
        updater.update_signal('eth_mean_reversion', pred_eth, actual_eth)
    
    # Get Omega diagonal
    signal_ids = ['btc_momentum', 'eth_mean_reversion']
    omega = updater.get_omega_diagonal(signal_ids)
    
    print(f"Signal confidences:")
    for sig_id in signal_ids:
        conf = updater.signals[sig_id].confidence
        sharpe = updater.signals[sig_id].sharpe_ratio
        print(f"  {sig_id}: confidence={conf:.3f}, sharpe={sharpe:.3f}")
    
    print(f"\nOmega diagonal: {omega}")
    print(f"Summary: {updater.get_summary_stats()}")
