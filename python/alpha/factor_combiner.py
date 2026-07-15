"""
Cross-sectional and time-series factor combination engine.
Dynamically weights factors based on rolling Information Coefficient (IC) and Information Ratio (IR).
Memory-efficient implementation to stay within 14GB RAM constraint.
"""

import numpy as np
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import warnings

warnings.filterwarnings('ignore')


@dataclass
class FactorMetrics:
    """Tracks rolling metrics for a single factor."""
    ic_window: deque = field(default_factory=lambda: deque(maxlen=252))
    returns_window: deque = field(default_factory=lambda: deque(maxlen=252))
    turnover_window: deque = field(default_factory=lambda: deque(maxlen=63))
    
    @property
    def ic_mean(self) -> float:
        if len(self.ic_window) < 10:
            return 0.0
        return np.mean(self.ic_window)
    
    @property
    def ic_std(self) -> float:
        if len(self.ic_window) < 10:
            return 1.0
        return np.std(self.ic_window) + 1e-8
    
    @property
    def information_ratio(self) -> float:
        ic_mean = self.ic_mean
        ic_std = self.ic_std
        return ic_mean / ic_std if ic_std > 0 else 0.0
    
    @property
    def avg_turnover(self) -> float:
        if len(self.turnover_window) < 5:
            return 0.5
        return np.mean(self.turnover_window)


class FactorCombiner:
    """
    Combines multiple alpha factors using dynamic weighting based on IC and IR.
    Implements both cross-sectional and time-series combination strategies.
    """
    
    def __init__(
        self,
        factor_names: List[str],
        ic_lookback: int = 252,
        ir_lookback: int = 63,
        min_samples: int = 20,
        turnover_penalty: float = 0.1,
    ):
        self.factor_names = factor_names
        self.ic_lookback = ic_lookback
        self.ir_lookback = ir_lookback
        self.min_samples = min_samples
        self.turnover_penalty = turnover_penalty
        
        # Per-factor metrics tracking
        self.metrics: Dict[str, FactorMetrics] = {
            name: FactorMetrics() for name in factor_names
        }
        
        # Factor values storage (circular buffer for memory efficiency)
        self.factor_values: Dict[str, deque] = {
            name: deque(maxlen=ic_lookback) for name in factor_names
        }
        
        # Forward returns storage
        self.forward_returns: deque = deque(maxlen=ic_lookback)
        
        # Previous factor values for turnover calculation
        self.prev_factor_values: Dict[str, np.ndarray] = {}
        
        # Current weights
        self.current_weights: Dict[str, float] = {name: 1.0 / len(factor_names) for name in factor_names}
        
    def update_metrics(
        self,
        factor_signals: Dict[str, np.ndarray],
        forward_return: np.ndarray,
        timestamp: int,
    ) -> Dict[str, float]:
        """
        Update factor metrics with new observations.
        
        Args:
            factor_signals: Dict mapping factor names to cross-sectional signals
            forward_return: Realized forward returns for the period
            timestamp: Current timestamp
            
        Returns:
            Updated factor weights
        """
        # Store forward return
        self.forward_returns.append(forward_return.copy())
        
        # Calculate IC for each factor
        for name, signals in factor_signals.items():
            # Standardize signals
            signals_std = self._standardize(signals)
            
            # Store factor values
            self.factor_values[name].append(signals_std.copy())
            
            # Calculate turnover
            if name in self.prev_factor_values:
                prev = self.prev_factor_values[name]
                if len(prev) == len(signals_std):
                    turnover = self._calculate_turnover(prev, signals_std)
                    self.metrics[name].turnover_window.append(turnover)
            
            self.prev_factor_values[name] = signals_std.copy()
            
            # Calculate IC if we have enough history
            if len(self.forward_returns) >= self.min_samples:
                ic = self._calculate_ic(signals_std, forward_return)
                self.metrics[name].ic_window.append(ic)
                
                # Store return contribution for IR calculation
                ret_contrib = np.dot(signals_std, forward_return) / len(signals_std)
                self.metrics[name].returns_window.append(ret_contrib)
        
        # Recalculate weights
        self._recalculate_weights()
        
        return self.current_weights.copy()
    
    def combine_factors(
        self,
        factor_signals: Dict[str, np.ndarray],
        method: str = 'ic_weighted',
    ) -> np.ndarray:
        """
        Combine multiple factors into a single composite signal.
        
        Args:
            factor_signals: Dict mapping factor names to signals
            method: Combination method ('equal', 'ic_weighted', 'ir_weighted', 'optimised')
            
        Returns:
            Combined signal array
        """
        if not factor_signals:
            return np.array([])
        
        # Get all signals as matrix (factors x assets)
        signal_matrix = []
        valid_factors = []
        
        for name in self.factor_names:
            if name in factor_signals:
                signals = factor_signals[name]
                if len(signals) > 0:
                    signal_matrix.append(self._standardize(signals))
                    valid_factors.append(name)
        
        if not signal_matrix:
            return np.array([])
        
        signal_matrix = np.array(signal_matrix)  # Shape: (n_factors, n_assets)
        
        if method == 'equal':
            weights = np.ones(len(valid_factors)) / len(valid_factors)
        elif method == 'ic_weighted':
            weights = self._get_ic_weights(valid_factors)
        elif method == 'ir_weighted':
            weights = self._get_ir_weights(valid_factors)
        elif method == 'optimised':
            weights = self._get_optimised_weights(valid_factors)
        else:
            raise ValueError(f"Unknown combination method: {method}")
        
        # Weighted combination
        combined = np.dot(weights, signal_matrix)
        
        # Final standardization
        return self._standardize(combined)
    
    def _get_ic_weights(self, factors: List[str]) -> np.ndarray:
        """Get weights proportional to absolute IC."""
        weights = []
        for name in factors:
            ic = abs(self.metrics[name].ic_mean)
            weights.append(ic)
        
        total = sum(weights) + 1e-8
        return np.array(weights) / total
    
    def _get_ir_weights(self, factors: List[str]) -> np.ndarray:
        """Get weights proportional to Information Ratio."""
        weights = []
        for name in factors:
            ir = self.metrics[name].information_ratio
            # Use max(0, IR) to avoid negative weights
            weights.append(max(0, ir))
        
        total = sum(weights) + 1e-8
        if total < 1e-6:
            return np.ones(len(factors)) / len(factors)
        return np.array(weights) / total
    
    def _get_optimised_weights(self, factors: List[str]) -> np.ndarray:
        """
        Get optimised weights considering IC, IR, and turnover.
        Uses shrinkage to improve stability.
        """
        n = len(factors)
        if n == 0:
            return np.array([])
        
        # Base weights from IR
        ir_weights = self._get_ir_weights(factors)
        
        # Turnover penalty adjustment
        penalties = []
        for name in factors:
            turnover = self.metrics[name].avg_turnover
            penalty = 1.0 - self.turnover_penalty * min(turnover, 1.0)
            penalties.append(penalty)
        
        penalties = np.array(penalties)
        
        # Combine IR weights with turnover penalty
        adjusted = ir_weights * penalties
        total = adjusted.sum() + 1e-8
        
        # Shrink towards equal weight for stability
        shrinkage = min(0.5, 1.0 / max(1, len(self.forward_returns) // 50))
        equal_weights = np.ones(n) / n
        
        return (1 - shrinkage) * adjusted / total + shrinkage * equal_weights
    
    def _calculate_ic(self, signals: np.ndarray, returns: np.ndarray) -> float:
        """Calculate rank Information Coefficient."""
        if len(signals) != len(returns) or len(signals) < 5:
            return 0.0
        
        # Rank correlation (Spearman)
        try:
            from scipy.stats import spearmanr
            ic, _ = spearmanr(signals, returns)
            return ic if not np.isnan(ic) else 0.0
        except ImportError:
            # Fallback to Pearson if scipy not available
            return np.corrcoef(signals, returns)[0, 1] if len(signals) > 1 else 0.0
    
    def _calculate_turnover(self, prev: np.ndarray, curr: np.ndarray) -> float:
        """Calculate factor turnover as mean absolute change."""
        if len(prev) != len(curr):
            return 0.5
        
        diff = np.abs(curr - prev)
        return np.mean(diff) if len(diff) > 0 else 0.0
    
    def _standardize(self, x: np.ndarray) -> np.ndarray:
        """Standardize array to zero mean, unit variance."""
        if len(x) < 2:
            return x
        
        mean = np.mean(x)
        std = np.std(x) + 1e-8
        return (x - mean) / std
    
    def _recalculate_weights(self):
        """Recalculate current weights for all factors."""
        weights = self._get_optimised_weights(self.factor_names)
        for i, name in enumerate(self.factor_names):
            self.current_weights[name] = weights[i] if i < len(weights) else 0.0
    
    def get_factor_summary(self) -> Dict[str, Dict]:
        """Get summary statistics for all factors."""
        summary = {}
        for name in self.factor_names:
            m = self.metrics[name]
            summary[name] = {
                'ic_mean': m.ic_mean,
                'ic_std': m.ic_std,
                'information_ratio': m.information_ratio,
                'avg_turnover': m.avg_turnover,
                'current_weight': self.current_weights.get(name, 0.0),
                'samples': len(m.ic_window),
            }
        return summary


class TimeSeriesFactorCombiner(FactorCombiner):
    """
    Extension for time-series factor combination.
    Adds autocorrelation adjustment and momentum/mean-reversion regime detection.
    """
    
    def __init__(self, *args, ar_lag: int = 5, **kwargs):
        super().__init__(*args, **kwargs)
        self.ar_lag = ar_lag
        self.factor_autocorr: Dict[str, deque] = {
            name: deque(maxlen=63) for name in self.factor_names
        }
    
    def update_metrics(
        self,
        factor_signals: Dict[str, np.ndarray],
        forward_return: np.ndarray,
        timestamp: int,
    ) -> Dict[str, float]:
        """Update metrics with autocorrelation tracking."""
        weights = super().update_metrics(factor_signals, forward_return, timestamp)
        
        # Update autocorrelation for each factor
        for name in self.factor_names:
            if name in factor_signals:
                signals = factor_signals[name]
                if len(signals) > self.ar_lag:
                    # Calculate lag-1 autocorrelation of factor returns
                    if len(self.factor_values[name]) > self.ar_lag:
                        prev_signal = self.factor_values[name][-self.ar_lag - 1]
                        curr_signal = signals
                        
                        # Simplified autocorr calculation
                        if len(prev_signal) == len(curr_signal):
                            corr = np.corrcoef(prev_signal, curr_signal)[0, 1]
                            if not np.isnan(corr):
                                self.factor_autocorr[name].append(abs(corr))
        
        return weights
    
    def get_regime(self, factor_name: str) -> str:
        """Detect if factor is in momentum or mean-reversion regime."""
        if factor_name not in self.factor_autocorr:
            return 'unknown'
        
        autocorr = self.factor_autocorr[factor_name]
        if len(autocorr) < 10:
            return 'insufficient_data'
        
        avg_autocorr = np.mean(autocorr)
        
        if avg_autocorr > 0.3:
            return 'momentum'
        elif avg_autocorr < -0.3:
            return 'mean_reversion'
        else:
            return 'neutral'


if __name__ == '__main__':
    # Example usage
    np.random.seed(42)
    
    combiner = FactorCombiner(
        factor_names=['momentum', 'value', 'volatility'],
        ic_lookback=100,
        ir_lookback=30,
    )
    
    n_assets = 50
    
    for t in range(150):
        # Generate random factor signals
        factor_signals = {
            'momentum': np.random.randn(n_assets),
            'value': np.random.randn(n_assets),
            'volatility': np.random.randn(n_assets),
        }
        
        # Generate random forward returns
        forward_return = np.random.randn(n_assets) * 0.01
        
        # Update metrics
        weights = combiner.update_metrics(factor_signals, forward_return, t)
        
        # Combine factors
        combined = combiner.combine_factors(factor_signals, method='ir_weighted')
        
        if t % 25 == 0:
            print(f"t={t}: weights={weights}, combined_signal_mean={combined.mean():.4f}")
    
    print("\nFactor Summary:")
    summary = combiner.get_factor_summary()
    for name, stats in summary.items():
        print(f"{name}: IC={stats['ic_mean']:.4f}, IR={stats['information_ratio']:.4f}, "
              f"Weight={stats['current_weight']:.4f}")
