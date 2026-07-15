"""
Walk-Forward Visualizer: Formats out-of-sample vs in-sample equity curves.
Calculates rolling efficiency ratios for overfitting detection.
Pushes lightweight JSON to frontend for visual analysis.
"""

import numpy as np
import polars as pl
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import json
from collections import deque

# Memory bounds
MAX_WINDOWS = 100
MAX_DATA_POINTS = 10000


@dataclass
class WalkForwardWindow:
    """Represents a single walk-forward window result."""
    window_id: int
    train_start: int
    train_end: int
    oos_start: int
    oos_end: int
    in_sample_sharpe: float
    out_of_sample_sharpe: float
    in_sample_return: float
    out_of_sample_return: float
    in_sample_drawdown: float
    out_of_sample_drawdown: float
    efficiency_ratio: float  # OOS Sharpe / IS Sharpe


class WalkForwardVisualizer:
    """
    Memory-efficient walk-forward analysis visualizer.
    Uses bounded deques and streaming calculations.
    """
    
    def __init__(self, max_windows: int = MAX_WINDOWS):
        self.max_windows = max_windows
        self._windows: deque[WalkForwardWindow] = deque(maxlen=max_windows)
        self._rolling_is_sharpe: deque[float] = deque(maxlen=max_windows)
        self._rolling_oos_sharpe: deque[float] = deque(maxlen=max_windows)
        self._rolling_efficiency: deque[float] = deque(maxlen=max_windows)
        
    def add_window(
        self,
        window_id: int,
        train_start: int,
        train_end: int,
        oos_start: int,
        oos_end: int,
        is_equity_curve: np.ndarray,
        oos_equity_curve: np.ndarray
    ) -> WalkForwardWindow:
        """
        Add a walk-forward window with equity curves.
        Calculates metrics and efficiency ratio.
        """
        # Calculate in-sample metrics
        is_returns = np.diff(is_equity_curve)
        is_sharpe = self._calculate_sharpe(is_returns)
        is_return = is_equity_curve[-1] / is_equity_curve[0] - 1 if len(is_equity_curve) > 0 else 0
        is_drawdown = self._calculate_max_drawdown(is_equity_curve)
        
        # Calculate out-of-sample metrics
        oos_returns = np.diff(oos_equity_curve)
        oos_sharpe = self._calculate_sharpe(oos_returns)
        oos_return = oos_equity_curve[-1] / oos_equity_curve[0] - 1 if len(oos_equity_curve) > 0 else 0
        oos_drawdown = self._calculate_max_drawdown(oos_equity_curve)
        
        # Efficiency ratio (OOS/IS Sharpe)
        efficiency_ratio = oos_sharpe / is_sharpe if abs(is_sharpe) > 1e-10 else 0.0
        
        window = WalkForwardWindow(
            window_id=window_id,
            train_start=train_start,
            train_end=train_end,
            oos_start=oos_start,
            oos_end=oos_end,
            in_sample_sharpe=is_sharpe,
            out_of_sample_sharpe=oos_sharpe,
            in_sample_return=is_return,
            out_of_sample_return=oos_return,
            in_sample_drawdown=is_drawdown,
            out_of_sample_drawdown=oos_drawdown,
            efficiency_ratio=efficiency_ratio
        )
        
        self._windows.append(window)
        self._rolling_is_sharpe.append(is_sharpe)
        self._rolling_oos_sharpe.append(oos_sharpe)
        self._rolling_efficiency.append(efficiency_ratio)
        
        return window
    
    def _calculate_sharpe(self, returns: np.ndarray, annualization: float = 252.0) -> float:
        """Calculate annualized Sharpe ratio (assuming risk-free rate = 0)."""
        if len(returns) == 0 or np.std(returns) < 1e-10:
            return 0.0
        return float(np.mean(returns) / np.std(returns) * np.sqrt(annualization))
    
    def _calculate_max_drawdown(self, equity_curve: np.ndarray) -> float:
        """Calculate maximum drawdown from equity curve."""
        if len(equity_curve) == 0:
            return 0.0
        
        running_max = np.maximum.accumulate(equity_curve)
        drawdowns = (equity_curve - running_max) / running_max
        return float(np.min(drawdowns))
    
    def get_overfitting_score(self) -> float:
        """
        Calculate overall overfitting score.
        Score of 1.0 = perfect (no overfitting), < 0.5 = severe overfitting.
        """
        if len(self._rolling_efficiency) == 0:
            return 1.0
        
        # Mean efficiency ratio
        mean_efficiency = np.mean(list(self._rolling_efficiency))
        
        # Variance of efficiency (lower is better)
        var_efficiency = np.var(list(self._rolling_efficiency)) if len(self._rolling_efficiency) > 1 else 0
        
        # Combined score
        score = mean_efficiency * np.exp(-var_efficiency)
        return float(np.clip(score, 0.0, 1.0))
    
    def get_robustness_metrics(self) -> Dict:
        """
        Get comprehensive robustness metrics for UI display.
        """
        if len(self._windows) == 0:
            return {
                'status': 'no_data',
                'overfitting_score': 1.0,
                'windows_count': 0
            }
        
        is_sharpes = list(self._rolling_is_sharpe)
        oos_sharpes = list(self._rolling_oos_sharpe)
        efficiencies = list(self._rolling_efficiency)
        
        return {
            'status': 'ok',
            'windows_count': len(self._windows),
            'mean_is_sharpe': float(np.mean(is_sharpes)),
            'mean_oos_sharpe': float(np.mean(oos_sharpes)),
            'std_is_sharpe': float(np.std(is_sharpes)),
            'std_oos_sharpe': float(np.std(oos_sharpes)),
            'mean_efficiency': float(np.mean(efficiencies)),
            'min_efficiency': float(np.min(efficiencies)),
            'max_efficiency': float(np.max(efficiencies)),
            'overfitting_score': self.get_overfitting_score(),
            'p_value_approx': self._approximate_p_value(is_sharpes, oos_sharpes)
        }
    
    def _approximate_p_value(self, is_sharpes: List[float], oos_sharpes: List[float]) -> float:
        """
        Approximate p-value for difference between IS and OOS Sharpe.
        Simplified t-test approximation.
        """
        if len(is_sharpes) < 2 or len(oos_sharpes) < 2:
            return 1.0
        
        diff = np.array(is_sharpes) - np.array(oos_sharpes)
        mean_diff = np.mean(diff)
        std_diff = np.std(diff, ddof=1) if len(diff) > 1 else 1.0
        
        if std_diff < 1e-10:
            return 1.0 if mean_diff < 1e-10 else 0.0
        
        t_stat = mean_diff / (std_diff / np.sqrt(len(diff)))
        # Approximate two-tailed p-value using normal distribution
        p_value = 2.0 * (1.0 - 0.5 * (1.0 + np.math.erf(abs(t_stat) / np.sqrt(2))))
        return float(np.clip(p_value, 0.0, 1.0))
    
    def export_to_json(self, include_curves: bool = False) -> str:
        """
        Export walk-forward analysis to compact JSON for UI.
        """
        windows_data = []
        for w in self._windows:
            windows_data.append({
                'id': w.window_id,
                'train_range': [w.train_start, w.train_end],
                'oos_range': [w.oos_start, w.oos_end],
                'is_sharpe': round(w.in_sample_sharpe, 4),
                'oos_sharpe': round(w.out_of_sample_sharpe, 4),
                'is_return': round(w.in_sample_return, 6),
                'oos_return': round(w.out_of_sample_return, 6),
                'is_dd': round(w.in_sample_drawdown, 6),
                'oos_dd': round(w.out_of_sample_drawdown, 6),
                'efficiency': round(w.efficiency_ratio, 4)
            })
        
        output = {
            'windows': windows_data,
            'metrics': self.get_robustness_metrics()
        }
        
        return json.dumps(output, separators=(',', ':'))
    
    def clear(self):
        """Clear all stored data."""
        self._windows.clear()
        self._rolling_is_sharpe.clear()
        self._rolling_oos_sharpe.clear()
        self._rolling_efficiency.clear()


def process_walk_forward_results(
    results_df: pl.DataFrame,
    train_col: str,
    oos_col: str,
    window_size: int = 252,
    step_size: int = 63
) -> str:
    """
    Process a full DataFrame of walk-forward results.
    Returns JSON string for UI consumption.
    """
    visualizer = WalkForwardVisualizer()
    
    # Group by window and process
    unique_windows = results_df['window_id'].unique()
    
    for wid in unique_windows:
        window_data = results_df.filter(pl.col('window_id') == wid)
        
        is_curve = window_data.filter(pl.col(train_col) == 1)['equity'].to_numpy()
        oos_curve = window_data.filter(pl.col(oos_col) == 1)['equity'].to_numpy()
        
        if len(is_curve) > 0 and len(oos_curve) > 0:
            visualizer.add_window(
                window_id=int(wid),
                train_start=0,
                train_end=len(is_curve),
                oos_start=0,
                oos_end=len(oos_curve),
                is_equity_curve=is_curve,
                oos_equity_curve=oos_curve
            )
    
    return visualizer.export_to_json()


if __name__ == '__main__':
    # Example usage with synthetic data
    visualizer = WalkForwardVisualizer()
    
    # Simulate 10 walk-forward windows
    for i in range(10):
        # Generate synthetic equity curves
        is_curve = 1000 * np.cumprod(1 + np.random.normal(0.001, 0.02, 252))
        oos_curve = is_curve[-1] * np.cumprod(1 + np.random.normal(0.0008, 0.025, 63))
        
        visualizer.add_window(
            window_id=i,
            train_start=0,
            train_end=252,
            oos_start=0,
            oos_end=63,
            is_equity_curve=is_curve,
            oos_equity_curve=oos_curve
        )
    
    metrics = visualizer.get_robustness_metrics()
    print(f"Overfitting Score: {metrics['overfitting_score']:.3f}")
    print(f"Mean IS Sharpe: {metrics['mean_is_sharpe']:.3f}")
    print(f"Mean OOS Sharpe: {metrics['mean_oos_sharpe']:.3f}")
    
    json_output = visualizer.export_to_json()
    print(f"JSON length: {len(json_output)} bytes")
