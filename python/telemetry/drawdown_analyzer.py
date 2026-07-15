"""
Drawdown Analyzer: Calculates underwater equity curves and recovery metrics.
Pushes compressed arrays to UI for interactive drawdown heatmaps.
Memory-bounded with streaming calculations.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import json
from collections import deque

# Memory bounds
MAX_EQUITY_POINTS = 100000
MAX_DRAWDOWN_WINDOWS = 100


@dataclass
class DrawdownPeriod:
    """Represents a single drawdown period."""
    start_idx: int
    end_idx: int
    recovery_idx: Optional[int]
    peak_equity: float
    trough_equity: float
    drawdown_pct: float
    duration_days: int
    recovery_days: Optional[int]
    underwater_duration: int


class DrawdownAnalyzer:
    """
    Memory-efficient drawdown analyzer with streaming support.
    Uses bounded deques and incremental calculations.
    """
    
    def __init__(self, max_points: int = MAX_EQUITY_POINTS):
        self.max_points = max_points
        self._equity_curve: deque[float] = deque(maxlen=max_points)
        self._timestamps: deque[int] = deque(maxlen=max_points)
        self._drawdowns: deque[float] = deque(maxlen=max_points)
        self._running_max: deque[float] = deque(maxlen=max_points)
        self._drawdown_periods: List[DrawdownPeriod] = []
        
    def add_point(self, equity: float, timestamp_us: int):
        """Add a single equity point (streaming)."""
        self._equity_curve.append(equity)
        self._timestamps.append(timestamp_us)
        
        # Update running max and drawdown
        if len(self._running_max) == 0:
            self._running_max.append(equity)
            self._drawdowns.append(0.0)
        else:
            prev_max = self._running_max[-1]
            new_max = max(prev_max, equity)
            self._running_max.append(new_max)
            
            dd = (equity - new_max) / new_max if new_max > 0 else 0.0
            self._drawdowns.append(dd)
    
    def add_batch(self, equities: np.ndarray, timestamps: np.ndarray):
        """Add a batch of equity points."""
        for eq, ts in zip(equities[-self.max_points:], timestamps[-self.max_points:]):
            self.add_point(float(eq), int(ts))
    
    def calculate_drawdown_periods(self, threshold: float = -0.05) -> List[DrawdownPeriod]:
        """
        Identify distinct drawdown periods below threshold.
        Returns list of DrawdownPeriod objects.
        """
        if len(self._drawdowns) < 2:
            return []
        
        dd_array = np.array(list(self._drawdowns))
        periods = []
        
        in_drawdown = False
        start_idx = 0
        peak_equity = 0.0
        trough_equity = float('inf')
        
        for i, dd in enumerate(dd_array):
            if dd < threshold and not in_drawdown:
                # Start of drawdown
                in_drawdown = True
                start_idx = i
                peak_equity = self._equity_curve[i] if i < len(self._equity_curve) else 0
                trough_equity = self._equity_curve[i]
            
            elif in_drawdown:
                current_eq = self._equity_curve[i] if i < len(self._equity_curve) else 0
                trough_equity = min(trough_equity, current_eq)
                
                if dd >= threshold:
                    # End of drawdown (recovery)
                    recovery_idx = i
                    
                    # Calculate metrics
                    dd_pct = (trough_equity - peak_equity) / peak_equity if peak_equity > 0 else 0
                    duration = recovery_idx - start_idx
                    
                    # Estimate recovery days (simplified)
                    recovery_days = None
                    if recovery_idx < len(dd_array):
                        # Look for full recovery to new high
                        for j in range(recovery_idx, len(dd_array)):
                            if dd_array[j] >= 0:
                                recovery_days = j - start_idx
                                break
                    
                    periods.append(DrawdownPeriod(
                        start_idx=start_idx,
                        end_idx=recovery_idx - 1,
                        recovery_idx=recovery_idx,
                        peak_equity=peak_equity,
                        trough_equity=trough_equity,
                        drawdown_pct=dd_pct,
                        duration_days=duration,
                        recovery_days=recovery_days,
                        underwater_duration=recovery_days if recovery_days else duration
                    ))
                    
                    in_drawdown = False
        
        # Handle ongoing drawdown
        if in_drawdown:
            dd_pct = (trough_equity - peak_equity) / peak_equity if peak_equity > 0 else 0
            duration = len(dd_array) - start_idx
            
            periods.append(DrawdownPeriod(
                start_idx=start_idx,
                end_idx=len(dd_array) - 1,
                recovery_idx=None,
                peak_equity=peak_equity,
                trough_equity=trough_equity,
                drawdown_pct=dd_pct,
                duration_days=duration,
                recovery_days=None,
                underwater_duration=duration
            ))
        
        self._drawdown_periods = periods
        return periods
    
    def get_underwater_curve(self) -> np.ndarray:
        """Get the underwater (drawdown) curve as numpy array."""
        return np.array(list(self._drawdowns))
    
    def get_max_drawdown(self) -> float:
        """Get maximum historical drawdown."""
        if len(self._drawdowns) == 0:
            return 0.0
        return float(min(self._drawdowns))
    
    def get_current_drawdown(self) -> float:
        """Get current drawdown from peak."""
        if len(self._drawdowns) == 0:
            return 0.0
        return float(self._drawdowns[-1])
    
    def get_time_in_drawdown(self, threshold: float = -0.05) -> float:
        """Calculate fraction of time spent in drawdown below threshold."""
        if len(self._drawdowns) == 0:
            return 0.0
        
        dd_array = np.array(list(self._drawdowns))
        time_below = np.sum(dd_array < threshold)
        return float(time_below / len(dd_array))
    
    def get_recovery_factor(self) -> float:
        """Calculate recovery factor (total return / max drawdown)."""
        if len(self._equity_curve) < 2:
            return 0.0
        
        total_return = (self._equity_curve[-1] - self._equity_curve[0]) / self._equity_curve[0]
        max_dd = abs(self.get_max_drawdown())
        
        if max_dd < 1e-10:
            return float('inf') if total_return > 0 else 0.0
        
        return total_return / max_dd
    
    def get_comprehensive_metrics(self) -> Dict:
        """Get all drawdown metrics for UI display."""
        periods = self.calculate_drawdown_periods()
        
        avg_dd = np.mean(list(self._drawdowns)) if len(self._drawdowns) > 0 else 0
        std_dd = np.std(list(self._drawdowns)) if len(self._drawdowns) > 0 else 0
        
        # Calculate average drawdown duration
        avg_duration = 0
        if periods:
            avg_duration = np.mean([p.duration_days for p in periods])
        
        # Calculate average recovery time
        recovery_times = [p.recovery_days for p in periods if p.recovery_days is not None]
        avg_recovery = np.mean(recovery_times) if recovery_times else None
        
        return {
            'max_drawdown': self.get_max_drawdown(),
            'current_drawdown': self.get_current_drawdown(),
            'avg_drawdown': float(avg_dd),
            'std_drawdown': float(std_dd),
            'time_in_drawdown': self.get_time_in_drawdown(),
            'recovery_factor': self.get_recovery_factor(),
            'num_drawdown_periods': len(periods),
            'avg_drawdown_duration': float(avg_duration),
            'avg_recovery_time': float(avg_recovery) if avg_recovery else None,
            'worst_period': {
                'drawdown': min((p.drawdown_pct for p in periods), default=0),
                'duration': max((p.duration_days for p in periods), default=0)
            } if periods else None
        }
    
    def export_to_json(self) -> str:
        """Export drawdown data to compact JSON for UI."""
        # Downsample for visualization (max 1000 points)
        dd_array = self.get_underwater_curve()
        eq_array = np.array(list(self._equity_curve))
        
        step = max(1, len(dd_array) // 1000)
        dd_downsampled = dd_array[::step].tolist()
        eq_downsampled = eq_array[::step].tolist()
        ts_downsampled = list(self._timestamps)[::step]
        
        output = {
            'underwater_curve': dd_downsampled,
            'equity_curve': eq_downsampled,
            'timestamps': ts_downsampled,
            'metrics': self.get_comprehensive_metrics(),
            'periods': [
                {
                    'start': p.start_idx,
                    'end': p.end_idx,
                    'recovery': p.recovery_idx,
                    'dd_pct': round(p.drawdown_pct, 6),
                    'duration': p.duration_days,
                    'recovery_days': p.recovery_days
                }
                for p in self._drawdown_periods[:20]  # Top 20 periods
            ]
        }
        
        return json.dumps(output, separators=(',', ':'))
    
    def clear(self):
        """Clear all stored data."""
        self._equity_curve.clear()
        self._timestamps.clear()
        self._drawdowns.clear()
        self._running_max.clear()
        self._drawdown_periods.clear()


if __name__ == '__main__':
    # Example usage with synthetic equity curve
    analyzer = DrawdownAnalyzer()
    
    # Generate synthetic equity with drawdowns
    np.random.seed(42)
    n_points = 1000
    returns = np.random.normal(0.0005, 0.02, n_points)
    
    # Add some drawdown periods
    returns[200:250] -= 0.03  # Drawdown 1
    returns[500:580] -= 0.04  # Drawdown 2
    returns[800:830] -= 0.02  # Drawdown 3
    
    equity = 1000 * np.cumprod(1 + returns)
    timestamps = np.arange(n_points) * 3600 * 1000000  # Hourly
    
    analyzer.add_batch(equity, timestamps)
    
    metrics = analyzer.get_comprehensive_metrics()
    print(f"Max Drawdown: {metrics['max_drawdown']:.2%}")
    print(f"Recovery Factor: {metrics['recovery_factor']:.2f}")
    print(f"Time in Drawdown: {metrics['time_in_drawdown']:.2%}")
    
    json_output = analyzer.export_to_json()
    print(f"JSON size: {len(json_output)} bytes")
