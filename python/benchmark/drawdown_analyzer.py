# python/benchmark/drawdown_analyzer.py
"""
Deep underwater equity curve analyzer.
Calculates Maximum Drawdown, Average Drawdown, Time-to-Recovery, and Ulcer Index.
Quantifies psychological and capital toll of losing streaks.
"""

from __future__ import annotations
import polars as pl
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any, Tuple
from collections import deque


@dataclass
class DrawdownEvent:
    """Represents a single drawdown event."""
    start_date: str
    trough_date: str
    end_date: Optional[str]  # None if still in drawdown
    peak_value: float
    trough_value: float
    drawdown_pct: float
    duration_days: int
    recovery_days: Optional[int]  # None if not recovered
    total_days: int  # From peak to recovery/end
    
    def to_dict(self) -> dict:
        return {
            "start_date": self.start_date,
            "trough_date": self.trough_date,
            "end_date": self.end_date,
            "peak_value": self.peak_value,
            "trough_value": self.trough_value,
            "drawdown_pct": self.drawdown_pct,
            "duration_days": self.duration_days,
            "recovery_days": self.recovery_days,
            "total_days": self.total_days,
        }


@dataclass
class DrawdownSummary:
    """Comprehensive drawdown analysis summary."""
    max_drawdown_pct: float
    max_drawdown_duration_days: int
    current_drawdown_pct: float
    current_drawdown_days: int
    average_drawdown_pct: float
    average_recovery_days: float
    ulcer_index: float
    pain_index: float
    number_of_drawdowns: int
    longest_recovery_days: int
    worst_5_drawdowns: List[DrawdownEvent]
    
    def to_dict(self) -> dict:
        return {
            "max_drawdown_pct": self.max_drawdown_pct,
            "max_drawdown_duration_days": self.max_drawdown_duration_days,
            "current_drawdown_pct": self.current_drawdown_pct,
            "current_drawdown_days": self.current_drawdown_days,
            "average_drawdown_pct": self.average_drawdown_pct,
            "average_recovery_days": self.average_recovery_days,
            "ulcer_index": self.ulcer_index,
            "pain_index": self.pain_index,
            "number_of_drawdowns": self.number_of_drawdowns,
            "longest_recovery_days": self.longest_recovery_days,
            "worst_5_drawdowns": [dd.to_dict() for dd in self.worst_5_drawdowns],
        }


class DrawdownAnalyzer:
    """
    Comprehensive drawdown analysis for trading strategies.
    
    Features:
    - Complete drawdown event tracking
    - Time-to-recovery analysis
    - Ulcer Index calculation (pain metric)
    - Underwater equity visualization data
    - Memory-efficient streaming computation
    """
    
    def __init__(self, max_history_days: int = 730):
        """
        Initialize the analyzer.
        
        Args:
            max_history_days: Maximum history to retain
        """
        self.max_history_days = max_history_days
        
        # Storage (memory-bounded)
        self._dates: deque = deque(maxlen=max_history_days)
        self._values: deque = deque(maxlen=max_history_days)
        self._returns: deque = deque(maxlen=max_history_days)
        
        # Running state
        self._peak_value: float = 0.0
        self._peak_date: str = ""
        self._in_drawdown: bool = False
        self._drawdown_start: Optional[str] = None
        
        # Historical drawdown events
        self._drawdown_events: List[DrawdownEvent] = []
    
    def update(
        self,
        date: str,
        portfolio_value: float,
        daily_return: Optional[float] = None,
    ) -> float:
        """
        Update with new portfolio value.
        
        Args:
            date: Date string
            portfolio_value: Current portfolio value
            daily_return: Optional daily return
            
        Returns:
            Current drawdown percentage
        """
        self._dates.append(date)
        self._values.append(portfolio_value)
        
        if daily_return is not None:
            self._returns.append(daily_return)
        
        # Update peak
        if portfolio_value > self._peak_value:
            # Check if we're exiting a drawdown
            if self._in_drawdown and self._drawdown_start:
                # Record completed drawdown
                pass  # Will be computed in batch
            self._peak_value = portfolio_value
            self._peak_date = date
            self._in_drawdown = False
        
        # Store running state for streaming
        return self._calculate_current_drawdown(portfolio_value)
    
    def _calculate_current_drawdown(self, current_value: float) -> float:
        """Calculate current drawdown from peak."""
        if self._peak_value <= 0:
            return 0.0
        return (current_value - self._peak_value) / self._peak_value
    
    def analyze_full_history(self) -> DrawdownSummary:
        """
        Perform complete drawdown analysis on all stored data.
        
        Returns:
            DrawdownSummary with all metrics
        """
        if len(self._values) < 2:
            return self._empty_summary()
        
        values = np.array(list(self._values))
        dates = list(self._dates)
        
        # Calculate running maximum
        running_max = np.maximum.accumulate(values)
        
        # Drawdown series
        drawdowns = (values - running_max) / running_max
        
        # Identify drawdown events
        events = self._identify_drawdown_events(drawdowns, dates, values)
        
        # Max drawdown
        max_dd = np.min(drawdowns)
        
        # Current drawdown
        current_dd = drawdowns[-1]
        
        # Days in current drawdown
        current_dd_days = 0
        if current_dd < 0:
            # Find when this drawdown started
            for i in range(len(drawdowns) - 1, -1, -1):
                if drawdowns[i] >= 0:
                    current_dd_days = len(drawdowns) - 1 - i
                    break
            else:
                current_dd_days = len(drawdowns)
        
        # Average drawdown (of significant DDs > 1%)
        significant_dd = [dd.drawdown_pct for dd in events if abs(dd.drawdown_pct) > 0.01]
        avg_dd = np.mean(significant_dd) if significant_dd else 0.0
        
        # Average recovery days
        recovered = [dd.recovery_days for dd in events if dd.recovery_days is not None]
        avg_recovery = np.mean(recovered) if recovered else 0.0
        
        # Longest recovery
        longest_recovery = max(recovered) if recovered else 0
        
        # Ulcer Index
        ulcer_index = self._calculate_ulcer_index(drawdowns)
        
        # Pain Index
        pain_index = self._calculate_pain_index(drawdowns)
        
        # Worst 5 drawdowns
        sorted_events = sorted(events, key=lambda x: x.drawdown_pct)[:5]
        
        return DrawdownSummary(
            max_drawdown_pct=max_dd,
            max_drawdown_duration_days=self._get_max_dd_duration(events),
            current_drawdown_pct=current_dd,
            current_drawdown_days=current_dd_days,
            average_drawdown_pct=avg_dd,
            average_recovery_days=avg_recovery,
            ulcer_index=ulcer_index,
            pain_index=pain_index,
            number_of_drawdowns=len(events),
            longest_recovery_days=longest_recovery,
            worst_5_drawdowns=sorted_events,
        )
    
    def _identify_drawdown_events(
        self,
        drawdowns: np.ndarray,
        dates: List[str],
        values: np.ndarray,
    ) -> List[DrawdownEvent]:
        """Identify individual drawdown events from the series."""
        events = []
        
        in_dd = False
        dd_start_idx = 0
        trough_idx = 0
        min_value = float('inf')
        
        threshold = -0.01  # 1% threshold for significance
        
        for i in range(len(drawdowns)):
            dd = drawdowns[i]
            
            if dd < threshold and not in_dd:
                # Start of new drawdown
                in_dd = True
                dd_start_idx = i
                trough_idx = i
                min_value = values[i]
            
            elif in_dd:
                if values[i] < min_value:
                    # New trough
                    trough_idx = i
                    min_value = values[i]
                
                if dd >= 0:
                    # Recovery - end of drawdown
                    event = DrawdownEvent(
                        start_date=dates[dd_start_idx],
                        trough_date=dates[trough_idx],
                        end_date=dates[i],
                        peak_value=values[dd_start_idx],
                        trough_value=min_value,
                        drawdown_pct=(min_value - values[dd_start_idx]) / values[dd_start_idx],
                        duration_days=trough_idx - dd_start_idx,
                        recovery_days=i - trough_idx,
                        total_days=i - dd_start_idx,
                    )
                    events.append(event)
                    in_dd = False
        
        # Handle ongoing drawdown
        if in_dd:
            event = DrawdownEvent(
                start_date=dates[dd_start_idx],
                trough_date=dates[trough_idx],
                end_date=None,
                peak_value=values[dd_start_idx],
                trough_value=min_value,
                drawdown_pct=(min_value - values[dd_start_idx]) / values[dd_start_idx],
                duration_days=trough_idx - dd_start_idx,
                recovery_days=None,
                total_days=len(drawdowns) - dd_start_idx,
            )
            events.append(event)
        
        return events
    
    def _calculate_ulcer_index(self, drawdowns: np.ndarray) -> float:
        """
        Calculate Ulcer Index (Martin Ratio denominator).
        
        UI = sqrt(sum(drawdown^2) / n)
        
        Measures both depth and duration of drawdowns.
        """
        if len(drawdowns) == 0:
            return 0.0
        
        # Square of drawdowns (only negative values contribute)
        squared_dd = np.where(drawdowns < 0, drawdowns ** 2, 0)
        
        return np.sqrt(np.mean(squared_dd))
    
    def _calculate_pain_index(self, drawdowns: np.ndarray) -> float:
        """
        Calculate Pain Index.
        
        PI = sum(|drawdown|) / n
        
        Simpler measure of cumulative pain.
        """
        if len(drawdowns) == 0:
            return 0.0
        
        return np.mean(np.abs(np.minimum(drawdowns, 0)))
    
    def _get_max_dd_duration(self, events: List[DrawdownEvent]) -> int:
        """Get the longest drawdown duration in days."""
        if not events:
            return 0
        return max(e.total_days for e in events)
    
    def _empty_summary(self) -> DrawdownSummary:
        """Return empty summary when insufficient data."""
        return DrawdownSummary(
            max_drawdown_pct=0.0,
            max_drawdown_duration_days=0,
            current_drawdown_pct=0.0,
            current_drawdown_days=0,
            average_drawdown_pct=0.0,
            average_recovery_days=0.0,
            ulcer_index=0.0,
            pain_index=0.0,
            number_of_drawdowns=0,
            longest_recovery_days=0,
            worst_5_drawdowns=[],
        )
    
    def get_underwater_data(self) -> Dict[str, List[Any]]:
        """
        Get data for plotting underwater equity curve.
        
        Returns:
            Dict with dates, drawdown percentages, and status
        """
        if len(self._values) == 0:
            return {"dates": [], "drawdown": [], "in_recovery": []}
        
        values = np.array(list(self._values))
        dates = list(self._dates)
        running_max = np.maximum.accumulate(values)
        drawdowns = (values - running_max) / running_max
        
        # Determine if each point is in recovery phase
        in_recovery = []
        current_max = 0.0
        for i, v in enumerate(values):
            if v > current_max:
                current_max = v
                in_recovery.append(False)
            else:
                in_recovery.append(v < current_max)
        
        return {
            "dates": dates,
            "drawdown": drawdowns.tolist(),
            "in_recovery": in_recovery,
        }
    
    def get_recovery_analysis(self) -> Dict[str, Any]:
        """
        Detailed analysis of recovery patterns.
        
        Returns:
            Dict with recovery statistics
        """
        events = self._identify_drawdown_events(
            np.array(list(self._values)),
            list(self._dates),
            np.array(list(self._values)),
        )
        
        recovered = [e for e in events if e.recovery_days is not None]
        ongoing = [e for e in events if e.recovery_days is None]
        
        if not recovered:
            return {
                "total_drawdowns": len(events),
                "recovered": 0,
                "ongoing": len(ongoing),
                "message": "No completed drawdowns to analyze",
            }
        
        recovery_times = [e.recovery_days for e in recovered]
        
        return {
            "total_drawdowns": len(events),
            "recovered_count": len(recovered),
            "ongoing_count": len(ongoing),
            "avg_recovery_days": np.mean(recovery_times),
            "median_recovery_days": np.median(recovery_times),
            "min_recovery_days": min(recovery_times),
            "max_recovery_days": max(recovery_times),
            "std_recovery_days": np.std(recovery_times),
            "recovery_rate": len(recovered) / len(events) if events else 0,
        }
    
    def to_polars(self) -> pl.DataFrame:
        """Export data to Polars DataFrame."""
        if len(self._values) == 0:
            return pl.DataFrame()
        
        values = np.array(list(self._values))
        running_max = np.maximum.accumulate(values)
        drawdowns = (values - running_max) / running_max
        
        return pl.DataFrame({
            "date": list(self._dates),
            "value": list(self._values),
            "running_max": running_max.tolist(),
            "drawdown": drawdowns.tolist(),
        })


def calculate_martin_ratio(
    returns: np.ndarray,
    risk_free_rate: float = 0.0,
    annualization: int = 365,
) -> float:
    """
    Calculate Martin Ratio (return / Ulcer Index).
    
    Alternative to Sharpe that uses drawdown-based risk measure.
    """
    if len(returns) < 10:
        return 0.0
    
    # Calculate cumulative values
    cum_values = np.cumprod(1 + returns)
    running_max = np.maximum.accumulate(cum_values)
    drawdowns = (cum_values - running_max) / running_max
    
    ulcer_index = np.sqrt(np.mean(drawdowns ** 2))
    
    if ulcer_index == 0:
        return float('inf') if np.mean(returns) > 0 else 0.0
    
    ann_return = np.mean(returns) * annualization
    
    return ann_return / ulcer_index


if __name__ == "__main__":
    import random
    
    analyzer = DrawdownAnalyzer(max_history_days=500)
    
    # Simulate equity curve with realistic drawdowns
    value = 100000.0
    for day in range(300):
        date = f"2024-{(day // 30) + 1:02d}-{(day % 30) + 1:02d}"
        
        # Random return with occasional bad streaks
        if random.random() < 0.1:
            ret = random.gauss(-0.03, 0.02)  # Bad day
        else:
            ret = random.gauss(0.001, 0.02)  # Normal day
        
        value *= (1 + ret)
        analyzer.update(date, value, ret)
    
    # Analyze
    summary = analyzer.analyze_full_history()
    
    print("Drawdown Analysis Summary:")
    print(f"  Max Drawdown: {summary.max_drawdown_pct:.2%}")
    print(f"  Current Drawdown: {summary.current_drawdown_pct:.2%} ({summary.current_drawdown_days} days)")
    print(f"  Ulcer Index: {summary.ulcer_index:.4f}")
    print(f"  Pain Index: {summary.pain_index:.4f}")
    print(f"  Number of Drawdowns: {summary.number_of_drawdowns}")
    print(f"  Avg Recovery Days: {summary.average_recovery_days:.1f}")
    print()
    
    print("Worst 5 Drawdowns:")
    for i, dd in enumerate(summary.worst_5_drawdowns, 1):
        print(f"  {i}. {dd.drawdown_pct:.2%} - {dd.duration_days} days to trough, "
              f"{dd.recovery_days or 'N/A'} days to recover")
    
    # Recovery analysis
    recovery = analyzer.get_recovery_analysis()
    print(f"\nRecovery Stats: {recovery}")
