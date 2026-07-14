"""
Time-series alignment engine for mapping low-frequency macro/on-chain data
to high-frequency microsecond tick timelines.
Uses forward-fill, exponential decay, and strict memory bounds.
"""

import time
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Tuple
from collections import deque
import threading


@dataclass
class AlignedDataPoint:
    """Aligned data point with original and aligned timestamps."""
    __slots__ = ('original_timestamp_ns', 'aligned_timestamp_ns', 'value', 'source', 'decay_factor')
    
    original_timestamp_ns: int
    aligned_timestamp_ns: int
    value: float
    source: str
    decay_factor: float = 1.0


class TimeSeriesAligner:
    """
    Aligns irregular, low-frequency time series to regular high-frequency grid.
    Supports forward-fill, linear interpolation, and exponential decay weighting.
    """
    
    __slots__ = ('_buffer', '_max_points', '_target_interval_ns', '_lock')
    
    def __init__(self, target_interval_ns: int = 1_000_000, max_points: int = 10_000):
        """
        Args:
            target_interval_ns: Target grid interval in nanoseconds (default 1ms)
            max_points: Maximum points to retain per source (memory bound)
        """
        self._buffer: Dict[str, deque] = {}
        self._max_points = max_points
        self._target_interval_ns = target_interval_ns
        self._lock = threading.Lock()
    
    def add_point(self, source: str, timestamp_ns: int, value: float) -> None:
        """Add a new data point from a source."""
        with self._lock:
            if source not in self._buffer:
                self._buffer[source] = deque(maxlen=self._max_points)
            
            self._buffer[source].append((timestamp_ns, value))
    
    def get_aligned_value(
        self, 
        source: str, 
        target_timestamp_ns: int,
        method: str = 'forward_fill'
    ) -> Optional[float]:
        """
        Get value aligned to target timestamp using specified method.
        
        Args:
            source: Data source identifier
            target_timestamp_ns: Target timestamp on high-freq grid
            method: 'forward_fill', 'linear_interp', or 'exponential_decay'
        
        Returns:
            Aligned value or None if insufficient data
        """
        with self._lock:
            buffer = self._buffer.get(source)
            if not buffer or len(buffer) < 1:
                return None
            
            data = list(buffer)
        
        if method == 'forward_fill':
            return self._forward_fill(data, target_timestamp_ns)
        elif method == 'linear_interp':
            return self._linear_interpolate(data, target_timestamp_ns)
        elif method == 'exponential_decay':
            return self._exponential_decay_weighted(data, target_timestamp_ns)
        else:
            return self._forward_fill(data, target_timestamp_ns)
    
    def _forward_fill(self, data: List[Tuple[int, float]], target_ns: int) -> Optional[float]:
        """Forward fill: use most recent value before target timestamp."""
        last_value = None
        last_ts = 0
        
        for ts, value in data:
            if ts <= target_ns:
                last_value = value
                last_ts = ts
            else:
                break
        
        return last_value
    
    def _linear_interpolate(self, data: List[Tuple[int, float]], target_ns: int) -> Optional[float]:
        """Linear interpolation between surrounding points."""
        # Find surrounding points
        before = None
        after = None
        
        for ts, value in data:
            if ts <= target_ns:
                before = (ts, value)
            elif ts > target_ns and after is None:
                after = (ts, value)
                break
        
        if before is None:
            return None
        if after is None:
            return before[1]  # No future point, use last
        
        # Interpolate
        t1, v1 = before
        t2, v2 = after
        
        if t2 == t1:
            return v1
        
        ratio = (target_ns - t1) / (t2 - t1)
        return v1 + ratio * (v2 - v1)
    
    def _exponential_decay_weighted(
        self, 
        data: List[Tuple[int, float]], 
        target_ns: int,
        half_life_ns: int = 60_000_000_000  # 60 seconds default
    ) -> Optional[float]:
        """
        Exponential decay weighting of historical values.
        More recent values have higher weight.
        """
        if not data:
            return None
        
        import math
        
        total_weight = 0.0
        weighted_sum = 0.0
        
        for ts, value in reversed(data):
            if ts > target_ns:
                continue
            
            # Calculate decay factor
            age_ns = target_ns - ts
            decay = math.exp(-age_ns * math.log(2) / half_life_ns)
            
            weighted_sum += value * decay
            total_weight += decay
            
            # Early termination when weights become negligible
            if decay < 0.001:
                break
        
        if total_weight == 0:
            return None
        
        return weighted_sum / total_weight
    
    def get_all_aligned(
        self, 
        target_timestamp_ns: int,
        sources: List[str] = None
    ) -> Dict[str, AlignedDataPoint]:
        """Get aligned values for all (or specified) sources at target timestamp."""
        result = {}
        
        with self._lock:
            source_list = sources if sources else list(self._buffer.keys())
            
            for source in source_list:
                buffer = self._buffer.get(source)
                if not buffer:
                    continue
                
                data = list(buffer)
                value = self._forward_fill(data, target_timestamp_ns)
                
                if value is not None:
                    # Find original timestamp
                    orig_ts = 0
                    for ts, v in data:
                        if ts <= target_timestamp_ns:
                            orig_ts = ts
                        else:
                            break
                    
                    # Calculate decay factor
                    age_ns = target_timestamp_ns - orig_ts
                    decay = 2 ** (-age_ns / 60_000_000_000)  # 60s half-life
                    
                    result[source] = AlignedDataPoint(
                        original_timestamp_ns=orig_ts,
                        aligned_timestamp_ns=target_timestamp_ns,
                        value=value,
                        source=source,
                        decay_factor=decay,
                    )
        
        return result
    
    def align_to_grid(
        self,
        start_ns: int,
        end_ns: int,
        sources: List[str] = None
    ) -> List[Dict[str, float]]:
        """
        Align all data to regular grid between start and end timestamps.
        
        Returns:
            List of dicts, each containing aligned values for one grid point
        """
        grid = []
        current_ns = start_ns
        
        while current_ns <= end_ns:
            aligned = self.get_all_aligned(current_ns, sources)
            row = {'timestamp_ns': current_ns}
            
            for source, point in aligned.items():
                row[source] = point.value
            
            grid.append(row)
            current_ns += self._target_interval_ns
        
        return grid
    
    def clear_source(self, source: str) -> None:
        """Clear data for a specific source."""
        with self._lock:
            if source in self._buffer:
                self._buffer[source].clear()
    
    def clear_all(self) -> None:
        """Clear all buffered data."""
        with self._lock:
            self._buffer.clear()


class MacroOnChainAligner:
    """
    Specialized aligner for macro and on-chain data streams.
    Handles different update frequencies and staleness detection.
    """
    
    __slots__ = ('_aligner', '_staleness_threshold_ns', '_frequency_map')
    
    # Expected update frequencies in nanoseconds
    FREQUENCY_MAP = {
        'cpi': 2_592_000_000_000_000,  # Monthly
        'ppi': 2_592_000_000_000_000,  # Monthly
        'dxy': 86_400_000_000_000,     # Daily
        'treasury_10y': 86_400_000_000_000,  # Daily
        'whale_tx': 60_000_000_000,    # ~1 minute
        'exchange_flow': 3_600_000_000_000,  # Hourly
        'tvl': 600_000_000_000,        # 10 minutes
        'gas_price': 12_000_000_000,   # ~12 seconds (ETH block time)
        'fear_greed': 3_600_000_000_000,  # Hourly
    }
    
    def __init__(self, target_interval_ns: int = 1_000_000):
        self._aligner = TimeSeriesAligner(target_interval_ns=target_interval_ns)
        self._staleness_threshold_ns = 300_000_000_000  # 5 minutes default
        self._frequency_map = self.FREQUENCY_MAP.copy()
    
    def add_macro_data(self, metric_type: str, timestamp_ns: int, value: float) -> None:
        """Add macroeconomic data point."""
        self._aligner.add_point(f'macro_{metric_type}', timestamp_ns, value)
    
    def add_onchain_data(self, metric_type: str, timestamp_ns: int, value: float) -> None:
        """Add on-chain data point."""
        self._aligner.add_point(f'onchain_{metric_type}', timestamp_ns, value)
    
    def get_aligned_features(
        self, 
        target_timestamp_ns: int,
        check_staleness: bool = True
    ) -> Dict[str, Any]:
        """
        Get all aligned features at target timestamp.
        
        Args:
            target_timestamp_ns: Target timestamp for alignment
            check_staleness: If True, mark stale features
        
        Returns:
            Dict with aligned values and metadata
        """
        aligned = self._aligner.get_all_aligned(target_timestamp_ns)
        
        features = {}
        stale_flags = {}
        
        for source, point in aligned.items():
            # Check staleness
            is_stale = False
            if check_staleness:
                expected_freq = self._frequency_map.get(
                    source.replace('macro_', '').replace('onchain_', ''),
                    self._staleness_threshold_ns
                )
                age = target_timestamp_ns - point.original_timestamp_ns
                is_stale = age > expected_freq
            
            features[source] = point.value
            stale_flags[source] = is_stale
            
            # Also include decay factor as confidence measure
            features[f'{source}_decay'] = point.decay_factor
        
        return {
            'timestamp_ns': target_timestamp_ns,
            'features': features,
            'stale_flags': stale_flags,
            'feature_count': len(features) // 2,  # Divide by 2 for value + decay
        }
    
    def set_staleness_threshold(self, threshold_ns: int) -> None:
        """Update staleness threshold."""
        self._staleness_threshold_ns = threshold_ns
    
    def get_feature_freshness(self, target_timestamp_ns: int) -> Dict[str, float]:
        """Get freshness score (0-1) for each feature."""
        aligned = self._aligner.get_all_aligned(target_timestamp_ns)
        freshness = {}
        
        for source, point in aligned.items():
            age_ns = target_timestamp_ns - point.original_timestamp_ns
            expected_freq = self._frequency_map.get(
                source.replace('macro_', '').replace('onchain_', ''),
                60_000_000_000
            )
            
            # Freshness = 1.0 if just updated, decays to 0.0 at 2x expected frequency
            freshness[source] = max(0.0, 1.0 - age_ns / (2 * expected_freq))
        
        return freshness


# Singleton instance
_aligner: Optional[MacroOnChainAligner] = None


def get_aligner() -> MacroOnChainAligner:
    """Get or create singleton aligner instance."""
    global _aligner
    if _aligner is None:
        _aligner = MacroOnChainAligner()
    return _aligner
