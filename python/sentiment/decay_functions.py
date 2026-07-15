"""
Exponential Time-Decay Functions for Sentiment Scores.
Applies decay where recent data has more weight than older data.
Implemented via highly efficient, fixed-size ring buffers for <14GB RAM.
"""

import math
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple


@dataclass
class DecayConfig:
    """Configuration for exponential decay function."""
    half_life_seconds: float  # Time for value to halve
    min_weight: float = 0.01  # Minimum weight threshold
    max_age_seconds: float = 3600  # Maximum age to consider
    
    @property
    def decay_constant(self) -> float:
        """Calculate decay constant lambda from half-life."""
        return math.log(2) / self.half_life_seconds
    
    def weight_for_age(self, age_seconds: float) -> float:
        """Calculate weight for a given age."""
        if age_seconds > self.max_age_seconds:
            return 0.0
        weight = math.exp(-self.decay_constant * age_seconds)
        return max(weight, self.min_weight)


@dataclass
class WeightedValue:
    """A value with its timestamp and calculated weight."""
    value: float
    timestamp_ms: int
    weight: float = 0.0
    source: str = ""


class ExponentialDecayBuffer:
    """
    Fixed-size ring buffer with exponential time-decay.
    Efficiently maintains weighted average of time-series data.
    """
    
    def __init__(
        self,
        capacity: int = 1000,
        config: Optional[DecayConfig] = None,
    ):
        self.capacity = capacity
        self.config = config or DecayConfig(half_life_seconds=300)  # 5 min default
        
        # Ring buffer for values
        self._buffer: Deque[WeightedValue] = deque(maxlen=capacity)
        
        # Running weighted sum and weight total for O(1) average calculation
        self._weighted_sum = 0.0
        self._weight_total = 0.0
        
        # Statistics
        self._count = 0
        self._last_update_ms = 0
    
    def add(self, value: float, source: str = "") -> None:
        """Add a new value with current timestamp."""
        now_ms = int(time.time() * 1000)
        
        weighted_value = WeightedValue(
            value=value,
            timestamp_ms=now_ms,
            source=source,
        )
        
        # Calculate initial weight (fresh value = weight 1.0)
        weighted_value.weight = 1.0
        
        # Remove oldest if at capacity
        if len(self._buffer) >= self.capacity:
            old = self._buffer[0]
            self._weighted_sum -= old.value * old.weight
            self._weight_total -= old.weight
        
        # Add new value
        self._buffer.append(weighted_value)
        self._weighted_sum += value * weighted_value.weight
        self._weight_total += weighted_value.weight
        self._count += 1
        self._last_update_ms = now_ms
    
    def recalculate_weights(self) -> None:
        """Recalculate all weights based on current time."""
        now_ms = int(time.time() * 1000)
        
        self._weighted_sum = 0.0
        self._weight_total = 0.0
        
        for wv in self._buffer:
            age_seconds = (now_ms - wv.timestamp_ms) / 1000.0
            wv.weight = self.config.weight_for_age(age_seconds)
            
            self._weighted_sum += wv.value * wv.weight
            self._weight_total += wv.weight
    
    def get_weighted_average(self) -> float:
        """Get current exponentially-weighted average."""
        # Recalculate weights first for accuracy
        self.recalculate_weights()
        
        if self._weight_total == 0:
            return 0.0
        
        return self._weighted_sum / self._weight_total
    
    def get_decay_factor(self) -> float:
        """Get overall decay factor applied to the buffer."""
        if not self._buffer:
            return 1.0
        
        avg_age = sum(
            (int(time.time() * 1000) - wv.timestamp_ms) / 1000.0
            for wv in self._buffer
        ) / len(self._buffer)
        
        return math.exp(-self.config.decay_constant * avg_age)
    
    def get_effective_sample_size(self) -> float:
        """
        Get effective sample size accounting for decay.
        Older samples contribute less, so effective N is smaller.
        """
        self.recalculate_weights()
        return self._weight_total  # Sum of weights is effective N
    
    def get_statistics(self) -> Dict[str, float]:
        """Get buffer statistics."""
        if not self._buffer:
            return {
                "count": 0,
                "weighted_avg": 0.0,
                "effective_n": 0.0,
                "decay_factor": 1.0,
                "oldest_age_seconds": 0.0,
            }
        
        now_ms = int(time.time() * 1000)
        oldest_age = (now_ms - self._buffer[0].timestamp_ms) / 1000.0
        
        return {
            "count": len(self._buffer),
            "weighted_avg": self.get_weighted_average(),
            "effective_n": self.get_effective_sample_size(),
            "decay_factor": self.get_decay_factor(),
            "oldest_age_seconds": oldest_age,
        }


class MultiSourceDecayAggregator:
    """
    Aggregates multiple sentiment sources with independent decay rates.
    Each source can have different half-life based on reliability/frequency.
    """
    
    # Default configs for different sources
    DEFAULT_CONFIGS = {
        "twitter": DecayConfig(half_life_seconds=60),    # 1 minute - very fast decay
        "reddit": DecayConfig(half_life_seconds=300),    # 5 minutes
        "news": DecayConfig(half_life_seconds=900),      # 15 minutes
        "macro": DecayConfig(half_life_seconds=3600),    # 1 hour - slow decay
        "fear_greed": DecayConfig(half_life_seconds=1800),  # 30 minutes
    }
    
    def __init__(self, buffer_capacity: int = 500):
        self._buffers: Dict[str, ExponentialDecayBuffer] = {}
        self._buffer_capacity = buffer_capacity
        
        # Initialize buffers for each source type
        for source_name, config in self.DEFAULT_CONFIGS.items():
            self._buffers[source_name] = ExponentialDecayBuffer(
                capacity=buffer_capacity,
                config=config,
            )
    
    def add_signal(self, source: str, value: float) -> None:
        """Add a signal from a specific source."""
        source_lower = source.lower()
        
        if source_lower not in self._buffers:
            # Create new buffer with default config
            self._buffers[source_lower] = ExponentialDecayBuffer(
                capacity=self._buffer_capacity,
                config=DecayConfig(half_life_seconds=300),
            )
        
        self._buffers[source_lower].add(value, source)
    
    def get_source_values(self) -> Dict[str, float]:
        """Get current weighted average for each source."""
        return {
            name: buffer.get_weighted_average()
            for name, buffer in self._buffers.items()
        }
    
    def get_composite_score(
        self,
        weights: Optional[Dict[str, float]] = None,
    ) -> float:
        """
        Calculate composite score across all sources.
        
        Args:
            weights: Optional custom weights per source.
                     Uses equal weights if not provided.
        """
        if weights is None:
            weights = {name: 1.0 for name in self._buffers.keys()}
        
        values = self.get_source_values()
        
        weighted_sum = 0.0
        weight_total = 0.0
        
        for name, value in values.items():
            w = weights.get(name, 0.0)
            weighted_sum += value * w
            weight_total += w
        
        if weight_total == 0:
            return 0.0
        
        return weighted_sum / weight_total
    
    def get_momentum(self, source: str, window_seconds: int = 60) -> float:
        """
        Calculate momentum (rate of change) for a source.
        Compares recent vs older weighted averages.
        """
        source_lower = source.lower()
        if source_lower not in self._buffers:
            return 0.0
        
        buffer = self._buffers[source_lower]
        if len(buffer._buffer) < 10:
            return 0.0
        
        now_ms = int(time.time() * 1000)
        cutoff_ms = now_ms - (window_seconds * 1000)
        
        # Split into recent and older
        recent_sum = 0.0
        recent_count = 0
        older_sum = 0.0
        older_count = 0
        
        for wv in buffer._buffer:
            if wv.timestamp_ms >= cutoff_ms:
                recent_sum += wv.value
                recent_count += 1
            else:
                older_sum += wv.value
                older_count += 1
        
        if recent_count == 0 or older_count == 0:
            return 0.0
        
        recent_avg = recent_sum / recent_count
        older_avg = older_sum / older_count
        
        return recent_avg - older_avg


def apply_decay_to_series(
    values: List[Tuple[int, float]],
    half_life_seconds: float,
    reference_time_ms: Optional[int] = None,
) -> List[Tuple[int, float, float]]:
    """
    Apply exponential decay to a time series.
    
    Args:
        values: List of (timestamp_ms, value) tuples
        half_life_seconds: Half-life for decay
        reference_time_ms: Reference point for age calculation (default=now)
    
    Returns:
        List of (timestamp_ms, original_value, decayed_value) tuples
    """
    if reference_time_ms is None:
        reference_time_ms = int(time.time() * 1000)
    
    decay_constant = math.log(2) / half_life_seconds
    
    result = []
    for ts_ms, value in values:
        age_seconds = (reference_time_ms - ts_ms) / 1000.0
        weight = math.exp(-decay_constant * age_seconds)
        decayed_value = value * weight
        result.append((ts_ms, value, decayed_value))
    
    return result


def main():
    """Example usage of decay functions."""
    print("Exponential Decay Functions")
    print("=" * 50)
    
    # Create decay buffer
    config = DecayConfig(half_life_seconds=60)  # 1 minute half-life
    buffer = ExponentialDecayBuffer(capacity=100, config=config)
    
    # Simulate adding values over time
    import random
    base_value = 0.5
    
    print("\nSimulating sentiment values with decay:")
    for i in range(10):
        value = base_value + random.gauss(0, 0.1)
        buffer.add(value, "test")
        
        stats = buffer.get_statistics()
        print(f"  Added {value:.3f} -> Weighted Avg: {stats['weighted_avg']:.3f}, "
              f"Effective N: {stats['effective_n']:.1f}")
    
    # Multi-source aggregator
    print("\nMulti-Source Aggregator:")
    aggregator = MultiSourceDecayAggregator()
    
    aggregator.add_signal("twitter", 0.8)
    aggregator.add_signal("twitter", 0.7)
    aggregator.add_signal("reddit", 0.6)
    aggregator.add_signal("news", 0.5)
    aggregator.add_signal("macro", 0.4)
    
    source_values = aggregator.get_source_values()
    for source, value in source_values.items():
        print(f"  {source}: {value:.3f}")
    
    composite = aggregator.get_composite_score()
    print(f"\nComposite Score: {composite:.3f}")
    
    print("\nDecay configuration initialized successfully!")


if __name__ == "__main__":
    main()
