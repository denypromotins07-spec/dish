"""
Statistical Anomaly Detector for Sentiment Volume and Polarity.
Flags "sentiment shocks" - sudden massive spikes in negative/positive news.
Triggers dynamic position flattening before price action materializes.
Designed for <14GB RAM with efficient ring buffers.
"""

import math
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple


@dataclass
class SentimentShock:
    """Detected sentiment anomaly/shock event."""
    timestamp_ms: int
    shock_type: str         # VOLUME_SPIKE, POLARITY_SHIFT, EXTREME_SENTIMENT
    severity: float         # 0-1 scale
    z_score: float          # Standard deviations from mean
    description: str
    recommended_action: str  # FLATTEN, REDUCE, MONITOR
    
    @property
    def is_critical(self) -> bool:
        """Check if shock requires immediate action."""
        return self.severity > 0.8 or abs(self.z_score) > 3.0


@dataclass
class StatisticalWindow:
    """Running statistics for a time window."""
    values: Deque[float] = field(default_factory=lambda: deque(maxlen=1000))
    _sum: float = 0.0
    _sum_sq: float = 0.0
    
    def add(self, value: float) -> Optional[float]:
        """Add value and return removed value if any."""
        removed = None
        if len(self.values) >= self.values.maxlen:
            removed = self.values[0]
            self._sum -= removed
            self._sum_sq -= removed * removed
        
        self.values.append(value)
        self._sum += value
        self._sum_sq += value * value
        
        return removed
    
    @property
    def count(self) -> int:
        return len(self.values)
    
    @property
    def mean(self) -> float:
        if self.count == 0:
            return 0.0
        return self._sum / self.count
    
    @property
    def variance(self) -> float:
        if self.count < 2:
            return 0.0
        mean = self.mean
        return (self._sum_sq / self.count) - (mean * mean)
    
    @property
    def std_dev(self) -> float:
        var = self.variance
        return math.sqrt(var) if var > 0 else 0.0
    
    def z_score(self, value: float) -> float:
        """Calculate z-score for a value."""
        std = self.std_dev
        if std == 0:
            return 0.0
        return (value - self.mean) / std
    
    def percentile(self, value: float) -> float:
        """Estimate percentile of value in distribution."""
        if self.count == 0:
            return 0.5
        
        # Using empirical rule for normal approximation
        z = self.z_score(value)
        # Approximate CDF of standard normal
        return 0.5 * (1 + math.erf(z / math.sqrt(2)))


class SentimentSpikeDetector:
    """
    Real-time detector for sentiment volume and polarity anomalies.
    Uses statistical methods (z-scores, moving averages) for detection.
    """
    
    def __init__(
        self,
        window_size: int = 500,
        volume_threshold: float = 3.0,      # Z-score threshold for volume spike
        polarity_threshold: float = 2.5,     # Z-score threshold for polarity shift
        extreme_threshold: float = 0.8,      # Absolute sentiment level
    ):
        self.window_size = window_size
        self.volume_threshold = volume_threshold
        self.polarity_threshold = polarity_threshold
        self.extreme_threshold = extreme_threshold
        
        # Volume tracking (messages per minute)
        self._volume_window = StatisticalWindow()
        self._volume_window.values = deque(maxlen=window_size)
        
        # Polarity tracking (net sentiment -1 to +1)
        self._polarity_window = StatisticalWindow()
        self._polarity_window.values = deque(maxlen=window_size)
        
        # Rate of change tracking
        self._momentum_window = StatisticalWindow()
        self._momentum_window.values = deque(maxlen=min(window_size, 100))
        
        # Recent shocks for cooldown
        self._recent_shocks: Deque[int] = deque(maxlen=100)
        
        # Statistics
        self._total_volume = 0
        self._shocks_detected = 0
        self._last_check_ms = 0
    
    def update_volume(self, count: int) -> Optional[SentimentShock]:
        """Update volume count and check for volume spike."""
        self._total_volume += count
        removed = self._volume_window.add(float(count))
        
        # Check for volume anomaly
        if self._volume_window.count >= 50:  # Need minimum samples
            z = self._volume_window.z_score(float(count))
            
            if z > self.volume_threshold:
                severity = min(1.0, z / 5.0)  # Normalize to 0-1
                
                # Check cooldown (don't alert more than once per minute)
                now_ms = int(time.time() * 1000)
                if self._is_in_cooldown(now_ms):
                    return None
                
                shock = SentimentShock(
                    timestamp_ms=now_ms,
                    shock_type="VOLUME_SPIKE",
                    severity=severity,
                    z_score=z,
                    description=f"Volume spike detected: {count} msgs (z={z:.2f})",
                    recommended_action=self._get_action(severity),
                )
                
                self._register_shock(now_ms)
                self._shocks_detected += 1
                
                return shock
        
        return None
    
    def update_polarity(self, sentiment: float) -> Optional[SentimentShock]:
        """
        Update polarity reading and check for extreme shifts.
        
        Args:
            sentiment: Net sentiment score (-1.0 to +1.0)
        """
        # Calculate momentum (rate of change)
        if self._polarity_window.count > 0:
            prev = self._polarity_window.values[-1] if self._polarity_window.values else 0
            momentum = sentiment - prev
            self._momentum_window.add(momentum)
        
        removed = self._polarity_window.add(sentiment)
        
        shocks = []
        now_ms = int(time.time() * 1000)
        
        # Check for extreme absolute sentiment
        if abs(sentiment) > self.extreme_threshold:
            z = self._polarity_window.z_score(sentiment)
            
            if abs(z) > self.polarity_threshold and not self._is_in_cooldown(now_ms):
                severity = min(1.0, abs(sentiment))
                
                shock = SentimentShock(
                    timestamp_ms=now_ms,
                    shock_type="EXTREME_SENTIMENT",
                    severity=severity,
                    z_score=z,
                    description=f"Extreme sentiment: {sentiment:.2f} ({'bearish' if sentiment < 0 else 'bullish'})",
                    recommended_action=self._get_action(severity),
                )
                
                self._register_shock(now_ms)
                self._shocks_detected += 1
                
                return shock
        
        # Check for rapid polarity shift
        if self._momentum_window.count >= 10:
            momentum_z = self._momentum_window.z_score(momentum)
            
            if abs(momentum_z) > 3.0 and not self._is_in_cooldown(now_ms):
                severity = min(1.0, abs(momentum) * 2)
                
                shock = SentimentShock(
                    timestamp_ms=now_ms,
                    shock_type="POLARITY_SHIFT",
                    severity=severity,
                    z_score=momentum_z,
                    description=f"Rapid polarity shift: {momentum:.2f} in single period",
                    recommended_action=self._get_action(severity),
                )
                
                self._register_shock(now_ms)
                self._shocks_detected += 1
                
                return shock
        
        return None
    
    def check_combined_signals(
        self,
        current_volume: int,
        current_sentiment: float,
    ) -> Optional[SentimentShock]:
        """
        Combined check for multi-factor sentiment shock.
        Detects situations where volume AND polarity are both anomalous.
        """
        now_ms = int(time.time() * 1000)
        
        if self._is_in_cooldown(now_ms):
            return None
        
        # Need sufficient history
        if self._volume_window.count < 50 or self._polarity_window.count < 50:
            return None
        
        volume_z = self._volume_window.z_score(float(current_volume))
        polarity_z = self._polarity_window.z_score(current_sentiment)
        
        # Combined score (both factors elevated)
        combined_severity = 0.0
        
        if volume_z > 2.0 and abs(polarity_z) > 2.0:
            # Both elevated - this is significant
            combined_severity = min(1.0, (abs(volume_z) + abs(polarity_z)) / 8.0)
            
            direction = "bearish" if current_sentiment < 0 else "bullish"
            
            shock = SentimentShock(
                timestamp_ms=now_ms,
                shock_type="COMBINED_SHOCK",
                severity=combined_severity,
                z_score=(volume_z + polarity_z) / 2,
                description=f"Combined shock: High volume + extreme {direction} sentiment",
                recommended_action="FLATTEN" if combined_severity > 0.7 else "REDUCE",
            )
            
            self._register_shock(now_ms)
            self._shocks_detected += 1
            
            return shock
        
        return None
    
    def _is_in_cooldown(self, now_ms: int) -> bool:
        """Check if we're in cooldown period after last shock."""
        if not self._recent_shocks:
            return False
        
        last_shock = self._recent_shocks[-1]
        cooldown_ms = 60000  # 1 minute cooldown
        
        return (now_ms - last_shock) < cooldown_ms
    
    def _register_shock(self, timestamp_ms: int):
        """Register a shock for cooldown tracking."""
        self._recent_shocks.append(timestamp_ms)
    
    def _get_action(self, severity: float) -> str:
        """Get recommended action based on severity."""
        if severity > 0.8:
            return "FLATTEN"
        elif severity > 0.6:
            return "REDUCE"
        elif severity > 0.4:
            return "HEDGE"
        else:
            return "MONITOR"
    
    def get_statistics(self) -> Dict[str, float]:
        """Get current statistical state."""
        return {
            "volume_mean": self._volume_window.mean,
            "volume_std": self._volume_window.std_dev,
            "polarity_mean": self._polarity_window.mean,
            "polarity_std": self._polarity_window.std_dev,
            "total_volume": self._total_volume,
            "shocks_detected": self._shocks_detected,
            "volume_samples": self._volume_window.count,
            "polarity_samples": self._polarity_window.count,
        }
    
    def get_risk_level(self) -> str:
        """Get current risk level based on recent statistics."""
        if self._shocks_detected > 5:
            return "CRITICAL"
        elif self._shocks_detected > 2:
            return "HIGH"
        elif self._polarity_window.std_dev > 0.3:
            return "ELEVATED"
        else:
            return "NORMAL"


class PositionFlattener:
    """
    Dynamic position management based on sentiment shocks.
    Triggers gradual or immediate position reduction.
    """
    
    def __init__(
        self,
        flatten_speed: str = "gradual",  # gradual, moderate, immediate
    ):
        self.flatten_speed = flatten_speed
        self._flatten_active = False
        self._target_reduction = 0.0
        self._current_reduction = 0.0
        self._start_time_ms = 0
        self._duration_ms = 0
    
    def trigger_flatten(self, shock: SentimentShock) -> Dict[str, float]:
        """
        Trigger position flattening based on shock.
        
        Returns:
            Dict with reduction parameters
        """
        self._flatten_active = True
        self._start_time_ms = int(time.time() * 1000)
        
        # Determine reduction amount based on shock type and severity
        base_reduction = shock.severity
        
        if shock.shock_type == "COMBINED_SHOCK":
            self._target_reduction = min(1.0, base_reduction * 1.2)
        elif shock.shock_type == "EXTREME_SENTIMENT":
            self._target_reduction = min(1.0, base_reduction)
        else:
            self._target_reduction = min(0.7, base_reduction * 0.8)
        
        # Set duration based on speed setting
        duration_map = {
            "immediate": 5000,    # 5 seconds
            "moderate": 30000,    # 30 seconds
            "gradual": 120000,    # 2 minutes
        }
        self._duration_ms = duration_map.get(self.flatten_speed, 60000)
        
        return {
            "active": True,
            "target_reduction": self._target_reduction,
            "duration_seconds": self._duration_ms / 1000,
            "speed": self.flatten_speed,
        }
    
    def get_current_reduction(self) -> float:
        """Get current reduction percentage based on elapsed time."""
        if not self._flatten_active:
            return 0.0
        
        now_ms = int(time.time() * 1000)
        elapsed = now_ms - self._start_time_ms
        
        if elapsed >= self._duration_ms:
            self._flatten_active = False
            return self._target_reduction
        
        # Linear interpolation
        progress = elapsed / self._duration_ms
        self._current_reduction = self._target_reduction * progress
        
        return self._current_reduction
    
    def reset(self):
        """Reset flattening state."""
        self._flatten_active = False
        self._target_reduction = 0.0
        self._current_reduction = 0.0


def main():
    """Example usage of sentiment spike detector."""
    print("Sentiment Spike Detector")
    print("=" * 50)
    
    detector = SentimentSpikeDetector(
        window_size=500,
        volume_threshold=3.0,
        polarity_threshold=2.5,
    )
    
    print(f"Window size: {detector.window_size}")
    print(f"Volume threshold (z-score): {detector.volume_threshold}")
    print(f"Polarity threshold (z-score): {detector.polarity_threshold}")
    
    # Simulate some data
    import random
    
    print("\nSimulating sentiment updates:")
    
    # Normal period
    for i in range(100):
        volume = random.randint(10, 30)
        sentiment = random.gauss(0.1, 0.2)
        
        detector.update_volume(volume)
        detector.update_polarity(sentiment)
    
    stats = detector.get_statistics()
    print(f"\nAfter normal period:")
    print(f"  Volume mean: {stats['volume_mean']:.1f}, std: {stats['volume_std']:.1f}")
    print(f"  Polarity mean: {stats['polarity_mean']:.3f}, std: {stats['polarity_std']:.3f}")
    print(f"  Risk level: {detector.get_risk_level()}")
    
    # Shock simulation
    print("\n--- SIMULATING SHOCK ---")
    shock_volume = detector.update_volume(500)  # Massive spike
    shock_sentiment = detector.update_polarity(-0.95)  # Extreme negative
    
    if shock_volume:
        print(f"Volume shock detected: {shock_volume.description}")
        print(f"  Severity: {shock_volume.severity:.2f}")
        print(f"  Action: {shock_volume.recommended_action}")
    
    if shock_sentiment:
        print(f"Sentiment shock detected: {shock_sentiment.description}")
        print(f"  Severity: {shock_sentiment.severity:.2f}")
        print(f"  Action: {shock_sentiment.recommended_action}")
    
    print(f"\nRisk level after shock: {detector.get_risk_level()}")
    print(f"Total shocks detected: {detector._shocks_detected}")
    
    print("\nDetector initialized successfully!")


if __name__ == "__main__":
    main()
