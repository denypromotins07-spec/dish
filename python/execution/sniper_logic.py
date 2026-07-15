"""
Liquidity sniping logic for detecting large market orders sweeping the book
Instantly places aggressive limit orders to capture microsecond momentum bounce
Optimized for AMD Ryzen AI 5 with minimal memory footprint
"""

import time
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, List
from collections import deque


@dataclass
class SweepEvent:
    """Represents a detected liquidity sweep event"""
    timestamp_us: int
    price: float
    volume: float
    side: str  # 'buy' or 'sell'
    levels_swept: int
    total_value: float
    momentum_score: float


@dataclass
class SniperSignal:
    """Sniper entry signal with timing and sizing"""
    timestamp_us: int
    direction: str  # 'long' or 'short'
    entry_price: float
    target_price: float
    stop_loss: float
    size_pct: float
    confidence: float
    expected_hold_us: int


class LiquiditySweepDetector:
    """
    Detects large market orders sweeping multiple price levels
    Uses fixed-size buffers to maintain <14GB RAM constraint
    """
    
    def __init__(self, window_size: int = 100, sweep_threshold: float = 1000000.0):
        self.window_size = window_size
        self.sweep_threshold = sweep_threshold
        
        # Fixed-size circular buffers (no heap growth)
        self.price_buffer: deque = deque(maxlen=window_size)
        self.volume_buffer: deque = deque(maxlen=window_size)
        self.timestamp_buffer: deque = deque(maxlen=window_size)
        
        # State tracking
        self.last_sweep_time: int = 0
        self.sweep_cooldown_us: int = 1000  # 1ms between sweeps
        
        # Statistics
        self.sweeps_detected: int = 0
        self.false_signals: int = 0
    
    def update(self, timestamp_us: int, price: float, volume: float, side: str) -> Optional[SweepEvent]:
        """
        Process new trade and detect potential sweep
        Returns SweepEvent if detected, None otherwise
        """
        self.price_buffer.append(price)
        self.volume_buffer.append(volume)
        self.timestamp_buffer.append(timestamp_us)
        
        if len(self.price_buffer) < 5:
            return None
        
        # Check for sweep conditions
        sweep = self._detect_sweep(side)
        
        if sweep:
            # Cooldown check
            if timestamp_us - self.last_sweep_time < self.sweep_cooldown_us:
                return None
            
            self.last_sweep_time = timestamp_us
            self.sweeps_detected += 1
            
        return sweep
    
    def _detect_sweep(self, side: str) -> Optional[SweepEvent]:
        """Internal sweep detection logic"""
        prices = np.array(self.price_buffer)
        volumes = np.array(self.volume_buffer)
        
        # Calculate price velocity (rate of change)
        price_changes = np.diff(prices)
        
        # Look for sustained directional movement with high volume
        if side == 'buy':
            # Buying sweep: consecutive upticks with increasing volume
            upticks = price_changes > 0
            if not np.all(upticks[-3:]):
                return None
            
            # Volume weighted momentum
            recent_volume = volumes[-3:].sum()
            avg_volume = np.mean(volumes[:-3]) if len(volumes) > 3 else recent_volume
            
            if recent_volume < self.sweep_threshold:
                return None
            
            volume_ratio = recent_volume / max(avg_volume, 1)
            
        else:
            # Selling sweep: consecutive downticks with increasing volume
            downticks = price_changes < 0
            if not np.all(downticks[-3:]):
                return None
            
            recent_volume = volumes[-3:].sum()
            avg_volume = np.mean(volumes[:-3]) if len(volumes) > 3 else recent_volume
            
            if recent_volume < self.sweep_threshold:
                return None
            
            volume_ratio = recent_volume / max(avg_volume, 1)
        
        # Count levels swept
        price_range = abs(prices[-1] - prices[-min(10, len(prices))])
        levels_swept = int(price_range / (prices[-1] * 0.0001))  # Assuming 1 bps tick
        
        if levels_swept < 2:
            return None
        
        # Calculate momentum score
        momentum_score = volume_ratio * levels_swept * 0.1
        
        total_value = recent_volume * np.mean(prices[-3:])
        
        return SweepEvent(
            timestamp_us=self.timestamp_buffer[-1],
            price=prices[-1],
            volume=recent_volume,
            side='buy' if side == 'buy' else 'sell',
            levels_swept=levels_swept,
            total_value=total_value,
            momentum_score=min(momentum_score, 10.0)
        )


class MomentumBouncePredictor:
    """
    Predicts momentum bounce after liquidity sweep
    Uses lightweight statistical models within RAM constraints
    """
    
    def __init__(self, lookback_samples: int = 50):
        self.lookback = lookback_samples
        
        # Historical bounce patterns (fixed size)
        self.bounce_returns: deque = deque(maxlen=lookback_samples)
        self.bounce_times: deque = deque(maxlen=lookback_samples)
        
        # Model parameters
        self.mean_bounce: float = 0.0
        self.std_bounce: float = 0.001
        self.mean_duration_us: int = 50000  # 50ms average
    
    def record_bounce(self, return_pct: float, duration_us: int):
        """Record observed bounce for model updating"""
        self.bounce_returns.append(return_pct)
        self.bounce_times.append(duration_us)
        
        if len(self.bounce_returns) >= 10:
            returns_arr = np.array(self.bounce_returns)
            times_arr = np.array(self.bounce_times)
            
            self.mean_bounce = np.mean(returns_arr)
            self.std_bounce = np.std(returns_arr)
            self.mean_duration_us = int(np.mean(times_arr))
    
    def predict_bounce(self, sweep: SweepEvent) -> Tuple[float, int, float]:
        """
        Predict bounce characteristics after sweep
        Returns: (expected_return_pct, expected_duration_us, confidence)
        """
        # Base prediction from historical patterns
        expected_return = self.mean_bounce
        
        # Adjust based on sweep intensity
        intensity_factor = min(sweep.momentum_score / 5.0, 2.0)
        expected_return *= intensity_factor
        
        # Direction based on sweep side
        if sweep.side == 'buy':
            expected_return = abs(expected_return)  # Expect upward bounce
        else:
            expected_return = -abs(expected_return)  # Expect downward bounce
        
        # Duration estimate
        expected_duration = self.mean_duration_us
        
        # Confidence based on sweep quality and model fit
        confidence = 0.5
        if sweep.levels_swept >= 5:
            confidence += 0.2
        if sweep.momentum_score >= 3.0:
            confidence += 0.2
        if self.std_bounce < 0.002:
            confidence += 0.1
        
        return expected_return, expected_duration, min(confidence, 0.95)


class SniperExecutor:
    """
    Executes sniper entries based on sweep detection and bounce prediction
    Implements risk controls and position sizing
    """
    
    def __init__(self, max_position_pct: float = 0.05, risk_per_trade: float = 0.001):
        self.max_position_pct = max_position_pct
        self.risk_per_trade = risk_per_trade
        
        # Active positions
        self.active_snipes: List[SniperSignal] = []
        
        # Performance tracking
        self.total_trades: int = 0
        self.winning_trades: int = 0
        self.total_pnl: float = 0.0
    
    def generate_signal(self, sweep: SweepEvent, bounce_prediction: Tuple[float, int, float],
                       current_price: float) -> Optional[SniperSignal]:
        """Generate sniper signal from sweep and prediction"""
        expected_return, duration_us, confidence = bounce_prediction
        
        if confidence < 0.6:
            return None
        
        # Calculate position size based on confidence and risk
        size_pct = self.max_position_pct * confidence
        
        # Entry: enter in direction of expected bounce
        direction = 'long' if expected_return > 0 else 'short'
        
        # Target: based on expected return
        if direction == 'long':
            target_price = current_price * (1 + abs(expected_return))
            stop_loss = current_price * (1 - self.risk_per_trade)
        else:
            target_price = current_price * (1 - abs(expected_return))
            stop_loss = current_price * (1 + self.risk_per_trade)
        
        return SniperSignal(
            timestamp_us=time.time_ns() // 1000,
            direction=direction,
            entry_price=current_price,
            target_price=target_price,
            stop_loss=stop_loss,
            size_pct=size_pct,
            confidence=confidence,
            expected_hold_us=duration_us
        )
    
    def record_outcome(self, signal: SniperSignal, exit_price: float, pnl: float):
        """Record trade outcome for performance tracking"""
        self.total_trades += 1
        self.total_pnl += pnl
        
        if pnl > 0:
            self.winning_trades += 1
    
    @property
    def win_rate(self) -> float:
        if self.total_trades == 0:
            return 0.0
        return self.winning_trades / self.total_trades
    
    @property
    def avg_pnl(self) -> float:
        if self.total_trades == 0:
            return 0.0
        return self.total_pnl / self.total_trades


class LiquiditySniper:
    """
    Main sniper class combining detection, prediction, and execution
    Memory-efficient design for <14GB RAM constraint
    """
    
    def __init__(self, sweep_threshold: float = 1000000.0):
        self.detector = LiquiditySweepDetector(sweep_threshold=sweep_threshold)
        self.predictor = MomentumBouncePredictor()
        self.executor = SniperExecutor()
        
        # Operating state
        self.enabled: bool = True
        self.last_action_time: int = 0
    
    def process_tick(self, timestamp_us: int, price: float, volume: float, side: str) -> Optional[SniperSignal]:
        """
        Process market tick and potentially generate sniper signal
        """
        if not self.enabled:
            return None
        
        # Detect sweep
        sweep = self.detector.update(timestamp_us, price, volume, side)
        
        if sweep is None:
            return None
        
        # Predict bounce
        bounce_pred = self.predictor.predict_bounce(sweep)
        
        # Generate signal
        signal = self.executor.generate_signal(sweep, bounce_pred, price)
        
        if signal:
            self.last_action_time = timestamp_us
        
        return signal
    
    def complete_snipe(self, signal: SniperSignal, exit_price: float):
        """Record completed snipe for learning"""
        # Calculate PnL
        if signal.direction == 'long':
            pnl = (exit_price - signal.entry_price) / signal.entry_price
        else:
            pnl = (signal.entry_price - exit_price) / signal.entry_price
        
        # Record for model improvement
        duration_us = (time.time_ns() // 1000) - signal.timestamp_us
        self.predictor.record_bounce(pnl, duration_us)
        
        # Track performance
        self.executor.record_outcome(signal, exit_price, pnl)
    
    def get_stats(self) -> dict:
        """Get sniper statistics"""
        return {
            'sweeps_detected': self.detector.sweeps_detected,
            'total_trades': self.executor.total_trades,
            'win_rate': self.executor.win_rate,
            'avg_pnl': self.executor.avg_pnl,
            'total_pnl': self.executor.total_pnl,
        }
    
    def set_enabled(self, enabled: bool):
        """Enable/disable sniper"""
        self.enabled = enabled


# Example usage pattern
if __name__ == "__main__":
    sniper = LiquiditySniper(sweep_threshold=500000.0)
    
    # Simulated market data processing loop
    # In production, this would integrate with NautilusTrader
    sample_data = [
        (1000000, 50000.0, 100000.0, 'buy'),
        (1000100, 50005.0, 200000.0, 'buy'),
        (1000200, 50010.0, 500000.0, 'buy'),  # Potential sweep
        (1000300, 50015.0, 800000.0, 'buy'),
        (1000400, 50020.0, 1000000.0, 'buy'),
    ]
    
    for ts, price, vol, side in sample_data:
        signal = sniper.process_tick(ts, price, vol, side)
        if signal:
            print(f"Sniper signal: {signal.direction} at {signal.entry_price}")
