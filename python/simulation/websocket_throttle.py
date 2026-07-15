"""
WebSocket Throttle Simulator - Simulates exchange-side WebSocket throttling and message queue backlogs.
Ensures bot's internal ring buffers can gracefully drop non-critical telemetry while prioritizing trades.
Memory-bounded using fixed-size ring buffers.
"""

import time
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable
from collections import deque
from enum import Enum, auto
import logging
import heapq

logger = logging.getLogger(__name__)


class MessageType(Enum):
    """Priority levels for WebSocket messages."""
    TRADE_EXECUTION = auto()  # Highest priority - never drop
    ORDER_UPDATE = auto()     # High priority - rarely drop
    ORDERBOOK_DELTA = auto()  # Medium priority - can drop if backlog
    TICKER = auto()           # Low priority - drop first under pressure
    FUNDING_RATE = auto()     # Low priority
    HEARTBEAT = auto()        # Lowest priority - drop immediately


@dataclass
class WSMessage:
    """WebSocket message with priority metadata."""
    msg_type: MessageType
    payload: bytes
    timestamp: float = field(default_factory=time.time)
    sequence: int = 0
    symbol: str = ""
    
    def __lt__(self, other):
        # For heap ordering - lower priority value = higher priority
        return self.msg_type.value < other.msg_type.value


@dataclass
class ThrottleConfig:
    """WebSocket throttle configuration."""
    max_queue_size: int = 10000  # Maximum messages in queue
    high_water_mark_pct: float = 80.0  # Start dropping at this fill %
    low_water_mark_pct: float = 50.0   # Stop dropping below this
    max_messages_per_second: int = 5000  # Rate limit
    burst_allowance: int = 1000  # Allow bursts above rate limit
    
    # Priority-based drop thresholds
    drop_ticker_at_pct: float = 70.0
    drop_orderbook_at_pct: float = 85.0
    drop_heartbeat_at_pct: float = 60.0


@dataclass
class ThrottleStats:
    """Statistics for throttle simulation."""
    messages_received: int = 0
    messages_processed: int = 0
    messages_dropped: int = 0
    messages_by_type_dropped: Dict[str, int] = field(default_factory=dict)
    max_queue_depth: int = 0
    current_queue_depth: int = 0
    throttle_events: int = 0
    avg_latency_ms: float = 0.0


class WebSocketThrottleSimulator:
    """
    Simulates WebSocket throttling and message queue management.
    Uses bounded memory with priority-based message dropping.
    """
    
    def __init__(self, config: ThrottleConfig):
        self.config = config
        
        # Priority queue for messages (heap for O(1) lowest priority access)
        self.message_queue: List[WSMessage] = []
        self.queue_lock = threading.Lock()
        
        # Ring buffer for processed messages (bounded)
        self.processed_buffer: deque = deque(maxlen=1000)
        
        # Rate limiting
        self.message_times: deque = deque(maxlen=config.max_messages_per_second)
        self.burst_tokens = config.burst_allowance
        
        # Statistics
        self.stats = ThrottleStats()
        self.latencies: deque = deque(maxlen=1000)
        
        # State
        self.running = False
        self.throttling = False
        self._drop_callback: Optional[Callable] = None
        
    def set_drop_callback(self, callback: Callable[[WSMessage], None]) -> None:
        """Set callback for dropped messages."""
        self._drop_callback = callback
        
    def enqueue_message(self, message: WSMessage) -> bool:
        """
        Enqueue a WebSocket message.
        Returns False if message was dropped due to backpressure.
        """
        with self.queue_lock:
            self.stats.messages_received += 1
            
            # Calculate current fill percentage
            fill_pct = (len(self.message_queue) / self.config.max_queue_size) * 100
            
            # Check if we need to drop based on priority
            if fill_pct >= self.config.high_water_mark_pct:
                if self._should_drop_message(message, fill_pct):
                    self._drop_message(message)
                    return False
            
            # Add to priority queue
            heapq.heappush(self.message_queue, message)
            
            # Update stats
            self.stats.current_queue_depth = len(self.message_queue)
            if self.stats.current_queue_depth > self.stats.max_queue_depth:
                self.stats.max_queue_depth = self.stats.current_queue_depth
            
            # Check throttle state
            if fill_pct >= self.config.high_water_mark_pct and not self.throttling:
                self.throttling = True
                self.stats.throttle_events += 1
                logger.warning(f"WebSocket throttle activated: {fill_pct:.1f}% full")
            elif fill_pct <= self.config.low_water_mark_pct and self.throttling:
                self.throttling = False
                logger.info("WebSocket throttle deactivated")
            
            return True
    
    def _should_drop_message(self, message: WSMessage, fill_pct: float) -> bool:
        """Determine if a message should be dropped based on priority and fill level."""
        msg_type = message.msg_type
        
        if msg_type == MessageType.HEARTBEAT:
            return fill_pct >= self.config.drop_heartbeat_at_pct
        
        if msg_type == MessageType.TICKER:
            return fill_pct >= self.config.drop_ticker_at_pct
        
        if msg_type == MessageType.ORDERBOOK_DELTA:
            return fill_pct >= self.config.drop_orderbook_at_pct
        
        # Never drop high priority messages unless absolutely full
        if msg_type in (MessageType.TRADE_EXECUTION, MessageType.ORDER_UPDATE):
            return fill_pct >= 95.0
        
        return False
    
    def _drop_message(self, message: WSMessage) -> None:
        """Handle dropped message."""
        self.stats.messages_dropped += 1
        
        type_name = message.msg_type.name
        if type_name not in self.stats.messages_by_type_dropped:
            self.stats.messages_by_type_dropped[type_name] = 0
        self.stats.messages_by_type_dropped[type_name] += 1
        
        if self._drop_callback:
            try:
                self._drop_callback(message)
            except Exception as e:
                logger.error(f"Drop callback error: {e}")
    
    def dequeue_message(self) -> Optional[WSMessage]:
        """
        Dequeue the highest priority message.
        Applies rate limiting.
        """
        with self.queue_lock:
            if not self.message_queue:
                return None
            
            # Rate limiting check
            now = time.time()
            self._clean_old_message_times(now)
            
            if len(self.message_times) >= self.config.max_messages_per_second:
                if self.burst_tokens > 0:
                    self.burst_tokens -= 1
                else:
                    # Rate limited - don't dequeue yet
                    return None
            
            # Get highest priority message
            message = heapq.heappop(self.message_queue)
            
            # Record processing
            self.message_times.append(now)
            self.processed_buffer.append(message)
            
            # Calculate latency
            latency_ms = (now - message.timestamp) * 1000
            self.latencies.append(latency_ms)
            self._update_latency_stats(latency_ms)
            
            self.stats.messages_processed += 1
            self.stats.current_queue_depth = len(self.message_queue)
            
            return message
    
    def _clean_old_message_times(self, now: float) -> None:
        """Remove message timestamps older than 1 second."""
        while self.message_times and now - self.message_times[0] > 1.0:
            self.message_times.popleft()
    
    def _update_latency_stats(self, latency_ms: float) -> None:
        """Update average latency statistics."""
        if self.latencies:
            total = sum(self.latencies)
            count = len(self.latencies)
            self.stats.avg_latency_ms = total / count
    
    def get_queue_depth(self) -> int:
        """Get current queue depth."""
        with self.queue_lock:
            return len(self.message_queue)
    
    def get_fill_percentage(self) -> float:
        """Get current queue fill percentage."""
        with self.queue_lock:
            return (len(self.message_queue) / self.config.max_queue_size) * 100
    
    def is_throttling(self) -> bool:
        """Check if currently throttling."""
        return self.throttling
    
    def get_stats(self) -> Dict:
        """Get current statistics."""
        with self.queue_lock:
            return {
                'messages_received': self.stats.messages_received,
                'messages_processed': self.stats.messages_processed,
                'messages_dropped': self.stats.messages_dropped,
                'messages_by_type_dropped': dict(self.stats.messages_by_type_dropped),
                'max_queue_depth': self.stats.max_queue_depth,
                'current_queue_depth': self.stats.current_queue_depth,
                'throttle_events': self.stats.throttle_events,
                'avg_latency_ms': self.stats.avg_latency_ms,
                'fill_percentage': self.get_fill_percentage(),
                'is_throttling': self.throttling,
                'rate_limit_usage': len(self.message_times) / self.config.max_messages_per_second,
            }
    
    def reset(self) -> None:
        """Reset simulator state."""
        with self.queue_lock:
            self.message_queue.clear()
            self.processed_buffer.clear()
            self.message_times.clear()
            self.burst_tokens = self.config.burst_allowance
            self.stats = ThrottleStats()
            self.latencies.clear()
            self.throttling = False


class MessagePrioritizer:
    """
    Dynamic message priority adjuster.
    Increases priority of critical messages during high volatility.
    """
    
    def __init__(self):
        self.volatility_multiplier = 1.0
        self.active_symbols: set = set()
        
    def update_volatility(self, symbol: str, volatility: float) -> None:
        """Update volatility for a symbol."""
        if volatility > 0.05:  # > 5% volatility
            self.active_symbols.add(symbol)
            self.volatility_multiplier = max(self.volatility_multiplier, 2.0)
        elif symbol in self.active_symbols and volatility < 0.02:
            self.active_symbols.discard(symbol)
            if not self.active_symbols:
                self.volatility_multiplier = 1.0
    
    def should_elevate_priority(self, message: WSMessage) -> bool:
        """Check if message priority should be elevated."""
        if message.symbol in self.active_symbols:
            if message.msg_type == MessageType.ORDERBOOK_DELTA:
                return True
        return False


# Example usage
if __name__ == "__main__":
    config = ThrottleConfig(
        max_queue_size=5000,
        high_water_mark_pct=80.0,
        max_messages_per_second=3000,
    )
    
    simulator = WebSocketThrottleSimulator(config)
    
    # Simulate message flood
    print("Simulating message flood...")
    for i in range(6000):
        msg_type = MessageType(i % 6 + 1)
        msg = WSMessage(
            msg_type=msg_type,
            payload=b"test",
            sequence=i,
            symbol="BTCUSDT"
        )
        success = simulator.enqueue_message(msg)
        if not success and i % 1000 == 0:
            print(f"Message {i} dropped due to backpressure")
    
    # Print stats
    stats = simulator.get_stats()
    print("\nWebSocket Throttle Stats:")
    for k, v in stats.items():
        print(f"  {k}: {v}")
