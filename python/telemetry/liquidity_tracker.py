"""
Liquidity tracker for identifying walls, icebergs, and spoofing events.
Pushes annotated markers to the UI for visual overlay on candlestick charts.
Strictly memory-bounded to prevent RAM bloat during high-frequency tracking.
"""

import asyncio
import time
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple
from collections import deque
import json


@dataclass
class LiquidityEvent:
    """Detected liquidity event for UI annotation."""
    timestamp_ms: int
    event_type: str  # 'wall', 'iceberg', 'spoof', 'large_cancel'
    price: float
    volume: float
    side: str  # 'bid' or 'ask'
    confidence: float  # 0.0 to 1.0
    duration_ms: int = 0
    metadata: Dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class OrderLevel:
    """Tracking data for a specific price level."""
    price: float
    cumulative_volume: float = 0.0
    order_count: int = 0
    first_seen_ms: int = 0
    last_update_ms: int = 0
    max_volume: float = 0.0
    cancel_volume: float = 0.0
    is_iceberg_suspect: bool = False
    iceberg_refill_count: int = 0


class LiquidityTracker:
    """
    Background tracker for liquidity anomalies.
    Identifies walls, icebergs, and spoofing patterns.
    """

    def __init__(
        self,
        wall_threshold: float = 1000000.0,  # Volume threshold for wall detection
        iceberg_refill_threshold: int = 3,  # Refills before flagging as iceberg
        spoof_duration_ms: int = 500,  # Max duration for spoof detection
        max_events: int = 1000,  # Memory bound for event history
        max_levels: int = 500,  # Memory bound for tracked levels
    ):
        self.wall_threshold = wall_threshold
        self.iceberg_refill_threshold = iceberg_refill_threshold
        self.spoof_duration_ms = spoof_duration_ms
        self.max_events = max_events
        self.max_levels = max_levels

        # Price level tracking (memory bounded)
        self.bid_levels: Dict[float, OrderLevel] = {}
        self.ask_levels: Dict[float, OrderLevel] = {}

        # Event history (bounded deque)
        self.events: deque[LiquidityEvent] = deque(maxlen=max_events)

        # Callback for pushing to UI
        self.ui_callback = None

        # Running state
        self._running = False
        self._task: Optional[asyncio.Task] = None

    def set_ui_callback(self, callback):
        """Set callback function for pushing events to UI."""
        self.ui_callback = callback

    def process_orderbook_update(
        self,
        bids: List[Tuple[float, float]],
        asks: List[Tuple[float, float]],
        timestamp_ms: int,
    ):
        """
        Process order book snapshot/update.
        bids/asks: List of (price, volume) tuples
        """
        # Process bids
        for price, volume in bids:
            self._update_level(
                price, volume, 'bid', timestamp_ms
            )

        # Process asks
        for price, volume in asks:
            self._update_level(
                price, volume, 'ask', timestamp_ms
            )

        # Check for cancellations (levels that disappeared)
        self._check_cancellations('bid', bids, timestamp_ms)
        self._check_cancellations('ask', asks, timestamp_ms)

    def _update_level(
        self,
        price: float,
        volume: float,
        side: str,
        timestamp_ms: int,
    ):
        """Update tracking for a specific price level."""
        levels = self.bid_levels if side == 'bid' else self.ask_levels

        # Enforce memory bound
        if len(levels) >= self.max_levels and price not in levels:
            # Remove oldest/least active level
            self._evict_oldest_level(side)

        if price not in levels:
            levels[price] = OrderLevel(
                price=price,
                cumulative_volume=volume,
                order_count=1,
                first_seen_ms=timestamp_ms,
                last_update_ms=timestamp_ms,
                max_volume=volume,
            )
        else:
            level = levels[price]
            prev_volume = level.cumulative_volume

            # Detect iceberg refill
            if volume > prev_volume and level.cancel_volume > 0:
                level.iceberg_refill_count += 1
                if level.iceberg_refill_count >= self.iceberg_refill_threshold:
                    level.is_iceberg_suspect = True
                    self._emit_iceberg_event(level, side, timestamp_ms)

            level.cumulative_volume = volume
            level.order_count += 1
            level.last_update_ms = timestamp_ms
            level.max_volume = max(level.max_volume, volume)

        # Check for liquidity wall
        if volume >= self.wall_threshold:
            self._emit_wall_event(price, volume, side, timestamp_ms)

    def _check_cancellations(
        self,
        side: str,
        current_levels: List[Tuple[float, float]],
        timestamp_ms: int,
    ):
        """Detect cancelled orders and potential spoofing."""
        levels = self.bid_levels if side == 'bid' else self.ask_levels
        current_prices = {price for price, _ in current_levels}

        to_remove = []
        for price, level in levels.items():
            if price not in current_prices:
                # Order was cancelled
                duration = timestamp_ms - level.first_seen_ms

                # Check for spoofing (large order, short duration)
                if (level.max_volume >= self.wall_threshold * 0.5 and
                        duration < self.spoof_duration_ms):
                    self._emit_spoof_event(level, side, timestamp_ms, duration)

                # Track cancellation volume
                level.cancel_volume = level.cumulative_volume
                to_remove.append(price)

        # Remove cancelled levels
        for price in to_remove:
            del levels[price]

    def _evict_oldest_level(self, side: str):
        """Remove the oldest tracked level to maintain memory bounds."""
        levels = self.bid_levels if side == 'bid' else self.ask_levels

        if not levels:
            return

        oldest_price = min(
            levels.keys(),
            key=lambda p: levels[p].first_seen_ms
        )
        del levels[oldest_price]

    def _emit_wall_event(
        self,
        price: float,
        volume: float,
        side: str,
        timestamp_ms: int,
    ):
        """Emit a liquidity wall detection event."""
        event = LiquidityEvent(
            timestamp_ms=timestamp_ms,
            event_type='wall',
            price=price,
            volume=volume,
            side=side,
            confidence=min(1.0, volume / (self.wall_threshold * 2)),
            metadata={'threshold': self.wall_threshold},
        )
        self._add_event(event)

    def _emit_iceberg_event(
        self,
        level: OrderLevel,
        side: str,
        timestamp_ms: int,
    ):
        """Emit an iceberg order detection event."""
        event = LiquidityEvent(
            timestamp_ms=timestamp_ms,
            event_type='iceberg',
            price=level.price,
            volume=level.max_volume,
            side=side,
            confidence=min(1.0, level.iceberg_refill_count / 5),
            duration_ms=timestamp_ms - level.first_seen_ms,
            metadata={
                'refill_count': level.iceberg_refill_count,
                'total_volume': level.cumulative_volume,
            },
        )
        self._add_event(event)

    def _emit_spoof_event(
        self,
        level: OrderLevel,
        side: str,
        timestamp_ms: int,
        duration_ms: int,
    ):
        """Emit a potential spoofing detection event."""
        event = LiquidityEvent(
            timestamp_ms=timestamp_ms,
            event_type='spoof',
            price=level.price,
            volume=level.max_volume,
            side=side,
            confidence=min(1.0, (self.spoof_duration_ms - duration_ms) / self.spoof_duration_ms),
            duration_ms=duration_ms,
            metadata={
                'cancel_volume': level.cancel_volume,
                'rapid_cancel': True,
            },
        )
        self._add_event(event)

    def _add_event(self, event: LiquidityEvent):
        """Add event to history and push to UI if callback is set."""
        self.events.append(event)

        if self.ui_callback:
            try:
                self.ui_callback(event.to_dict())
            except Exception as e:
                print(f"Error pushing event to UI: {e}")

    def get_recent_events(
        self,
        limit: int = 100,
        event_type: Optional[str] = None,
    ) -> List[Dict]:
        """Get recent liquidity events, optionally filtered by type."""
        events = list(self.events)

        if event_type:
            events = [e for e in events if e.event_type == event_type]

        # Sort by timestamp descending
        events.sort(key=lambda e: e.timestamp_ms, reverse=True)

        return [e.to_dict() for e in events[:limit]]

    def get_statistics(self) -> Dict:
        """Get current tracking statistics."""
        return {
            'tracked_bid_levels': len(self.bid_levels),
            'tracked_ask_levels': len(self.ask_levels),
            'total_events': len(self.events),
            'walls_detected': sum(1 for e in self.events if e.event_type == 'wall'),
            'icebergs_detected': sum(1 for e in self.events if e.event_type == 'iceberg'),
            'spoofs_detected': sum(1 for e in self.events if e.event_type == 'spoof'),
        }

    def clear(self):
        """Clear all tracking data."""
        self.bid_levels.clear()
        self.ask_levels.clear()
        self.events.clear()


# Example usage with async background task
async def run_liquidity_tracker(tracker: LiquidityTracker, update_interval_ms: int = 100):
    """Run the liquidity tracker as a background async task."""
    tracker._running = True

    while tracker._running:
        await asyncio.sleep(update_interval_ms / 1000.0)

        # Periodic cleanup or analysis could go here
        stats = tracker.get_statistics()

        if stats['total_events'] > 0 and stats['total_events'] % 100 == 0:
            print(f"Liquidity Tracker Stats: {stats}")


def create_tracker_with_ui_bridge(
    ws_send_function,
    wall_threshold: float = 1000000.0,
) -> LiquidityTracker:
    """
    Create a liquidity tracker with WebSocket bridge to frontend.

    Args:
        ws_send_function: Async function to send messages to WebSocket clients
        wall_threshold: Volume threshold for wall detection

    Returns:
        Configured LiquidityTracker instance
    """
    tracker = LiquidityTracker(wall_threshold=wall_threshold)

    def ui_callback(event_dict: dict):
        """Callback to push events to frontend via WebSocket."""
        message = {
            'type': 'liquidity_event',
            'data': event_dict,
        }
        # Fire and forget - don't block the tracker
        asyncio.create_task(ws_send_function(json.dumps(message)))

    tracker.set_ui_callback(ui_callback)
    return tracker


if __name__ == '__main__':
    # Test/demo mode
    tracker = LiquidityTracker(wall_threshold=100000.0)

    # Simulate order book updates
    test_bids = [(50000.0, 150000.0), (49999.0, 50000.0)]
    test_asks = [(50001.0, 200000.0), (50002.0, 75000.0)]

    tracker.process_orderbook_update(test_bids, test_asks, int(time.time() * 1000))

    print("Events detected:")
    for event in tracker.get_recent_events():
        print(f"  {event['event_type']} at {event['price']}: {event['volume']}")

    print(f"\nStatistics: {tracker.get_statistics()}")
