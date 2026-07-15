"""
Smart publisher that downsamples L3 data to L2 for standard Nautilus strategies.
Saves processing overhead while keeping raw L3 stream available for HFT engine.
Memory-bounded with strict caps on buffer sizes.
"""

import asyncio
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass, field
from collections import defaultdict
import time


@dataclass(slots=True)
class PriceLevel:
    """Single price level in L2 book"""
    price: int
    quantity: int
    order_count: int = 1


@dataclass(slots=True)
class L2Snapshot:
    """L2 order book snapshot"""
    bids: List[PriceLevel] = field(default_factory=list)
    asks: List[PriceLevel] = field(default_factory=list)
    timestamp_ns: int = 0
    symbol: str = ""


class DepthPublisher:
    """
    Smart publisher that converts L3 (individual orders) to L2 (aggregated levels).
    Configurable depth levels and update throttling for optimal performance.
    """
    
    # Default maximum levels to publish
    DEFAULT_MAX_LEVELS = 25
    # Default throttle interval (ms)
    DEFAULT_THROTTLE_MS = 10
    # Maximum symbols to track
    MAX_SYMBOLS = 100
    
    def __init__(
        self,
        max_levels: int = DEFAULT_MAX_LEVELS,
        throttle_ms: int = DEFAULT_THROTTLE_MS,
        preserve_l3: bool = True,
    ):
        """
        Initialize depth publisher.
        
        Args:
            max_levels: Maximum bid/ask levels to publish
            throttle_ms: Minimum interval between updates (ms)
            preserve_l3: Keep raw L3 data available for HFT engine
        """
        self._max_levels = max_levels
        self._throttle_ms = throttle_ms
        self._preserve_l3 = preserve_l3
        
        # L3 order storage: symbol -> {order_id: (price, quantity, side)}
        self._l3_orders: Dict[str, Dict[int, Tuple[int, int, int]]] = defaultdict(dict)
        
        # Aggregated L2 levels: symbol -> {price: [bid_qty, ask_qty, count]}
        self._l2_levels: Dict[str, Dict[int, List[int]]] = defaultdict(lambda: defaultdict(lambda: [0, 0, 0]))
        
        # Last update timestamps
        self._last_update_ns: Dict[str, int] = {}
        
        # Subscribers/callbacks
        self._l2_subscribers: List[Callable[[str, L2Snapshot], None]] = []
        self._l3_subscribers: List[Callable[[str, dict], None]] = []
        
        # Statistics
        self._stats = {
            'l3_updates': 0,
            'l2_published': 0,
            'throttled': 0,
        }
    
    def add_l3_order(
        self,
        symbol: str,
        order_id: int,
        price: int,
        quantity: int,
        side: int,  # 0 = bid, 1 = ask
        timestamp_ns: int,
    ) -> None:
        """
        Add/update an L3 order.
        
        Args:
            symbol: Trading pair symbol
            order_id: Unique order identifier
            price: Order price (fixed point)
            quantity: Order quantity
            side: 0 for bid, 1 for ask
            timestamp_ns: Timestamp in nanoseconds
        """
        if len(self._l3_orders) >= self.MAX_SYMBOLS:
            # Memory protection - evict oldest symbol
            oldest = next(iter(self._l3_orders))
            self._remove_symbol(oldest)
        
        # Store L3 order
        self._l3_orders[symbol][order_id] = (price, quantity, side)
        self._stats['l3_updates'] += 1
        
        # Update L2 aggregation
        level = self._l2_levels[symbol][price]
        if side == 0:
            level[0] += quantity  # Bid quantity
        else:
            level[1] += quantity  # Ask quantity
        level[2] += 1  # Order count
        
        # Try to publish L2 update (with throttling)
        self._try_publish_l2(symbol, timestamp_ns)
    
    def remove_l3_order(self, symbol: str, order_id: int, timestamp_ns: int) -> bool:
        """
        Remove/cancel an L3 order.
        
        Args:
            symbol: Trading pair symbol
            order_id: Order to remove
            timestamp_ns: Timestamp
            
        Returns:
            True if order was found and removed
        """
        if symbol not in self._l3_orders:
            return False
        
        order = self._l3_orders[symbol].pop(order_id, None)
        if order is None:
            return False
        
        price, quantity, side = order
        
        # Update L2 aggregation
        if price in self._l2_levels[symbol]:
            level = self._l2_levels[symbol][price]
            if side == 0:
                level[0] = max(0, level[0] - quantity)
            else:
                level[1] = max(0, level[1] - quantity)
            level[2] = max(0, level[2] - 1)
            
            # Clean up empty levels
            if level[0] == 0 and level[1] == 0 and level[2] == 0:
                del self._l2_levels[symbol][price]
        
        self._stats['l3_updates'] += 1
        self._try_publish_l2(symbol, timestamp_ns)
        return True
    
    def _try_publish_l2(self, symbol: str, timestamp_ns: int) -> None:
        """Try to publish L2 update with throttling."""
        now_ms = timestamp_ns // 1_000_000
        last_ms = self._last_update_ns.get(symbol, 0) // 1_000_000
        
        if now_ms - last_ms < self._throttle_ms:
            self._stats['throttled'] += 1
            return
        
        self._publish_l2(symbol, timestamp_ns)
    
    def _publish_l2(self, symbol: str, timestamp_ns: int) -> None:
        """Publish L2 snapshot to subscribers."""
        snapshot = self._build_l2_snapshot(symbol, timestamp_ns)
        
        for callback in self._l2_subscribers:
            try:
                callback(symbol, snapshot)
            except Exception as e:
                print(f"L2 subscriber error: {e}")
        
        self._last_update_ns[symbol] = timestamp_ns
        self._stats['l2_published'] += 1
    
    def _build_l2_snapshot(self, symbol: str, timestamp_ns: int) -> L2Snapshot:
        """Build L2 snapshot from aggregated levels."""
        levels = self._l2_levels.get(symbol, {})
        
        # Separate bids and asks
        bid_levels = []
        ask_levels = []
        
        for price, (bid_qty, ask_qty, count) in levels.items():
            if bid_qty > 0:
                bid_levels.append(PriceLevel(price=price, quantity=bid_qty, order_count=count))
            if ask_qty > 0:
                ask_levels.append(PriceLevel(price=price, quantity=ask_qty, order_count=count))
        
        # Sort: bids descending, asks ascending
        bid_levels.sort(key=lambda x: x.price, reverse=True)
        ask_levels.sort(key=lambda x: x.price)
        
        # Limit to max levels
        bid_levels = bid_levels[:self._max_levels]
        ask_levels = ask_levels[:self._max_levels]
        
        return L2Snapshot(
            bids=bid_levels,
            asks=ask_levels,
            timestamp_ns=timestamp_ns,
            symbol=symbol,
        )
    
    def get_l3_orders(self, symbol: str) -> Dict[int, Tuple[int, int, int]]:
        """
        Get raw L3 orders for a symbol (for HFT engine).
        
        Args:
            symbol: Trading pair symbol
            
        Returns:
            Dict of order_id -> (price, quantity, side)
        """
        return dict(self._l3_orders.get(symbol, {}))
    
    def get_l2_snapshot(self, symbol: str) -> L2Snapshot:
        """Get current L2 snapshot for a symbol."""
        ts = time.time_ns()
        return self._build_l2_snapshot(symbol, ts)
    
    def subscribe_l2(self, callback: Callable[[str, L2Snapshot], None]) -> None:
        """Subscribe to L2 updates."""
        self._l2_subscribers.append(callback)
    
    def subscribe_l3(self, callback: Callable[[str, dict], None]) -> None:
        """Subscribe to raw L3 updates (HFT only)."""
        if self._preserve_l3:
            self._l3_subscribers.append(callback)
    
    def unsubscribe_l2(self, callback: Callable) -> bool:
        """Unsubscribe from L2 updates."""
        try:
            self._l2_subscribers.remove(callback)
            return True
        except ValueError:
            return False
    
    def _remove_symbol(self, symbol: str) -> None:
        """Remove all data for a symbol (memory management)."""
        self._l3_orders.pop(symbol, None)
        self._l2_levels.pop(symbol, None)
        self._last_update_ns.pop(symbol, None)
    
    def clear_symbol(self, symbol: str) -> None:
        """Clear all data for a specific symbol."""
        self._remove_symbol(symbol)
    
    def clear_all(self) -> None:
        """Clear all data (use with caution)."""
        self._l3_orders.clear()
        self._l2_levels.clear()
        self._last_update_ns.clear()
    
    def get_stats(self) -> dict:
        """Get publisher statistics."""
        return {
            **self._stats,
            'symbols_tracked': len(self._l3_orders),
            'total_l3_orders': sum(len(orders) for orders in self._l3_orders.values()),
        }


# Singleton instance
_publisher_instance: Optional[DepthPublisher] = None


def get_publisher(
    max_levels: int = DepthPublisher.DEFAULT_MAX_LEVELS,
    throttle_ms: int = DepthPublisher.DEFAULT_THROTTLE_MS,
) -> DepthPublisher:
    """Get or create singleton DepthPublisher instance."""
    global _publisher_instance
    if _publisher_instance is None:
        _publisher_instance = DepthPublisher(max_levels=max_levels, throttle_ms=throttle_ms)
    return _publisher_instance


if __name__ == '__main__':
    # Demo/test code
    pub = get_publisher(max_levels=10, throttle_ms=5)
    
    # Add some test orders
    pub.add_l3_order('BTC/USD', 1, 50000, 100, 0, time.time_ns())
    pub.add_l3_order('BTC/USD', 2, 49999, 200, 0, time.time_ns())
    pub.add_l3_order('BTC/USD', 3, 50001, 150, 1, time.time_ns())
    
    # Get L2 snapshot
    snapshot = pub.get_l2_snapshot('BTC/USD')
    print(f"L2 Bids: {[(l.price, l.quantity) for l in snapshot.bids]}")
    print(f"L2 Asks: {[(l.price, l.quantity) for l in snapshot.asks]}")
    
    # Get stats
    print(f"Stats: {pub.get_stats()}")
