"""
Background sweeper for phantom/orphaned orders.
Queries the exchange for any orders that slipped past WebSocket execution reports.
Forcefully cancels them to guarantee a clean state.
"""

from __future__ import annotations
import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional, Callable, Awaitable, List, Dict, Set
from enum import Enum

logger = logging.getLogger(__name__)


class OrderStatus(Enum):
    """Order status from exchange."""
    NEW = "NEW"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCELED = "CANCELED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"


@dataclass
class ExchangeOrder:
    """Order as reported by the exchange API."""
    order_id: str
    client_order_id: str
    symbol: str
    side: str  # BUY or SELL
    order_type: str
    quantity: float
    filled_quantity: float
    price: float
    status: OrderStatus
    created_at: datetime
    updated_at: datetime


@dataclass
class PhantomOrder:
    """An order that exists on exchange but not in local state."""
    exchange_order: ExchangeOrder
    detected_at: datetime
    severity: str  # LOW, MEDIUM, HIGH
    reason: str


@dataclass
class SweepResult:
    """Result of a phantom order sweep."""
    sweep_timestamp: datetime
    total_orders_checked: int
    phantom_orders_found: int
    phantom_orders_canceled: int
    cancel_failures: int
    sweep_duration_ms: float


class PhantomOrderSweeper:
    """
    Background sweeper that detects and cancels phantom orders.
    
    Ensures no orphaned orders remain on the exchange that could
    cause unintended executions or capital lock-up.
    """
    
    def __init__(
        self,
        fetch_orders_callback: Optional[Callable[..., Awaitable[List[ExchangeOrder]]]] = None,
        cancel_order_callback: Optional[Callable[[str], Awaitable[bool]]] = None,
        known_order_ids: Optional[Set[str]] = None,
    ):
        self.fetch_orders_callback = fetch_orders_callback
        self.cancel_order_callback = cancel_order_callback
        self._known_order_ids: Set[str] = known_order_ids or set()
        
        # State tracking
        self._phantom_orders: List[PhantomOrder] = []
        self._sweep_history: List[SweepResult] = []
        self._max_history_size = 1000
        
        # Configuration
        self._sweep_interval_seconds = 60  # Run every minute
        self._min_severity_to_cancel = "LOW"  # Cancel all phantoms by default
        self._excluded_symbols: Set[str] = set()
        
        # Running state
        self._is_running = False
        self._sweep_task: Optional[asyncio.Task] = None
    
    def register_known_order(self, order_id: str) -> None:
        """Register an order ID that we know about locally."""
        self._known_order_ids.add(order_id)
    
    def unregister_order(self, order_id: str) -> None:
        """Remove an order ID from known orders (after fill/cancel)."""
        self._known_order_ids.discard(order_id)
    
    def clear_known_orders(self) -> None:
        """Clear all known order IDs."""
        self._known_order_ids.clear()
    
    async def sweep_once(self) -> SweepResult:
        """
        Perform a single sweep for phantom orders.
        
        Returns SweepResult with statistics.
        """
        start_time = datetime.utcnow()
        
        if self.fetch_orders_callback is None:
            logger.warning("No fetch_orders_callback configured")
            return SweepResult(
                sweep_timestamp=start_time,
                total_orders_checked=0,
                phantom_orders_found=0,
                phantom_orders_canceled=0,
                cancel_failures=0,
                sweep_duration_ms=0,
            )
        
        try:
            # Fetch all open orders from exchange
            exchange_orders = await self.fetch_orders_callback()
            
            phantom_orders = []
            canceled_count = 0
            cancel_failures = 0
            
            for order in exchange_orders:
                # Skip if we know about this order
                if order.order_id in self._known_order_ids:
                    continue
                
                if order.client_order_id in self._known_order_ids:
                    continue
                
                # Skip excluded symbols
                if order.symbol in self._excluded_symbols:
                    continue
                
                # This is a phantom order!
                severity = self._assess_phantom_severity(order)
                
                phantom = PhantomOrder(
                    exchange_order=order,
                    detected_at=start_time,
                    severity=severity,
                    reason=self._get_phantom_reason(order),
                )
                phantom_orders.append(phantom)
                
                # Attempt to cancel if severity meets threshold
                if self._should_cancel(phantom):
                    if self.cancel_order_callback:
                        try:
                            success = await self.cancel_order_callback(order.order_id)
                            if success:
                                canceled_count += 1
                                logger.info(
                                    f"Canceled phantom order: {order.order_id}, "
                                    f"symbol={order.symbol}, qty={order.quantity}"
                                )
                            else:
                                cancel_failures += 1
                                logger.warning(f"Failed to cancel phantom order: {order.order_id}")
                        except Exception as e:
                            cancel_failures += 1
                            logger.error(f"Error canceling phantom order: {e}")
            
            # Store phantom orders
            self._phantom_orders.extend(phantom_orders)
            
            # Trim history if needed
            if len(self._phantom_orders) > 10000:
                self._phantom_orders = self._phantom_orders[-10000:]
            
            end_time = datetime.utcnow()
            duration_ms = (end_time - start_time).total_seconds() * 1000
            
            result = SweepResult(
                sweep_timestamp=start_time,
                total_orders_checked=len(exchange_orders),
                phantom_orders_found=len(phantom_orders),
                phantom_orders_canceled=canceled_count,
                cancel_failures=cancel_failures,
                sweep_duration_ms=duration_ms,
            )
            
            self._sweep_history.append(result)
            if len(self._sweep_history) > self._max_history_size:
                self._sweep_history = self._sweep_history[-self._max_history_size:]
            
            if phantom_orders:
                logger.warning(
                    f"Sweep found {len(phantom_orders)} phantom orders, "
                    f"canceled {canceled_count}"
                )
            
            return result
            
        except Exception as e:
            logger.error(f"Sweep failed: {e}")
            return SweepResult(
                sweep_timestamp=start_time,
                total_orders_checked=0,
                phantom_orders_found=0,
                phantom_orders_canceled=0,
                cancel_failures=0,
                sweep_duration_ms=0,
            )
    
    def _assess_phantom_severity(self, order: ExchangeOrder) -> str:
        """Assess the severity of a phantom order."""
        notional_value = order.quantity * order.price
        
        if order.status == OrderStatus.FILLED:
            return "HIGH"  # Filled order we didn't track is serious
        
        if notional_value > 100000:  # > $100k
            return "HIGH"
        
        if notional_value > 10000:  # > $10k
            return "MEDIUM"
        
        return "LOW"
    
    def _get_phantom_reason(self, order: ExchangeOrder) -> str:
        """Determine likely reason for phantom order."""
        if order.status == OrderStatus.FILLED:
            return "FILL_NOT_REPORTED"
        
        if order.status == OrderStatus.PARTIALLY_FILLED:
            return "PARTIAL_FILL_NOT_REPORTED"
        
        age = datetime.utcnow() - order.created_at
        if age > timedelta(hours=24):
            return "STALE_ORDER"
        
        return "EXECUTION_REPORT_LOST"
    
    def _should_cancel(self, phantom: PhantomOrder) -> bool:
        """Determine if a phantom order should be canceled."""
        severity_levels = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}
        threshold = severity_levels.get(self._min_severity_to_cancel, 0)
        phantom_level = severity_levels.get(phantom.severity, 0)
        return phantom_level >= threshold
    
    async def start_background_sweep(self, interval_seconds: Optional[int] = None) -> None:
        """Start background sweeping task."""
        if self._is_running:
            return
        
        if interval_seconds:
            self._sweep_interval_seconds = interval_seconds
        
        self._is_running = True
        self._sweep_task = asyncio.create_task(self._sweep_loop())
        logger.info(f"Started phantom order sweeper (interval={self._sweep_interval_seconds}s)")
    
    async def _sweep_loop(self) -> None:
        """Background sweep loop."""
        while self._is_running:
            try:
                await self.sweep_once()
            except Exception as e:
                logger.error(f"Sweep loop error: {e}")
            
            await asyncio.sleep(self._sweep_interval_seconds)
    
    def stop_background_sweep(self) -> None:
        """Stop background sweeping task."""
        self._is_running = False
        if self._sweep_task:
            self._sweep_task.cancel()
            self._sweep_task = None
        logger.info("Stopped phantom order sweeper")
    
    def get_phantom_orders(self, limit: int = 100) -> List[PhantomOrder]:
        """Get recently detected phantom orders."""
        return self._phantom_orders[-limit:]
    
    def get_sweep_history(self, limit: int = 100) -> List[SweepResult]:
        """Get recent sweep results."""
        return self._sweep_history[-limit:]
    
    def get_statistics(self) -> Dict:
        """Get sweeper statistics."""
        total_phantoms = len(self._phantom_orders)
        total_sweeps = len(self._sweep_history)
        
        avg_phantoms_per_sweep = 0
        avg_duration_ms = 0
        
        if self._sweep_history:
            avg_phantoms_per_sweep = sum(s.phantom_orders_found for s in self._sweep_history) / total_sweeps
            avg_duration_ms = sum(s.sweep_duration_ms for s in self._sweep_history) / total_sweeps
        
        return {
            "total_phantom_orders_detected": total_phantoms,
            "total_sweeps_performed": total_sweeps,
            "avg_phantoms_per_sweep": avg_phantoms_per_sweep,
            "avg_sweep_duration_ms": avg_duration_ms,
            "known_order_count": len(self._known_order_ids),
            "is_running": self._is_running,
        }


# Example usage
if __name__ == "__main__":
    import asyncio
    
    async def mock_fetch():
        # Simulate fetching orders from exchange
        return [
            ExchangeOrder(
                order_id="unknown-123",
                client_order_id="client-123",
                symbol="BTC-USDT",
                side="BUY",
                order_type="LIMIT",
                quantity=1.0,
                filled_quantity=0,
                price=50000.0,
                status=OrderStatus.NEW,
                created_at=datetime.utcnow() - timedelta(minutes=5),
                updated_at=datetime.utcnow() - timedelta(minutes=5),
            ),
        ]
    
    async def mock_cancel(order_id: str) -> bool:
        print(f"Canceling order: {order_id}")
        return True
    
    async def main():
        sweeper = PhantomOrderSweeper(
            fetch_orders_callback=mock_fetch,
            cancel_order_callback=mock_cancel,
        )
        
        # Note: We intentionally don't register the order, so it appears as phantom
        result = await sweeper.sweep_once()
        
        print(f"Sweep result:")
        print(f"  Orders checked: {result.total_orders_checked}")
        print(f"  Phantoms found: {result.phantom_orders_found}")
        print(f"  Phantoms canceled: {result.phantom_orders_canceled}")
    
    asyncio.run(main())
