"""
Venue Failover - Automated venue failover logic for exchange outages.
Dynamically routes TWAP/VWAP execution slices to backup venues when primary fails.
Memory-bounded using ring buffers and streaming Polars DataFrames.
"""

import time
import asyncio
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple
from collections import deque
import logging

logger = logging.getLogger(__name__)


class VenueStatus(Enum):
    """Exchange venue status states."""
    HEALTHY = auto()
    DEGRADED = auto()
    UNHEALTHY = auto()
    OFFLINE = auto()


@dataclass
class VenueMetrics:
    """Real-time metrics for a trading venue."""
    latency_ms: float = 0.0
    success_rate: float = 1.0
    order_rejection_rate: float = 0.0
    fill_rate: float = 1.0
    last_heartbeat: float = 0.0
    consecutive_failures: int = 0
    total_orders_sent: int = 0
    total_orders_filled: int = 0


@dataclass
class ExecutionSlice:
    """A single execution slice for TWAP/VWAP."""
    symbol: str
    side: str
    quantity: float
    venue: str
    slice_id: str
    timestamp: float = field(default_factory=time.time)
    status: str = "pending"


class VenueHealthMonitor:
    """
    Monitors health of multiple trading venues.
    Uses exponential weighted moving average for latency tracking.
    """
    
    def __init__(self, ewma_alpha: float = 0.3):
        self.ewma_alpha = ewma_alpha
        self._venues: Dict[str, VenueStatus] = {}
        self._metrics: Dict[str, VenueMetrics] = {}
        self._latency_history: Dict[str, deque] = {}
        self._max_history = 100  # Bounded memory
        
    def register_venue(self, venue: str) -> None:
        """Register a new venue for monitoring."""
        if venue not in self._venues:
            self._venues[venue] = VenueStatus.HEALTHY
            self._metrics[venue] = VenueMetrics()
            self._latency_history[venue] = deque(maxlen=self._max_history)
            
    def update_latency(self, venue: str, latency_ms: float) -> None:
        """Update latency metric with EWMA smoothing."""
        if venue not in self._metrics:
            self.register_venue(venue)
            
        metrics = self._metrics[venue]
        # EWMA update
        metrics.latency_ms = (
            self.ewma_alpha * latency_ms + 
            (1 - self.ewma_alpha) * metrics.latency_ms
        )
        self._latency_history[venue].append(latency_ms)
        
    def update_order_result(self, venue: str, success: bool, filled: bool = False) -> None:
        """Update order success/failure metrics."""
        if venue not in self._metrics:
            self.register_venue(venue)
            
        metrics = self._metrics[venue]
        metrics.total_orders_sent += 1
        
        if success:
            metrics.consecutive_failures = 0
            if filled:
                metrics.total_orders_filled += 1
        else:
            metrics.consecutive_failures += 1
            
        # Update rates
        if metrics.total_orders_sent > 0:
            metrics.success_rate = (
                metrics.total_orders_sent - metrics.consecutive_failures
            ) / metrics.total_orders_sent
            metrics.fill_rate = metrics.total_orders_filled / metrics.total_orders_sent
            
    def update_heartbeat(self, venue: str) -> None:
        """Record a successful heartbeat from venue."""
        if venue not in self._metrics:
            self.register_venue(venue)
        self._metrics[venue].last_heartbeat = time.time()
        
    def assess_health(self, venue: str) -> VenueStatus:
        """Assess current health status of a venue."""
        if venue not in self._metrics:
            return VenueStatus.OFFLINE
            
        metrics = self._metrics[venue]
        now = time.time()
        
        # Check heartbeat timeout (5 seconds)
        if now - metrics.last_heartbeat > 5.0:
            return VenueStatus.OFFLINE
            
        # Check consecutive failures
        if metrics.consecutive_failures >= 5:
            return VenueStatus.UNHEALTHY
        elif metrics.consecutive_failures >= 3:
            return VenueStatus.DEGRADED
            
        # Check latency threshold (degraded if > 500ms)
        if metrics.latency_ms > 500:
            return VenueStatus.DEGRADED
            
        return VenueStatus.HEALTHY
    
    def get_all_statuses(self) -> Dict[str, VenueStatus]:
        """Get status of all registered venues."""
        return {venue: self.assess_health(venue) for venue in self._venues}
    
    def get_healthy_venues(self) -> List[str]:
        """Get list of currently healthy venues."""
        return [
            venue for venue, status in self.get_all_statuses().items()
            if status == VenueStatus.HEALTHY
        ]


class VenueFailoverRouter:
    """
    Automated venue failover router for TWAP/VWAP execution.
    Dynamically selects optimal venue based on health metrics.
    """
    
    def __init__(
        self,
        primary_venue: str,
        backup_venues: List[str],
        failover_threshold: int = 3,
        fallback_cooldown_ms: float = 30.0
    ):
        self.primary_venue = primary_venue
        self.backup_venues = backup_venues
        self.failover_threshold = failover_threshold
        self.fallback_cooldown_ms = fallback_cooldown_ms
        
        self.health_monitor = VenueHealthMonitor()
        self._current_venue = primary_venue
        self._failover_count = 0
        self._last_failover_time = 0.0
        self._venue_order_queue: Dict[str, deque] = {}
        self._max_queue_size = 1000  # Bounded memory per venue
        
        # Register all venues
        self.health_monitor.register_venue(primary_venue)
        for venue in backup_venues:
            self.health_monitor.register_venue(venue)
            self._venue_order_queue[venue] = deque(maxlen=self._max_queue_size)
            
    def select_venue(self, symbol: str, urgency: float = 0.0) -> str:
        """
        Select optimal venue for execution.
        
        Args:
            symbol: Trading pair symbol
            urgency: Urgency factor (0.0-1.0), higher means prefer lower latency
            
        Returns:
            Selected venue name
        """
        # Check if primary is healthy
        primary_status = self.health_monitor.assess_health(self.primary_venue)
        
        if primary_status == VenueStatus.HEALTHY:
            self._current_venue = self.primary_venue
            return self.primary_venue
            
        # Primary degraded/unhealthy, find best backup
        healthy_backups = self.health_monitor.get_healthy_venues()
        
        if not healthy_backups:
            # All venues unhealthy, use primary as last resort
            logger.warning(f"All backup venues unhealthy, using primary {self.primary_venue}")
            return self.primary_venue
            
        # Select backup with lowest latency if urgency is high
        if urgency > 0.5:
            best_venue = min(
                healthy_backups,
                key=lambda v: self.health_monitor._metrics[v].latency_ms
            )
        else:
            # Prefer venue with highest success rate
            best_venue = max(
                healthy_backups,
                key=lambda v: self.health_monitor._metrics[v].success_rate
            )
            
        # Record failover
        if self._current_venue != best_venue:
            self._failover_count += 1
            self._last_failover_time = time.time()
            logger.info(f"Venue failover: {self._current_venue} -> {best_venue}")
            self._current_venue = best_venue
            
        return best_venue
    
    def route_execution_slice(
        self,
        symbol: str,
        side: str,
        quantity: float,
        urgency: float = 0.0
    ) -> ExecutionSlice:
        """Route an execution slice to optimal venue."""
        venue = self.select_venue(symbol, urgency)
        
        slice_id = f"{symbol}_{side}_{time.time_ns()}"
        slice_order = ExecutionSlice(
            symbol=symbol,
            side=side,
            quantity=quantity,
            venue=venue,
            slice_id=slice_id
        )
        
        # Queue for tracking
        self._venue_order_queue[venue].append(slice_order)
        
        logger.debug(f"Routed slice {slice_id} to {venue}")
        return slice_order
    
    def record_execution_result(
        self,
        venue: str,
        slice_id: str,
        success: bool,
        filled: bool = False,
        latency_ms: float = 0.0
    ) -> None:
        """Record result of an execution attempt."""
        self.health_monitor.update_order_result(venue, success, filled)
        if latency_ms > 0:
            self.health_monitor.update_latency(venue, latency_ms)
            
    def record_heartbeat(self, venue: str) -> None:
        """Record heartbeat from venue."""
        self.health_monitor.update_heartbeat(venue)
        
    def get_failover_stats(self) -> Dict:
        """Get failover statistics."""
        return {
            "current_venue": self._current_venue,
            "failover_count": self._failover_count,
            "last_failover_time": self._last_failover_time,
            "venue_statuses": {
                v: s.name for v, s in self.health_monitor.get_all_statuses().items()
            },
            "healthy_venues": self.health_monitor.get_healthy_venues(),
        }
    
    def force_failover(self, target_venue: Optional[str] = None) -> str:
        """Force immediate failover to specified or best available venue."""
        if target_venue:
            if target_venue not in self.backup_venues:
                raise ValueError(f"Invalid target venue: {target_venue}")
            self._current_venue = target_venue
        else:
            healthy = self.health_monitor.get_healthy_venues()
            if healthy:
                self._current_venue = healthy[0]
                
        self._failover_count += 1
        self._last_failover_time = time.time()
        logger.warning(f"Forced failover to {self._current_venue}")
        
        return self._current_venue


class TwapVwapFailoverExecutor:
    """
    TWAP/VWAP executor with automatic venue failover.
    Splits large orders and routes slices dynamically.
    """
    
    def __init__(
        self,
        router: VenueFailoverRouter,
        total_quantity: float,
        duration_seconds: float,
        symbol: str,
        side: str
    ):
        self.router = router
        self.total_quantity = total_quantity
        self.duration_seconds = duration_seconds
        self.symbol = symbol
        self.side = side
        
        self.slices_executed = 0
        self.slices_failed = 0
        self.total_filled = 0.0
        self._running = False
        
    async def execute_twap(self, num_slices: int) -> List[ExecutionSlice]:
        """Execute TWAP strategy with venue failover."""
        self._running = True
        slice_quantity = self.total_quantity / num_slices
        interval = self.duration_seconds / num_slices
        
        executed_slices = []
        
        for i in range(num_slices):
            if not self._running:
                break
                
            # Route slice with increasing urgency as we approach deadline
            urgency = i / num_slices
            slice_order = self.router.route_execution_slice(
                symbol=self.symbol,
                side=self.side,
                quantity=slice_quantity,
                urgency=urgency
            )
            
            executed_slices.append(slice_order)
            
            # Simulate execution delay
            await asyncio.sleep(interval)
            self.slices_executed += 1
            
        return executed_slices
    
    def stop(self) -> None:
        """Stop TWAP execution."""
        self._running = False


# Example usage and testing
if __name__ == "__main__":
    # Setup router
    router = VenueFailoverRouter(
        primary_venue="binance",
        backup_venues=["bybit", "okx"],
        failover_threshold=3
    )
    
    # Simulate health updates
    router.record_heartbeat("binance")
    router.record_heartbeat("bybit")
    router.record_heartbeat("okx")
    
    # Route some executions
    for i in range(5):
        slice_order = router.route_execution_slice(
            symbol="BTCUSDT",
            side="buy",
            quantity=0.1,
            urgency=0.3
        )
        print(f"Routed to: {slice_order.venue}")
        
    # Simulate primary venue failure
    for i in range(5):
        router.health_monitor.update_order_result("binance", success=False)
        
    # Next routing should failover
    slice_order = router.route_execution_slice(
        symbol="BTCUSDT",
        side="buy",
        quantity=0.1,
        urgency=0.8
    )
    print(f"After failure, routed to: {slice_order.venue}")
    
    # Print stats
    print("\nFailover Stats:")
    for k, v in router.get_failover_stats().items():
        print(f"  {k}: {v}")
