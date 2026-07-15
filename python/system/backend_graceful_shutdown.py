"""
Master graceful shutdown orchestrator.
Upon receiving SIGTERM or UI "Kill Switch" command:
- Safely flattens positions
- Cancels all open orders
- Flushes RAM to LMDB
- Gracefully tears down the Ray cluster
"""

from __future__ import annotations
import asyncio
import logging
import signal
import sys
import time
from datetime import datetime, timezone
from typing import Optional, Callable, Awaitable, List, Dict, Any
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class ShutdownPhase(Enum):
    """Phases of the graceful shutdown process."""
    INITIATED = "initiated"
    CANCELING_ORDERS = "canceling_orders"
    FLATTENING_POSITIONS = "flattening_positions"
    FLUSHING_TO_DISK = "flushing_to_disk"
    STOPPING_STRATEGIES = "stopping_strategies"
    TEARDOWN_RAY = "teardown_ray"
    COMPLETED = "completed"


@dataclass
class ShutdownStatus:
    """Current shutdown status."""
    phase: ShutdownPhase
    initiated_at: datetime
    initiated_by: str  # SIGTERM, UI_KILL, API
    progress_percent: float
    current_operation: str
    errors: List[str] = field(default_factory=list)
    completed_at: Optional[datetime] = None


class BackendGracefulShutdown:
    """
    Master graceful shutdown orchestrator for the trading backend.
    
    Ensures clean teardown of all components while preserving
    capital and maintaining audit compliance.
    """
    
    def __init__(
        self,
        cancel_orders_callback: Optional[Callable[..., Awaitable[bool]]] = None,
        flatten_positions_callback: Optional[Callable[..., Awaitable[bool]]] = None,
        flush_to_lmdb_callback: Optional[Callable[..., Awaitable[bool]]] = None,
        stop_strategies_callback: Optional[Callable[..., Awaitable[None]]] = None,
        teardown_ray_callback: Optional[Callable[..., Awaitable[None]]] = None,
    ):
        self.cancel_orders_callback = cancel_orders_callback
        self.flatten_positions_callback = flatten_positions_callback
        self.flush_to_lmdb_callback = flush_to_lmdb_callback
        self.stop_strategies_callback = stop_strategies_callback
        self.teardown_ray_callback = teardown_ray_callback
        
        # Shutdown state
        self._is_shutting_down = False
        self._shutdown_status: Optional[ShutdownStatus] = None
        self._shutdown_task: Optional[asyncio.Task] = None
        
        # Registered signal handlers
        self._signals_registered = False
        
        # Progress callbacks for UI
        self._progress_callbacks: List[Callable[[ShutdownStatus], None]] = []
    
    def register_signal_handlers(self) -> None:
        """Register OS signal handlers for graceful shutdown."""
        if self._signals_registered:
            return
        
        def signal_handler(signum, frame):
            sig_name = signal.Signals(signum).name
            logger.warning(f"Received signal {sig_name}, initiating graceful shutdown")
            
            # Schedule async shutdown
            if self._shutdown_task is None or self._shutdown_task.done():
                self._shutdown_task = asyncio.create_task(
                    self.initiate_shutdown(initiated_by=f"SIGNAL:{sig_name}")
                )
        
        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)
        
        # Windows-specific
        if sys.platform == "win32":
            try:
                signal.signal(signal.SIGBREAK, signal_handler)
            except Exception:
                pass
        
        self._signals_registered = True
        logger.info("Signal handlers registered for graceful shutdown")
    
    def add_progress_callback(self, callback: Callable[[ShutdownStatus], None]) -> None:
        """Add a callback to receive shutdown progress updates."""
        self._progress_callbacks.append(callback)
    
    def _notify_progress(self) -> None:
        """Notify all progress callbacks of current status."""
        if self._shutdown_status:
            for callback in self._progress_callbacks:
                try:
                    callback(self._shutdown_status)
                except Exception as e:
                    logger.error(f"Progress callback error: {e}")
    
    async def initiate_shutdown(self, initiated_by: str = "UNKNOWN") -> ShutdownStatus:
        """
        Initiate the graceful shutdown sequence.
        
        Args:
            initiated_by: Source of shutdown request (SIGTERM, UI_KILL, API)
        
        Returns:
            Final shutdown status
        """
        if self._is_shutting_down:
            logger.warning("Shutdown already in progress")
            return self._shutdown_status
        
        self._is_shutting_down = True
        
        self._shutdown_status = ShutdownStatus(
            phase=ShutdownPhase.INITIATED,
            initiated_at=datetime.now(timezone.utc),
            initiated_by=initiated_by,
            progress_percent=0.0,
            current_operation="Initializing shutdown sequence",
        )
        
        logger.critical(
            f"GRACEFUL SHUTDOWN INITIATED by {initiated_by} at {self._shutdown_status.initiated_at}"
        )
        
        try:
            # Phase 1: Cancel all open orders
            await self._phase_cancel_orders()
            
            # Phase 2: Flatten positions (if requested)
            if initiated_by.startswith("UI_KILL"):
                await self._phase_flatten_positions()
            
            # Phase 3: Flush RAM to persistent storage
            await self._phase_flush_to_disk()
            
            # Phase 4: Stop all strategies
            await self._phase_stop_strategies()
            
            # Phase 5: Tear down Ray cluster
            await self._phase_teardown_ray()
            
            # Mark completed
            self._shutdown_status.phase = ShutdownPhase.COMPLETED
            self._shutdown_status.completed_at = datetime.now(timezone.utc)
            self._shutdown_status.progress_percent = 100.0
            self._shutdown_status.current_operation = "Shutdown complete"
            
            elapsed = (self._shutdown_status.completed_at - self._shutdown_status.initiated_at).total_seconds()
            logger.info(f"Graceful shutdown completed in {elapsed:.2f} seconds")
            
        except Exception as e:
            logger.exception(f"Error during shutdown: {e}")
            self._shutdown_status.errors.append(str(e))
            raise
        
        finally:
            self._notify_progress()
        
        return self._shutdown_status
    
    async def _phase_cancel_orders(self) -> bool:
        """Phase 1: Cancel all open orders."""
        self._shutdown_status.phase = ShutdownPhase.CANCELING_ORDERS
        self._shutdown_status.current_operation = "Canceling all open orders"
        self._shutdown_status.progress_percent = 10.0
        self._notify_progress()
        
        start_time = time.time()
        
        if self.cancel_orders_callback is None:
            logger.warning("No cancel_orders_callback configured, skipping order cancellation")
            return True
        
        try:
            success = await self.cancel_orders_callback()
            
            if success:
                elapsed = time.time() - start_time
                logger.info(f"All orders canceled in {elapsed:.2f}s")
                return True
            else:
                self._shutdown_status.errors.append("Order cancellation returned failure")
                logger.error("Order cancellation failed")
                return False
                
        except Exception as e:
            self._shutdown_status.errors.append(f"Order cancellation error: {e}")
            logger.exception("Order cancellation failed")
            return False
    
    async def _phase_flatten_positions(self) -> bool:
        """Phase 2: Flatten all positions (emergency only)."""
        self._shutdown_status.phase = ShutdownPhase.FLATTENING_POSITIONS
        self._shutdown_status.current_operation = "Flattening all positions"
        self._shutdown_status.progress_percent = 30.0
        self._notify_progress()
        
        start_time = time.time()
        
        if self.flatten_positions_callback is None:
            logger.warning("No flatten_positions_callback configured, skipping position flattening")
            return True
        
        try:
            success = await self.flatten_positions_callback()
            
            if success:
                elapsed = time.time() - start_time
                logger.info(f"All positions flattened in {elapsed:.2f}s")
                return True
            else:
                self._shutdown_status.errors.append("Position flattening returned failure")
                logger.error("Position flattening failed")
                return False
                
        except Exception as e:
            self._shutdown_status.errors.append(f"Position flattening error: {e}")
            logger.exception("Position flattening failed")
            return False
    
    async def _phase_flush_to_disk(self) -> bool:
        """Phase 3: Flush all RAM state to LMDB/persistent storage."""
        self._shutdown_status.phase = ShutdownPhase.FLUSHING_TO_DISK
        self._shutdown_status.current_operation = "Flushing RAM to persistent storage"
        self._shutdown_status.progress_percent = 50.0
        self._notify_progress()
        
        start_time = time.time()
        
        if self.flush_to_lmdb_callback is None:
            logger.warning("No flush_to_lmdb_callback configured, skipping disk flush")
            return True
        
        try:
            success = await self.flush_to_lmdb_callback()
            
            if success:
                elapsed = time.time() - start_time
                logger.info(f"RAM flushed to disk in {elapsed:.2f}s")
                return True
            else:
                self._shutdown_status.errors.append("Disk flush returned failure")
                logger.error("Disk flush failed")
                return False
                
        except Exception as e:
            self._shutdown_status.errors.append(f"Disk flush error: {e}")
            logger.exception("Disk flush failed")
            return False
    
    async def _phase_stop_strategies(self) -> bool:
        """Phase 4: Stop all trading strategies."""
        self._shutdown_status.phase = ShutdownPhase.STOPPING_STRATEGIES
        self._shutdown_status.current_operation = "Stopping all strategies"
        self._shutdown_status.progress_percent = 70.0
        self._notify_progress()
        
        start_time = time.time()
        
        if self.stop_strategies_callback is None:
            logger.warning("No stop_strategies_callback configured, skipping strategy shutdown")
            return True
        
        try:
            await self.stop_strategies_callback()
            
            elapsed = time.time() - start_time
            logger.info(f"All strategies stopped in {elapsed:.2f}s")
            return True
            
        except Exception as e:
            self._shutdown_status.errors.append(f"Strategy shutdown error: {e}")
            logger.exception("Strategy shutdown failed")
            return False
    
    async def _phase_teardown_ray(self) -> bool:
        """Phase 5: Tear down Ray cluster."""
        self._shutdown_status.phase = ShutdownPhase.TEARDOWN_RAY
        self._shutdown_status.current_operation = "Tearing down Ray cluster"
        self._shutdown_status.progress_percent = 90.0
        self._notify_progress()
        
        start_time = time.time()
        
        if self.teardown_ray_callback is None:
            logger.warning("No teardown_ray_callback configured, skipping Ray teardown")
            return True
        
        try:
            await self.teardown_ray_callback()
            
            elapsed = time.time() - start_time
            logger.info(f"Ray cluster torn down in {elapsed:.2f}s")
            return True
            
        except Exception as e:
            self._shutdown_status.errors.append(f"Ray teardown error: {e}")
            logger.exception("Ray teardown failed")
            return False
    
    def get_status(self) -> Optional[ShutdownStatus]:
        """Get current shutdown status."""
        return self._shutdown_status
    
    def is_shutting_down(self) -> bool:
        """Check if shutdown is in progress."""
        return self._is_shutting_down
    
    def is_complete(self) -> bool:
        """Check if shutdown has completed."""
        return (
            self._shutdown_status is not None 
            and self._shutdown_status.phase == ShutdownPhase.COMPLETED
        )
    
    def get_errors(self) -> List[str]:
        """Get any errors that occurred during shutdown."""
        if self._shutdown_status:
            return self._shutdown_status.errors.copy()
        return []


# Example usage and testing
if __name__ == "__main__":
    async def mock_cancel():
        print("Canceling orders...")
        await asyncio.sleep(0.5)
        print("Orders canceled")
        return True
    
    async def mock_flush():
        print("Flushing to LMDB...")
        await asyncio.sleep(0.3)
        print("Flushed")
        return True
    
    async def mock_stop_strategies():
        print("Stopping strategies...")
        await asyncio.sleep(0.2)
        return True
    
    async def mock_teardown_ray():
        print("Tearing down Ray...")
        await asyncio.sleep(0.2)
        return True
    
    async def main():
        shutdown = BackendGracefulShutdown(
            cancel_orders_callback=mock_cancel,
            flush_to_lmdb_callback=mock_flush,
            stop_strategies_callback=mock_stop_strategies,
            teardown_ray_callback=mock_teardown_ray,
        )
        
        # Register signal handlers
        shutdown.register_signal_handlers()
        
        # Add progress callback
        def on_progress(status: ShutdownStatus):
            print(f"Progress: {status.phase.value} - {status.progress_percent}%")
        
        shutdown.add_progress_callback(on_progress)
        
        # Simulate shutdown
        print("\nInitiating graceful shutdown...")
        result = await shutdown.initiate_shutdown(initiated_by="TEST")
        
        print(f"\nShutdown completed!")
        print(f"Phase: {result.phase.value}")
        print(f"Duration: {(result.completed_at - result.initiated_at).total_seconds():.2f}s")
        print(f"Errors: {result.errors}")
    
    asyncio.run(main())
