"""
Custom Asyncio Runner with uvloop for High-Throughput Network I/O

This module provides a tuned asyncio event loop optimized for:
- Binance WebSocket connections (high-frequency tick data)
- Minimal memory footprint
- Maximum throughput on AMD Ryzen AI 5

Uses uvloop for significant performance improvements over default asyncio.
"""

import os
import sys
import gc
import asyncio
import signal
import threading
from typing import Optional, Callable, Any, Dict, List, Set
from dataclasses import dataclass
from contextlib import asynccontextmanager

import psutil

# Try to import uvloop, fall back to default if not available
try:
    import uvloop
    UVLOOP_AVAILABLE = True
except ImportError:
    UVLOOP_AVAILABLE = False
    print("[ASYNCIO_RUNNER] uvloop not available, using default event loop")


# Constants
DEFAULT_MEMORY_LIMIT_MB = 14 * 1024  # 14GB total system limit
MEMORY_CHECK_INTERVAL_SEC = 5.0
GC_THRESHOLD_PERCENT = 90  # Trigger GC at 90% of limit
MAX_ASYNCIO_QUEUE_SIZE = 100000
WEBSOCKET_RECONNECT_DELAY_SEC = 0.05  # 50ms for fast reconnection


@dataclass(slots=True)
class RunnerConfig:
    """Configuration for the asyncio runner."""
    memory_limit_mb: int = DEFAULT_MEMORY_LIMIT_MB
    gc_threshold_percent: float = GC_THRESHOLD_PERCENT
    max_queue_size: int = MAX_ASYNCIO_QUEUE_SIZE
    enable_uvloop: bool = True
    debug: bool = False
    websocket_reconnect_delay: float = WEBSOCKET_RECONNECT_DELAY_SEC


class MemoryMonitor:
    """Monitors process memory usage and triggers GC when needed."""
    
    def __init__(self, limit_mb: int, threshold_percent: float):
        self.limit_bytes = limit_mb * 1024 * 1024
        self.threshold = threshold_percent / 100.0
        self._process = psutil.Process(os.getpid())
        self._gc_triggered_count = 0
        self._last_gc_time = 0.0
    
    def check_and_maybe_gc(self) -> bool:
        """Check memory usage and trigger GC if above threshold."""
        import time
        
        current_mem = self._process.memory_info().rss
        threshold_bytes = self.limit_bytes * self.threshold
        
        if current_mem > threshold_bytes:
            # Rate limit GC to avoid excessive pauses
            now = time.time()
            if now - self._last_gc_time > 1.0:  # At most once per second
                gc.collect()
                self._gc_triggered_count += 1
                self._last_gc_time = now
                return True
        
        return False
    
    @property
    def current_memory_mb(self) -> float:
        """Get current process memory in MB."""
        return self._process.memory_info().rss / (1024 * 1024)
    
    @property
    def memory_percent(self) -> float:
        """Get memory usage as percentage of limit."""
        current = self._process.memory_info().rss
        return (current / self.limit_bytes) * 100.0
    
    @property
    def stats(self) -> Dict[str, Any]:
        return {
            'current_memory_mb': self.current_memory_mb,
            'memory_percent': self.memory_percent,
            'gc_triggered_count': self._gc_triggered_count,
            'limit_mb': self.limit_bytes / (1024 * 1024),
        }


class AsyncioRunner:
    """
    Custom asyncio runner with uvloop and memory management.
    
    Features:
    - uvloop for high-performance event loop
    - Automatic memory monitoring and GC triggering
    - Graceful shutdown handling
    - WebSocket connection management
    """
    
    def __init__(self, config: Optional[RunnerConfig] = None):
        self.config = config or RunnerConfig()
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._running = False
        self._tasks: Set[asyncio.Task] = set()
        self._memory_monitor: Optional[MemoryMonitor] = None
        self._memory_check_task: Optional[asyncio.Task] = None
        self._shutdown_event: Optional[asyncio.Event] = None
        self._callbacks: Dict[str, List[Callable]] = {
            'start': [],
            'stop': [],
            'error': [],
        }
    
    def _setup_loop(self):
        """Configure and install the event loop."""
        if self.config.enable_uvloop and UVLOOP_AVAILABLE:
            asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
        
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        
        # Configure loop for high throughput
        if hasattr(self._loop, 'set_debug'):
            self._loop.set_debug(self.config.debug)
    
    def _setup_signal_handlers(self):
        """Set up signal handlers for graceful shutdown."""
        def handle_signal(signum, frame):
            if self._shutdown_event:
                self._loop.call_soon_threadsafe(self._shutdown_event.set)
        
        signal.signal(signal.SIGINT, handle_signal)
        signal.signal(signal.SIGTERM, handle_signal)
    
    async def _memory_check_loop(self):
        """Background task to monitor memory and trigger GC."""
        while self._running:
            if self._memory_monitor:
                self._memory_monitor.check_and_maybe_gc()
            await asyncio.sleep(MEMORY_CHECK_INTERVAL_SEC)
    
    def _run_callbacks(self, name: str):
        """Run all callbacks for a given event."""
        for callback in self._callbacks.get(name, []):
            try:
                if asyncio.iscoroutinefunction(callback):
                    asyncio.create_task(callback())
                else:
                    callback()
            except Exception as e:
                print(f"[RUNNER] Callback error ({name}): {e}")
    
    def on_start(self, callback: Callable):
        """Register a callback to run on startup."""
        self._callbacks['start'].append(callback)
        return callback
    
    def on_stop(self, callback: Callable):
        """Register a callback to run on shutdown."""
        self._callbacks['stop'].append(callback)
        return callback
    
    def on_error(self, callback: Callable):
        """Register a callback to run on error."""
        self._callbacks['error'].append(callback)
        return callback
    
    async def _run_main(self, main_coro):
        """Run the main coroutine with monitoring."""
        self._running = True
        self._shutdown_event = asyncio.Event()
        
        # Start memory monitoring
        self._memory_check_task = asyncio.create_task(self._memory_check_loop())
        
        try:
            # Run startup callbacks
            self._run_callbacks('start')
            
            # Run main coroutine concurrently with shutdown wait
            main_task = asyncio.create_task(main_coro)
            
            # Wait for either main to complete or shutdown signal
            done, pending = await asyncio.wait(
                [main_task, asyncio.create_task(self._shutdown_event.wait())],
                return_when=asyncio.FIRST_COMPLETED
            )
            
            # Cancel pending tasks
            for task in pending:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
            
            # Check if main task completed with exception
            if main_task in done:
                exc = main_task.exception()
                if exc:
                    raise exc
            
        except Exception as e:
            print(f"[RUNNER] Error: {e}")
            self._run_callbacks('error')
            raise
        finally:
            self._running = False
            
            # Cleanup
            if self._memory_check_task:
                self._memory_check_task.cancel()
                try:
                    await self._memory_check_task
                except asyncio.CancelledError:
                    pass
            
            # Cancel all remaining tasks
            for task in self._tasks:
                task.cancel()
            
            if self._tasks:
                await asyncio.gather(*self._tasks, return_exceptions=True)
            
            # Run shutdown callbacks
            self._run_callbacks('stop')
    
    def run(self, main_coro):
        """Run the asyncio event loop with the given main coroutine."""
        self._setup_loop()
        self._setup_signal_handlers()
        
        # Initialize memory monitor
        self._memory_monitor = MemoryMonitor(
            self.config.memory_limit_mb,
            self.config.gc_threshold_percent
        )
        
        try:
            self._loop.run_until_complete(self._run_main(main_coro))
        finally:
            # Cleanup
            try:
                # Cancel all running tasks
                pending = asyncio.all_tasks(self._loop)
                for task in pending:
                    task.cancel()
                
                if pending:
                    self._loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
                
                self._loop.run_until_complete(self._loop.shutdown_asyncgens())
            finally:
                self._loop.close()
    
    @property
    def loop(self) -> asyncio.AbstractEventLoop:
        """Get the current event loop."""
        if self._loop is None:
            raise RuntimeError("Event loop not initialized. Call run() first.")
        return self._loop
    
    @property
    def is_running(self) -> bool:
        return self._running
    
    @property
    def memory_stats(self) -> Dict[str, Any]:
        """Get memory statistics."""
        if self._memory_monitor:
            return self._memory_monitor.stats
        return {}


class WebSocketManager:
    """
    Manages WebSocket connections with automatic reconnection.
    Optimized for Binance/Coinbase high-frequency tick data.
    """
    
    def __init__(self, runner: AsyncioRunner):
        self.runner = runner
        self._connections: Dict[str, Any] = {}  # url -> websocket
        self._reconnecting: Set[str] = set()
    
    async def connect(
        self,
        url: str,
        on_message: Callable,
        on_error: Optional[Callable] = None,
        reconnect: bool = True,
    ):
        """Connect to a WebSocket URL with automatic reconnection."""
        import websockets
        
        while True:
            try:
                async with websockets.connect(
                    url,
                    ping_interval=30,
                    ping_timeout=10,
                    max_size=10 * 1024 * 1024,  # 10MB max message size
                    max_queue=self.runner.config.max_queue_size,
                ) as ws:
                    self._connections[url] = ws
                    print(f"[WS] Connected to {url}")
                    
                    async for message in ws:
                        try:
                            await on_message(message)
                        except Exception as e:
                            print(f"[WS] Message handler error: {e}")
                            if on_error:
                                await on_error(e)
                    
            except Exception as e:
                print(f"[WS] Connection error for {url}: {e}")
                if on_error:
                    await on_error(e)
            
            del self._connections[url]
            
            if not reconnect:
                break
            
            # Reconnect with delay
            print(f"[WS] Reconnecting to {url} in {self.runner.config.websocket_reconnect_delay}s")
            await asyncio.sleep(self.runner.config.websocket_reconnect_delay)
    
    async def send(self, url: str, message: str):
        """Send a message to a WebSocket connection."""
        if url in self._connections:
            await self._connections[url].send(message)
        else:
            raise RuntimeError(f"Not connected to {url}")
    
    async def close(self, url: str):
        """Close a WebSocket connection."""
        if url in self._connections:
            await self._connections[url].close()
            del self._connections[url]


@asynccontextmanager
async def create_runner(config: Optional[RunnerConfig] = None):
    """Context manager for creating and running an AsyncioRunner."""
    runner = AsyncioRunner(config)
    try:
        yield runner
    finally:
        pass


if __name__ == "__main__":
    # Demo/test code
    async def main():
        print("[DEMO] Starting asyncio runner demo")
        
        async def tick_processor():
            count = 0
            while True:
                count += 1
                if count % 10000 == 0:
                    print(f"[DEMO] Processed {count} ticks")
                await asyncio.sleep(0.0001)  # Simulate tick processing
        
        # Run for a few seconds
        await asyncio.sleep(3)
        
        print("[DEMO] Demo complete")
    
    config = RunnerConfig(
        memory_limit_mb=1024,  # 1GB for demo
        gc_threshold_percent=80,
        enable_uvloop=True,
    )
    
    runner = AsyncioRunner(config)
    runner.run(main())
    
    print("Final memory stats:", runner.memory_stats)
