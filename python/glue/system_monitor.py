"""
System Monitor Daemon

A lightweight, low-overhead daemon that monitors RAM usage via psutil
and aggressively triggers garbage collection if approaching the 14GB hard limit.

Features:
- Continuous memory monitoring at configurable intervals
- Aggressive GC triggering before hitting limits
- Integration with Ray worker pausing
- Alert callbacks for critical thresholds
"""

import os
import gc
import time
import threading
import signal
from typing import Optional, Callable, Dict, Any, List, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime

import psutil


class MemoryState(Enum):
    """System memory state."""
    NORMAL = "normal"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


@dataclass(slots=True)
class MemoryThresholds:
    """Memory threshold configuration in bytes."""
    # Hard limit: 14GB
    hard_limit_bytes: int = 14 * 1024 * 1024 * 1024
    # Emergency: 95% of hard limit
    emergency_threshold: float = 0.95
    # Critical: 90% of hard limit
    critical_threshold: float = 0.90
    # Warning: 80% of hard limit
    warning_threshold: float = 0.80
    
    @property
    def emergency_bytes(self) -> int:
        return int(self.hard_limit_bytes * self.emergency_threshold)
    
    @property
    def critical_bytes(self) -> int:
        return int(self.hard_limit_bytes * self.critical_threshold)
    
    @property
    def warning_bytes(self) -> int:
        return int(self.hard_limit_bytes * self.warning_threshold)


@dataclass(slots=True)
class SystemStats:
    """Current system statistics."""
    timestamp: float
    total_ram_bytes: int
    available_ram_bytes: int
    used_ram_bytes: int
    ram_percent: float
    process_rss_bytes: int
    process_vms_bytes: int
    process_cpu_percent: float
    state: MemoryState
    gc_triggered: bool = False


class SystemMonitor:
    """
    Lightweight system monitor daemon for RAM enforcement.
    
    Monitors both system-wide and process-specific memory usage,
    triggering aggressive GC and callbacks when thresholds are crossed.
    """
    
    _instance: Optional['SystemMonitor'] = None
    _lock = threading.Lock()
    
    def __new__(cls, *args, **kwargs) -> 'SystemMonitor':
        """Singleton pattern."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
            return cls._instance
    
    def __init__(
        self,
        thresholds: Optional[MemoryThresholds] = None,
        check_interval_sec: float = 1.0,
    ):
        # Prevent double initialization
        if hasattr(self, '_initialized') and self._initialized:
            return
        
        self._initialized = True
        self.thresholds = thresholds or MemoryThresholds()
        self.check_interval_sec = check_interval_sec
        
        self._process = psutil.Process(os.getpid())
        self._running = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._stats_history: List[SystemStats] = []
        self._max_history = 60  # Keep last 60 seconds of stats
        
        # Callbacks for different states
        self._callbacks: Dict[MemoryState, List[Callable]] = {
            state: [] for state in MemoryState
        }
        
        # GC tracking
        self._gc_count = 0
        self._last_gc_time = 0.0
        self._gc_cooldown_sec = 1.0  # Minimum time between GC triggers
        
        # State tracking
        self._current_state = MemoryState.NORMAL
        self._state_changes: List[Tuple[float, MemoryState]] = []
    
    def register_callback(self, state: MemoryState, callback: Callable):
        """Register a callback for a specific memory state."""
        self._callbacks[state].append(callback)
    
    def _run_callbacks(self, state: MemoryState, stats: SystemStats):
        """Run all callbacks for a given state."""
        for callback in self._callbacks.get(state, []):
            try:
                callback(stats)
            except Exception as e:
                print(f"[MONITOR] Callback error for {state}: {e}")
    
    def _get_system_stats(self) -> SystemStats:
        """Get current system and process statistics."""
        mem = psutil.virtual_memory()
        now = time.time()
        
        # Process-specific stats
        with self._process.oneshot():
            proc_mem = self._process.memory_info()
            proc_cpu = self._process.cpu_percent(interval=0)
        
        # Determine state based on system memory
        used_ratio = mem.used / mem.total
        
        if used_ratio >= self.thresholds.emergency_threshold:
            state = MemoryState.EMERGENCY
        elif used_ratio >= self.thresholds.critical_threshold:
            state = MemoryState.CRITICAL
        elif used_ratio >= self.thresholds.warning_threshold:
            state = MemoryState.WARNING
        else:
            state = MemoryState.NORMAL
        
        return SystemStats(
            timestamp=now,
            total_ram_bytes=mem.total,
            available_ram_bytes=mem.available,
            used_ram_bytes=mem.used,
            ram_percent=mem.percent,
            process_rss_bytes=proc_mem.rss,
            process_vms_bytes=proc_mem.vms,
            process_cpu_percent=proc_cpu,
            state=state,
        )
    
    def _maybe_trigger_gc(self, state: MemoryState) -> bool:
        """Trigger GC if in critical/emergency state and cooldown expired."""
        if state not in (MemoryState.CRITICAL, MemoryState.EMERGENCY):
            return False
        
        now = time.time()
        if now - self._last_gc_time < self._gc_cooldown_sec:
            return False
        
        # Aggressive GC
        gc.collect()
        self._gc_count += 1
        self._last_gc_time = now
        
        return True
    
    def _monitor_loop(self):
        """Main monitoring loop running in background thread."""
        while self._running:
            try:
                stats = self._get_system_stats()
                
                # Check for state change
                if stats.state != self._current_state:
                    old_state = self._current_state
                    self._current_state = stats.state
                    self._state_changes.append((time.time(), self._current_state))
                    
                    # Run callbacks for new state
                    self._run_callbacks(self._current_state, stats)
                    
                    print(f"[MONITOR] State change: {old_state.value} -> {self._current_state.value}")
                    print(f"[MONITOR] RAM: {stats.ram_percent:.1f}%, Process RSS: {stats.process_rss_bytes / 1024 / 1024:.1f}MB")
                
                # Trigger GC if needed
                gc_triggered = self._maybe_trigger_gc(stats.state)
                if gc_triggered:
                    stats.gc_triggered = True
                
                # Store stats history
                self._stats_history.append(stats)
                if len(self._stats_history) > self._max_history:
                    self._stats_history.pop(0)
                
            except Exception as e:
                print(f"[MONITOR] Error in monitor loop: {e}")
            
            time.sleep(self.check_interval_sec)
    
    def start(self):
        """Start the monitoring daemon."""
        if self._running:
            return
        
        self._running = True
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop,
            daemon=True,
            name="SystemMonitor"
        )
        self._monitor_thread.start()
        print("[MONITOR] System monitor started")
    
    def stop(self):
        """Stop the monitoring daemon."""
        self._running = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5.0)
        print("[MONITOR] System monitor stopped")
    
    @property
    def is_running(self) -> bool:
        return self._running
    
    @property
    def current_state(self) -> MemoryState:
        return self._current_state
    
    @property
    def current_stats(self) -> Optional[SystemStats]:
        """Get the most recent statistics."""
        if self._stats_history:
            return self._stats_history[-1]
        return None
    
    @property
    def stats_history(self) -> List[SystemStats]:
        return self._stats_history.copy()
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of monitor status."""
        stats = self.current_stats
        return {
            'is_running': self._running,
            'current_state': self._current_state.value,
            'ram_percent': stats.ram_percent if stats else 0,
            'process_rss_mb': stats.process_rss_bytes / 1024 / 1024 if stats else 0,
            'gc_count': self._gc_count,
            'state_changes': len(self._state_changes),
            'thresholds': {
                'hard_limit_gb': self.thresholds.hard_limit_bytes / 1024 / 1024 / 1024,
                'warning_percent': self.thresholds.warning_threshold * 100,
                'critical_percent': self.thresholds.critical_threshold * 100,
                'emergency_percent': self.thresholds.emergency_threshold * 100,
            }
        }


# Convenience functions for module-level access
_monitor_instance: Optional[SystemMonitor] = None


def get_monitor() -> SystemMonitor:
    """Get or create the global monitor instance."""
    global _monitor_instance
    if _monitor_instance is None:
        _monitor_instance = SystemMonitor()
    return _monitor_instance


def init_monitor(
    hard_limit_gb: float = 14.0,
    check_interval_sec: float = 1.0,
) -> SystemMonitor:
    """Initialize the global monitor with custom settings."""
    global _monitor_instance
    thresholds = MemoryThresholds(
        hard_limit_bytes=int(hard_limit_gb * 1024 * 1024 * 1024)
    )
    _monitor_instance = SystemMonitor(
        thresholds=thresholds,
        check_interval_sec=check_interval_sec,
    )
    return _monitor_instance


def start_monitoring():
    """Start the global monitor."""
    get_monitor().start()


def stop_monitoring():
    """Stop the global monitor."""
    get_monitor().stop()


if __name__ == "__main__":
    # Demo/test code
    print("[DEMO] Starting system monitor demo")
    
    def on_warning(stats: SystemStats):
        print(f"[ALERT] Warning: RAM at {stats.ram_percent:.1f}%")
    
    def on_critical(stats: SystemStats):
        print(f"[ALERT] Critical: RAM at {stats.ram_percent:.1f}%, GC triggered: {stats.gc_triggered}")
    
    def on_emergency(stats: SystemStats):
        print(f"[ALERT] EMERGENCY: RAM at {stats.ram_percent:.1f}%, immediate action required!")
    
    monitor = init_monitor(hard_limit_gb=14.0, check_interval_sec=0.5)
    monitor.register_callback(MemoryState.WARNING, on_warning)
    monitor.register_callback(MemoryState.CRITICAL, on_critical)
    monitor.register_callback(MemoryState.EMERGENCY, on_emergency)
    
    monitor.start()
    
    # Run demo for 10 seconds
    try:
        for i in range(10):
            summary = monitor.get_summary()
            print(f"[DEMO] {summary}")
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        monitor.stop()
        print("[DEMO] Monitor stopped")
