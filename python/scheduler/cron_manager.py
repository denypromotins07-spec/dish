"""
Lightweight Cron Manager for Periodic Trading Tasks.
Respects market hours and system load limits.
Designed for low overhead on 14GB RAM constrained systems.
"""

from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass
from datetime import datetime, time, timedelta
import asyncio
import logging
import psutil
import schedule
import threading

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ScheduledTask:
    """A scheduled task with execution constraints."""
    name: str
    func: Callable
    cron_expression: str  # Simple: "every_minute", "hourly", "daily_00:00"
    max_memory_gb: float = 0.5
    allowed_hours: tuple = (0, 24)  # Allowed hour range
    allowed_days: tuple = None  # Allowed days (0=Monday, 6=Sunday)
    skip_if_trading: bool = True  # Skip if active positions exist
    last_run: Optional[datetime] = None
    next_run: Optional[datetime] = None
    enabled: bool = True


class CronManager:
    """
    Lightweight cron manager for periodic trading tasks.
    Ensures tasks don't run during critical trading periods or when memory is tight.
    """
    
    MAX_SYSTEM_RAM_GB = 14.0
    RESERVED_RAM_GB = 2.0
    
    def __init__(self):
        self.tasks: Dict[str, ScheduledTask] = {}
        self.scheduler = schedule.Scheduler()
        self.running = False
        self._thread: Optional[threading.Thread] = None
        
    def add_task(
        self,
        name: str,
        func: Callable,
        schedule_str: str,
        max_memory_gb: float = 0.5,
        allowed_hours: tuple = (0, 24),
        allowed_days: tuple = None,
        skip_if_trading: bool = True,
    ):
        """Add a scheduled task."""
        task = ScheduledTask(
            name=name,
            func=func,
            cron_expression=schedule_str,
            max_memory_gb=max_memory_gb,
            allowed_hours=allowed_hours,
            allowed_days=allowed_days,
            skip_if_trading=skip_if_trading,
        )
        
        self.tasks[name] = task
        self._schedule_task(task)
        logger.info(f"Scheduled task: {name} ({schedule_str})")
        
    def _schedule_task(self, task: ScheduledTask):
        """Map simple schedule strings to schedule library calls."""
        expr = task.cron_expression.lower()
        
        if expr == "every_minute":
            self.scheduler.every().minute.do(self._execute_task_wrapper, task)
        elif expr == "every_5_minutes":
            self.scheduler.every(5).minutes.do(self._execute_task_wrapper, task)
        elif expr == "every_15_minutes":
            self.scheduler.every(15).minutes.do(self._execute_task_wrapper, task)
        elif expr == "hourly":
            self.scheduler.every().hour.do(self._execute_task_wrapper, task)
        elif expr.startswith("daily_"):
            time_str = expr.split("_")[1]
            self.scheduler.every().day.at(time_str).do(self._execute_task_wrapper, task)
        elif expr.startswith("weekly_"):
            parts = expr.split("_")
            day = int(parts[1])
            time_str = parts[2] if len(parts) > 2 else "00:00"
            getattr(self.scheduler.every(), f"week{day}s" if day > 1 else "monday").at(time_str).do(
                self._execute_task_wrapper, task
            )
        else:
            logger.warning(f"Unknown schedule expression: {expr}")
    
    def _check_preconditions(self, task: ScheduledTask) -> bool:
        """Check if task should run based on constraints."""
        now = datetime.now()
        
        # Check allowed hours
        if not (task.allowed_hours[0] <= now.hour < task.allowed_hours[1]):
            return False
        
        # Check allowed days
        if task.allowed_days is not None and now.weekday() not in task.allowed_days:
            return False
        
        # Check memory availability
        mem = psutil.virtual_memory()
        available_gb = mem.available / (1024 ** 3)
        if available_gb < (self.RESERVED_RAM_GB + task.max_memory_gb):
            logger.warning(f"Skipping {task.name}: insufficient memory")
            return False
        
        # Check if trading is active (skip if needed)
        if task.skip_if_trading and self._is_trading_active():
            logger.info(f"Skipping {task.name}: trading active")
            return False
        
        return True
    
    def _is_trading_active(self) -> bool:
        """Check if trading is currently active (placeholder)."""
        # In production, check actual position/risk state
        now = datetime.now()
        # Crypto trades 24/7, but we might want to skip during major news
        return True  # Assume always active for crypto
    
    def _execute_task_wrapper(self, task: ScheduledTask):
        """Wrapper to execute task with precondition checks."""
        if not task.enabled:
            return
            
        if not self._check_preconditions(task):
            return
        
        try:
            logger.info(f"Executing task: {task.name}")
            task.func()
            task.last_run = datetime.now()
        except Exception as e:
            logger.error(f"Task {task.name} failed: {e}")
    
    def start(self, interval_sec: float = 1.0):
        """Start the scheduler in a background thread."""
        if self.running:
            return
            
        self.running = True
        
        def run_scheduler():
            while self.running:
                self.scheduler.run_pending()
                asyncio.sleep(interval_sec)
        
        self._thread = threading.Thread(target=run_scheduler, daemon=True)
        self._thread.start()
        logger.info("Cron manager started")
    
    def stop(self):
        """Stop the scheduler."""
        self.running = False
        if self._thread:
            self._thread.join(timeout=5.0)
        logger.info("Cron manager stopped")
    
    def enable_task(self, name: str):
        """Enable a task."""
        if name in self.tasks:
            self.tasks[name].enabled = True
            
    def disable_task(self, name: str):
        """Disable a task."""
        if name in self.tasks:
            self.tasks[name].enabled = False


# Example periodic tasks for trading bot
def setup_trading_cron():
    """Setup common trading bot periodic tasks."""
    cron = CronManager()
    
    # Funding rate harvesting (every 15 minutes)
    def harvest_funding_rates():
        logger.info("Harvesting funding rates...")
        # Implementation here
        
    # Daily reconciliation (midnight UTC)
    def daily_reconciliation():
        logger.info("Running daily reconciliation...")
        # Implementation here
        
    # Memory cleanup (every hour)
    def cleanup_memory():
        import gc
        gc.collect()
        logger.info("Memory cleanup completed")
    
    # Add tasks
    cron.add_task(
        "funding_harvest",
        harvest_funding_rates,
        "every_15_minutes",
        max_memory_gb=0.25,
        skip_if_trading=False,  # Always run
    )
    
    cron.add_task(
        "daily_recon",
        daily_reconciliation,
        "daily_00:00",
        max_memory_gb=1.0,
        allowed_hours=(0, 6),  # Only run between midnight and 6am
    )
    
    cron.add_task(
        "memory_cleanup",
        cleanup_memory,
        "hourly",
        max_memory_gb=0.1,
        skip_if_trading=False,
    )
    
    return cron


if __name__ == "__main__":
    cron = setup_trading_cron()
    cron.start()
    
    # Keep running
    try:
        while True:
            asyncio.sleep(1)
    except KeyboardInterrupt:
        cron.stop()
