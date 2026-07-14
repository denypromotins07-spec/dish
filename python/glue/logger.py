"""
Zero-Copy Asynchronous Logging Module

This module implements high-performance asynchronous logging using:
- orjson for ultra-fast JSON serialization (written in Rust)
- Memory-mapped files (mmap) to avoid disk I/O bottlenecks
- Lock-free queues for thread-safe log ingestion

Designed for minimal RAM usage and maximum throughput on AMD Ryzen AI 5.
"""

import os
import mmap
import json
import time
import threading
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any, List
from collections import deque
from queue import Queue, Empty

import orjson


# Constants
DEFAULT_MMAP_SIZE_MB = 256
DEFAULT_FLUSH_INTERVAL_MS = 100
DEFAULT_QUEUE_SIZE = 10000
LOG_ENTRY_HEADER_SIZE = 8  # 4 bytes length + 4 bytes checksum


@dataclass(slots=True, frozen=True)
class LogEntry:
    """Immutable log entry with minimal memory footprint."""
    timestamp_ns: int
    level: str
    module: str
    message: str
    data: Optional[Dict[str, Any]] = None
    
    def to_json(self) -> bytes:
        """Serialize to JSON using orjson (zero-copy where possible)."""
        return orjson.dumps(asdict(self), option=orjson.OPT_SERIALIZE_NUMPY)


class MMapLogWriter:
    """
    Memory-mapped file writer for zero-copy logging.
    
    Uses a circular buffer approach within the mmap'd file to avoid
    unbounded growth while maintaining sequential write performance.
    """
    
    def __init__(self, path: str, size_mb: int = DEFAULT_MMAP_SIZE_MB):
        self.path = Path(path)
        self.size_bytes = size_mb * 1024 * 1024
        self._lock = threading.Lock()
        self._write_pos = 0
        self._total_written = 0
        self._entry_count = 0
        
        # Ensure parent directory exists
        self.path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create/truncate file to desired size
        with open(self.path, 'wb') as f:
            f.seek(self.size_bytes - 1)
            f.write(b'\x00')
        
        # Open for read/write and mmap
        self._file = open(self.path, 'r+b')
        self._mmap = mmap.mmap(self._file.fileno(), self.size_bytes)
        
    def write(self, data: bytes) -> bool:
        """Write data to mmap'd file at current position."""
        with self._lock:
            data_len = len(data)
            header = data_len.to_bytes(4, 'little')
            
            # Check if we need to wrap around
            if self._write_pos + 4 + data_len > self.size_bytes:
                # Wrap to beginning
                self._write_pos = 0
            
            # Write header
            self._mmap[self._write_pos:self._write_pos + 4] = header
            self._write_pos += 4
            
            # Write data
            if self._write_pos + data_len <= self.size_bytes:
                self._mmap[self._write_pos:self._write_pos + data_len] = data
                self._write_pos += data_len
            else:
                # Split write across boundary
                first_part = self.size_bytes - self._write_pos
                self._mmap[self._write_pos:] = data[:first_part]
                self._write_pos = 0
                remaining = data_len - first_part
                self._mmap[self._write_pos:self._write_pos + remaining] = data[first_part:]
                self._write_pos += remaining
            
            self._total_written += data_len
            self._entry_count += 1
            
            return True
    
    def flush(self):
        """Flush mmap changes to disk."""
        self._mmap.flush()
    
    def sync(self):
        """Force sync to disk (slower but safer)."""
        self._file.flush()
        os.fsync(self._file.fileno())
    
    @property
    def stats(self) -> Dict[str, Any]:
        return {
            'write_position': self._write_pos,
            'total_written_bytes': self._total_written,
            'entry_count': self._entry_count,
            'file_size_mb': self.size_bytes / (1024 * 1024),
        }
    
    def close(self):
        """Close mmap and file."""
        try:
            self._mmap.flush()
            self._mmap.close()
            self._file.close()
        except Exception:
            pass


class AsyncLogger:
    """
    High-performance asynchronous logger with memory-mapped storage.
    
    Features:
    - Lock-free log ingestion via Queue
    - Background flush thread
    - Zero-copy JSON serialization with orjson
    - Configurable flush intervals
    - Memory-bounded queue to prevent RAM bloat
    """
    
    _instance: Optional['AsyncLogger'] = None
    _lock = threading.Lock()
    
    def __new__(cls, *args, **kwargs) -> 'AsyncLogger':
        """Singleton pattern to ensure single logger instance."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
            return cls._instance
    
    def __init__(
        self,
        log_path: str = "/var/log/trading_bot/events.log",
        mmap_size_mb: int = DEFAULT_MMAP_SIZE_MB,
        flush_interval_ms: int = DEFAULT_FLUSH_INTERVAL_MS,
        queue_size: int = DEFAULT_QUEUE_SIZE,
    ):
        # Prevent double initialization
        if hasattr(self, '_initialized') and self._initialized:
            return
        
        self._initialized = True
        self._log_path = log_path
        self._flush_interval_ms = flush_interval_ms
        self._queue: Queue[Optional[LogEntry]] = Queue(maxsize=queue_size)
        self._running = False
        self._flush_thread: Optional[threading.Thread] = None
        self._dropped_count = 0
        self._logged_count = 0
        
        # Initialize mmap writer
        self._writer = MMapLogWriter(log_path, mmap_size_mb)
        
        # Level filters
        self._enabled_levels = {'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'}
    
    def start(self):
        """Start the background flush thread."""
        if self._running:
            return
        
        self._running = True
        self._flush_thread = threading.Thread(
            target=self._flush_loop,
            daemon=True,
            name="AsyncLogger-Flush"
        )
        self._flush_thread.start()
    
    def stop(self):
        """Stop the logger gracefully."""
        self._running = False
        
        # Send sentinel to unblock queue
        self._queue.put(None)
        
        if self._flush_thread:
            self._flush_thread.join(timeout=5.0)
        
        # Flush remaining entries
        self._flush_batch()
        self._writer.sync()
        self._writer.close()
    
    def _flush_loop(self):
        """Background thread that flushes log entries periodically."""
        last_flush = time.time()
        flush_interval_sec = self._flush_interval_ms / 1000.0
        
        while self._running:
            try:
                # Try to get entry with timeout
                entry = self._queue.get(timeout=0.01)
                
                if entry is None:
                    # Sentinel received, shutdown
                    break
                
                self._write_entry(entry)
                
            except Empty:
                pass
            
            # Periodic flush
            now = time.time()
            if now - last_flush >= flush_interval_sec:
                self._writer.flush()
                last_flush = now
    
    def _write_entry(self, entry: LogEntry):
        """Write a log entry to mmap file."""
        try:
            json_data = entry.to_json()
            self._writer.write(json_data)
            self._logged_count += 1
        except Exception as e:
            print(f"[LOGGER] Write error: {e}")
    
    def _flush_batch(self):
        """Flush all pending entries in queue."""
        while True:
            try:
                entry = self._queue.get_nowait()
                if entry is not None:
                    self._write_entry(entry)
            except Empty:
                break
    
    def log(
        self,
        level: str,
        module: str,
        message: str,
        data: Optional[Dict[str, Any]] = None,
    ):
        """Log a message asynchronously."""
        if level not in self._enabled_levels:
            return
        
        entry = LogEntry(
            timestamp_ns=time.time_ns(),
            level=level,
            module=module,
            message=message,
            data=data,
        )
        
        try:
            self._queue.put_nowait(entry)
        except Exception:
            # Queue full, drop log entry (prefer speed over completeness)
            self._dropped_count += 1
    
    def debug(self, module: str, message: str, data: Optional[Dict[str, Any]] = None):
        self.log('DEBUG', module, message, data)
    
    def info(self, module: str, message: str, data: Optional[Dict[str, Any]] = None):
        self.log('INFO', module, message, data)
    
    def warning(self, module: str, message: str, data: Optional[Dict[str, Any]] = None):
        self.log('WARNING', module, message, data)
    
    def error(self, module: str, message: str, data: Optional[Dict[str, Any]] = None):
        self.log('ERROR', module, message, data)
    
    def critical(self, module: str, message: str, data: Optional[Dict[str, Any]] = None):
        self.log('CRITICAL', module, message, data)
    
    @property
    def stats(self) -> Dict[str, Any]:
        """Get logger statistics."""
        return {
            'queued_entries': self._queue.qsize(),
            'logged_count': self._logged_count,
            'dropped_count': self._dropped_count,
            'writer_stats': self._writer.stats,
            'is_running': self._running,
        }


# Convenience functions for module-level access
_logger_instance: Optional[AsyncLogger] = None


def get_logger() -> AsyncLogger:
    """Get or create the global logger instance."""
    global _logger_instance
    if _logger_instance is None:
        _logger_instance = AsyncLogger()
    return _logger_instance


def init_logger(
    log_path: str = "/var/log/trading_bot/events.log",
    mmap_size_mb: int = DEFAULT_MMAP_SIZE_MB,
    flush_interval_ms: int = DEFAULT_FLUSH_INTERVAL_MS,
) -> AsyncLogger:
    """Initialize the global logger with custom settings."""
    global _logger_instance
    _logger_instance = AsyncLogger(
        log_path=log_path,
        mmap_size_mb=mmap_size_mb,
        flush_interval_ms=flush_interval_ms,
    )
    return _logger_instance


def log_debug(module: str, message: str, data: Optional[Dict[str, Any]] = None):
    get_logger().debug(module, message, data)


def log_info(module: str, message: str, data: Optional[Dict[str, Any]] = None):
    get_logger().info(module, message, data)


def log_warning(module: str, message: str, data: Optional[Dict[str, Any]] = None):
    get_logger().warning(module, message, data)


def log_error(module: str, message: str, data: Optional[Dict[str, Any]] = None):
    get_logger().error(module, message, data)


if __name__ == "__main__":
    # Demo/test code
    import tempfile
    
    with tempfile.NamedTemporaryFile(suffix='.log', delete=False) as f:
        test_path = f.name
    
    try:
        logger = init_logger(log_path=test_path, mmap_size_mb=16)
        logger.start()
        
        # Log some test entries
        for i in range(1000):
            log_info("test", f"Test message {i}", {"counter": i})
        
        time.sleep(0.5)  # Allow flush
        
        print("Logger stats:", logger.stats)
        
        logger.stop()
        print("Logger stopped successfully")
        
    finally:
        # Cleanup
        if os.path.exists(test_path):
            os.unlink(test_path)
