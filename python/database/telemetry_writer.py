"""
Telemetry Writer for System Metrics
Asynchronous writer for CPU, RAM, latency histograms, network jitter
Stores in lightweight SQLite for frontend monitoring dashboard
Strict memory efficiency for <14GB RAM constraint
"""

import asyncio
import sqlite3
import time
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any
from collections import deque
import psutil
import threading
from contextlib import contextmanager


@dataclass
class MetricSample:
    """Single metric sample with timestamp"""
    timestamp: float
    name: str
    value: float
    tags: Dict[str, str] = field(default_factory=dict)


@dataclass
class LatencyHistogram:
    """Fixed-size latency histogram using ring buffer"""
    buckets: deque
    max_samples: int = 10000
    
    def __post_init__(self):
        self.buckets = deque(maxlen=self.max_samples)
    
    def add(self, latency_us: float):
        """Add latency sample in microseconds"""
        self.buckets.append(latency_us)
    
    def get_percentiles(self) -> Dict[str, float]:
        """Calculate p50, p95, p99 latencies"""
        if not self.buckets:
            return {"p50": 0.0, "p95": 0.0, "p99": 0.0}
        
        sorted_samples = sorted(self.buckets)
        n = len(sorted_samples)
        
        return {
            "p50": sorted_samples[int(n * 0.50)],
            "p95": sorted_samples[int(n * 0.95)],
            "p99": sorted_samples[int(n * 0.99)],
            "min": sorted_samples[0],
            "max": sorted_samples[-1],
            "avg": sum(sorted_samples) / n,
        }
    
    def reset(self):
        """Clear histogram"""
        self.buckets.clear()


class TelemetryDatabase:
    """SQLite-backed telemetry storage with connection pooling"""
    
    def __init__(self, db_path: str = "telemetry.db"):
        self.db_path = db_path
        self._local = threading.local()
        self._init_schema()
    
    @contextmanager
    def get_connection(self):
        """Thread-local connection getter"""
        if not hasattr(self._local, 'conn'):
            self._local.conn = sqlite3.connect(
                self.db_path,
                check_same_thread=False,
                timeout=30.0
            )
            self._local.conn.execute("PRAGMA journal_mode=WAL")
            self._local.conn.execute("PRAGMA synchronous=NORMAL")
            self._local.conn.execute("PRAGMA cache_size=-64000")  # 64MB cache
        yield self._local.conn
    
    def _init_schema(self):
        """Initialize database schema"""
        with self.get_connection() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS system_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    metric_name TEXT NOT NULL,
                    value REAL NOT NULL,
                    tags TEXT
                );
                
                CREATE TABLE IF NOT EXISTS latency_histograms (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    component TEXT NOT NULL,
                    p50 REAL,
                    p95 REAL,
                    p99 REAL,
                    min REAL,
                    max REAL,
                    avg REAL,
                    sample_count INTEGER
                );
                
                CREATE TABLE IF NOT EXISTS exchange_stats (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    exchange TEXT NOT NULL,
                    messages_sent INTEGER,
                    messages_received INTEGER,
                    bytes_sent INTEGER,
                    bytes_received INTEGER,
                    latency_p99 REAL
                );
                
                CREATE INDEX IF NOT EXISTS idx_metrics_time 
                ON system_metrics(timestamp);
                
                CREATE INDEX IF NOT EXISTS idx_latency_time 
                ON latency_histograms(timestamp);
                
                CREATE INDEX IF NOT EXISTS idx_exchange_time 
                ON exchange_stats(timestamp);
            """)
            conn.commit()
    
    def insert_metric(self, sample: MetricSample):
        """Insert single metric sample"""
        with self.get_connection() as conn:
            tags_str = ",".join(f"{k}={v}" for k, v in sample.tags.items())
            conn.execute(
                """INSERT INTO system_metrics 
                   (timestamp, metric_name, value, tags)
                   VALUES (?, ?, ?, ?)""",
                (sample.timestamp, sample.name, sample.value, tags_str)
            )
            conn.commit()
    
    def insert_batch_metrics(self, samples: List[MetricSample]):
        """Batch insert for efficiency"""
        if not samples:
            return
        
        with self.get_connection() as conn:
            data = [
                (s.timestamp, s.name, s.value, 
                 ",".join(f"{k}={v}" for k, v in s.tags.items()))
                for s in samples
            ]
            conn.executemany(
                """INSERT INTO system_metrics 
                   (timestamp, metric_name, value, tags)
                   VALUES (?, ?, ?, ?)""",
                data
            )
            conn.commit()
    
    def insert_latency_histogram(self, component: str, histogram: LatencyHistogram):
        """Store latency histogram percentiles"""
        percentiles = histogram.get_percentiles()
        
        with self.get_connection() as conn:
            conn.execute(
                """INSERT INTO latency_histograms 
                   (timestamp, component, p50, p95, p99, min, max, avg, sample_count)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    time.time(),
                    component,
                    percentiles["p50"],
                    percentiles["p95"],
                    percentiles["p99"],
                    percentiles["min"],
                    percentiles["max"],
                    percentiles["avg"],
                    len(histogram.buckets),
                )
            )
            conn.commit()
    
    def insert_exchange_stat(
        self,
        exchange: str,
        messages_sent: int,
        messages_received: int,
        bytes_sent: int,
        bytes_received: int,
        latency_p99: float
    ):
        """Store exchange connection statistics"""
        with self.get_connection() as conn:
            conn.execute(
                """INSERT INTO exchange_stats 
                   (timestamp, exchange, messages_sent, messages_received,
                    bytes_sent, bytes_received, latency_p99)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    time.time(),
                    exchange,
                    messages_sent,
                    messages_received,
                    bytes_sent,
                    bytes_received,
                    latency_p99,
                )
            )
            conn.commit()
    
    def get_recent_metrics(
        self,
        metric_name: str,
        since: float,
        limit: int = 1000
    ) -> List[MetricSample]:
        """Query recent metrics"""
        with self.get_connection() as conn:
            cursor = conn.execute(
                """SELECT timestamp, metric_name, value, tags
                   FROM system_metrics
                   WHERE metric_name = ? AND timestamp > ?
                   ORDER BY timestamp DESC
                   LIMIT ?""",
                (metric_name, since, limit)
            )
            
            samples = []
            for row in cursor:
                tags = {}
                if row[3]:
                    for pair in row[3].split(","):
                        if "=" in pair:
                            k, v = pair.split("=", 1)
                            tags[k] = v
                
                samples.append(MetricSample(
                    timestamp=row[0],
                    name=row[1],
                    value=row[2],
                    tags=tags
                ))
            
            return samples
    
    def cleanup_old_data(self, max_age_hours: float = 24.0):
        """Remove old telemetry data to prevent disk bloat"""
        cutoff = time.time() - (max_age_hours * 3600)
        
        with self.get_connection() as conn:
            conn.execute(
                "DELETE FROM system_metrics WHERE timestamp < ?",
                (cutoff,)
            )
            conn.execute(
                "DELETE FROM latency_histograms WHERE timestamp < ?",
                (cutoff,)
            )
            conn.execute(
                "DELETE FROM exchange_stats WHERE timestamp < ?",
                (cutoff,)
            )
            conn.commit()


class TelemetryCollector:
    """Collects system metrics asynchronously"""
    
    def __init__(self, db: TelemetryDatabase, sample_interval: float = 1.0):
        self.db = db
        self.sample_interval = sample_interval
        self.running = False
        self.latency_histograms: Dict[str, LatencyHistogram] = {}
        self._metrics_buffer: List[MetricSample] = []
        self._buffer_lock = asyncio.Lock()
        self._flush_threshold = 100
    
    def record_latency(self, component: str, latency_us: float):
        """Record latency sample for a component"""
        if component not in self.latency_histograms:
            self.latency_histograms[component] = LatencyHistogram()
        self.latency_histograms[component].add(latency_us)
    
    async def collect_system_metrics(self) -> List[MetricSample]:
        """Collect CPU, RAM, and network metrics"""
        timestamp = time.time()
        samples = []
        
        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=0.1)
        samples.append(MetricSample(
            timestamp=timestamp,
            name="cpu_usage_percent",
            value=cpu_percent,
        ))
        
        # Per-core CPU usage
        for i, core_percent in enumerate(psutil.cpu_percent(percpu=True, interval=0.1)):
            samples.append(MetricSample(
                timestamp=timestamp,
                name="cpu_core_usage_percent",
                value=core_percent,
                tags={"core": str(i)},
            ))
        
        # Memory usage
        mem = psutil.virtual_memory()
        samples.extend([
            MetricSample(
                timestamp=timestamp,
                name="memory_used_mb",
                value=mem.used / (1024 * 1024),
            ),
            MetricSample(
                timestamp=timestamp,
                name="memory_available_mb",
                value=mem.available / (1024 * 1024),
            ),
            MetricSample(
                timestamp=timestamp,
                name="memory_percent",
                value=mem.percent,
            ),
        ])
        
        # Strict RAM constraint monitoring
        if mem.percent > 87.5:  # 14GB/16GB threshold
            samples.append(MetricSample(
                timestamp=timestamp,
                name="ram_constraint_warning",
                value=1.0,
                tags={"threshold": "14GB"},
            ))
        
        # Network I/O
        net_io = psutil.net_io_counters()
        samples.extend([
            MetricSample(
                timestamp=timestamp,
                name="network_bytes_sent",
                value=net_io.bytes_sent,
            ),
            MetricSample(
                timestamp=timestamp,
                name="network_bytes_recv",
                value=net_io.bytes_recv,
            ),
            MetricSample(
                timestamp=timestamp,
                name="network_packets_sent",
                value=net_io.packets_sent,
            ),
            MetricSample(
                timestamp=timestamp,
                name="network_packets_recv",
                value=net_io.packets_recv,
            ),
        ])
        
        # Disk I/O for SSD monitoring
        disk_io = psutil.disk_io_counters()
        if disk_io:
            samples.extend([
                MetricSample(
                    timestamp=timestamp,
                    name="disk_read_mb",
                    value=disk_io.read_bytes / (1024 * 1024),
                ),
                MetricSample(
                    timestamp=timestamp,
                    name="disk_write_mb",
                    value=disk_io.write_bytes / (1024 * 1024),
                ),
            ])
        
        return samples
    
    async def flush_metrics(self):
        """Flush buffered metrics to database"""
        async with self._buffer_lock:
            if self._metrics_buffer:
                await asyncio.to_thread(
                    self.db.insert_batch_metrics,
                    self._metrics_buffer.copy()
                )
                self._metrics_buffer.clear()
    
    async def run_collector(self):
        """Main collector loop"""
        self.running = True
        last_cleanup = time.time()
        
        while self.running:
            try:
                # Collect system metrics
                samples = await self.collect_system_metrics()
                
                async with self._buffer_lock:
                    self._metrics_buffer.extend(samples)
                    
                    if len(self._metrics_buffer) >= self._flush_threshold:
                        await self.flush_metrics()
                
                # Store latency histograms periodically
                for component, histogram in self.latency_histograms.items():
                    if len(histogram.buckets) > 0:
                        await asyncio.to_thread(
                            self.db.insert_latency_histogram,
                            component,
                            histogram
                        )
                        histogram.reset()
                
                # Periodic cleanup
                if time.time() - last_cleanup > 3600:  # Hourly
                    await asyncio.to_thread(self.db.cleanup_old_data, 24.0)
                    last_cleanup = time.time()
                
                await asyncio.sleep(self.sample_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Telemetry collection error: {e}")
                await asyncio.sleep(1.0)
        
        # Final flush
        await self.flush_metrics()
    
    def stop(self):
        """Stop collector"""
        self.running = False


class TelemetryWriter:
    """High-level telemetry writer interface"""
    
    def __init__(self, db_path: str = "telemetry.db"):
        self.db = TelemetryDatabase(db_path)
        self.collector = TelemetryCollector(self.db)
        self._task: Optional[asyncio.Task] = None
    
    async def start(self):
        """Start telemetry collection"""
        self._task = asyncio.create_task(self.collector.run_collector())
    
    async def stop(self):
        """Stop telemetry collection"""
        self.collector.stop()
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
    
    def record_metric(self, name: str, value: float, tags: Optional[Dict[str, str]] = None):
        """Record a custom metric"""
        sample = MetricSample(
            timestamp=time.time(),
            name=name,
            value=value,
            tags=tags or {},
        )
        asyncio.create_task(self._store_sample(sample))
    
    async def _store_sample(self, sample: MetricSample):
        """Store single sample asynchronously"""
        async with self.collector._buffer_lock:
            self.collector._metrics_buffer.append(sample)
            
            if len(self.collector._metrics_buffer) >= self.collector._flush_threshold:
                await self.collector.flush_metrics()
    
    def record_latency(self, component: str, latency_us: float):
        """Record latency for a component"""
        self.collector.record_latency(component, latency_us)
    
    def record_exchange_stats(
        self,
        exchange: str,
        messages_sent: int,
        messages_received: int,
        bytes_sent: int,
        bytes_received: int,
        latency_p99: float
    ):
        """Record exchange connection statistics"""
        asyncio.create_task(asyncio.to_thread(
            self.db.insert_exchange_stat,
            exchange,
            messages_sent,
            messages_received,
            bytes_sent,
            bytes_received,
            latency_p99
        ))


# Example usage
async def main():
    writer = TelemetryWriter()
    await writer.start()
    
    # Simulate metrics
    for i in range(100):
        writer.record_metric("test_metric", float(i), {"source": "test"})
        writer.record_latency("order_execution", 100.0 + i * 10)
        await asyncio.sleep(0.1)
    
    await writer.stop()


if __name__ == "__main__":
    asyncio.run(main())
