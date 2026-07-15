"""
High-resolution HDR histogram aggregator for latency tracking
Tracks end-to-end latency from WebSocket packet receipt to exchange acknowledgment
Pushes telemetry to frontend dashboard with minimal overhead
Memory-efficient design for <14GB RAM constraint
"""

import time
import threading
import numpy as np
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple
from collections import deque


@dataclass
class LatencyBucket:
    """Single latency bucket in HDR histogram"""
    start_us: int  # Start of bucket (microseconds)
    end_us: int    # End of bucket
    count: int     # Number of samples in bucket
    percentile: float  # Calculated percentile


class HDRHistogram:
    """
    High Dynamic Range Histogram for latency measurement
    Provides accurate percentile calculations with fixed memory footprint
    """
    
    def __init__(self, min_value_us: int = 1, max_value_us: int = 60_000_000,
                 significant_digits: int = 3):
        """
        Initialize HDR histogram
        
        Args:
            min_value_us: Minimum trackable latency in microseconds
            max_value_us: Maximum trackable latency in microseconds  
            significant_digits: Number of significant decimal digits
        """
        self.min_value = min_value_us
        self.max_value = max_value_us
        self.significant_digits = significant_digits
        
        # Calculate number of buckets needed
        self.bucket_count = self._calculate_bucket_count()
        
        # Fixed-size bucket array (no heap growth)
        self.buckets: np.ndarray = np.zeros(self.bucket_count, dtype=np.int64)
        
        # Statistics
        self.total_count: int = 0
        self.sum: int = 0
        self.min: int = max_value_us
        self.max: int = min_value_us
        
        # Thread safety
        self.lock = threading.Lock()
    
    def _calculate_bucket_count(self) -> int:
        """Calculate number of buckets for given range and precision"""
        # HDR histogram uses logarithmic bucket spacing
        # Formula: buckets = log2(max/min) * 2^significant_digits
        import math
        sub_buckets = 2 ** self.significant_digits
        max_power = math.ceil(math.log2(self.max_value / self.min_value))
        return max_power * sub_buckets
    
    def _get_bucket_index(self, value: int) -> int:
        """Get bucket index for a value"""
        if value <= self.min_value:
            return 0
        if value >= self.max_value:
            return self.bucket_count - 1
        
        import math
        # Logarithmic bucket calculation
        ratio = value / self.min_value
        log_ratio = math.log2(ratio)
        sub_bucket = int((log_ratio % 1) * (2 ** self.significant_digits))
        major = int(log_ratio)
        
        index = major * (2 ** self.significant_digits) + sub_bucket
        return min(index, self.bucket_count - 1)
    
    def record(self, value_us: int):
        """Record a latency value"""
        if value_us < 0:
            return
        
        bucket_idx = self._get_bucket_index(value_us)
        
        with self.lock:
            self.buckets[bucket_idx] += 1
            self.total_count += 1
            self.sum += value_us
            self.min = min(self.min, value_us)
            self.max = max(self.max, value_us)
    
    def record_batch(self, values: List[int]):
        """Record multiple values efficiently"""
        with self.lock:
            for value in values:
                if value < 0:
                    continue
                bucket_idx = self._get_bucket_index(value)
                self.buckets[bucket_idx] += 1
                self.total_count += 1
                self.sum += value
                self.min = min(self.min, value)
                self.max = max(self.max, value)
    
    def get_percentile(self, percentile: float) -> int:
        """Get latency at given percentile"""
        if self.total_count == 0:
            return 0
        
        target_count = (percentile / 100.0) * self.total_count
        cumulative = 0
        
        for i, count in enumerate(self.buckets):
            cumulative += count
            if cumulative >= target_count:
                return self._bucket_to_value(i)
        
        return self.max
    
    def _bucket_to_value(self, bucket_idx: int) -> int:
        """Convert bucket index back to approximate value"""
        import math
        sub_buckets = 2 ** self.significant_digits
        major = bucket_idx // sub_buckets
        sub = bucket_idx % sub_buckets
        
        ratio = 2 ** (major + sub / sub_buckets)
        return int(self.min_value * ratio)
    
    def get_statistics(self) -> Dict:
        """Get comprehensive statistics"""
        if self.total_count == 0:
            return {}
        
        return {
            'count': self.total_count,
            'mean': self.sum / self.total_count,
            'min': self.min,
            'max': self.max,
            'p50': self.get_percentile(50),
            'p90': self.get_percentile(90),
            'p95': self.get_percentile(95),
            'p99': self.get_percentile(99),
            'p999': self.get_percentile(99.9),
        }
    
    def get_distribution(self, num_buckets: int = 50) -> List[LatencyBucket]:
        """Get latency distribution for visualization"""
        if self.total_count == 0:
            return []
        
        # Sample buckets evenly
        step = max(1, len(self.buckets) // num_buckets)
        result = []
        cumulative = 0
        
        for i in range(0, len(self.buckets), step):
            count = int(self.buckets[i])
            cumulative += count
            percentile = (cumulative / self.total_count) * 100
            
            result.append(LatencyBucket(
                start_us=self._bucket_to_value(i),
                end_us=self._bucket_to_value(min(i + step, len(self.buckets) - 1)),
                count=count,
                percentile=percentile,
            ))
        
        return result
    
    def reset(self):
        """Reset histogram"""
        with self.lock:
            self.buckets.fill(0)
            self.total_count = 0
            self.sum = 0
            self.min = self.max_value
            self.max = self.min_value


class LatencyTracker:
    """
    Track end-to-end latency for trading operations
    From WebSocket receipt to exchange acknowledgment
    """
    
    def __init__(self, window_size: int = 10000):
        # HDR histogram for full distribution
        self.histogram = HDRHistogram(min_value_us=1, max_value_us=60_000_000)
        
        # Fixed-size window for recent latencies
        self.window_size = window_size
        self.recent_latencies: deque = deque(maxlen=window_size)
        
        # Per-operation histograms
        self.operation_histograms: Dict[str, HDRHistogram] = {}
        
        # Start timestamps for pending operations
        self.pending_ops: Dict[str, int] = {}
        
        # Thread safety
        self.lock = threading.Lock()
    
    def start_operation(self, op_id: str):
        """Mark start of an operation"""
        with self.lock:
            self.pending_ops[op_id] = time.time_ns() // 1000
    
    def end_operation(self, op_id: str, operation_type: str = 'default') -> Optional[int]:
        """Mark end of operation and record latency"""
        with self.lock:
            start_ns = self.pending_ops.pop(op_id, None)
            if start_ns is None:
                return None
            
            latency_us = (time.time_ns() // 1000) - start_ns
            
            # Record in main histogram
            self.histogram.record(latency_us)
            
            # Record in operation-specific histogram
            if operation_type not in self.operation_histograms:
                self.operation_histograms[operation_type] = HDRHistogram()
            self.operation_histograms[operation_type].record(latency_us)
            
            # Record in recent window
            self.recent_latencies.append((latency_us, time.time(), operation_type))
            
            return latency_us
    
    def get_recent_stats(self, window_seconds: float = 60.0) -> Dict:
        """Get statistics for recent latencies"""
        now = time.time()
        cutoff = now - window_seconds
        
        recent = [lat for lat, ts, _ in self.recent_latencies if ts >= cutoff]
        
        if not recent:
            return {}
        
        arr = np.array(recent)
        return {
            'count': len(recent),
            'mean': float(np.mean(arr)),
            'std': float(np.std(arr)),
            'min': float(np.min(arr)),
            'max': float(np.max(arr)),
            'p50': float(np.percentile(arr, 50)),
            'p95': float(np.percentile(arr, 95)),
            'p99': float(np.percentile(arr, 99)),
        }
    
    def get_operation_stats(self, operation_type: str) -> Dict:
        """Get statistics for specific operation type"""
        if operation_type not in self.operation_histograms:
            return {}
        return self.operation_histograms[operation_type].get_statistics()
    
    def get_all_stats(self) -> Dict:
        """Get comprehensive statistics"""
        return {
            'overall': self.histogram.get_statistics(),
            'recent_60s': self.get_recent_stats(60.0),
            'by_operation': {
                op_type: hist.get_statistics() 
                for op_type, hist in self.operation_histograms.items()
            },
        }
    
    def reset(self):
        """Reset all tracking data"""
        self.histogram.reset()
        self.recent_latencies.clear()
        self.operation_histograms.clear()
        self.pending_ops.clear()


class TelemetryPusher:
    """
    Push latency telemetry to frontend dashboard
    Implements rate limiting and batching for efficiency
    """
    
    def __init__(self, tracker: LatencyTracker, push_interval_ms: int = 1000):
        self.tracker = tracker
        self.push_interval_ms = push_interval_ms
        self.last_push_time: float = 0
        self.enabled: bool = True
        
        # Dashboard endpoint (would be configured in production)
        self.dashboard_url: str = ""
        
        # Batch buffer
        self.batch_buffer: List[Dict] = []
        self.max_batch_size: int = 100
    
    def should_push(self) -> bool:
        """Check if it's time to push"""
        now = time.time() * 1000  # milliseconds
        return now - self.last_push_time >= self.push_interval_ms
    
    def prepare_telemetry(self) -> Dict:
        """Prepare telemetry data for dashboard"""
        stats = self.tracker.get_all_stats()
        
        return {
            'timestamp': time.time_ns() // 1000,
            'latency': {
                'overall': stats.get('overall', {}),
                'recent': stats.get('recent_60s', {}),
            },
            'operations': stats.get('by_operation', {}),
            'system': {
                'memory_mb': self._get_memory_usage(),
                'cpu_pct': self._get_cpu_usage(),
            },
        }
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        try:
            import resource
            usage = resource.getrusage(resource.RUSAGE_SELF)
            return usage.ru_maxrss / 1024  # Convert to MB
        except:
            return 0.0
    
    def _get_cpu_usage(self) -> float:
        """Get CPU usage percentage"""
        try:
            import psutil
            return psutil.cpu_percent(interval=0.1)
        except:
            return 0.0
    
    def push(self) -> bool:
        """Push telemetry to dashboard"""
        if not self.enabled or not self.should_push():
            return False
        
        telemetry = self.prepare_telemetry()
        self.batch_buffer.append(telemetry)
        
        # Push when batch is full or interval elapsed
        if len(self.batch_buffer) >= self.max_batch_size:
            self._send_batch()
            self.last_push_time = time.time() * 1000
            return True
        
        return False
    
    def _send_batch(self):
        """Send batch to dashboard"""
        # In production, this would use HTTP/gRPC to send to dashboard
        # For now, just clear the batch
        self.batch_buffer.clear()
    
    def set_enabled(self, enabled: bool):
        """Enable/disable telemetry pushing"""
        self.enabled = enabled


class LatencyAlertManager:
    """
    Monitor latencies and generate alerts for anomalies
    """
    
    def __init__(self, tracker: LatencyTracker):
        self.tracker = tracker
        
        # Alert thresholds (microseconds)
        self.warning_threshold: int = 1000  # 1ms
        self.critical_threshold: int = 10000  # 10ms
        self.severe_threshold: int = 100000  # 100ms
        
        # Alert history (fixed size)
        self.alert_history: deque = deque(maxlen=100)
        
        # Cooldown between similar alerts (seconds)
        self.alert_cooldown: float = 5.0
        self.last_alert_time: Dict[str, float] = {}
    
    def check_and_alert(self) -> List[Dict]:
        """Check for alert conditions and generate alerts"""
        alerts = []
        stats = self.tracker.histogram.get_statistics()
        
        if not stats:
            return alerts
        
        p99 = stats.get('p99', 0)
        max_lat = stats.get('max', 0)
        
        now = time.time()
        
        # Check max latency
        if max_lat >= self.severe_threshold:
            alert = self._create_alert('SEVERE', f'Max latency {max_lat}us exceeds {self.severe_threshold}us')
            if alert:
                alerts.append(alert)
        elif max_lat >= self.critical_threshold:
            alert = self._create_alert('CRITICAL', f'Max latency {max_lat}us exceeds {self.critical_threshold}us')
            if alert:
                alerts.append(alert)
        elif max_lat >= self.warning_threshold:
            alert = self._create_alert('WARNING', f'Max latency {max_lat}us exceeds {self.warning_threshold}us')
            if alert:
                alerts.append(alert)
        
        # Check p99 latency
        if p99 >= self.critical_threshold:
            alert = self._create_alert('CRITICAL', f'P99 latency {p99}us exceeds {self.critical_threshold}us')
            if alert:
                alerts.append(alert)
        
        return alerts
    
    def _create_alert(self, severity: str, message: str) -> Optional[Dict]:
        """Create an alert with cooldown checking"""
        key = f"{severity}:{message}"
        last_time = self.last_alert_time.get(key, 0)
        
        if time.time() - last_time < self.alert_cooldown:
            return None  # Still in cooldown
        
        self.last_alert_time[key] = time.time()
        
        alert = {
            'severity': severity,
            'message': message,
            'timestamp': time.time_ns() // 1000,
        }
        
        self.alert_history.append(alert)
        return alert
    
    def get_recent_alerts(self, count: int = 10) -> List[Dict]:
        """Get recent alerts"""
        return list(self.alert_history)[-count:]


if __name__ == "__main__":
    # Example usage
    tracker = LatencyTracker()
    pusher = TelemetryPusher(tracker)
    alerter = LatencyAlertManager(tracker)
    
    # Simulate some operations
    for i in range(100):
        op_id = f"op_{i}"
        tracker.start_operation(op_id)
        time.sleep(0.0001)  # Simulate work
        tracker.end_operation(op_id, 'trade_execution')
    
    # Get statistics
    stats = tracker.get_all_stats()
    print(f"Overall stats: {stats['overall']}")
    print(f"Recent stats: {stats['recent_60s']}")
    
    # Check for alerts
    alerts = alerter.check_and_alert()
    if alerts:
        print(f"Alerts: {alerts}")
    
    # Push telemetry
    if pusher.push():
        print("Telemetry pushed to dashboard")
