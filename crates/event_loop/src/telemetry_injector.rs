//! Non-blocking Telemetry Injector for Main Event Loop
//! Samples latency and throughput metrics every N ticks without pausing execution.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use log::{debug, info};

/// Metric sample structure (fixed size, no allocations)
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct MetricSample {
    pub tick_id: u64,
    pub timestamp_ns: u64,
    pub latency_ns: u32,      // Event processing latency
    pub queue_depth: u16,     // Current event queue depth
    pub memory_used_mb: u16,  // Current memory usage
    pub flags: u8,            // Status flags
    pub strategy_id: u8,      // Active strategy ID
}

impl MetricSample {
    pub const SIZE: usize = std::mem::size_of::<Self>();
}

/// Ring buffer for metric samples (lock-free, single producer/single consumer)
pub struct MetricRingBuffer {
    buffer: Box<[MetricSample]>,
    head: Arc<AtomicU64>,
    tail: Arc<AtomicU64>,
    capacity: usize,
}

impl MetricRingBuffer {
    pub fn new(capacity: usize) -> Self {
        let mut buffer = vec![MetricSample::default(); capacity];
        buffer.shrink_to_fit();
        
        Self {
            buffer: buffer.into_boxed_slice(),
            head: Arc::new(AtomicU64::new(0)),
            tail: Arc::new(AtomicU64::new(0)),
            capacity,
        }
    }

    /// Push a sample (non-blocking, drops oldest if full)
    pub fn push(&self, sample: MetricSample) -> bool {
        let head = self.head.load(Ordering::Relaxed);
        let next_head = (head + 1) % self.capacity as u64;

        // Check if buffer is full
        let tail = self.tail.load(Ordering::Relaxed);
        if next_head == tail {
            // Buffer full, advance tail to drop oldest
            self.tail.store((tail + 1) % self.capacity as u64, Ordering::Relaxed);
        }

        self.buffer[head as usize] = sample;
        self.head.store(next_head, Ordering::Release);
        true
    }

    /// Read all available samples (consumer side)
    pub fn read_samples<F>(&self, mut callback: F) -> usize
    where
        F: FnMut(&MetricSample),
    {
        let head = self.head.load(Ordering::Acquire);
        let mut tail = self.tail.load(Ordering::Relaxed);
        let mut count = 0;

        while tail != head {
            callback(&self.buffer[tail as usize]);
            tail = (tail + 1) % self.capacity as u64;
            count += 1;
        }

        self.tail.store(tail, Ordering::Release);
        count
    }

    /// Get current fill level
    pub fn fill_level(&self) -> usize {
        let head = self.head.load(Ordering::Acquire);
        let tail = self.tail.load(Ordering::Relaxed);
        if head >= tail {
            (head - tail) as usize
        } else {
            self.capacity - (tail - head) as usize
        }
    }
}

/// Aggregated statistics computed from samples
#[derive(Debug, Clone, Default)]
pub struct TelemetryStats {
    pub sample_count: u64,
    pub avg_latency_ns: f64,
    pub min_latency_ns: u32,
    pub max_latency_ns: u32,
    pub p50_latency_ns: u32,
    pub p95_latency_ns: u32,
    pub p99_latency_ns: u32,
    pub events_per_second: f64,
    pub bytes_per_second: f64,
    pub last_update_ns: u64,
}

/// Telemetry injector configuration
#[derive(Debug, Clone)]
pub struct TelemetryConfig {
    /// Sample every N ticks
    pub sample_interval_ticks: u64,
    /// Ring buffer capacity
    pub buffer_capacity: usize,
    /// Stats computation interval in milliseconds
    pub stats_interval_ms: u64,
    /// Enable memory tracking
    pub enable_memory_tracking: bool,
}

impl Default for TelemetryConfig {
    fn default() -> Self {
        Self {
            sample_interval_ticks: 1000,
            buffer_capacity: 65536,
            stats_interval_ms: 1000,
            enable_memory_tracking: true,
        }
    }
}

/// Main telemetry injector hooked into the event loop
pub struct TelemetryInjector {
    config: TelemetryConfig,
    ring_buffer: Arc<MetricRingBuffer>,
    tick_counter: Arc<AtomicU64>,
    total_latency_ns: Arc<AtomicU64>,
    last_sample_tick: Arc<AtomicU64>,
    is_running: Arc<AtomicBool>,
    stats: Arc<parking_lot::RwLock<TelemetryStats>>,
    start_time_ns: Arc<AtomicU64>,
}

impl TelemetryInjector {
    pub fn new(config: TelemetryConfig) -> Self {
        let ring_buffer = Arc::new(MetricRingBuffer::new(config.buffer_capacity));
        
        Self {
            config,
            ring_buffer,
            tick_counter: Arc::new(AtomicU64::new(0)),
            total_latency_ns: Arc::new(AtomicU64::new(0)),
            last_sample_tick: Arc::new(AtomicU64::new(0)),
            is_running: Arc::new(AtomicBool::new(false)),
            stats: Arc::new(parking_lot::RwLock::new(TelemetryStats::default())),
            start_time_ns: Arc::new(AtomicU64::new(0)),
        }
    }

    /// Record a tick (called from main loop, zero-allocation path)
    #[inline(always)]
    pub fn record_tick(&self, latency_ns: u32, queue_depth: u16, strategy_id: u8) {
        let tick = self.tick_counter.fetch_add(1, Ordering::Relaxed);
        self.total_latency_ns.fetch_add(latency_ns as u64, Ordering::Relaxed);

        // Sample periodically
        if tick % self.config.sample_interval_ticks == 0 {
            let timestamp_ns = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos() as u64;

            let memory_mb = if self.config.enable_memory_tracking {
                // Fast memory estimate (platform-specific)
                self.get_memory_usage_mb() as u16
            } else {
                0
            };

            let sample = MetricSample {
                tick_id: tick,
                timestamp_ns,
                latency_ns,
                queue_depth,
                memory_used_mb: memory_mb,
                flags: 0,
                strategy_id,
            };

            self.ring_buffer.push(sample);
            self.last_sample_tick.store(tick, Ordering::Relaxed);
        }
    }

    /// Get current memory usage in MB (fast approximation)
    fn get_memory_usage_mb(&self) -> u64 {
        #[cfg(target_os = "linux")]
        {
            use std::fs;
            if let Ok(status) = fs::read_to_string("/proc/self/status") {
                for line in status.lines() {
                    if line.starts_with("VmRSS:") {
                        let parts: Vec<&str> = line.split_whitespace().collect();
                        if parts.len() >= 2 {
                            if let Ok(kb) = parts[1].parse::<u64>() {
                                return kb / 1024;
                            }
                        }
                    }
                }
            }
        }
        0
    }

    /// Start background stats computation thread
    pub fn start_stats_thread(&self) {
        self.is_running.store(true, Ordering::SeqCst);
        self.start_time_ns.store(
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos() as u64,
            Ordering::Relaxed,
        );

        let ring_buffer = self.ring_buffer.clone();
        let stats = self.stats.clone();
        let is_running = self.is_running.clone();
        let total_latency = self.total_latency_ns.clone();
        let tick_counter = self.tick_counter.clone();
        let start_time = self.start_time_ns.clone();

        std::thread::spawn(move || {
            let interval = Duration::from_millis(100);
            let mut latencies: Vec<u32> = Vec::with_capacity(10000);

            while is_running.load(Ordering::SeqCst) {
                latencies.clear();

                ring_buffer.read_samples(|sample| {
                    latencies.push(sample.latency_ns);
                });

                if !latencies.is_empty() {
                    latencies.sort_unstable();

                    let sum: u64 = latencies.iter().map(|&x| x as u64).sum();
                    let count = latencies.len() as u64;
                    let avg = sum as f64 / count as f64;

                    let elapsed_ns = std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .unwrap()
                        .as_nanos() as u64;
                    let duration_sec = (elapsed_ns - start_time.load(Ordering::Relaxed)) as f64 / 1e9;
                    let eps = if duration_sec > 0.0 {
                        tick_counter.load(Ordering::Relaxed) as f64 / duration_sec
                    } else {
                        0.0
                    };

                    let p50_idx = latencies.len() / 2;
                    let p95_idx = (latencies.len() * 95) / 100;
                    let p99_idx = (latencies.len() * 99) / 100;

                    let new_stats = TelemetryStats {
                        sample_count: count,
                        avg_latency_ns: avg,
                        min_latency_ns: *latencies.first().unwrap_or(&0),
                        max_latency_ns: *latencies.last().unwrap_or(&0),
                        p50_latency_ns: latencies.get(p50_idx).copied().unwrap_or(0),
                        p95_latency_ns: latencies.get(p95_idx).copied().unwrap_or(0),
                        p99_latency_ns: latencies.get(p99_idx).copied().unwrap_or(0),
                        events_per_second: eps,
                        bytes_per_second: 0.0,
                        last_update_ns: elapsed_ns,
                    };

                    *stats.write() = new_stats;

                    debug!(
                        "Telemetry - Avg: {:.2}ns, P95: {}ns, P99: {}ns, EPS: {:.0}",
                        avg,
                        new_stats.p95_latency_ns,
                        new_stats.p99_latency_ns,
                        eps
                    );
                }

                std::thread::sleep(interval);
            }
        });
    }

    /// Get current statistics
    pub fn get_stats(&self) -> TelemetryStats {
        self.stats.read().clone()
    }

    /// Get ring buffer for direct access
    pub fn get_ring_buffer(&self) -> Arc<MetricRingBuffer> {
        self.ring_buffer.clone()
    }

    /// Stop the telemetry injector
    pub fn stop(&self) {
        self.is_running.store(false, Ordering::SeqCst);
        info!("Telemetry injector stopped");
    }
}

impl Default for TelemetryInjector {
    fn default() -> Self {
        Self::new(TelemetryConfig::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metric_ring_buffer() {
        let buffer = MetricRingBuffer::new(1024);
        
        for i in 0..2000 {
            let sample = MetricSample {
                tick_id: i,
                timestamp_ns: i * 1000,
                latency_ns: 100,
                queue_depth: 10,
                memory_used_mb: 1000,
                flags: 0,
                strategy_id: 1,
            };
            buffer.push(sample);
        }

        assert!(buffer.fill_level() > 0);
        assert!(buffer.fill_level() <= 1024);
    }

    #[test]
    fn test_telemetry_injector() {
        let config = TelemetryConfig {
            sample_interval_ticks: 10,
            buffer_capacity: 4096,
            stats_interval_ms: 100,
            enable_memory_tracking: false,
        };

        let injector = TelemetryInjector::new(config);
        injector.start_stats_thread();

        // Simulate ticks
        for i in 0..1000 {
            injector.record_tick(100 + (i % 50) as u32, 10, 1);
        }

        std::thread::sleep(Duration::from_millis(200));

        let stats = injector.get_stats();
        assert!(stats.sample_count > 0);
        assert!(stats.avg_latency_ns > 0.0);

        injector.stop();
    }
}
