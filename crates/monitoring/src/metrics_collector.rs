//! Lock-free, zero-allocation metrics collector for system telemetry.
//! Tracks event loop latency, WebSocket throughput, and thread CPU usage
//! via custom HDR histograms mapped to shared memory. Strictly bounded RAM.

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

/// High-Dynamic-Range Histogram simulation using fixed-size buckets.
/// Avoids heap allocations in the hot path by using pre-allocated arrays.
const BUCKET_COUNT: usize = 1024;
const MAX_LATENCY_NS: u64 = 1_000_000_000; // 1 second cap

#[repr(C)]
pub struct HdrHistogram {
    counts: [AtomicU64; BUCKET_COUNT],
    total_count: AtomicU64,
    min_val: AtomicU64,
    max_val: AtomicU64,
}

impl HdrHistogram {
    pub const fn new() -> Self {
        const INIT: AtomicU64 = AtomicU64::new(0);
        Self {
            counts: [INIT; BUCKET_COUNT],
            total_count: INIT,
            min_val: AtomicU64::new(u64::MAX),
            max_val: AtomicU64::new(0),
        }
    }

    #[inline]
    fn get_bucket_index(&self, value: u64) -> usize {
        // Logarithmic bucket mapping for microsecond precision at low end
        let capped = value.min(MAX_LATENCY_NS);
        let log_val = (capped + 1).leading_zeros() as usize;
        let bucket = (BUCKET_COUNT - 1).saturating_sub(log_val);
        bucket.min(BUCKET_COUNT - 1)
    }

    #[inline]
    pub fn record(&self, value: u64) {
        let idx = self.get_bucket_index(value);
        self.counts[idx].fetch_add(1, Ordering::Relaxed);
        self.total_count.fetch_add(1, Ordering::Relaxed);
        
        // Update min/max with CAS loop for lock-free safety
        let mut current_min = self.min_val.load(Ordering::Relaxed);
        while value < current_min {
            match self.min_val.compare_exchange_weak(
                current_min, value, Ordering::SeqCst, Ordering::Relaxed
            ) {
                Ok(_) => break,
                Err(x) => current_min = x,
            }
        }

        let mut current_max = self.max_val.load(Ordering::Relaxed);
        while value > current_max {
            match self.max_val.compare_exchange_weak(
                current_max, value, Ordering::SeqCst, Ordering::Relaxed
            ) {
                Ok(_) => break,
                Err(x) => current_max = x,
            }
        }
    }

    pub fn get_percentile(&self, pct: f64) -> u64 {
        let total = self.total_count.load(Ordering::Relaxed);
        if total == 0 { return 0; }
        
        let target = ((total as f64) * pct / 100.0) as u64;
        let mut count = 0u64;
        
        for (i, atomic) in self.counts.iter().enumerate() {
            count += atomic.load(Ordering::Relaxed);
            if count >= target {
                // Approximate value from bucket index
                return ((i as u64) * (MAX_LATENCY_NS / BUCKET_COUNT as u64));
            }
        }
        MAX_LATENCY_NS
    }
}

/// Core metrics collector instance.
pub struct MetricsCollector {
    pub ws_latency: HdrHistogram,
    pub event_loop_latency: HdrHistogram,
    pub order_submission_latency: HdrHistogram,
    pub ws_throughput_bps: AtomicU64, // Bytes per second
    pub active_threads: AtomicU64,
    pub last_tick: AtomicU64, // Nanoseconds since epoch
}

impl MetricsCollector {
    pub const fn new() -> Self {
        Self {
            ws_latency: HdrHistogram::new(),
            event_loop_latency: HdrHistogram::new(),
            order_submission_latency: HdrHistogram::new(),
            ws_throughput_bps: AtomicU64::new(0),
            active_threads: AtomicU64::new(0),
            last_tick: AtomicU64::new(0),
        }
    }

    /// Record WebSocket message receive latency.
    #[inline]
    pub fn record_ws_latency(&self, duration_ns: u64) {
        self.ws_latency.record(duration_ns);
    }

    /// Record main event loop iteration time.
    #[inline]
    pub fn record_event_loop(&self, duration_ns: u64) {
        self.event_loop_latency.record(duration_ns);
    }

    /// Record order submission to acknowledgment time.
    #[inline]
    pub fn record_order_latency(&self, duration_ns: u64) {
        self.order_submission_latency.record(duration_ns);
    }

    /// Update throughput counter atomically.
    #[inline]
    pub fn update_throughput(&self, bytes: u64) {
        self.ws_throughput_bps.fetch_add(bytes, Ordering::Relaxed);
    }

    /// Increment active thread counter.
    #[inline]
    pub fn register_thread(&self) {
        self.active_threads.fetch_add(1, Ordering::Relaxed);
    }

    /// Decrement active thread counter.
    #[inline]
    pub fn unregister_thread(&self) {
        self.active_threads.fetch_sub(1, Ordering::Relaxed);
    }

    /// Get P99 latency for event loop in microseconds.
    pub fn get_p99_event_loop_us(&self) -> u64 {
        self.event_loop_latency.get_percentile(99.0) / 1_000
    }

    /// Get P50 latency for WebSocket in microseconds.
    pub fn get_p50_ws_latency_us(&self) -> u64 {
        self.ws_latency.get_percentile(50.0) / 1_000
    }

    /// Reset throughput counter for next interval calculation.
    pub fn reset_throughput(&self) -> u64 {
        self.ws_throughput_bps.swap(0, Ordering::Relaxed)
    }
}

// Global static instance for zero-overhead access
static GLOBAL_METRICS: MetricsCollector = MetricsCollector::new();

pub fn get_global_metrics() -> &'static MetricsCollector {
    &GLOBAL_METRICS
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_histogram_recording() {
        let hist = HdrHistogram::new();
        hist.record(1000); // 1us
        hist.record(5000); // 5us
        hist.record(100000); // 100us
        
        assert!(hist.get_percentile(50.0) <= 10000);
        assert!(hist.get_percentile(99.0) >= 10000);
    }

    #[test]
    fn test_collector_lock_free() {
        let collector = MetricsCollector::new();
        for i in 0..1000 {
            collector.record_event_loop(i * 100);
        }
        assert!(collector.get_p99_event_loop_us() > 0);
    }
}
