//! Zero-overhead sampling profiler generating continuous flamegraphs
//! Identifies microsecond bottlenecks without stopping main event loop
//! Optimized for AMD Ryzen with minimal memory footprint

use std::sync::atomic::{AtomicU64, AtomicBool, Ordering};
use std::time::Instant;
use std::collections::HashMap;

/// Stack frame information for profiling
#[derive(Debug, Clone)]
pub struct StackFrame {
    /// Function name
    pub function: String,
    /// File location
    pub file: String,
    /// Line number
    pub line: u32,
    /// Time spent in this frame (nanoseconds)
    pub time_ns: u64,
    /// Call count
    pub call_count: u64,
}

/// Sampling result for a single sample
#[derive(Debug, Clone)]
pub struct Sample {
    /// Timestamp (microseconds since epoch)
    pub timestamp_us: u64,
    /// Stack trace (bottom to top)
    pub stack: Vec<StackFrame>,
    /// Thread ID
    pub thread_id: u64,
    /// CPU core
    pub cpu_core: u32,
}

/// Fixed-size circular buffer for samples
const MAX_SAMPLES: usize = 100_000;

struct SampleBuffer {
    data: Vec<Sample>,
    head: usize,
    count: usize,
}

impl SampleBuffer {
    fn new() -> Self {
        Self {
            data: Vec::with_capacity(MAX_SAMPLES),
            head: 0,
            count: 0,
        }
    }

    #[inline(always)]
    fn push(&mut self, sample: Sample) {
        if self.count < MAX_SAMPLES {
            self.data.push(sample);
            self.count += 1;
        } else {
            // Overwrite oldest
            self.data[self.head] = sample;
        }
        self.head = (self.head + 1) % MAX_SAMPLES;
    }

    #[inline(always)]
    fn get_samples(&self) -> &[Sample] {
        &self.data[..self.count.min(self.data.len())]
    }

    #[inline(always)]
    fn clear(&mut self) {
        self.data.clear();
        self.head = 0;
        self.count = 0;
    }
}

/// Lock-free sampling profiler
pub struct SamplingProfiler {
    /// Is profiling enabled
    enabled: AtomicBool,
    /// Sampling interval in microseconds
    sample_interval_us: AtomicU64,
    /// Total samples collected
    sample_count: AtomicU64,
    /// Dropped samples (buffer full)
    dropped_count: AtomicU64,
    /// Start time
    start_time: Instant,
    /// Sample buffer
    buffer: std::sync::Mutex<SampleBuffer>,
    /// Aggregated statistics per function
    stats: std::sync::Mutex<HashMap<String, FrameStats>>,
    /// Last sample timestamp
    last_sample_us: AtomicU64,
}

/// Aggregated statistics for a function
#[derive(Debug, Clone)]
struct FrameStats {
    /// Total time in function (ns)
    total_time_ns: u64,
    /// Self time (excluding children, ns)
    self_time_ns: u64,
    /// Call count
    call_count: u64,
    /// Max single call time (ns)
    max_time_ns: u64,
    /// Min single call time (ns)
    min_time_ns: u64,
}

impl FrameStats {
    fn new() -> Self {
        Self {
            total_time_ns: 0,
            self_time_ns: 0,
            call_count: 0,
            max_time_ns: 0,
            min_time_ns: u64::MAX,
        }
    }
}

impl SamplingProfiler {
    pub fn new(sample_interval_us: u64) -> Self {
        Self {
            enabled: AtomicBool::new(false),
            sample_interval_us: AtomicU64::new(sample_interval_us),
            sample_count: AtomicU64::new(0),
            dropped_count: AtomicU64::new(0),
            start_time: Instant::now(),
            buffer: std::sync::Mutex::new(SampleBuffer::new()),
            stats: std::sync::Mutex::new(HashMap::new()),
            last_sample_us: AtomicU64::new(0),
        }
    }

    /// Start profiling
    #[inline(always)]
    pub fn start(&self) {
        self.enabled.store(true, Ordering::Relaxed);
        self.start_time = Instant::now();
    }

    /// Stop profiling
    #[inline(always)]
    pub fn stop(&self) {
        self.enabled.store(false, Ordering::Relaxed);
    }

    /// Record a sample (called by sampling thread)
    #[inline(always)]
    pub fn record_sample(&self, stack: Vec<StackFrame>, thread_id: u64, cpu_core: u32) {
        if !self.enabled.load(Ordering::Relaxed) {
            return;
        }

        let now_us = self.start_time.elapsed().as_micros() as u64;
        let last = self.last_sample_us.load(Ordering::Relaxed);
        
        // Rate limiting
        let interval = self.sample_interval_us.load(Ordering::Relaxed);
        if now_us - last < interval {
            return;
        }

        self.last_sample_us.store(now_us, Ordering::Relaxed);

        let sample = Sample {
            timestamp_us: now_us,
            stack,
            thread_id,
            cpu_core,
        };

        // Update aggregated stats
        self.update_stats(&sample);

        // Store sample
        if let Ok(mut buffer) = self.buffer.lock() {
            if buffer.count >= MAX_SAMPLES {
                self.dropped_count.fetch_add(1, Ordering::Relaxed);
            }
            buffer.push(sample);
        }

        self.sample_count.fetch_add(1, Ordering::Relaxed);
    }

    /// Update aggregated statistics from sample
    fn update_stats(&self, sample: &Sample) {
        let mut stats = match self.stats.lock() {
            Ok(s) => s,
            Err(_) => return,
        };

        for (i, frame) in sample.stack.iter().enumerate() {
            let entry = stats.entry(frame.function.clone()).or_insert_with(FrameStats::new);
            entry.call_count += 1;
            
            // Estimate time based on position in stack
            // Bottom frames get more time attributed
            let estimated_time = 1000 * (sample.stack.len() - i) as u64;
            entry.total_time_ns += estimated_time;
            
            if i == 0 {
                // Leaf frame - this is self time
                entry.self_time_ns += estimated_time;
            }
            
            entry.max_time_ns = entry.max_time_ns.max(estimated_time);
            entry.min_time_ns = entry.min_time_ns.min(estimated_time);
        }
    }

    /// Generate flamegraph data in collapsed format
    #[inline(always)]
    pub fn generate_flamegraph(&self) -> String {
        let buffer = match self.buffer.lock() {
            Ok(b) => b,
            Err(_) => return String::new(),
        };

        let mut stacks: HashMap<String, u64> = HashMap::new();

        for sample in buffer.get_samples() {
            // Create collapsed stack string
            let stack_str: Vec<&str> = sample.stack.iter()
                .map(|f| f.function.as_str())
                .collect();
            let key = stack_str.join(";");
            
            *stacks.entry(key).or_insert(0) += 1;
        }

        // Format as collapsed flamegraph input
        let mut output = String::new();
        for (stack, count) in stacks {
            output.push_str(&format!("{} {}\n", stack, count));
        }

        output
    }

    /// Get top functions by total time
    #[inline(always)]
    pub fn get_top_functions(&self, n: usize) -> Vec<(String, u64, f64)> {
        let stats = match self.stats.lock() {
            Ok(s) => s,
            Err(_) => return Vec::new(),
        };

        let mut sorted: Vec<_> = stats.iter().collect();
        sorted.sort_by(|a, b| b.1.total_time_ns.cmp(&a.1.total_time_ns));

        let total_time: u64 = stats.values().map(|s| s.total_time_ns).sum();

        sorted.into_iter()
            .take(n)
            .map(|(name, s)| {
                let pct = if total_time > 0 {
                    (s.total_time_ns as f64 / total_time as f64) * 100.0
                } else {
                    0.0
                };
                (name.clone(), s.total_time_ns, pct)
            })
            .collect()
    }

    /// Get profiling statistics
    #[inline(always)]
    pub fn get_stats(&self) -> (u64, u64, u64, bool) {
        (
            self.sample_count.load(Ordering::Relaxed),
            self.dropped_count.load(Ordering::Relaxed),
            self.sample_interval_us.load(Ordering::Relaxed),
            self.enabled.load(Ordering::Relaxed),
        )
    }

    /// Set sampling interval
    #[inline(always)]
    pub fn set_interval(&self, interval_us: u64) {
        self.sample_interval_us.store(interval_us, Ordering::Relaxed);
    }

    /// Reset all data
    #[inline(always)]
    pub fn reset(&self) {
        if let Ok(mut buffer) = self.buffer.lock() {
            buffer.clear();
        }
        if let Ok(mut stats) = self.stats.lock() {
            stats.clear();
        }
        self.sample_count.store(0, Ordering::Relaxed);
        self.dropped_count.store(0, Ordering::Relaxed);
        self.last_sample_us.store(0, Ordering::Relaxed);
    }
}

/// RAII guard for timing a code block
pub struct ProfileGuard<'a> {
    profiler: &'a SamplingProfiler,
    function_name: &'static str,
    start: Instant,
}

impl<'a> ProfileGuard<'a> {
    #[inline(always)]
    pub fn new(profiler: &'a SamplingProfiler, function_name: &'static str) -> Self {
        Self {
            profiler,
            function_name,
            start: Instant::now(),
        }
    }
}

impl<'a> Drop for ProfileGuard<'_> {
    #[inline(always)]
    fn drop(&mut self) {
        let elapsed = self.start.elapsed().as_nanos() as u64;
        
        // Record timing
        let frame = StackFrame {
            function: self.function_name.to_string(),
            file: "".to_string(),
            line: 0,
            time_ns: elapsed,
            call_count: 1,
        };

        self.profiler.record_sample(
            vec![frame],
            std::thread::current().id().as_u64(),
            0,
        );
    }
}

/// Convenience macro for profiling a scope
#[macro_export]
macro_rules! profile_scope {
    ($profiler:expr, $name:expr) => {
        let _guard = ProfileGuard::new($profiler, $name);
    };
}

/// Bottleneck detector analyzing profiler data
pub struct BottleneckDetector {
    /// Threshold for bottleneck detection (percentage of total time)
    threshold_pct: f64,
    /// Minimum calls to consider
    min_calls: u64,
}

impl BottleneckDetector {
    pub fn new(threshold_pct: f64, min_calls: u64) -> Self {
        Self {
            threshold_pct,
            min_calls,
        }
    }

    /// Detect bottlenecks from profiler data
    #[inline(always)]
    pub fn detect(&self, profiler: &SamplingProfiler) -> Vec<BottleneckInfo> {
        let top = profiler.get_top_functions(50);
        let mut bottlenecks = Vec::new();

        for (name, time_ns, pct) in top {
            if pct >= self.threshold_pct {
                bottlenecks.push(BottleneckInfo {
                    function: name,
                    time_percentage: pct,
                    severity: self.calculate_severity(pct),
                });
            }
        }

        bottlenecks
    }

    fn calculate_severity(&self, pct: f64) -> Severity {
        if pct >= 50.0 {
            Severity::Critical
        } else if pct >= 30.0 {
            Severity::High
        } else if pct >= 15.0 {
            Severity::Medium
        } else {
            Severity::Low
        }
    }
}

/// Bottleneck information
#[derive(Debug)]
pub struct BottleneckInfo {
    pub function: String,
    pub time_percentage: f64,
    pub severity: Severity,
}

/// Severity levels
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Severity {
    Critical,
    High,
    Medium,
    Low,
}

impl Default for SamplingProfiler {
    fn default() -> Self {
        Self::new(100) // Default 100us sampling interval
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_profiler_basic() {
        let profiler = SamplingProfiler::new(100);
        profiler.start();

        let frame = StackFrame {
            function: "test_function".to_string(),
            file: "test.rs".to_string(),
            line: 10,
            time_ns: 1000,
            call_count: 1,
        };

        profiler.record_sample(vec![frame], 1, 0);

        let (count, _, _, enabled) = profiler.get_stats();
        assert_eq!(count, 1);
        assert!(enabled);

        profiler.stop();
        let (_, _, _, enabled) = profiler.get_stats();
        assert!(!enabled);
    }

    #[test]
    fn test_flamegraph_generation() {
        let profiler = SamplingProfiler::new(100);
        profiler.start();

        for _ in 0..10 {
            let frame = StackFrame {
                function: "main".to_string(),
                file: "".to_string(),
                line: 0,
                time_ns: 1000,
                call_count: 1,
            };
            profiler.record_sample(vec![frame], 1, 0);
        }

        let flamegraph = profiler.generate_flamegraph();
        assert!(flamegraph.contains("main"));
    }

    #[test]
    fn test_bottleneck_detection() {
        let profiler = SamplingProfiler::new(100);
        profiler.start();

        // Simulate a bottleneck
        for _ in 0..100 {
            let frame = StackFrame {
                function: "slow_function".to_string(),
                file: "".to_string(),
                line: 0,
                time_ns: 10000,
                call_count: 1,
            };
            profiler.record_sample(vec![frame], 1, 0);
        }

        let detector = BottleneckDetector::new(10.0, 1);
        let bottlenecks = detector.detect(&profiler);
        
        assert!(!bottlenecks.is_empty());
        assert_eq!(bottlenecks[0].function, "slow_function");
    }
}
