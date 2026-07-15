// crates/telemetry/src/tca_streamer.rs
// Streams aggregated TCA metrics to frontend WebSocket
// Implements LTTB (Largest-Triangle-Three-Buckets) downsampling

use std::sync::atomic::{AtomicU64, AtomicBool, Ordering};
use std::time::{Duration, Instant};

/// Aggregated TCA metric for streaming
#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct TcaMetric {
    /// Nanosecond timestamp
    pub timestamp_ns: u64,
    /// Implementation Shortfall in bps
    pub is_bps: f64,
    /// Slippage in bps
    pub slippage_bps: f64,
    /// VWAP performance in bps
    pub vs_vwap_bps: f64,
    /// TWAP performance in bps
    pub vs_twap_bps: f64,
    /// Execution quality score (0-100)
    pub quality_score: f64,
    /// Fill count in period
    pub fill_count: u32,
    /// Total volume
    pub volume: f64,
}

impl TcaMetric {
    #[inline]
    pub const fn new() -> Self {
        Self {
            timestamp_ns: 0,
            is_bps: 0.0,
            slippage_bps: 0.0,
            vs_vwap_bps: 0.0,
            vs_twap_bps: 0.0,
            quality_score: 0.0,
            fill_count: 0,
            volume: 0.0,
        }
    }

    #[inline]
    pub fn aggregate(&mut self, other: &TcaMetric) {
        // Weighted average based on volume
        let total_vol = self.volume + other.volume;
        if total_vol > 0.0 {
            let self_weight = self.volume / total_vol;
            let other_weight = other.volume / total_vol;

            self.is_bps = self.is_bps * self_weight + other.is_bps * other_weight;
            self.slippage_bps = self.slippage_bps * self_weight + other.slippage_bps * other_weight;
            self.vs_vwap_bps = self.vs_vwap_bps * self_weight + other.vs_vwap_bps * other_weight;
            self.vs_twap_bps = self.vs_twap_bps * self_weight + other.vs_twap_bps * other_weight;
            self.quality_score = self.quality_score * self_weight + other.quality_score * other_weight;
        }

        self.fill_count += other.fill_count;
        self.volume = total_vol;
        
        // Use latest timestamp
        if other.timestamp_ns > self.timestamp_ns {
            self.timestamp_ns = other.timestamp_ns;
        }
    }
}

impl Default for TcaMetric {
    fn default() -> Self {
        Self::new()
    }
}

/// LTTB downsampling point
#[derive(Clone, Copy, Debug)]
#[repr(C)]
struct Point {
    x: f64,  // Timestamp (normalized)
    y: f64,  // Value
}

/// LTTB (Largest-Triangle-Three-Buckets) downsampler
/// Reduces data points while preserving visual characteristics
pub struct LttbDownsampler {
    /// Output buffer (pre-allocated)
    output: Vec<Point>,
    /// Input buffer reference
    input: Vec<Point>,
}

impl LttbDownsampler {
    #[inline]
    pub fn new(max_output_points: usize) -> Self {
        Self {
            output: Vec::with_capacity(max_output_points),
            input: Vec::new(),
        }
    }

    /// Downsample a series of TCA metrics
    /// Returns indices of selected points
    #[inline]
    pub fn downsample_metrics(
        &mut self,
        metrics: &[TcaMetric],
        target_points: usize,
        value_selector: fn(&TcaMetric) -> f64,
    ) -> Vec<usize> {
        if metrics.is_empty() || target_points >= metrics.len() {
            return (0..metrics.len()).collect();
        }

        // Convert to points
        self.input.clear();
        for (i, m) in metrics.iter().enumerate() {
            self.input.push(Point {
                x: i as f64,
                y: value_selector(m),
            });
        }

        // Run LTTB
        let indices = self.lttb(target_points);
        indices
    }

    /// Core LTTB algorithm
    fn lttb(&mut self, threshold: usize) -> Vec<usize> {
        self.output.clear();
        let n = self.input.len();

        if threshold >= n {
            return (0..n).collect();
        }

        // Bucket size
        let bucket_size = (n - 2) as f64 / (threshold - 2) as f64;

        // Always include first point
        self.output.push(self.input[0]);
        let mut result_indices = vec![0];

        let mut current_idx = 0;

        for i in 0..threshold - 2 {
            let bucket_start = (1.0 + (i as f64) * bucket_size) as usize;
            let bucket_end = (1.0 + ((i + 1) as f64) * bucket_size) as usize;
            let next_bucket_start = bucket_end;
            let next_bucket_end = ((bucket_end as f64 + bucket_size) as usize).min(n - 1);

            let avg_x = if next_bucket_end > next_bucket_start {
                let mut sum_x = 0.0;
                for j in next_bucket_start..next_bucket_end {
                    sum_x += self.input[j].x;
                }
                sum_x / (next_bucket_end - next_bucket_start) as f64
            } else {
                self.input[next_bucket_start].x
            };
            let avg_y = if next_bucket_end > next_bucket_start {
                let mut sum_y = 0.0;
                for j in next_bucket_start..next_bucket_end {
                    sum_y += self.input[j].y;
                }
                sum_y / (next_bucket_end - next_bucket_start) as f64
            } else {
                self.input[next_bucket_start].y
            };

            // Find point in current bucket that maximizes triangle area
            let mut max_area = -1.0;
            let mut max_idx = bucket_start;

            for j in bucket_start..bucket_end.min(n) {
                // Triangle area with base from previous selected point to average of next bucket
                let area = self.triangle_area(
                    self.input[current_idx],
                    self.input[j],
                    Point { x: avg_x, y: avg_y },
                );

                if area > max_area {
                    max_area = area;
                    max_idx = j;
                }
            }

            self.output.push(self.input[max_idx]);
            result_indices.push(max_idx);
            current_idx = max_idx;
        }

        // Always include last point
        self.output.push(self.input[n - 1]);
        result_indices.push(n - 1);

        result_indices
    }

    #[inline]
    fn triangle_area(&self, a: Point, b: Point, c: Point) -> f64 {
        // Area = 0.5 * |x1(y2 - y3) + x2(y3 - y1) + x3(y1 - y2)|
        0.5 * (a.x * (b.y - c.y) + b.x * (c.y - a.y) + c.x * (a.y - b.y)).abs()
    }

    #[inline]
    pub fn get_downsampled_values(&self) -> &[Point] {
        &self.output
    }
}

/// Streaming aggregator for TCA metrics
pub struct TcaStreamAggregator {
    /// Recent metrics buffer (ring buffer style)
    metrics: Vec<TcaMetric>,
    /// Maximum metrics to retain
    max_metrics: usize,
    /// Current write index
    write_idx: usize,
    /// Downsampler instance
    downsampler: LttbDownsampler,
    /// Last flush time
    last_flush: Instant,
    /// Flush interval
    flush_interval: Duration,
    /// Active flag
    active: AtomicBool,
    /// Total metrics processed
    total_processed: AtomicU64,
}

impl TcaStreamAggregator {
    #[inline]
    pub fn new(max_metrics: usize, flush_interval_ms: u64) -> Self {
        Self {
            metrics: Vec::with_capacity(max_metrics),
            max_metrics,
            write_idx: 0,
            downsampler: LttbDownsampler::new(100), // Default to 100 points for UI
            last_flush: Instant::now(),
            flush_interval: Duration::from_millis(flush_interval_ms),
            active: AtomicBool::new(true),
            total_processed: AtomicU64::new(0),
        }
    }

    /// Add a new metric to the stream
    #[inline]
    pub fn push_metric(&mut self, metric: TcaMetric) {
        if !self.active.load(Ordering::Relaxed) {
            return;
        }

        if self.metrics.len() < self.max_metrics {
            self.metrics.push(metric);
        } else {
            // Overwrite oldest
            self.metrics[self.write_idx] = metric;
            self.write_idx = (self.write_idx + 1) % self.max_metrics;
        }

        self.total_processed.fetch_add(1, Ordering::Relaxed);
    }

    /// Get downsampled metrics for UI rendering
    #[inline]
    pub fn get_downsampled_for_ui(
        &mut self,
        target_points: usize,
        value_selector: fn(&TcaMetric) -> f64,
    ) -> Vec<TcaMetric> {
        if self.metrics.is_empty() {
            return vec![];
        }

        // Get indices of selected points
        let indices = self.downsampler.downsample_metrics(
            &self.metrics,
            target_points,
            value_selector,
        );

        // Return selected metrics
        indices.into_iter().map(|i| self.metrics[i]).collect()
    }

    /// Get recent metrics without downsampling
    #[inline]
    pub fn get_recent(&self, count: usize) -> &[TcaMetric] {
        if count >= self.metrics.len() {
            return &self.metrics;
        }

        // Return most recent 'count' metrics
        let start = self.metrics.len().saturating_sub(count);
        &self.metrics[start..]
    }

    /// Aggregate metrics over a time window
    #[inline]
    pub fn aggregate_window(
        &self,
        window_start_ns: u64,
        window_end_ns: u64,
    ) -> Option<TcaMetric> {
        let mut aggregate = TcaMetric::new();
        let mut found = false;

        for metric in &self.metrics {
            if metric.timestamp_ns >= window_start_ns 
                && metric.timestamp_ns <= window_end_ns 
            {
                if !found {
                    aggregate = *metric;
                    found = true;
                } else {
                    aggregate.aggregate(metric);
                }
            }
        }

        if found { Some(aggregate) } else { None }
    }

    /// Check if flush is needed
    #[inline]
    pub fn should_flush(&self) -> bool {
        self.last_flush.elapsed() >= self.flush_interval
    }

    /// Mark as flushed
    #[inline]
    pub fn mark_flushed(&mut self) {
        self.last_flush = Instant::now();
    }

    /// Get statistics
    #[inline]
    pub fn get_stats(&self) -> StreamStats {
        StreamStats {
            total_metrics: self.metrics.len(),
            total_processed: self.total_processed.load(Ordering::Relaxed),
            is_active: self.active.load(Ordering::Relaxed),
            write_idx: self.write_idx,
        }
    }

    /// Pause/Resume streaming
    #[inline]
    pub fn set_active(&self, active: bool) {
        self.active.store(active, Ordering::Relaxed);
    }
}

/// Stream statistics
#[derive(Clone, Copy, Debug, Default)]
#[repr(C)]
pub struct StreamStats {
    pub total_metrics: usize,
    pub total_processed: u64,
    pub is_active: bool,
    pub write_idx: usize,
}

/// WebSocket message builder for TCA data
pub struct TcaWebSocketBuilder {
    buffer: Vec<u8>,
}

impl TcaWebSocketBuilder {
    #[inline]
    pub fn new() -> Self {
        Self {
            buffer: Vec::with_capacity(4096),
        }
    }

    /// Build JSON-like payload for WebSocket
    #[inline]
    pub fn build_metrics_payload(&mut self, metrics: &[TcaMetric]) -> &[u8] {
        self.buffer.clear();
        self.buffer.push(b'{');
        self.buffer.extend_from_slice(b"\"metrics\":[");

        for (i, m) in metrics.iter().enumerate() {
            if i > 0 {
                self.buffer.push(b',');
            }
            
            // Compact JSON format
            self.buffer.push(b'{');
            self.buffer.extend_from_slice(b"\"ts\":");
            self.buffer.extend_from_slice(m.timestamp_ns.to_string().as_bytes());
            self.buffer.extend_from_slice(b",\"is\":");
            self.buffer.extend_from_slice(format!("{:.4}", m.is_bps).as_bytes());
            self.buffer.extend_from_slice(b",\"slip\":");
            self.buffer.extend_from_slice(format!("{:.4}", m.slippage_bps).as_bytes());
            self.buffer.extend_from_slice(b",\"vwap\":");
            self.buffer.extend_from_slice(format!("{:.4}", m.vs_vwap_bps).as_bytes());
            self.buffer.extend_from_slice(b",\"qual\":");
            self.buffer.extend_from_slice(format!("{:.2}", m.quality_score).as_bytes());
            self.buffer.push(b'}');
        }

        self.buffer.extend_from_slice(b"]}");
        &self.buffer
    }

    /// Build summary payload
    #[inline]
    pub fn build_summary_payload(
        &mut self,
        avg_is: f64,
        avg_slippage: f64,
        avg_quality: f64,
        trade_count: u64,
    ) -> &[u8] {
        self.buffer.clear();
        self.buffer.extend_from_slice(b"{\"summary\":{\"avg_is\":");
        self.buffer.extend_from_slice(format!("{:.4}", avg_is).as_bytes());
        self.buffer.extend_from_slice(b",\"avg_slip\":");
        self.buffer.extend_from_slice(format!("{:.4}", avg_slippage).as_bytes());
        self.buffer.extend_from_slice(b",\"avg_qual\":");
        self.buffer.extend_from_slice(format!("{:.2}", avg_quality).as_bytes());
        self.buffer.extend_from_slice(b",\"trades\":");
        self.buffer.extend_from_slice(trade_count.to_string().as_bytes());
        self.buffer.extend_from_slice(b"}}");
        &self.buffer
    }
}

impl Default for TcaWebSocketBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lttb_downsampling() {
        let mut sampler = LttbDownsampler::new(50);
        
        // Create sine wave pattern
        let mut metrics = Vec::new();
        for i in 0..1000 {
            metrics.push(TcaMetric {
                timestamp_ns: i as u64 * 1000000,
                is_bps: (i as f64 * 0.1).sin() * 10.0,
                ..TcaMetric::new()
            });
        }

        let indices = sampler.downsample_metrics(&metrics, 50, |m| m.is_bps);
        
        // Should have exactly 50 points (including first and last)
        assert_eq!(indices.len(), 50);
        assert_eq!(indices[0], 0);
        assert_eq!(indices[indices.len() - 1], 999);
    }

    #[test]
    fn test_stream_aggregator() {
        let mut aggregator = TcaStreamAggregator::new(1000, 100);

        for i in 0..100 {
            aggregator.push_metric(TcaMetric {
                timestamp_ns: i * 1000000000,
                is_bps: i as f64 * 0.1,
                ..TcaMetric::new()
            });
        }

        let downsampled = aggregator.get_downsampled_for_ui(20, |m| m.is_bps);
        assert!(downsampled.len() <= 20);

        let stats = aggregator.get_stats();
        assert_eq!(stats.total_metrics, 100);
    }
}
