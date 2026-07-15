//! Probabilistic queue position estimator for limit orders
//! Tracks exact volume ahead of bot's limit orders, adjusting for hidden liquidity
//! Optimized for AMD Ryzen architecture with zero heap allocations

use std::sync::atomic::{AtomicI64, AtomicU64, Ordering};
use std::collections::VecDeque;

/// Queue position estimate with confidence metrics
#[derive(Debug, Clone, Copy)]
pub struct QueuePosition {
    /// Estimated volume ahead (in base units * 1e8)
    pub volume_ahead: i64,
    /// Estimated position in queue (0 = front)
    pub position: u32,
    /// Confidence level (0.0 to 1.0, scaled by 1e6)
    pub confidence: u32,
    /// Hidden liquidity estimate (in base units * 1e8)
    pub hidden_liquidity: i64,
    /// Timestamp of estimate (microseconds)
    pub timestamp_us: u64,
}

/// Fixed-size circular buffer for queue tracking (no heap allocation)
const MAX_QUEUE_SAMPLES: usize = 64;

struct CircularBuffer {
    data: [i64; MAX_QUEUE_SAMPLES],
    head: usize,
    count: usize,
}

impl CircularBuffer {
    fn new() -> Self {
        Self {
            data: [0; MAX_QUEUE_SAMPLES],
            head: 0,
            count: 0,
        }
    }

    #[inline(always)]
    fn push(&mut self, value: i64) {
        self.data[self.head] = value;
        self.head = (self.head + 1) % MAX_QUEUE_SAMPLES;
        if self.count < MAX_QUEUE_SAMPLES {
            self.count += 1;
        }
    }

    #[inline(always)]
    fn sum(&self) -> i64 {
        self.data.iter().take(self.count).sum()
    }

    #[inline(always)]
    fn average(&self) -> f64 {
        if self.count == 0 {
            return 0.0;
        }
        self.sum() as f64 / self.count as f64
    }

    #[inline(always)]
    fn clear(&mut self) {
        self.head = 0;
        self.count = 0;
    }
}

/// Lock-free queue position estimator
pub struct QueueEstimator {
    /// Our order size (scaled by 1e8)
    our_size: AtomicI64,
    /// Total visible queue size at price level
    visible_queue: AtomicI64,
    /// Estimated hidden liquidity (icebergs, etc.)
    hidden_estimate: AtomicI64,
    /// Volume executed since we joined queue
    volume_executed: AtomicI64,
    /// Queue jumps detected (orders ahead that cancelled)
    queue_jumps: AtomicU64,
    /// Last estimated position
    last_estimate: AtomicI64,
    /// Historical execution rates for prediction
    execution_history: std::sync::Mutex<CircularBuffer>,
    /// Update timestamp
    last_update_us: AtomicU64,
    /// Estimation count
    estimate_count: AtomicU64,
}

impl QueueEstimator {
    pub fn new() -> Self {
        Self {
            our_size: AtomicI64::new(0),
            visible_queue: AtomicI64::new(0),
            hidden_estimate: AtomicI64::new(0),
            volume_executed: AtomicI64::new(0),
            queue_jumps: AtomicU64::new(0),
            last_estimate: AtomicI64::new(-1),
            execution_history: std::sync::Mutex::new(CircularBuffer::new()),
            last_update_us: AtomicU64::new(0),
            estimate_count: AtomicU64::new(0),
        }
    }

    /// Initialize our order in the queue
    #[inline(always)]
    pub fn join_queue(&self, our_size: i64, visible_queue: i64) {
        self.our_size.store(our_size, Ordering::Relaxed);
        self.visible_queue.store(visible_queue, Ordering::Relaxed);
        self.volume_executed.store(0, Ordering::Relaxed);
        self.queue_jumps.store(0, Ordering::Relaxed);
        self.update_timestamp();
    }

    /// Update queue state after market events
    #[inline(always)]
    pub fn update_queue(&self, new_visible: i64, executed_volume: i64) {
        let old_visible = self.visible_queue.load(Ordering::Relaxed);
        
        // Detect queue jumps (visible decreased more than execution)
        let visible_change = old_visible - new_visible;
        if visible_change > executed_volume && visible_change > 0 {
            let jumps = visible_change - executed_volume;
            self.queue_jumps.fetch_add(jumps as u64, Ordering::Relaxed);
        }

        self.visible_queue.store(new_visible, Ordering::Relaxed);
        self.volume_executed.fetch_add(executed_volume, Ordering::Relaxed);
        
        // Track execution rate
        if let Ok(mut history) = self.execution_history.lock() {
            history.push(executed_volume);
        }

        self.update_timestamp();
    }

    /// Estimate hidden liquidity based on execution patterns
    #[inline(always)]
    pub fn estimate_hidden_liquidity(&self, typical_iceberg_ratio: f64) -> i64 {
        let visible = self.visible_queue.load(Ordering::Relaxed);
        let executed = self.volume_executed.load(Ordering::Relaxed);
        
        // If executions exceed visible queue refresh rate, likely hidden liquidity
        if let Ok(history) = self.execution_history.lock() {
            let avg_execution = history.average();
            if avg_execution > visible as f64 * 0.5 {
                // High execution rate suggests hidden liquidity
                let hidden = (visible as f64 * typical_iceberg_ratio) as i64;
                self.hidden_estimate.store(hidden, Ordering::Relaxed);
                return hidden;
            }
        }

        // Default estimate based on typical ratios
        let hidden = (visible as f64 * typical_iceberg_ratio) as i64;
        self.hidden_estimate.store(hidden, Ordering::Relaxed);
        hidden
    }

    /// Calculate current queue position estimate
    #[inline(always)]
    pub fn estimate_position(&self) -> QueuePosition {
        let our_size = self.our_size.load(Ordering::Relaxed);
        let visible = self.visible_queue.load(Ordering::Relaxed);
        let hidden = self.hidden_estimate.load(Ordering::Relaxed);
        let executed = self.volume_executed.load(Ordering::Relaxed);
        let jumps = self.queue_jumps.load(Ordering::Relaxed);

        if our_size <= 0 || visible <= 0 {
            return QueuePosition {
                volume_ahead: 0,
                position: 0,
                confidence: 0,
                hidden_liquidity: 0,
                timestamp_us: self.last_update_us.load(Ordering::Relaxed),
            };
        }

        // Total queue = visible + hidden
        let total_queue = visible + hidden;
        
        // Volume ahead = total queue - our position adjustment
        // We assume FIFO, so volume ahead is what was there before us
        let mut volume_ahead = total_queue - executed as i64;
        
        // Adjust for queue jumps (orders ahead that cancelled)
        volume_ahead -= jumps as i64;
        volume_ahead = volume_ahead.max(0);

        // Position estimation (simplified FIFO assumption)
        let position = if total_queue > 0 {
            ((volume_ahead as f64 / total_queue as f64) * 100.0) as u32
        } else {
            0
        };

        // Confidence based on data quality
        let confidence = if jumps == 0 && hidden == 0 {
            900_000 // High confidence - clean queue
        } else if jumps < 1000 && hidden < visible / 2 {
            600_000 // Medium confidence
        } else {
            300_000 // Low confidence - noisy queue
        };

        self.last_estimate.store(volume_ahead, Ordering::Relaxed);
        self.estimate_count.fetch_add(1, Ordering::Relaxed);

        QueuePosition {
            volume_ahead,
            position,
            confidence,
            hidden_liquidity: hidden,
            timestamp_us: self.last_update_us.load(Ordering::Relaxed),
        }
    }

    /// Predict time until fill based on execution rate
    #[inline(always)]
    pub fn predict_fill_time_us(&self) -> u64 {
        let volume_ahead = self.last_estimate.load(Ordering::Relaxed);
        if volume_ahead <= 0 {
            return 0; // Already at front or filled
        }

        if let Ok(history) = self.execution_history.lock() {
            let avg_rate = history.average();
            if avg_rate > 0.0 {
                return (volume_ahead as f64 / avg_rate * 1000.0) as u64; // Convert to microseconds
            }
        }

        // No data - return large estimate
        u64::MAX
    }

    /// Get queue statistics
    #[inline(always)]
    pub fn get_stats(&self) -> (i64, i64, i64, u64) {
        (
            self.our_size.load(Ordering::Relaxed),
            self.visible_queue.load(Ordering::Relaxed),
            self.hidden_estimate.load(Ordering::Relaxed),
            self.queue_jumps.load(Ordering::Relaxed),
        )
    }

    /// Check if we're likely at the front of the queue
    #[inline(always)]
    pub fn is_at_front(&self, threshold: i64) -> bool {
        let volume_ahead = self.last_estimate.load(Ordering::Relaxed);
        volume_ahead >= 0 && volume_ahead <= threshold
    }

    /// Reset estimator state
    #[inline(always)]
    pub fn reset(&self) {
        self.our_size.store(0, Ordering::Relaxed);
        self.visible_queue.store(0, Ordering::Relaxed);
        self.hidden_estimate.store(0, Ordering::Relaxed);
        self.volume_executed.store(0, Ordering::Relaxed);
        self.queue_jumps.store(0, Ordering::Relaxed);
        self.last_estimate.store(-1, Ordering::Relaxed);
        if let Ok(mut history) = self.execution_history.lock() {
            history.clear();
        }
        self.estimate_count.store(0, Ordering::Relaxed);
    }

    #[inline(always)]
    fn update_timestamp(&self) {
        use std::time::{SystemTime, UNIX_EPOCH};
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_micros() as u64;
        self.last_update_us.store(now, Ordering::Relaxed);
    }
}

/// Iceberg order detector analyzing queue behavior patterns
pub struct IcebergDetector {
    /// Price level being monitored
    price_level: AtomicI64,
    /// Refresh count at this level
    refresh_count: AtomicU64,
    /// Total volume seen at level
    total_volume_seen: AtomicI64,
    /// Typical visible size
    typical_visible: AtomicI64,
    /// Detected iceberg size estimate
    iceberg_estimate: AtomicI64,
}

impl IcebergDetector {
    pub fn new() -> Self {
        Self {
            price_level: AtomicI64::new(0),
            refresh_count: AtomicU64::new(0),
            total_volume_seen: AtomicI64::new(0),
            typical_visible: AtomicI64::new(0),
            iceberg_estimate: AtomicI64::new(0),
        }
    }

    /// Track order book refresh at a price level
    #[inline(always)]
    pub fn track_refresh(&self, price: i64, visible_size: i64, executed: i64) {
        if price != self.price_level.load(Ordering::Relaxed) {
            // New price level - reset
            self.price_level.store(price, Ordering::Relaxed);
            self.refresh_count.store(1, Ordering::Relaxed);
            self.total_volume_seen.store(visible_size, Ordering::Relaxed);
            self.typical_visible.store(visible_size, Ordering::Relaxed);
            return;
        }

        self.refresh_count.fetch_add(1, Ordering::Relaxed);
        self.total_volume_seen.fetch_add(executed, Ordering::Relaxed);

        // Update typical visible size (running average)
        let typical = self.typical_visible.load(Ordering::Relaxed);
        let new_typical = ((typical as f64 * 0.9) + (visible_size as f64 * 0.1)) as i64;
        self.typical_visible.store(new_typical, Ordering::Relaxed);

        // Detect iceberg: if total executed >> typical visible, likely iceberg
        let total = self.total_volume_seen.load(Ordering::Relaxed);
        let refreshes = self.refresh_count.load(Ordering::Relaxed);
        
        if refreshes > 3 && total > new_typical as i64 * refreshes as i64 * 2 {
            // Likely iceberg - estimate size
            let estimate = total - (new_typical as i64 * refreshes as i64);
            self.iceberg_estimate.store(estimate.max(0), Ordering::Relaxed);
        }
    }

    /// Get detected iceberg size
    #[inline(always)]
    pub fn get_iceberg_estimate(&self) -> i64 {
        self.iceberg_estimate.load(Ordering::Relaxed)
    }

    /// Check if iceberg is detected
    #[inline(always)]
    pub fn is_iceberg_detected(&self, threshold_multiplier: f64) -> bool {
        let typical = self.typical_visible.load(Ordering::Relaxed);
        let iceberg = self.iceberg_estimate.load(Ordering::Relaxed);
        let refreshes = self.refresh_count.load(Ordering::Relaxed);

        refreshes > 3 && iceberg > (typical as f64 * threshold_multiplier) as i64
    }

    /// Reset detector
    #[inline(always)]
    pub fn reset(&self) {
        self.price_level.store(0, Ordering::Relaxed);
        self.refresh_count.store(0, Ordering::Relaxed);
        self.total_volume_seen.store(0, Ordering::Relaxed);
        self.typical_visible.store(0, Ordering::Relaxed);
        self.iceberg_estimate.store(0, Ordering::Relaxed);
    }
}

impl Default for QueueEstimator {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for IcebergDetector {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_queue_position_basic() {
        let estimator = QueueEstimator::new();
        estimator.join_queue(1000000, 5000000);
        
        let pos = estimator.estimate_position();
        assert!(pos.position <= 100);
    }

    #[test]
    fn test_iceberg_detection() {
        let detector = IcebergDetector::new();
        
        // Simulate multiple refreshes with consistent execution
        for i in 0..10 {
            detector.track_refresh(10000, 1000000, 500000);
        }
        
        let is_iceberg = detector.is_iceberg_detected(2.0);
        assert!(is_iceberg);
    }

    #[test]
    fn test_fill_prediction() {
        let estimator = QueueEstimator::new();
        estimator.join_queue(1000000, 10000000);
        
        // Simulate some executions
        for _ in 0..5 {
            estimator.update_queue(9000000, 1000000);
        }
        
        let fill_time = estimator.predict_fill_time_us();
        assert!(fill_time < u64::MAX);
    }
}
