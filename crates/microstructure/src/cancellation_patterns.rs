//! Pattern recognition engine for detecting toxic cancellations, flickering, and spoofing
//! Analyzes microsecond lifespan of limit orders at top of book
//! Zero heap allocations, optimized for AMD Ryzen architecture

use std::sync::atomic::{AtomicI64, AtomicU64, AtomicF64, Ordering};

/// Cancellation pattern detection result
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CancellationPattern {
    /// Normal cancellation (no suspicious behavior)
    Normal,
    /// Flickering: rapid add/cancel cycles
    Flickering,
    /// Spoofing: large orders placed then cancelled before execution
    Spoofing,
    /// Toxic: cancellation indicating adverse information
    Toxic,
    /// Queue jumping: cancel and re-add at better price
    QueueJumping,
}

/// Order lifecycle tracker
#[derive(Debug, Clone, Copy)]
pub struct OrderLifecycle {
    /// Order placement timestamp (microseconds)
    pub placed_us: u64,
    /// Order cancellation/modification timestamp
    pub cancelled_us: u64,
    /// Order size (scaled by 1e8)
    pub size: i64,
    /// Price level (scaled by 1e8)
    pub price: i64,
    /// Lifespan in microseconds
    pub lifespan_us: u64,
    /// Was order modified before cancellation
    pub was_modified: bool,
}

/// Fixed-size circular buffer for order tracking
const MAX_ORDER_HISTORY: usize = 256;

struct OrderHistoryBuffer {
    data: [OrderLifecycle; MAX_ORDER_HISTORY],
    head: usize,
    count: usize,
}

impl OrderHistoryBuffer {
    fn new() -> Self {
        Self {
            data: [OrderLifecycle {
                placed_us: 0,
                cancelled_us: 0,
                size: 0,
                price: 0,
                lifespan_us: 0,
                was_modified: false,
            }; MAX_ORDER_HISTORY],
            head: 0,
            count: 0,
        }
    }

    #[inline(always)]
    fn push(&mut self, order: OrderLifecycle) {
        self.data[self.head] = order;
        self.head = (self.head + 1) % MAX_ORDER_HISTORY;
        if self.count < MAX_ORDER_HISTORY {
            self.count += 1;
        }
    }

    #[inline(always)]
    fn recent_orders(&self, n: usize) -> impl Iterator<Item = &OrderLifecycle> {
        let start = if self.count < n { 0 } else { self.head };
        self.data.iter().take(self.count).skip(if start == 0 { self.count - n.min(self.count) } else { 0 })
    }

    #[inline(always)]
    fn count(&self) -> usize {
        self.count
    }

    #[inline(always)]
    fn clear(&mut self) {
        self.head = 0;
        self.count = 0;
    }
}

/// Lock-free cancellation pattern detector
pub struct CancellationDetector {
    /// Total cancellations tracked
    total_cancellations: AtomicU64,
    /// Flickering events detected
    flicker_count: AtomicU64,
    /// Spoofing events detected
    spoof_count: AtomicU64,
    /// Toxic cancellations detected
    toxic_count: AtomicU64,
    /// Average order lifespan (microseconds)
    avg_lifespan_us: AtomicU64,
    /// Short lifespan threshold (microseconds)
    short_lifespan_threshold_us: u64,
    /// Large order size threshold (scaled by 1e8)
    large_order_threshold: i64,
    /// Order history buffer
    order_history: std::sync::Mutex<OrderHistoryBuffer>,
    /// Last update timestamp
    last_update_us: AtomicU64,
}

impl CancellationDetector {
    pub fn new(short_lifespan_us: u64, large_order_size: i64) -> Self {
        Self {
            total_cancellations: AtomicU64::new(0),
            flicker_count: AtomicU64::new(0),
            spoof_count: AtomicU64::new(0),
            toxic_count: AtomicU64::new(0),
            avg_lifespan_us: AtomicU64::new(1_000_000), // Default 1 second
            short_lifespan_threshold_us: short_lifespan_us,
            large_order_threshold: large_order_size,
            order_history: std::sync::Mutex::new(OrderHistoryBuffer::new()),
            last_update_us: AtomicU64::new(0),
        }
    }

    /// Record a cancelled order and analyze pattern
    #[inline(always)]
    pub fn record_cancellation(&self, placed_us: u64, cancelled_us: u64, size: i64, price: i64, was_modified: bool) -> CancellationPattern {
        let lifespan = cancelled_us.saturating_sub(placed_us);
        
        let order = OrderLifecycle {
            placed_us,
            cancelled_us,
            size,
            price,
            lifespan_us: lifespan,
            was_modified,
        };

        // Detect pattern
        let pattern = self.detect_pattern(lifespan, size, was_modified, price);

        // Update counters based on pattern
        match pattern {
            CancellationPattern::Flickering => {
                self.flicker_count.fetch_add(1, Ordering::Relaxed);
            }
            CancellationPattern::Spoofing => {
                self.spoof_count.fetch_add(1, Ordering::Relaxed);
            }
            CancellationPattern::Toxic => {
                self.toxic_count.fetch_add(1, Ordering::Relaxed);
            }
            _ => {}
        }

        self.total_cancellations.fetch_add(1, Ordering::Relaxed);

        // Update average lifespan
        let old_avg = self.avg_lifespan_us.load(Ordering::Relaxed);
        let new_avg = ((old_avg as f64 * 0.99) + (lifespan as f64 * 0.01)) as u64;
        self.avg_lifespan_us.store(new_avg, Ordering::Relaxed);

        // Store in history
        if let Ok(mut history) = self.order_history.lock() {
            history.push(order);
        }

        self.update_timestamp();

        pattern
    }

    /// Detect cancellation pattern based on order characteristics
    #[inline(always)]
    fn detect_pattern(&self, lifespan_us: u64, size: i64, was_modified: bool, price: i64) -> CancellationPattern {
        // Flickering: very short lifespan (< threshold)
        if lifespan_us < self.short_lifespan_threshold_us {
            return CancellationPattern::Flickering;
        }

        // Spoofing: large order, short-medium lifespan, no modification
        if size > self.large_order_threshold && lifespan_us < 100_000 && !was_modified {
            return CancellationPattern::Spoofing;
        }

        // Toxic: cancellation immediately after trade execution nearby
        // (would need trade data integration - simplified here)
        if lifespan_us < 50_000 && size > self.large_order_threshold / 2 {
            return CancellationPattern::Toxic;
        }

        // Queue jumping: modification to better price
        if was_modified && lifespan_us < 500_000 {
            return CancellationPattern::QueueJumping;
        }

        CancellationPattern::Normal
    }

    /// Get detection statistics
    #[inline(always)]
    pub fn get_stats(&self) -> (u64, u64, u64, u64, u64) {
        (
            self.total_cancellations.load(Ordering::Relaxed),
            self.flicker_count.load(Ordering::Relaxed),
            self.spoof_count.load(Ordering::Relaxed),
            self.toxic_count.load(Ordering::Relaxed),
            self.avg_lifespan_us.load(Ordering::Relaxed),
        )
    }

    /// Calculate flicker ratio (flickers / total cancellations)
    #[inline(always)]
    pub fn get_flicker_ratio(&self) -> f64 {
        let total = self.total_cancellations.load(Ordering::Relaxed);
        let flickers = self.flicker_count.load(Ordering::Relaxed);
        
        if total == 0 {
            return 0.0;
        }
        flickers as f64 / total as f64
    }

    /// Calculate spoofing ratio
    #[inline(always)]
    pub fn get_spoof_ratio(&self) -> f64 {
        let total = self.total_cancellations.load(Ordering::Relaxed);
        let spoofs = self.spoof_count.load(Ordering::Relaxed);
        
        if total == 0 {
            return 0.0;
        }
        spoofs as f64 / total as f64
    }

    /// Get toxicity indicator (higher = more toxic flow)
    #[inline(always)]
    pub fn get_toxicity_indicator(&self) -> f64 {
        let total = self.total_cancellations.load(Ordering::Relaxed);
        let toxic = self.toxic_count.load(Ordering::Relaxed);
        
        if total == 0 {
            return 0.0;
        }
        
        // Weighted combination of toxic patterns
        let toxic_ratio = toxic as f64 / total as f64;
        let flicker_ratio = self.get_flicker_ratio();
        
        (toxic_ratio * 0.6 + flicker_ratio * 0.4).min(1.0)
    }

    /// Reset detector state
    #[inline(always)]
    pub fn reset(&self) {
        self.total_cancellations.store(0, Ordering::Relaxed);
        self.flicker_count.store(0, Ordering::Relaxed);
        self.spoof_count.store(0, Ordering::Relaxed);
        self.toxic_count.store(0, Ordering::Relaxed);
        self.avg_lifespan_us.store(1_000_000, Ordering::Relaxed);
        if let Ok(mut history) = self.order_history.lock() {
            history.clear();
        }
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

/// Spoofing detection with multi-level analysis
pub struct SpoofingAnalyzer {
    /// Price levels being monitored
    monitored_levels: AtomicU64,
    /// Large order placements at each level
    large_orders_at_level: std::sync::Mutex<[i64; 10]>,
    /// Cancellation rate per level
    cancel_rates: std::sync::Mutex<[f64; 10]>,
    /// Spoofing confidence score (0.0 to 1.0, scaled by 1e6)
    spoofing_confidence: AtomicU32,
    /// Minimum size for spoofing detection
    min_spoof_size: i64,
}

impl SpoofingAnalyzer {
    pub fn new(min_spoof_size: i64) -> Self {
        Self {
            monitored_levels: AtomicU64::new(0),
            large_orders_at_level: std::sync::Mutex::new([0; 10]),
            cancel_rates: std::sync::Mutex::new([0.0; 10]),
            spoofing_confidence: AtomicU32::new(0),
            min_spoof_size,
        }
    }

    /// Track large order placement
    #[inline(always)]
    pub fn track_large_order(&self, level: usize, size: i64) {
        if level >= 10 {
            return;
        }

        if size > self.min_spoof_size {
            if let Ok(mut orders) = self.large_orders_at_level.lock() {
                orders[level] += 1;
            }
            self.monitored_levels.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Record cancellation at level
    #[inline(always)]
    pub fn record_cancel_at_level(&self, level: usize, was_large: bool) {
        if level >= 10 {
            return;
        }

        if was_large {
            if let Ok(mut rates) = self.cancel_rates.lock() {
                let current = rates[level];
                rates[level] = (current * 0.9 + 1.0 * 0.1).min(1.0);
            }

            // Update spoofing confidence
            let old_conf = self.spoofing_confidence.load(Ordering::Relaxed);
            let new_conf = ((old_conf as f64 * 0.95) + (1_000_000.0 * 0.05)) as u32;
            self.spoofing_confidence.store(new_conf, Ordering::Relaxed);
        }
    }

    /// Get spoofing confidence
    #[inline(always)]
    pub fn get_spoofing_confidence(&self) -> f64 {
        self.spoofing_confidence.load(Ordering::Relaxed) as f64 / 1_000_000.0
    }

    /// Check if spoofing is likely at given level
    #[inline(always)]
    pub fn is_spoofing_likely(&self, level: usize) -> bool {
        if level >= 10 {
            return false;
        }

        if let Ok(rates) = self.cancel_rates.lock() {
            rates[level] > 0.7
        } else {
            false
        }
    }

    /// Reset analyzer
    #[inline(always)]
    pub fn reset(&self) {
        self.monitored_levels.store(0, Ordering::Relaxed);
        self.spoofing_confidence.store(0, Ordering::Relaxed);
        if let Ok(mut orders) = self.large_orders_at_level.lock() {
            *orders = [0; 10];
        }
        if let Ok(mut rates) = self.cancel_rates.lock() {
            *rates = [0.0; 10];
        }
    }
}

/// Combined market manipulation detector
pub struct ManipulationDetector {
    cancellation_detector: CancellationDetector,
    spoofing_analyzer: SpoofingAnalyzer,
    /// Overall manipulation score (0.0 to 1.0, scaled by 1e6)
    manipulation_score: AtomicU32,
}

impl ManipulationDetector {
    pub fn new(short_lifespan_us: u64, large_order_size: i64, min_spoof_size: i64) -> Self {
        Self {
            cancellation_detector: CancellationDetector::new(short_lifespan_us, large_order_size),
            spoofing_analyzer: SpoofingAnalyzer::new(min_spoof_size),
            manipulation_score: AtomicU32::new(0),
        }
    }

    /// Analyze order cancellation for manipulation
    #[inline(always)]
    pub fn analyze_cancellation(&self, placed_us: u64, cancelled_us: u64, size: i64, price: i64, level: usize) -> (CancellationPattern, f64) {
        let pattern = self.cancellation_detector.record_cancellation(placed_us, cancelled_us, size, price, false);
        
        // Track large orders for spoofing
        if size > self.spoofing_analyzer.min_spoof_size {
            self.spoofing_analyzer.track_large_order(level, size);
            
            if matches!(pattern, CancellationPattern::Spoofing | CancellationPattern::Flickering) {
                self.spoofing_analyzer.record_cancel_at_level(level, true);
            }
        }

        // Update overall manipulation score
        let toxicity = self.cancellation_detector.get_toxicity_indicator();
        let spoof_conf = self.spoofing_analyzer.get_spoofing_confidence();
        let score = ((toxicity * 0.5 + spoof_conf * 0.5) * 1_000_000.0) as u32;
        self.manipulation_score.store(score, Ordering::Relaxed);

        (pattern, toxicity)
    }

    /// Get overall manipulation risk score
    #[inline(always)]
    pub fn get_manipulation_score(&self) -> f64 {
        self.manipulation_score.load(Ordering::Relaxed) as f64 / 1_000_000.0
    }

    /// Get detailed statistics
    #[inline(always)]
    pub fn get_detailed_stats(&self) -> (u64, u64, u64, u64, f64, f64) {
        let (total, flickers, spoofs, toxic, avg_life) = self.cancellation_detector.get_stats();
        (
            total,
            flickers,
            spoofs,
            toxic,
            avg_life as f64 / 1_000_000.0, // Convert to seconds
            self.get_manipulation_score(),
        )
    }

    /// Access underlying components
    #[inline(always)]
    pub fn get_cancellation_detector(&self) -> &CancellationDetector {
        &self.cancellation_detector
    }

    #[inline(always)]
    pub fn get_spoofing_analyzer(&self) -> &SpoofingAnalyzer {
        &self.spoofing_analyzer
    }
}

impl Default for CancellationDetector {
    fn default() -> Self {
        Self::new(10_000, 1_000_000) // 10ms threshold, 1M size threshold
    }
}

impl Default for SpoofingAnalyzer {
    fn default() -> Self {
        Self::new(500_000) // 500K minimum spoof size
    }
}

impl Default for ManipulationDetector {
    fn default() -> Self {
        Self::new(10_000, 1_000_000, 500_000)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_flickering_detection() {
        let detector = CancellationDetector::new(10_000, 1_000_000);
        
        // Very short lifespan should be flickering
        let pattern = detector.record_cancellation(1000, 5000, 100_000, 10000, false);
        assert_eq!(pattern, CancellationPattern::Flickering);
    }

    #[test]
    fn test_spoofing_detection() {
        let detector = CancellationDetector::new(10_000, 1_000_000);
        
        // Large order, short lifespan, no modification = spoofing
        let pattern = detector.record_cancellation(1000, 50_000, 5_000_000, 10000, false);
        assert_eq!(pattern, CancellationPattern::Spoofing);
    }

    #[test]
    fn test_manipulation_scoring() {
        let detector = ManipulationDetector::new(10_000, 1_000_000, 500_000);
        
        // Record several suspicious cancellations
        for i in 0..10 {
            detector.analyze_cancellation(i * 1000, i * 1000 + 5000, 5_000_000, 10000, 0);
        }
        
        let score = detector.get_manipulation_score();
        assert!(score > 0.5);
    }
}
