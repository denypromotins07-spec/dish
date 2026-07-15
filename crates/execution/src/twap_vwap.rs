//! Microsecond TWAP and VWAP execution logic.
//! Slices large parent orders into micro-child orders based on time intervals and historical volume profiles.

use std::sync::atomic::{AtomicF64, AtomicU64, AtomicBool, Ordering};
use std::time::{Instant, SystemTime, UNIX_EPOCH, Duration};

/// TWAP (Time-Weighted Average Price) execution engine
pub struct TwapExecutor {
    /// Total quantity to execute
    total_quantity: AtomicF64,
    /// Quantity already executed
    executed_quantity: AtomicF64,
    /// Start timestamp (nanoseconds)
    start_time_ns: AtomicU64,
    /// End timestamp (nanoseconds)
    end_time_ns: AtomicU64,
    /// Number of slices
    num_slices: u32,
    /// Quantity per slice
    slice_quantity: f64,
    /// Interval between slices (milliseconds)
    slice_interval_ms: u64,
    /// Last execution timestamp
    last_execution_ts: AtomicU64,
    /// Active flag
    is_active: AtomicBool,
}

/// VWAP (Volume-Weighted Average Price) execution engine with volume profile
pub struct VwapExecutor {
    /// Total quantity to execute
    total_quantity: AtomicF64,
    /// Quantity already executed
    executed_quantity: AtomicF64,
    /// Start timestamp
    start_time_ns: AtomicU64,
    /// End timestamp
    end_time_ns: AtomicU64,
    /// Historical volume profile by time bucket (e.g., 5-minute buckets)
    volume_profile: Vec<f64>,
    /// Current bucket index
    current_bucket: AtomicU64,
    /// Target quantity for current bucket
    bucket_target: AtomicF64,
    /// Number of buckets
    num_buckets: usize,
    /// Active flag
    is_active: AtomicBool,
}

/// Execution result for a single slice
#[derive(Debug, Clone)]
pub struct SliceExecution {
    pub slice_number: u32,
    pub quantity: f64,
    pub price: f64,
    pub timestamp_ns: u64,
    pub filled: bool,
    pub fill_price: Option<f64>,
    pub fill_quantity: Option<f64>,
}

/// Parent order status
#[derive(Debug, Clone)]
pub struct ParentOrderStatus {
    pub total_quantity: f64,
    pub executed_quantity: f64,
    pub remaining_quantity: f64,
    pub average_price: f64,
    pub progress_pct: f64,
    pub estimated_completion_ns: u64,
    pub slices_completed: u32,
    pub total_slices: u32,
}

impl TwapExecutor {
    /// Create new TWAP executor
    /// 
    /// # Arguments
    /// * `total_quantity` - Total quantity to execute
    /// * `duration_seconds` - Total execution duration
    /// * `num_slices` - Number of child orders to split into
    pub fn new(total_quantity: f64, duration_seconds: u64, num_slices: u32) -> Self {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64;
        
        let duration_ns = duration_seconds * 1_000_000_000;
        let slice_interval_ms = (duration_seconds * 1000) / num_slices as u64;
        
        Self {
            total_quantity: AtomicF64::new(total_quantity),
            executed_quantity: AtomicF64::new(0.0),
            start_time_ns: AtomicU64::new(now),
            end_time_ns: AtomicU64::new(now + duration_ns),
            num_slices,
            slice_quantity: total_quantity / num_slices as f64,
            slice_interval_ms,
            last_execution_ts: AtomicU64::new(0),
            is_active: AtomicBool::new(false),
        }
    }

    /// Start TWAP execution
    #[inline(always)]
    pub fn start(&self) {
        self.is_active.store(true, Ordering::Relaxed);
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64;
        self.start_time_ns.store(now, Ordering::Relaxed);
    }

    /// Stop TWAP execution
    #[inline(always)]
    pub fn stop(&self) {
        self.is_active.store(false, Ordering::Relaxed);
    }

    /// Check if next slice should be executed
    #[inline(always)]
    pub fn should_execute_slice(&self) -> bool {
        if !self.is_active.load(Ordering::Relaxed) {
            return false;
        }
        
        let executed = self.executed_quantity.load(Ordering::Relaxed);
        let total = self.total_quantity.load(Ordering::Relaxed);
        
        if executed >= total {
            return false;
        }
        
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64;
        
        let last_exec = self.last_execution_ts.load(Ordering::Relaxed);
        let elapsed_ms = (now - last_exec) / 1_000_000;
        
        elapsed_ms >= self.slice_interval_ms
    }

    /// Get next slice quantity to execute
    #[inline(always)]
    pub fn get_next_slice_quantity(&self) -> f64 {
        let executed = self.executed_quantity.load(Ordering::Relaxed);
        let total = self.total_quantity.load(Ordering::Relaxed);
        let remaining = total - executed;
        
        // Return minimum of slice quantity and remaining
        self.slice_quantity.min(remaining).max(0.0)
    }

    /// Record slice execution
    #[inline(always)]
    pub fn record_slice_execution(&self, quantity: f64) {
        self.executed_quantity.fetch_add(quantity, Ordering::Relaxed);
        
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64;
        self.last_execution_ts.store(now, Ordering::Relaxed);
    }

    /// Get current execution status
    pub fn get_status(&self) -> ParentOrderStatus {
        let total = self.total_quantity.load(Ordering::Relaxed);
        let executed = self.executed_quantity.load(Ordering::Relaxed);
        let remaining = total - executed;
        let progress = if total > 0.0 { executed / total } else { 0.0 };
        
        let slices_completed = (executed / self.slice_quantity) as u32;
        
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64;
        
        let estimated_completion = if progress > 0.0 && progress < 1.0 {
            let elapsed = now - self.start_time_ns.load(Ordering::Relaxed);
            (elapsed as f64 / progress) as u64
        } else {
            self.end_time_ns.load(Ordering::Relaxed)
        };
        
        ParentOrderStatus {
            total_quantity: total,
            executed_quantity: executed,
            remaining_quantity: remaining,
            average_price: 0.0, // Would track separately
            progress_pct: progress * 100.0,
            estimated_completion_ns: estimated_completion,
            slices_completed,
            total_slices: self.num_slices,
        }
    }

    /// Update total quantity (dynamic adjustment)
    #[inline(always)]
    pub fn update_total_quantity(&self, new_total: f64) {
        self.total_quantity.store(new_total, Ordering::Relaxed);
        self.slice_quantity = new_total / self.num_slices as f64;
    }
}

impl VwapExecutor {
    /// Create new VWAP executor with volume profile
    /// 
    /// # Arguments
    /// * `total_quantity` - Total quantity to execute
    /// * `duration_seconds` - Total execution duration
    /// * `volume_profile` - Historical volume distribution (should sum to 1.0)
    pub fn new(total_quantity: f64, duration_seconds: u64, volume_profile: Vec<f64>) -> Self {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64;
        
        let duration_ns = duration_seconds * 1_000_000_000;
        let num_buckets = volume_profile.len();
        
        // Normalize volume profile
        let sum: f64 = volume_profile.iter().sum();
        let normalized_profile: Vec<f64> = volume_profile.iter().map(|&v| v / sum).collect();
        
        Self {
            total_quantity: AtomicF64::new(total_quantity),
            executed_quantity: AtomicF64::new(0.0),
            start_time_ns: AtomicU64::new(now),
            end_time_ns: AtomicU64::new(now + duration_ns),
            volume_profile: normalized_profile,
            current_bucket: AtomicU64::new(0),
            bucket_target: AtomicF64::new(0.0),
            num_buckets,
            is_active: AtomicBool::new(false),
        }
    }

    /// Start VWAP execution
    #[inline(always)]
    pub fn start(&self) {
        self.is_active.store(true, Ordering::Relaxed);
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64;
        self.start_time_ns.store(now, Ordering::Relaxed);
        
        // Initialize first bucket target
        if !self.volume_profile.is_empty() {
            let total = self.total_quantity.load(Ordering::Relaxed);
            self.bucket_target.store(total * self.volume_profile[0], Ordering::Relaxed);
        }
    }

    /// Stop VWAP execution
    #[inline(always)]
    pub fn stop(&self) {
        self.is_active.store(false, Ordering::Relaxed);
    }

    /// Get current bucket's target quantity
    #[inline(always)]
    pub fn get_bucket_target(&self) -> f64 {
        self.bucket_target.load(Ordering::Relaxed)
    }

    /// Check if should execute based on volume profile participation
    pub fn should_execute_slice(&self, current_volume: f64, market_volume: f64) -> bool {
        if !self.is_active.load(Ordering::Relaxed) {
            return false;
        }
        
        let executed = self.executed_quantity.load(Ordering::Relaxed);
        let total = self.total_quantity.load(Ordering::Relaxed);
        
        if executed >= total {
            return false;
        }
        
        // Calculate participation rate
        if market_volume <= 0.0 {
            return false;
        }
        
        let target_participation = self.get_current_target_participation();
        let actual_participation = executed / total;
        
        // Execute if we're behind target participation
        actual_participation < target_participation
    }

    /// Get target participation rate for current time bucket
    fn get_current_target_participation(&self) -> f64 {
        let bucket = self.current_bucket.load(Ordering::Relaxed) as usize;
        if bucket >= self.volume_profile.len() {
            return 1.0;
        }
        
        // Sum of profile up to current bucket
        self.volume_profile[..=bucket].iter().sum()
    }

    /// Get next slice quantity based on volume participation
    #[inline(always)]
    pub fn get_next_slice_quantity(&self, market_volume: f64, participation_rate: f64) -> f64 {
        let executed = self.executed_quantity.load(Ordering::Relaxed);
        let total = self.total_quantity.load(Ordering::Relaxed);
        let remaining = total - executed;
        
        // Target quantity based on market volume and participation
        let target_qty = market_volume * participation_rate;
        
        target_qty.min(remaining).max(0.0)
    }

    /// Record slice execution
    #[inline(always)]
    pub fn record_slice_execution(&self, quantity: f64) {
        self.executed_quantity.fetch_add(quantity, Ordering::Relaxed);
    }

    /// Advance to next volume bucket
    #[inline(always)]
    pub fn advance_bucket(&self) {
        let current = self.current_bucket.load(Ordering::Relaxed);
        if (current as usize) < self.num_buckets - 1 {
            let next = current + 1;
            self.current_bucket.store(next, Ordering::Relaxed);
            
            // Update bucket target
            let total = self.total_quantity.load(Ordering::Relaxed);
            let executed = self.executed_quantity.load(Ordering::Relaxed);
            let remaining = total - executed;
            
            if (next as usize) < self.volume_profile.len() {
                let bucket_allocation = remaining * self.volume_profile[next as usize];
                self.bucket_target.store(bucket_allocation, Ordering::Relaxed);
            }
        }
    }

    /// Get current execution status
    pub fn get_status(&self) -> ParentOrderStatus {
        let total = self.total_quantity.load(Ordering::Relaxed);
        let executed = self.executed_quantity.load(Ordering::Relaxed);
        let remaining = total - executed;
        let progress = if total > 0.0 { executed / total } else { 0.0 };
        
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64;
        
        let estimated_completion = if progress > 0.0 && progress < 1.0 {
            let elapsed = now - self.start_time_ns.load(Ordering::Relaxed);
            (elapsed as f64 / progress) as u64
        } else {
            self.end_time_ns.load(Ordering::Relaxed)
        };
        
        ParentOrderStatus {
            total_quantity: total,
            executed_quantity: executed,
            remaining_quantity: remaining,
            average_price: 0.0,
            progress_pct: progress * 100.0,
            estimated_completion_ns: estimated_completion,
            slices_completed: self.current_bucket.load(Ordering::Relaxed) as u32,
            total_slices: self.num_buckets as u32,
        }
    }
}

/// Iceberg order manager for hiding large order footprints
pub struct IcebergOrder {
    /// Total parent order quantity
    total_quantity: AtomicF64,
    /// Visible clip size
    visible_size: AtomicF64,
    /// Hidden quantity
    hidden_quantity: AtomicF64,
    /// Executed quantity
    executed_quantity: AtomicF64,
    /// Current visible clip remaining
    clip_remaining: AtomicF64,
    /// Number of clips executed
    clips_executed: AtomicU64,
    /// Active flag
    is_active: AtomicBool,
}

impl IcebergOrder {
    /// Create new iceberg order
    pub fn new(total_quantity: f64, visible_size: f64) -> Self {
        let hidden = total_quantity - visible_size;
        
        Self {
            total_quantity: AtomicF64::new(total_quantity),
            visible_size: AtomicF64::new(visible_size),
            hidden_quantity: AtomicF64::new(hidden.max(0.0)),
            executed_quantity: AtomicF64::new(0.0),
            clip_remaining: AtomicF64::new(visible_size),
            clips_executed: AtomicU64::new(0),
            is_active: AtomicBool::new(false),
        }
    }

    /// Start iceberg execution
    #[inline(always)]
    pub fn start(&self) {
        self.is_active.store(true, Ordering::Relaxed);
    }

    /// Get current visible order size to place
    #[inline(always)]
    pub fn get_visible_size(&self) -> f64 {
        let remaining_total = self.total_quantity.load(Ordering::Relaxed) 
            - self.executed_quantity.load(Ordering::Relaxed);
        let hidden = self.hidden_quantity.load(Ordering::Relaxed);
        let visible = self.visible_size.load(Ordering::Relaxed);
        
        // Return minimum of configured visible size and what's left in hidden + current clip
        visible.min(remaining_total)
    }

    /// Record fill on current clip
    #[inline(always)]
    pub fn record_fill(&self, fill_quantity: f64) {
        self.executed_quantity.fetch_add(fill_quantity, Ordering::Relaxed);
        
        let clip_rem = self.clip_remaining.load(Ordering::Relaxed);
        let new_clip_rem = clip_rem - fill_quantity;
        
        if new_clip_rem <= 0.0 {
            // Clip exhausted, reload from hidden
            let hidden = self.hidden_quantity.load(Ordering::Relaxed);
            let visible = self.visible_size.load(Ordering::Relaxed);
            
            if hidden > 0.0 {
                let reload_size = visible.min(hidden);
                self.clip_remaining.store(reload_size, Ordering::Relaxed);
                self.hidden_quantity.fetch_sub(reload_size, Ordering::Relaxed);
                self.clips_executed.fetch_add(1, Ordering::Relaxed);
            } else {
                // No more hidden quantity
                self.clip_remaining.store(0.0, Ordering::Relaxed);
            }
        } else {
            self.clip_remaining.store(new_clip_rem, Ordering::Relaxed);
        }
    }

    /// Check if order is complete
    #[inline(always)]
    pub fn is_complete(&self) -> bool {
        let executed = self.executed_quantity.load(Ordering::Relaxed);
        let total = self.total_quantity.load(Ordering::Relaxed);
        executed >= total
    }

    /// Get remaining quantity
    #[inline(always)]
    pub fn get_remaining(&self) -> f64 {
        self.total_quantity.load(Ordering::Relaxed) - self.executed_quantity.load(Ordering::Relaxed)
    }

    /// Dynamically adjust visible size based on liquidity
    #[inline(always)]
    pub fn adjust_visible_size(&self, new_visible: f64, order_book_depth: f64) {
        // Adjust visible size based on order book depth
        // Larger depth allows larger visible size without moving market
        let depth_ratio = (order_book_depth / new_visible).min(10.0).max(1.0);
        let adjusted_visible = new_visible * (depth_ratio / 10.0);
        
        self.visible_size.store(adjusted_visible, Ordering::Relaxed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_twap_execution() {
        let twap = TwapExecutor::new(10.0, 60, 10); // 10 BTC over 60 seconds, 10 slices
        twap.start();
        
        assert!(twap.should_execute_slice());
        
        let qty = twap.get_next_slice_quantity();
        assert!((qty - 1.0).abs() < 0.0001); // 1 BTC per slice
        
        twap.record_slice_execution(qty);
        
        let status = twap.get_status();
        assert_eq!(status.slices_completed, 1);
        assert!((status.progress_pct - 10.0).abs() < 0.1);
    }

    #[test]
    fn test_vwap_with_profile() {
        let profile = vec![0.1, 0.2, 0.3, 0.2, 0.2]; // Volume distribution
        let vwap = VwapExecutor::new(10.0, 300, profile);
        vwap.start();
        
        let target = vwap.get_bucket_target();
        assert!((target - 1.0).abs() < 0.001); // First bucket: 10% of 10 = 1
        
        vwap.advance_bucket();
        let status = vwap.get_status();
        assert_eq!(status.slices_completed, 1);
    }

    #[test]
    fn test_iceberg_order() {
        let iceberg = IcebergOrder::new(100.0, 10.0); // 100 total, 10 visible
        iceberg.start();
        
        let visible = iceberg.get_visible_size();
        assert!((visible - 10.0).abs() < 0.001);
        
        // Fill the first clip
        iceberg.record_fill(10.0);
        
        // Should reload another 10 from hidden
        let visible_after = iceberg.get_visible_size();
        assert!((visible_after - 10.0).abs() < 0.001);
        
        assert_eq!(iceberg.clips_executed.load(Ordering::Relaxed), 1);
    }
}
