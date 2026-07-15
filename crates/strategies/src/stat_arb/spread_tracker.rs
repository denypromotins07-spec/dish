//! Lock-free Spread Calculation and Z-Score Normalization Engine
//! For cointegrated crypto pairs with rolling ring buffers

use std::sync::atomic::{AtomicF64, AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

/// Fixed-size ring buffer for lock-free rolling statistics
pub struct RingBuffer {
    data: Vec<AtomicF64>,
    capacity: usize,
    head: AtomicU64,
    count: AtomicU64,
}

impl RingBuffer {
    pub fn new(capacity: usize) -> Self {
        let mut data = Vec::with_capacity(capacity);
        for _ in 0..capacity {
            data.push(AtomicF64::new(0.0));
        }
        Self {
            data,
            capacity,
            head: AtomicU64::new(0),
            count: AtomicU64::new(0),
        }
    }

    #[inline]
    pub fn push(&self, value: f64) {
        let head = self.head.fetch_add(1, Ordering::Relaxed);
        let index = (head % self.capacity as u64) as usize;
        self.data[index].store(value, Ordering::Relaxed);
        
        let count = self.count.load(Ordering::Relaxed);
        if count < self.capacity as u64 {
            self.count.fetch_add(1, Ordering::Relaxed);
        }
    }

    #[inline]
    pub fn get(&self, index: usize) -> Option<f64> {
        let count = self.count.load(Ordering::Relaxed) as usize;
        if index >= count || index >= self.capacity {
            return None;
        }
        let head = self.head.load(Ordering::Relaxed) as usize;
        let actual_index = (head - count + index) % self.capacity;
        Some(self.data[actual_index].load(Ordering::Relaxed))
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.count.load(Ordering::Relaxed) as usize
    }

    #[inline]
    pub fn is_full(&self) -> bool {
        self.len() == self.capacity
    }

    /// Calculate mean of all elements
    #[inline]
    pub fn mean(&self) -> Option<f64> {
        let count = self.len();
        if count == 0 { return None; }
        
        let mut sum = 0.0;
        for i in 0..count {
            sum += self.get(i)?;
        }
        Some(sum / count as f64)
    }

    /// Calculate standard deviation
    #[inline]
    pub fn std_dev(&self) -> Option<f64> {
        let count = self.len();
        if count < 2 { return None; }
        
        let mean = self.mean()?;
        let mut variance_sum = 0.0;
        for i in 0..count {
            let val = self.get(i)?;
            variance_sum += (val - mean).powi(2);
        }
        Some((variance_sum / (count as f64 - 1.0)).sqrt())
    }

    /// Calculate min and max
    #[inline]
    pub fn min_max(&self) -> Option<(f64, f64)> {
        let count = self.len();
        if count == 0 { return None; }
        
        let mut min = f64::MAX;
        let mut max = f64::MIN;
        for i in 0..count {
            let val = self.get(i)?;
            if val < min { min = val; }
            if val > max { max = val; }
        }
        Some((min, max))
    }
}

/// Spread tracker for cointegrated pairs
pub struct SpreadTracker {
    /// Price of asset A
    pub price_a: AtomicF64,
    /// Price of asset B
    pub price_b: AtomicF64,
    /// Hedge ratio (units of B per unit of A)
    pub hedge_ratio: AtomicF64,
    /// Rolling spread values
    pub spread_buffer: RingBuffer,
    /// Current z-score
    pub z_score: AtomicF64,
    /// Rolling mean of spread
    pub spread_mean: AtomicF64,
    /// Rolling std dev of spread
    pub spread_std: AtomicF64,
    /// Last update timestamp
    pub last_update_ns: AtomicU64,
}

impl SpreadTracker {
    pub fn new(buffer_size: usize, initial_hedge_ratio: f64) -> Self {
        let now_ns = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64;
        
        Self {
            price_a: AtomicF64::new(0.0),
            price_b: AtomicF64::new(0.0),
            hedge_ratio: AtomicF64::new(initial_hedge_ratio),
            spread_buffer: RingBuffer::new(buffer_size),
            z_score: AtomicF64::new(0.0),
            spread_mean: AtomicF64::new(0.0),
            spread_std: AtomicF64::new(1.0),
            last_update_ns: AtomicU64::new(now_ns),
        }
    }

    /// Update prices and calculate spread
    #[inline]
    pub fn update_prices(&self, price_a: f64, price_b: f64) {
        self.price_a.store(price_a, Ordering::Relaxed);
        self.price_b.store(price_b, Ordering::Relaxed);
        
        let hedge = self.hedge_ratio.load(Ordering::Relaxed);
        // Spread = price_a - hedge_ratio * price_b
        let spread = price_a - hedge * price_b;
        
        self.spread_buffer.push(spread);
        self.update_statistics();
        
        self.last_update_ns.store(
            SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos() as u64,
            Ordering::Relaxed,
        );
    }

    /// Update rolling statistics
    #[inline]
    fn update_statistics(&self) {
        if let Some(mean) = self.spread_buffer.mean() {
            self.spread_mean.store(mean, Ordering::Relaxed);
        }
        if let Some(std) = self.spread_buffer.std_dev() {
            self.spread_std.store(std.max(1e-10), Ordering::Relaxed);
        }
        
        // Update z-score
        let current_spread = self.current_spread();
        let mean = self.spread_mean.load(Ordering::Relaxed);
        let std = self.spread_std.load(Ordering::Relaxed);
        let z = (current_spread - mean) / std;
        self.z_score.store(z, Ordering::Relaxed);
    }

    #[inline]
    pub fn current_spread(&self) -> f64 {
        let a = self.price_a.load(Ordering::Relaxed);
        let b = self.price_b.load(Ordering::Relaxed);
        let hedge = self.hedge_ratio.load(Ordering::Relaxed);
        a - hedge * b
    }

    #[inline]
    pub fn get_z_score(&self) -> f64 {
        self.z_score.load(Ordering::Relaxed)
    }

    #[inline]
    pub fn get_mean(&self) -> f64 {
        self.spread_mean.load(Ordering::Relaxed)
    }

    #[inline]
    pub fn get_std(&self) -> f64 {
        self.spread_std.load(Ordering::Relaxed)
    }

    /// Update hedge ratio dynamically
    #[inline]
    pub fn update_hedge_ratio(&self, ratio: f64) {
        self.hedge_ratio.store(ratio, Ordering::Relaxed);
        // Recalculate all spreads with new ratio
        // Note: In production, you'd want to rebuild the buffer
    }

    /// Check if buffer is ready (full)
    #[inline]
    pub fn is_ready(&self) -> bool {
        self.spread_buffer.is_full()
    }

    /// Get entry signals based on z-score thresholds
    #[inline]
    pub fn get_signal(&self, entry_threshold: f64, exit_threshold: f64) -> Signal {
        let z = self.get_z_score();
        
        if z > entry_threshold {
            Signal::ShortSpread  // Spread too high, short A / long B
        } else if z < -entry_threshold {
            Signal::LongSpread   // Spread too low, long A / short B
        } else if z.abs() < exit_threshold {
            Signal::Exit         // Close position
        } else {
            Signal::Hold
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Signal {
    LongSpread,   // Expect spread to increase
    ShortSpread,  // Expect spread to decrease
    Exit,         // Close position
    Hold,         // No action
}

/// Pair state for monitoring
#[derive(Clone, Copy, Debug)]
pub struct PairState {
    pub price_a: f64,
    pub price_b: f64,
    pub spread: f64,
    pub z_score: f64,
    pub mean: f64,
    pub std: f64,
    pub signal: Signal,
    pub timestamp_ns: u64,
}

impl PairState {
    pub fn from_tracker(tracker: &SpreadTracker, entry_thresh: f64, exit_thresh: f64) -> Self {
        Self {
            price_a: tracker.price_a.load(Ordering::Relaxed),
            price_b: tracker.price_b.load(Ordering::Relaxed),
            spread: tracker.current_spread(),
            z_score: tracker.get_z_score(),
            mean: tracker.get_mean(),
            std: tracker.get_std(),
            signal: tracker.get_signal(entry_thresh, exit_thresh),
            timestamp_ns: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos() as u64,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ring_buffer_basic() {
        let buf = RingBuffer::new(5);
        assert_eq!(buf.len(), 0);
        assert!(!buf.is_full());
        
        buf.push(1.0);
        buf.push(2.0);
        buf.push(3.0);
        
        assert_eq!(buf.len(), 3);
        assert_eq!(buf.get(0), Some(1.0));
        assert_eq!(buf.get(2), Some(3.0));
    }

    #[test]
    fn test_ring_buffer_wrap() {
        let buf = RingBuffer::new(3);
        buf.push(1.0);
        buf.push(2.0);
        buf.push(3.0);
        buf.push(4.0); // Should overwrite 1.0
        
        assert_eq!(buf.len(), 3);
        assert_eq!(buf.get(0), Some(2.0));
        assert_eq!(buf.get(2), Some(4.0));
    }

    #[test]
    fn test_spread_tracker() {
        let tracker = SpreadTracker::new(10, 1.0);
        
        // Feed some data
        for i in 0..15 {
            let price_a = 100.0 + (i as f64 * 0.1);
            let price_b = 100.0 + (i as f64 * 0.05);
            tracker.update_prices(price_a, price_b);
        }
        
        assert!(tracker.is_ready());
        assert_ne!(tracker.get_z_score(), 0.0);
    }

    #[test]
    fn test_signal_generation() {
        let tracker = SpreadTracker::new(20, 1.0);
        
        // Create extreme spread values
        for i in 0..25 {
            let price_a = 100.0;
            let price_b = if i < 10 { 100.0 } else { 80.0 }; // Large negative spread
            tracker.update_prices(price_a, price_b);
        }
        
        let signal = tracker.get_signal(2.0, 0.5);
        assert!(signal == Signal::LongSpread || signal == Signal::Hold);
    }
}
