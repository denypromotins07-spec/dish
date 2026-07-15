//! Volume-Synchronized Probability of Informed Trading (VPIN) calculator.
//! Uses L3 trade aggressor data and volume buckets to detect toxic, informed flow in real-time.
//! Optimized for AMD Ryzen AI 5 with zero heap allocations.

use std::sync::atomic::{AtomicU64, AtomicBool, Ordering};

/// VPIN calculation parameters
#[repr(C, align(64))]
pub struct VpinCalculator {
    /// Target bucket size (in volume units)
    bucket_size: AtomicU64,
    /// Current bucket buy volume
    current_buy_volume: AtomicU64,
    /// Current bucket sell volume
    current_sell_volume: AtomicU64,
    /// Number of buckets processed
    buckets_processed: AtomicU64,
    /// Sum of absolute order flow imbalance across all buckets
    sum_abs_ofi: AtomicU64,
    /// Total volume processed
    total_volume: AtomicU64,
    /// Current VPIN value (fixed point: * 10000)
    current_vpin: AtomicU64,
    /// Is calculation active
    is_active: AtomicBool,
    _padding: [u8; 23],
}

unsafe impl Send for VpinCalculator {}
unsafe impl Sync for VpinCalculator {}

impl VpinCalculator {
    pub fn new(bucket_size: u64) -> Self {
        Self {
            bucket_size: AtomicU64::new(bucket_size),
            current_buy_volume: AtomicU64::new(0),
            current_sell_volume: AtomicU64::new(0),
            buckets_processed: AtomicU64::new(0),
            sum_abs_ofi: AtomicU64::new(0),
            total_volume: AtomicU64::new(0),
            current_vpin: AtomicU64::new(0),
            is_active: AtomicBool::new(true),
            _padding: [0u8; 23],
        }
    }
    
    /// Process a single trade - O(1) operation
    /// is_buyer_aggressor: true if buyer initiated the trade
    #[inline]
    pub fn process_trade(&self, volume: u64, is_buyer_aggressor: bool) {
        if !self.is_active.load(Ordering::Relaxed) {
            return;
        }
        
        // Update current bucket volumes
        if is_buyer_aggressor {
            self.current_buy_volume.fetch_add(volume, Ordering::Relaxed);
        } else {
            self.current_sell_volume.fetch_add(volume, Ordering::Relaxed);
        }
        
        self.total_volume.fetch_add(volume, Ordering::Relaxed);
        
        // Check if bucket is full
        let buy_vol = self.current_buy_volume.load(Ordering::Relaxed);
        let sell_vol = self.current_sell_volume.load(Ordering::Relaxed);
        let bucket_size = self.bucket_size.load(Ordering::Relaxed);
        
        if buy_vol + sell_vol >= bucket_size {
            self.close_bucket(buy_vol, sell_vol);
        }
    }
    
    /// Close current bucket and update VPIN - O(1)
    #[inline]
    fn close_bucket(&self, buy_vol: u64, sell_vol: u64) {
        // Calculate order flow imbalance for this bucket
        let total_bucket_vol = buy_vol + sell_vol;
        if total_bucket_vol == 0 {
            self.reset_bucket();
            return;
        }
        
        // OFI = |buy_vol - sell_vol| / total_bucket_vol
        let abs_diff = if buy_vol > sell_vol {
            buy_vol - sell_vol
        } else {
            sell_vol - buy_vol
        };
        
        // Scale to fixed point (* 10000)
        let ofi_scaled = (abs_diff * 10000) / total_bucket_vol;
        
        // Update running sums
        self.sum_abs_ofi.fetch_add(ofi_scaled, Ordering::Relaxed);
        let buckets = self.buckets_processed.fetch_add(1, Ordering::Relaxed);
        
        // Calculate VPIN = sum(|OFI|) / N
        let new_buckets = buckets + 1;
        let vpin = self.sum_abs_ofi.load(Ordering::Relaxed) / new_buckets;
        self.current_vpin.store(vpin, Ordering::Relaxed);
        
        self.reset_bucket();
    }
    
    /// Reset bucket counters
    #[inline]
    fn reset_bucket(&self) {
        self.current_buy_volume.store(0, Ordering::Relaxed);
        self.current_sell_volume.store(0, Ordering::Relaxed);
    }
    
    /// Get current VPIN value (0-10000 scale)
    #[inline]
    pub fn get_vpin(&self) -> f64 {
        (self.current_vpin.load(Ordering::Relaxed) as f64) / 10000.0
    }
    
    /// Get VPIN as scaled integer
    #[inline]
    pub fn get_vpin_scaled(&self) -> u64 {
        self.current_vpin.load(Ordering::Relaxed)
    }
    
    /// Check if VPIN exceeds threshold (toxic flow detected)
    #[inline]
    pub fn is_toxic(&self, threshold: f64) -> bool {
        self.get_vpin() > threshold
    }
    
    /// Update bucket size
    #[inline]
    pub fn set_bucket_size(&self, size: u64) {
        self.bucket_size.store(size, Ordering::Relaxed);
    }
    
    /// Reset all statistics
    #[inline]
    pub fn reset(&self) {
        self.current_buy_volume.store(0, Ordering::Relaxed);
        self.current_sell_volume.store(0, Ordering::Relaxed);
        self.buckets_processed.store(0, Ordering::Relaxed);
        self.sum_abs_ofi.store(0, Ordering::Relaxed);
        self.total_volume.store(0, Ordering::Relaxed);
        self.current_vpin.store(0, Ordering::Relaxed);
    }
    
    /// Get statistics
    #[inline]
    pub fn get_stats(&self) -> VpinStats {
        VpinStats {
            buckets_processed: self.buckets_processed.load(Ordering::Relaxed),
            total_volume: self.total_volume.load(Ordering::Relaxed),
            current_vpin: self.get_vpin(),
        }
    }
}

/// VPIN statistics snapshot
#[derive(Clone, Copy, Debug)]
pub struct VpinStats {
    pub buckets_processed: u64,
    pub total_volume: u64,
    pub current_vpin: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_vpin_basic() {
        let calc = VpinCalculator::new(1000);
        
        // Add balanced trades
        calc.process_trade(500, true);  // Buy
        calc.process_trade(500, false); // Sell
        
        // Bucket should be full now (1000 total)
        // VPIN should be 0 (perfectly balanced)
        assert_eq!(calc.get_vpin(), 0.0);
    }
    
    #[test]
    fn test_vpin_imbalanced() {
        let calc = VpinCalculator::new(1000);
        
        // Add imbalanced trades (all buys)
        calc.process_trade(1000, true);
        
        // VPIN should be 1.0 (completely imbalanced)
        assert!((calc.get_vpin() - 1.0).abs() < 0.01);
    }
    
    #[test]
    fn test_toxic_detection() {
        let calc = VpinCalculator::new(1000);
        
        // Create toxic flow (all one-sided)
        for _ in 0..10 {
            calc.process_trade(100, true);
        }
        
        assert!(calc.is_toxic(0.5)); // Should be toxic above 0.5 threshold
    }
}
