//! Dynamic spread optimizer factoring in real-time queue depletion, short-term alpha decay, and inventory risk.
//! Adjusts quote offset in CPU registers before writing to memory for minimum latency.
//! Optimized for AMD Ryzen AI 5 architecture.

use std::sync::atomic::{AtomicU64, AtomicI64, AtomicBool, Ordering};

/// Spread optimization parameters
#[repr(C, align(64))]
pub struct SpreadOptimizer {
    /// Base spread in basis points
    base_spread_bps: AtomicU64,
    /// Current optimized spread
    current_spread_bps: AtomicU64,
    /// Inventory position (signed, in base units)
    inventory: AtomicI64,
    /// Inventory skew factor (bps per unit of inventory)
    inventory_skew_bps: AtomicU64,
    /// Queue depletion rate (orders per second * 1000)
    queue_depletion_rate: AtomicU64,
    /// Alpha decay factor (0-1000, where 1000 = no decay)
    alpha_decay_factor: AtomicU64,
    /// Volatility adjustment (bps)
    vol_adjustment_bps: AtomicU64,
    /// Is optimization active
    is_active: AtomicBool,
    _padding: [u8; 15],
}

impl SpreadOptimizer {
    pub fn new(base_spread_bps: u64, inventory_skew_bps: u64) -> Self {
        Self {
            base_spread_bps: AtomicU64::new(base_spread_bps),
            current_spread_bps: AtomicU64::new(base_spread_bps),
            inventory: AtomicI64::new(0),
            inventory_skew_bps: AtomicU64::new(inventory_skew_bps),
            queue_depletion_rate: AtomicU64::new(1000), // Normal rate
            alpha_decay_factor: AtomicU64::new(1000),   // No decay
            vol_adjustment_bps: AtomicU64::new(0),
            is_active: AtomicBool::new(true),
            _padding: [0u8; 15],
        }
    }
    
    /// Calculate optimized spread - pure register arithmetic, O(1)
    #[inline]
    pub fn calculate_optimal_spread(&self, mid_price: u64) -> u64 {
        if !self.is_active.load(Ordering::Relaxed) {
            return self.base_spread_bps.load(Ordering::Relaxed);
        }
        
        let mut spread = self.base_spread_bps.load(Ordering::Relaxed) as i128;
        
        // 1. Inventory skew adjustment
        let inv = self.inventory.load(Ordering::Relaxed);
        if inv != 0 {
            let skew_bps = self.inventory_skew_bps.load(Ordering::Relaxed) as i128;
            let inv_abs = inv.abs() as i128;
            // Skew increases spread on side we want to avoid, decreases on side we want
            let inventory_adj = (inv_abs * skew_bps) / 1000;
            spread += inventory_adj;
        }
        
        // 2. Queue depletion adjustment
        // High depletion rate = tighter spread to get filled faster
        let depletion = self.queue_depletion_rate.load(Ordering::Relaxed) as i128;
        if depletion > 2000 {
            // Depletion is high, reduce spread by up to 20%
            let reduction = (spread * 20) / 100;
            spread -= reduction.min((depletion - 2000) / 100);
        } else if depletion < 500 {
            // Depletion is low, increase spread to protect against adverse selection
            let increase = (spread * 10) / 100;
            spread += increase;
        }
        
        // 3. Alpha decay adjustment
        // High alpha decay = wider spread to compensate for reduced predictability
        let alpha = self.alpha_decay_factor.load(Ordering::Relaxed) as i128;
        if alpha < 1000 {
            let decay_penalty = ((1000 - alpha) * spread) / 2000;
            spread += decay_penalty;
        }
        
        // 4. Volatility adjustment
        let vol_adj = self.vol_adjustment_bps.load(Ordering::Relaxed) as i128;
        spread += vol_adj;
        
        // Ensure spread is within reasonable bounds (1-1000 bps)
        spread = spread.max(1).min(1000);
        
        self.current_spread_bps.store(spread as u64, Ordering::Relaxed);
        spread as u64
    }
    
    /// Update inventory position
    #[inline]
    pub fn update_inventory(&self, delta: i64) {
        self.inventory.fetch_add(delta, Ordering::Relaxed);
    }
    
    /// Get current inventory
    #[inline]
    pub fn get_inventory(&self) -> i64 {
        self.inventory.load(Ordering::Relaxed)
    }
    
    /// Update queue depletion rate
    #[inline]
    pub fn update_queue_depletion(&self, rate: u64) {
        self.queue_depletion_rate.store(rate, Ordering::Relaxed);
    }
    
    /// Update alpha decay factor (0-1000)
    #[inline]
    pub fn update_alpha_decay(&self, factor: u64) {
        self.alpha_decay_factor.store(factor.min(1000), Ordering::Relaxed);
    }
    
    /// Update volatility adjustment
    #[inline]
    pub fn update_vol_adjustment(&self, bps: u64) {
        self.vol_adjustment_bps.store(bps, Ordering::Relaxed);
    }
    
    /// Get bid-side adjusted price (wider spread if long inventory)
    #[inline]
    pub fn get_bid_adjustment(&self, mid_price: u64, base_spread: u64) -> u64 {
        let inv = self.inventory.load(Ordering::Relaxed);
        let skew = self.inventory_skew_bps.load(Ordering::Relaxed);
        
        if inv > 0 {
            // Long inventory - widen bid spread to discourage more buys
            let adj = ((inv as u64 * skew) / 1000) as u64;
            base_spread.saturating_add(adj)
        } else {
            // Short or neutral - normal or tighter spread
            base_spread.saturating_sub(((inv.abs() as u64 * skew) / 2000) as u64)
        }
    }
    
    /// Get ask-side adjusted price (wider spread if short inventory)
    #[inline]
    pub fn get_ask_adjustment(&self, mid_price: u64, base_spread: u64) -> u64 {
        let inv = self.inventory.load(Ordering::Relaxed);
        let skew = self.inventory_skew_bps.load(Ordering::Relaxed);
        
        if inv < 0 {
            // Short inventory - widen ask spread to discourage more sells
            let adj = ((inv.abs() as u64 * skew) / 1000) as u64;
            base_spread.saturating_add(adj)
        } else {
            // Long or neutral - normal or tighter spread
            base_spread.saturating_sub(((inv as u64 * skew) / 2000) as u64)
        }
    }
    
    /// Reset optimizer to defaults
    #[inline]
    pub fn reset(&self) {
        self.current_spread_bps.store(self.base_spread_bps.load(Ordering::Relaxed), Ordering::Relaxed);
        self.inventory.store(0, Ordering::Relaxed);
        self.queue_depletion_rate.store(1000, Ordering::Relaxed);
        self.alpha_decay_factor.store(1000, Ordering::Relaxed);
        self.vol_adjustment_bps.store(0, Ordering::Relaxed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_basic_spread() {
        let opt = SpreadOptimizer::new(10, 1);
        let spread = opt.calculate_optimal_spread(10000);
        assert_eq!(spread, 10); // Base spread with no adjustments
    }
    
    #[test]
    fn test_inventory_skew() {
        let opt = SpreadOptimizer::new(10, 5);
        opt.update_inventory(1000); // Long position
        
        let bid_adj = opt.get_bid_adjustment(10000, 10);
        let ask_adj = opt.get_ask_adjustment(10000, 10);
        
        assert!(bid_adj > 10); // Wider bid spread when long
        assert!(ask_adj <= 10); // Tighter or normal ask spread
    }
    
    #[test]
    fn test_alpha_decay() {
        let opt = SpreadOptimizer::new(10, 0);
        opt.update_alpha_decay(500); // 50% decay
        
        let spread = opt.calculate_optimal_spread(10000);
        assert!(spread > 10); // Spread should increase with decay
    }
}
