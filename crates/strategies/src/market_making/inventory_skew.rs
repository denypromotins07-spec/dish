//! Dynamic Bid-Ask Skew Logic
//! Shifts midpoint of quotes based on real-time position, funding rates, and target inventory

use std::sync::atomic::{AtomicF64, AtomicBool, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

/// Configuration for inventory skew
#[derive(Clone, Copy, Debug)]
pub struct SkewConfig {
    /// Base skew factor (1.0 = symmetric)
    pub base_skew: f64,
    /// Max skew multiplier
    pub max_skew: f64,
    /// Inventory threshold to start applying skew (as fraction of max)
    pub threshold: f64,
    /// Funding rate sensitivity
    pub funding_sensitivity: f64,
    /// Target inventory ratio (0.0 = flat, >0 = prefer long, <0 = prefer short)
    pub target_inventory_ratio: f64,
}

impl Default for SkewConfig {
    fn default() -> Self {
        Self {
            base_skew: 1.0,
            max_skew: 3.0,
            threshold: 0.3,
            funding_sensitivity: 0.5,
            target_inventory_ratio: 0.0,
        }
    }
}

/// Lock-free inventory skew engine
pub struct InventorySkewEngine {
    /// Current mid-price
    pub mid_price: AtomicF64,
    /// Current inventory (signed)
    pub inventory: AtomicF64,
    /// Max inventory limit
    pub max_inventory: AtomicF64,
    /// Current funding rate (annualized)
    pub funding_rate: AtomicF64,
    /// Configuration
    pub config: SkewConfig,
    /// Enabled flag
    pub enabled: AtomicBool,
    /// Last update timestamp
    pub last_update_ns: AtomicU64,
}

impl InventorySkewEngine {
    pub fn new(mid_price: f64, max_inventory: f64, config: SkewConfig) -> Self {
        let now_ns = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64;
        
        Self {
            mid_price: AtomicF64::new(mid_price),
            inventory: AtomicF64::new(0.0),
            max_inventory: AtomicF64::new(max_inventory),
            funding_rate: AtomicF64::new(0.0),
            config,
            enabled: AtomicBool::new(true),
            last_update_ns: AtomicU64::new(now_ns),
        }
    }

    #[inline]
    pub fn update_mid_price(&self, price: f64) {
        self.mid_price.store(price, Ordering::Relaxed);
        self.last_update_ns.store(
            SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos() as u64,
            Ordering::Relaxed,
        );
    }

    #[inline]
    pub fn update_inventory(&self, qty: f64) {
        self.inventory.fetch_add(qty, Ordering::Relaxed);
    }

    #[inline]
    pub fn set_inventory(&self, qty: f64) {
        self.inventory.store(qty, Ordering::Relaxed);
    }

    #[inline]
    pub fn update_funding_rate(&self, rate: f64) {
        self.funding_rate.store(rate, Ordering::Relaxed);
    }

    /// Calculate inventory ratio (-1 to 1)
    #[inline]
    fn inventory_ratio(&self) -> f64 {
        let inv = self.inventory.load(Ordering::Relaxed);
        let max = self.max_inventory.load(Ordering::Relaxed);
        if max == 0.0 { return 0.0; }
        (inv / max).clamp(-1.0, 1.0)
    }

    /// Calculate target-adjusted inventory deviation
    #[inline]
    fn target_deviation(&self) -> f64 {
        let current_ratio = self.inventory_ratio();
        let target = self.config.target_inventory_ratio.clamp(-1.0, 1.0);
        current_ratio - target
    }

    /// Calculate skew factor based on inventory
    #[inline]
    fn inventory_skew(&self) -> f64 {
        let deviation = self.target_deviation();
        let threshold = self.config.threshold;
        
        // No skew if within threshold
        if deviation.abs() < threshold {
            return 1.0;
        }
        
        // Linear scaling beyond threshold
        let excess = (deviation.abs() - threshold) / (1.0 - threshold);
        let skew_multiplier = 1.0 + excess * (self.config.max_skew - 1.0);
        
        if deviation > 0.0 {
            // Long bias: widen ask, narrow bid (skew > 1)
            skew_multiplier
        } else {
            // Short bias: narrow ask, widen bid (skew < 1)
            1.0 / skew_multiplier
        }
    }

    /// Calculate funding-based skew adjustment
    #[inline]
    fn funding_skew(&self) -> f64 {
        let funding = self.funding_rate.load(Ordering::Relaxed);
        let sensitivity = self.config.funding_sensitivity;
        
        // Positive funding: prefer short (skew down)
        // Negative funding: prefer long (skew up)
        let funding_adjustment = 1.0 - funding * sensitivity;
        funding_adjustment.clamp(0.5, 2.0)
    }

    /// Get combined skew factor
    #[inline]
    pub fn get_skew_factor(&self) -> f64 {
        if !self.enabled.load(Ordering::Relaxed) {
            return 1.0;
        }
        
        let inv_skew = self.inventory_skew();
        let fund_skew = self.funding_skew();
        
        // Combine multiplicatively
        let combined = inv_skew * fund_skew;
        combined.clamp(1.0 / self.config.max_skew, self.config.max_skew)
    }

    /// Apply skew to bid-ask spread
    /// Returns (bid_offset_ratio, ask_offset_ratio) where 0.5 = symmetric
    #[inline]
    pub fn get_bid_ask_offsets(&self) -> (f64, f64) {
        let skew = self.get_skew_factor();
        
        // skew > 1: bid closer to mid, ask farther
        // skew < 1: bid farther from mid, ask closer
        
        let total = skew + 1.0;
        let bid_ratio = skew / total;  // Higher skew = higher bid ratio = closer to mid
        let ask_ratio = 1.0 / total;   // Higher skew = lower ask ratio = farther from mid
        
        (bid_ratio, ask_ratio)
    }

    /// Calculate skewed bid and ask prices from a base spread
    #[inline]
    pub fn apply_skew(&self, base_bid: f64, base_ask: f64) -> (f64, f64) {
        let mid = (base_bid + base_ask) / 2.0;
        let half_spread = (base_ask - base_bid) / 2.0;
        
        let (bid_ratio, ask_ratio) = self.get_bid_ask_offsets();
        
        let skewed_bid = mid - half_spread * (2.0 * bid_ratio);
        let skewed_ask = mid + half_spread * (2.0 * ask_ratio);
        
        (skewed_bid, skewed_ask)
    }

    /// Enable/disable skew logic
    #[inline]
    pub fn set_enabled(&self, enabled: bool) {
        self.enabled.store(enabled, Ordering::Relaxed);
    }

    /// Update configuration
    #[inline]
    pub fn update_config(&mut self, config: SkewConfig) {
        self.config = config;
    }
}

/// Skewed quote result
#[derive(Clone, Copy, Debug)]
pub struct SkewedQuote {
    pub original_bid: f64,
    pub original_ask: f64,
    pub skewed_bid: f64,
    pub skewed_ask: f64,
    pub skew_factor: f64,
    pub timestamp_ns: u64,
}

impl SkewedQuote {
    pub fn new(orig_bid: f64, orig_ask: f64, skw_bid: f64, skw_ask: f64, skew: f64) -> Self {
        Self {
            original_bid: orig_bid,
            original_ask: orig_ask,
            skewed_bid: skw_bid,
            skewed_ask: skw_ask,
            skew_factor: skew,
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
    fn test_symmetric_case() {
        let config = SkewConfig::default();
        let engine = InventorySkewEngine::new(50000.0, 100.0, config);
        
        let skew = engine.get_skew_factor();
        assert!((skew - 1.0).abs() < 0.001); // Near symmetric with zero inventory
        
        let (bid_off, ask_off) = engine.get_bid_ask_offsets();
        assert!((bid_off - 0.5).abs() < 0.001);
        assert!((ask_off - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_long_inventory_skew() {
        let config = SkewConfig {
            base_skew: 1.0,
            max_skew: 3.0,
            threshold: 0.3,
            funding_sensitivity: 0.0,
            target_inventory_ratio: 0.0,
        };
        let engine = InventorySkewEngine::new(50000.0, 100.0, config);
        
        // Set high long inventory
        engine.set_inventory(80.0);
        
        let skew = engine.get_skew_factor();
        assert!(skew > 1.0); // Should skew to encourage selling
        
        let (bid_off, ask_off) = engine.get_bid_ask_offsets();
        assert!(bid_off > ask_off); // Bid closer to mid than ask
    }

    #[test]
    fn test_short_inventory_skew() {
        let config = SkewConfig::default();
        let engine = InventorySkewEngine::new(50000.0, 100.0, config);
        
        // Set high short inventory
        engine.set_inventory(-80.0);
        
        let skew = engine.get_skew_factor();
        assert!(skew < 1.0); // Should skew to encourage buying
    }

    #[test]
    fn test_funding_rate_effect() {
        let config = SkewConfig {
            base_skew: 1.0,
            max_skew: 3.0,
            threshold: 0.3,
            funding_sensitivity: 0.5,
            target_inventory_ratio: 0.0,
        };
        let engine = InventorySkewEngine::new(50000.0, 100.0, config);
        
        // Positive funding should discourage long positions
        engine.update_funding_rate(0.001); // 0.1% per period
        let skew_positive = engine.get_skew_factor();
        
        engine.update_funding_rate(-0.001); // Negative funding
        let skew_negative = engine.get_skew_factor();
        
        assert!(skew_positive < skew_negative);
    }

    #[test]
    fn test_apply_skew() {
        let config = SkewConfig::default();
        let engine = InventorySkewEngine::new(50000.0, 100.0, config);
        
        engine.set_inventory(80.0);
        
        let (skewed_bid, skewed_ask) = engine.apply_skew(49990.0, 50010.0);
        
        // With long inventory, bid should move up (closer to mid), ask should move up more
        assert!(skewed_bid > 49990.0);
        assert!(skewed_ask > 50010.0);
        assert!(skewed_bid < skewed_ask);
    }
}
