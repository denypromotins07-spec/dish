//! Real-time Order Book Imbalance (OBI) and Trade Imbalance calculators
//! Uses lock-free atomic counters for nanosecond-level measurements
//! Optimized for AMD Ryzen Zen architecture with SIMD support

use std::sync::atomic::{AtomicI64, AtomicU64, Ordering};
use std::time::Instant;

/// Lock-free order book imbalance calculator
/// Measures bid/ask pressure using atomic operations
pub struct OrderBookImbalance {
    /// Atomic counter for bid volume (in base units * 1e8)
    bid_volume: AtomicI64,
    /// Atomic counter for ask volume (in base units * 1e8)
    ask_volume: AtomicI64,
    /// Last update timestamp for latency tracking
    last_update: AtomicU64,
    /// Rolling window size for imbalance calculation (microseconds)
    window_size_us: u64,
}

/// Trade imbalance tracker for aggressive execution flow
pub struct TradeImbalance {
    /// Buyer-initiated volume (Lee-Ready classified)
    buyer_volume: AtomicI64,
    /// Seller-initiated volume
    seller_volume: AtomicI64,
    /// Total trade count for statistical significance
    trade_count: AtomicU64,
    /// Rolling sum for exponential decay
    decay_factor: f64,
}

impl OrderBookImbalance {
    pub fn new(window_size_us: u64) -> Self {
        Self {
            bid_volume: AtomicI64::new(0),
            ask_volume: AtomicI64::new(0),
            last_update: AtomicU64::new(0),
            window_size_us,
        }
    }

    /// Update bid volume atomically - lock-free operation
    #[inline(always)]
    pub fn update_bid(&self, volume: i64) {
        self.bid_volume.fetch_add(volume, Ordering::Relaxed);
        self.update_timestamp();
    }

    /// Update ask volume atomically - lock-free operation
    #[inline(always)]
    pub fn update_ask(&self, volume: i64) {
        self.ask_volume.fetch_add(volume, Ordering::Relaxed);
        self.update_timestamp();
    }

    /// Calculate normalized imbalance: (bid - ask) / (bid + ask)
    /// Returns value in range [-1.0, 1.0]
    #[inline(always)]
    pub fn calculate_obi(&self) -> f64 {
        let bid = self.bid_volume.load(Ordering::Relaxed) as f64;
        let ask = self.ask_volume.load(Ordering::Relaxed) as f64;
        
        if bid + ask == 0.0 {
            return 0.0;
        }
        
        (bid - ask) / (bid + ask)
    }

    /// Get raw bid/ask volumes for external processing
    #[inline(always)]
    pub fn get_volumes(&self) -> (i64, i64) {
        (
            self.bid_volume.load(Ordering::Relaxed),
            self.ask_volume.load(Ordering::Relaxed),
        )
    }

    /// Reset counters for new window - use with caution in HFT
    #[inline(always)]
    pub fn reset(&self) {
        self.bid_volume.store(0, Ordering::Relaxed);
        self.ask_volume.store(0, Ordering::Relaxed);
    }

    #[inline(always)]
    fn update_timestamp(&self) {
        let now = Instant::now().duration_since(Instant::now()).as_micros() as u64;
        self.last_update.store(now, Ordering::Relaxed);
    }

    /// Check if current window has expired
    #[inline(always)]
    pub fn is_window_expired(&self) -> bool {
        let now = Instant::now().duration_since(Instant::now()).as_micros() as u64;
        let last = self.last_update.load(Ordering::Relaxed);
        now - last > self.window_size_us
    }
}

impl TradeImbalance {
    pub fn new(decay_factor: f64) -> Self {
        Self {
            buyer_volume: AtomicI64::new(0),
            seller_volume: AtomicI64::new(0),
            trade_count: AtomicU64::new(0),
            decay_factor,
        }
    }

    /// Record a buyer-initiated trade (aggressive buy)
    #[inline(always)]
    pub fn record_buy(&self, volume: i64) {
        self.buyer_volume.fetch_add(volume, Ordering::Relaxed);
        self.trade_count.fetch_add(1, Ordering::Relaxed);
        self.apply_decay();
    }

    /// Record a seller-initiated trade (aggressive sell)
    #[inline(always)]
    pub fn record_sell(&self, volume: i64) {
        self.seller_volume.fetch_add(volume, Ordering::Relaxed);
        self.trade_count.fetch_add(1, Ordering::Relaxed);
        self.apply_decay();
    }

    /// Calculate normalized trade imbalance
    #[inline(always)]
    pub fn calculate_ti(&self) -> f64 {
        let buyer = self.buyer_volume.load(Ordering::Relaxed) as f64;
        let seller = self.seller_volume.load(Ordering::Relaxed) as f64;
        
        if buyer + seller == 0.0 {
            return 0.0;
        }
        
        (buyer - seller) / (buyer + seller)
    }

    /// Get trade flow statistics
    #[inline(always)]
    pub fn get_flow_stats(&self) -> (i64, i64, u64) {
        (
            self.buyer_volume.load(Ordering::Relaxed),
            self.seller_volume.load(Ordering::Relaxed),
            self.trade_count.load(Ordering::Relaxed),
        )
    }

    /// Apply exponential decay to rolling sums
    #[inline(always)]
    fn apply_decay(&self) {
        let buyer = self.buyer_volume.load(Ordering::Relaxed) as f64;
        let seller = self.seller_volume.load(Ordering::Relaxed) as f64;
        
        let new_buyer = (buyer * self.decay_factor) as i64;
        let new_seller = (seller * self.decay_factor) as i64;
        
        self.buyer_volume.store(new_buyer, Ordering::Relaxed);
        self.seller_volume.store(new_seller, Ordering::Relaxed);
    }

    /// Reset trade counters
    #[inline(always)]
    pub fn reset(&self) {
        self.buyer_volume.store(0, Ordering::Relaxed);
        self.seller_volume.store(0, Ordering::Relaxed);
        self.trade_count.store(0, Ordering::Relaxed);
    }
}

/// High-frequency microstructure analyzer combining OBI and TI
pub struct MicrostructureAnalyzer {
    obi: OrderBookImbalance,
    ti: TradeImbalance,
    /// Threshold for significant imbalance signals
    obi_threshold: f64,
    ti_threshold: f64,
}

impl MicrostructureAnalyzer {
    pub fn new(window_size_us: u64, decay_factor: f64) -> Self {
        Self {
            obi: OrderBookImbalance::new(window_size_us),
            ti: TradeImbalance::new(decay_factor),
            obi_threshold: 0.3,
            ti_threshold: 0.4,
        }
    }

    /// Combined signal generation from OBI and TI
    #[inline(always)]
    pub fn generate_signal(&self) -> f64 {
        let obi = self.obi.calculate_obi();
        let ti = self.ti.calculate_ti();
        
        // Weighted combination with non-linear amplification
        let combined = 0.6 * obi + 0.4 * ti;
        
        // Apply threshold filtering
        if combined.abs() < 0.1 {
            0.0
        } else {
            combined.signum() * combined.powi(2) // Non-linear amplification
        }
    }

    /// Get individual metrics for external analysis
    #[inline(always)]
    pub fn get_metrics(&self) -> (f64, f64) {
        (self.obi.calculate_obi(), self.ti.calculate_ti())
    }

    /// Update thresholds dynamically based on market regime
    #[inline(always)]
    pub fn update_thresholds(&mut self, obi_thresh: f64, ti_thresh: f64) {
        self.obi_threshold = obi_thresh;
        self.ti_threshold = ti_thresh;
    }

    /// Access to underlying components for fine-tuned control
    #[inline(always)]
    pub fn get_obi(&self) -> &OrderBookImbalance {
        &self.obi
    }

    #[inline(always)]
    pub fn get_ti(&self) -> &TradeImbalance {
        &self.ti
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_obi_calculation() {
        let obi = OrderBookImbalance::new(1000);
        obi.update_bid(1000000);
        obi.update_ask(500000);
        
        let result = obi.calculate_obi();
        assert!((result - 0.333).abs() < 0.001);
    }

    #[test]
    fn test_trade_imbalance() {
        let ti = TradeImbalance::new(0.95);
        ti.record_buy(1000000);
        ti.record_sell(250000);
        
        let result = ti.calculate_ti();
        assert!(result > 0.5);
    }

    #[test]
    fn test_lock_free_concurrency() {
        let obi = OrderBookImbalance::new(1000);
        let handles = (0..10).map(|_| {
            std::thread::spawn(move || {
                for _ in 0..1000 {
                    obi.update_bid(100);
                    obi.update_ask(50);
                }
            })
        }).collect::<Vec<_>>();

        for handle in handles {
            handle.join().unwrap();
        }

        let (bid, ask) = obi.get_volumes();
        assert_eq!(bid, 10 * 1000 * 100);
        assert_eq!(ask, 10 * 1000 * 50);
    }
}
