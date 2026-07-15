//! Micro-price and weighted mid-price calculator
//! Adjusts theoretical fair value based on queue depth and bid/ask volume ratios
//! Prevents adverse selection in HFT strategies

use std::sync::atomic::{AtomicI64, AtomicU64, Ordering};

/// Micro-price calculation result with metadata
#[derive(Debug, Clone, Copy)]
pub struct MicroPrice {
    /// Calculated micro-price (scaled by 1e8)
    pub price: i64,
    /// Standard mid-price for comparison
    pub mid_price: i64,
    /// Bid-ask spread (scaled by 1e8)
    pub spread: i64,
    /// Weight factor applied (0.0 to 1.0, scaled by 1e6)
    pub weight_factor: u32,
    /// Timestamp of calculation (microseconds since epoch)
    pub timestamp_us: u64,
}

/// Lock-free micro-price calculator
pub struct MicroPriceCalculator {
    /// Best bid price (scaled by 1e8)
    best_bid: AtomicI64,
    /// Best ask price (scaled by 1e8)
    best_ask: AtomicI64,
    /// Bid queue size (in base units * 1e8)
    bid_queue_size: AtomicI64,
    /// Ask queue size (in base units * 1e8)
    ask_queue_size: AtomicI64,
    /// Last calculated micro-price
    last_micro_price: AtomicI64,
    /// Last update timestamp
    last_update_us: AtomicU64,
    /// Calculation count for statistics
    calc_count: AtomicU64,
}

impl MicroPriceCalculator {
    pub fn new() -> Self {
        Self {
            best_bid: AtomicI64::new(0),
            best_ask: AtomicI64::new(0),
            bid_queue_size: AtomicI64::new(0),
            ask_queue_size: AtomicI64::new(0),
            last_micro_price: AtomicI64::new(0),
            last_update_us: AtomicU64::new(0),
            calc_count: AtomicU64::new(0),
        }
    }

    /// Update order book state atomically
    #[inline(always)]
    pub fn update_book(&self, bid: i64, ask: i64, bid_size: i64, ask_size: i64) {
        self.best_bid.store(bid, Ordering::Relaxed);
        self.best_ask.store(ask, Ordering::Relaxed);
        self.bid_queue_size.store(bid_size, Ordering::Relaxed);
        self.ask_queue_size.store(ask_size, Ordering::Relaxed);
        self.update_timestamp();
    }

    /// Calculate micro-price using queue size weighting
    /// Formula: micro_price = (bid * ask_size + ask * bid_size) / (bid_size + ask_size)
    #[inline(always)]
    pub fn calculate_micro_price(&self) -> MicroPrice {
        let bid = self.best_bid.load(Ordering::Relaxed);
        let ask = self.best_ask.load(Ordering::Relaxed);
        let bid_size = self.bid_queue_size.load(Ordering::Relaxed);
        let ask_size = self.ask_queue_size.load(Ordering::Relaxed);

        if bid == 0 || ask == 0 || bid_size <= 0 || ask_size <= 0 {
            // Cannot calculate - return zero values
            return MicroPrice {
                price: 0,
                mid_price: 0,
                spread: 0,
                weight_factor: 0,
                timestamp_us: self.last_update_us.load(Ordering::Relaxed),
            };
        }

        let mid_price = (bid + ask) / 2;
        let spread = ask - bid;

        // Queue-weighted micro-price calculation
        // More weight to side with larger queue (more pressure)
        let total_size = bid_size + ask_size;
        let micro_price = (bid * ask_size + ask * bid_size) / total_size;

        // Calculate weight factor (how much micro-price deviates from mid)
        let deviation = (micro_price - mid_price).abs();
        let weight_factor = if spread > 0 {
            ((deviation * 1_000_000) / spread) as u32
        } else {
            0
        };

        // Store result
        self.last_micro_price.store(micro_price, Ordering::Relaxed);
        self.calc_count.fetch_add(1, Ordering::Relaxed);

        MicroPrice {
            price: micro_price,
            mid_price,
            spread,
            weight_factor,
            timestamp_us: self.last_update_us.load(Ordering::Relaxed),
        }
    }

    /// Calculate micro-price with volume imbalance adjustment
    /// Uses exponential decay for recent volume emphasis
    #[inline(always)]
    pub fn calculate_weighted_micro_price(&self, imbalance_factor: f64) -> MicroPrice {
        let bid = self.best_bid.load(Ordering::Relaxed);
        let ask = self.best_ask.load(Ordering::Relaxed);
        let bid_size = self.bid_queue_size.load(Ordering::Relaxed) as f64;
        let ask_size = self.ask_queue_size.load(Ordering::Relaxed) as f64;

        if bid == 0 || ask == 0 || bid_size <= 0.0 || ask_size <= 0.0 {
            return MicroPrice {
                price: 0,
                mid_price: 0,
                spread: 0,
                weight_factor: 0,
                timestamp_us: self.last_update_us.load(Ordering::Relaxed),
            };
        }

        let mid_price = (bid + ask) / 2;
        let spread = ask - bid;

        // Volume imbalance: (bid_size - ask_size) / (bid_size + ask_size)
        let volume_imbalance = (bid_size - ask_size) / (bid_size + ask_size);

        // Adjust micro-price based on imbalance and external factor
        let adjustment = volume_imbalance * imbalance_factor * spread as f64;
        let micro_price = mid_price as f64 + adjustment;

        let weight_factor = ((adjustment.abs() * 1_000_000.0) / spread as f64) as u32;

        self.last_micro_price.store(micro_price as i64, Ordering::Relaxed);
        self.calc_count.fetch_add(1, Ordering::Relaxed);

        MicroPrice {
            price: micro_price as i64,
            mid_price,
            spread,
            weight_factor: weight_factor.min(1_000_000),
            timestamp_us: self.last_update_us.load(Ordering::Relaxed),
        }
    }

    /// Get fair value estimate accounting for adverse selection
    /// Returns adjusted price that accounts for toxic flow risk
    #[inline(always)]
    pub fn get_adverse_selection_adjusted_price(&self, toxicity_estimate: f64) -> i64 {
        let micro_price = self.last_micro_price.load(Ordering::Relaxed);
        let spread = self.best_ask.load(Ordering::Relaxed) - self.best_bid.load(Ordering::Relaxed);

        if micro_price == 0 || spread <= 0 {
            return micro_price;
        }

        // Adjust away from toxic side
        // Higher toxicity = larger adjustment toward safer side
        let adjustment = (spread as f64 * toxicity_estimate * 0.5) as i64;

        // If bid queue is much larger, risk is on bid side (toxic buy flow)
        let bid_size = self.bid_queue_size.load(Ordering::Relaxed);
        let ask_size = self.ask_queue_size.load(Ordering::Relaxed);

        if bid_size > ask_size {
            micro_price - adjustment // Adjust downward
        } else {
            micro_price + adjustment // Adjust upward
        }
    }

    /// Get current bid-ask spread in basis points
    #[inline(always)]
    pub fn get_spread_bps(&self) -> f64 {
        let bid = self.best_bid.load(Ordering::Relaxed);
        let ask = self.best_ask.load(Ordering::Relaxed);

        if bid == 0 || ask == 0 {
            return 0.0;
        }

        let mid = (bid + ask) as f64 / 2.0;
        let spread = (ask - bid) as f64;

        (spread / mid) * 10000.0 // Basis points
    }

    /// Get queue size ratio (bid/ask)
    #[inline(always)]
    pub fn get_queue_ratio(&self) -> f64 {
        let bid_size = self.bid_queue_size.load(Ordering::Relaxed) as f64;
        let ask_size = self.ask_queue_size.load(Ordering::Relaxed) as f64;

        if ask_size == 0.0 {
            return f64::INFINITY;
        }

        bid_size / ask_size
    }

    /// Get calculation statistics
    #[inline(always)]
    pub fn get_stats(&self) -> (u64, i64, u64) {
        (
            self.calc_count.load(Ordering::Relaxed),
            self.last_micro_price.load(Ordering::Relaxed),
            self.last_update_us.load(Ordering::Relaxed),
        )
    }

    /// Reset calculator state
    #[inline(always)]
    pub fn reset(&self) {
        self.best_bid.store(0, Ordering::Relaxed);
        self.best_ask.store(0, Ordering::Relaxed);
        self.bid_queue_size.store(0, Ordering::Relaxed);
        self.ask_queue_size.store(0, Ordering::Relaxed);
        self.last_micro_price.store(0, Ordering::Relaxed);
        self.last_update_us.store(0, Ordering::Relaxed);
        self.calc_count.store(0, Ordering::Relaxed);
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

/// Weighted mid-price calculator with multiple weighting schemes
pub struct WeightedMidPrice {
    calculator: MicroPriceCalculator,
    /// Tick size for rounding (scaled by 1e8)
    tick_size: i64,
}

impl WeightedMidPrice {
    pub fn new(tick_size: i64) -> Self {
        Self {
            calculator: MicroPriceCalculator::new(),
            tick_size,
        }
    }

    /// Calculate weighted mid-price with custom weights
    #[inline(always)]
    pub fn calculate(&self, bid_weight: f64, ask_weight: f64) -> i64 {
        let bid = self.calculator.best_bid.load(Ordering::Relaxed);
        let ask = self.calculator.best_ask.load(Ordering::Relaxed);

        if bid == 0 || ask == 0 {
            return 0;
        }

        let total_weight = bid_weight + ask_weight;
        if total_weight == 0.0 {
            return (bid + ask) / 2;
        }

        let weighted_price = (bid as f64 * ask_weight + ask as f64 * bid_weight) / total_weight;
        let rounded = ((weighted_price / self.tick_size as f64).round() * self.tick_size as f64) as i64;

        self.calculator.last_micro_price.store(rounded, Ordering::Relaxed);
        self.calculator.calc_count.fetch_add(1, Ordering::Relaxed);

        rounded
    }

    /// Calculate with volume-based weights (default behavior)
    #[inline(always)]
    pub fn calculate_volume_weighted(&self) -> i64 {
        let result = self.calculator.calculate_micro_price();
        result.price
    }

    /// Get underlying calculator for fine-tuned control
    #[inline(always)]
    pub fn get_calculator(&self) -> &MicroPriceCalculator {
        &self.calculator
    }

    /// Update tick size dynamically
    #[inline(always)]
    pub fn set_tick_size(&mut self, tick_size: i64) {
        self.tick_size = tick_size;
    }
}

/// Fair value estimator combining multiple pricing models
pub struct FairValueEstimator {
    micro_price_calc: MicroPriceCalculator,
    weighted_mid: WeightedMidPrice,
    /// Confidence level in current estimate (0.0 to 1.0, scaled by 1e6)
    confidence: AtomicU32,
}

impl FairValueEstimator {
    pub fn new(tick_size: i64) -> Self {
        Self {
            micro_price_calc: MicroPriceCalculator::new(),
            weighted_mid: WeightedMidPrice::new(tick_size),
            confidence: AtomicU32::new(500_000), // Start at 50% confidence
        }
    }

    /// Update market data and recalculate fair value
    #[inline(always)]
    pub fn update_and_estimate(&self, bid: i64, ask: i64, bid_size: i64, ask_size: i64) -> i64 {
        self.micro_price_calc.update_book(bid, ask, bid_size, ask_size);

        // Calculate multiple estimates
        let micro_price = self.micro_price_calc.calculate_micro_price().price;
        let weighted_price = self.weighted_mid.calculate_volume_weighted();

        // Blend estimates based on queue balance
        let queue_ratio = self.micro_price_calc.get_queue_ratio();
        let blend_weight = if queue_ratio > 1.0 {
            0.6 // More weight to micro-price when bid queue dominates
        } else {
            0.4 // More weight to weighted mid when ask queue dominates
        };

        let fair_value = ((micro_price as f64 * blend_weight) + (weighted_price as f64 * (1.0 - blend_weight))) as i64;

        // Update confidence based on spread and queue sizes
        let spread_bps = self.micro_price_calc.get_spread_bps();
        let new_confidence = if spread_bps < 5.0 && bid_size > 0 && ask_size > 0 {
            800_000 // High confidence in tight markets
        } else if spread_bps < 20.0 {
            600_000 // Medium confidence
        } else {
            300_000 // Low confidence in wide markets
        };

        self.confidence.store(new_confidence, Ordering::Relaxed);

        fair_value
    }

    /// Get current confidence level
    #[inline(always)]
    pub fn get_confidence(&self) -> f64 {
        self.confidence.load(Ordering::Relaxed) as f64 / 1_000_000.0
    }

    /// Get all fair value estimates
    #[inline(always)]
    pub fn get_all_estimates(&self) -> (i64, i64, i64) {
        let micro = self.micro_price_calc.calculate_micro_price().price;
        let weighted = self.weighted_mid.calculate_volume_weighted();
        let confidence = self.confidence.load(Ordering::Relaxed) as f64 / 1_000_000.0;
        let blended = ((micro as f64 * confidence) + (weighted as f64 * (1.0 - confidence))) as i64;

        (micro, weighted, blended)
    }
}

impl Default for MicroPriceCalculator {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for WeightedMidPrice {
    fn default() -> Self {
        Self::new(1) // Default tick size of 1
    }
}

impl Default for FairValueEstimator {
    fn default() -> Self {
        Self::new(1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_micro_price_basic() {
        let calc = MicroPriceCalculator::new();
        calc.update_book(10000, 10010, 1000000, 500000);

        let result = calc.calculate_micro_price();
        
        assert!(result.price > 0);
        assert_eq!(result.mid_price, 10005);
        assert_eq!(result.spread, 10);
    }

    #[test]
    fn test_queue_ratio_impact() {
        let calc = MicroPriceCalculator::new();
        
        // Large bid queue should push micro-price toward ask
        calc.update_book(10000, 10010, 10000000, 1000000);
        let result1 = calc.calculate_micro_price();
        
        // Large ask queue should push micro-price toward bid
        calc.update_book(10000, 10010, 1000000, 10000000);
        let result2 = calc.calculate_micro_price();
        
        assert!(result1.price > result2.price);
    }

    #[test]
    fn test_fair_value_estimator() {
        let estimator = FairValueEstimator::new(1);
        
        let fv = estimator.update_and_estimate(10000, 10010, 5000000, 5000000);
        
        assert!(fv > 0);
        assert!(estimator.get_confidence() > 0.0);
    }
}
