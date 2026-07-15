//! High-speed Tick Rule and Lee-Ready algorithm implementation
//! Classifies trades as buyer- or seller-initiated with zero memory allocation
//! Optimized for nanosecond processing on AMD Ryzen architecture

use std::sync::atomic::{AtomicI64, AtomicU64, Ordering};

/// Tick classification result
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TickType {
    /// Uptick: price moved up from previous trade
    Uptick,
    /// Downtick: price moved down from previous trade
    Downtick,
    /// Zero uptick: same price but previous was higher
    ZeroUptick,
    /// Zero downtick: same price but previous was lower
    ZeroDowntick,
    /// Undefined: first trade or insufficient data
    Undefined,
}

/// Trade classification result (Lee-Ready)
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TradeClassification {
    /// Buyer-initiated (aggressive buy)
    BuyerInitiated,
    /// Seller-initiated (aggressive sell)
    SellerInitiated,
    /// Unclassified (insufficient data or ambiguous)
    Unclassified,
}

/// Lock-free tick rule processor
pub struct TickRuleProcessor {
    /// Previous trade price (scaled by 1e8 to avoid floats)
    prev_price: AtomicI64,
    /// Price before previous (for zero tick resolution)
    prev_prev_price: AtomicI64,
    /// Current tick type
    current_tick: AtomicU64,
    /// Trade count for statistics
    trade_count: AtomicU64,
    /// Uptick count
    uptick_count: AtomicU64,
    /// Downtick count
    downtick_count: AtomicU64,
}

impl TickRuleProcessor {
    pub fn new() -> Self {
        Self {
            prev_price: AtomicI64::new(0),
            prev_prev_price: AtomicI64::new(0),
            current_tick: AtomicU64::new(TickType::Undefined as u64),
            trade_count: AtomicU64::new(0),
            uptick_count: AtomicU64::new(0),
            downtick_count: AtomicU64::new(0),
        }
    }

    /// Process a new trade and classify the tick - zero allocation
    #[inline(always)]
    pub fn process_tick(&self, price: i64) -> TickType {
        let prev = self.prev_price.load(Ordering::Relaxed);
        let prev_prev = self.prev_prev_price.load(Ordering::Relaxed);

        let tick_type = if self.trade_count.load(Ordering::Relaxed) == 0 {
            TickType::Undefined
        } else if price > prev {
            TickType::Uptick
        } else if price < prev {
            TickType::Downtick
        } else {
            // Zero tick - resolve using previous price movement
            if prev > prev_prev {
                TickType::ZeroUptick
            } else if prev < prev_prev {
                TickType::ZeroDowntick
            } else {
                TickType::Undefined
            }
        };

        // Update state atomically
        self.prev_prev_price.store(prev, Ordering::Relaxed);
        self.prev_price.store(price, Ordering::Relaxed);
        self.current_tick.store(tick_type as u64, Ordering::Relaxed);
        self.trade_count.fetch_add(1, Ordering::Relaxed);

        // Update counters
        match tick_type {
            TickType::Uptick | TickType::ZeroUptick => {
                self.uptick_count.fetch_add(1, Ordering::Relaxed);
            }
            TickType::Downtick | TickType::ZeroDowntick => {
                self.downtick_count.fetch_add(1, Ordering::Relaxed);
            }
            _ => {}
        }

        tick_type
    }

    /// Get current tick type
    #[inline(always)]
    pub fn get_current_tick(&self) -> TickType {
        match self.current_tick.load(Ordering::Relaxed) {
            0 => TickType::Uptick,
            1 => TickType::Downtick,
            2 => TickType::ZeroUptick,
            3 => TickType::ZeroDowntick,
            _ => TickType::Undefined,
        }
    }

    /// Get tick statistics
    #[inline(always)]
    pub fn get_stats(&self) -> (u64, u64, u64) {
        (
            self.trade_count.load(Ordering::Relaxed),
            self.uptick_count.load(Ordering::Relaxed),
            self.downtick_count.load(Ordering::Relaxed),
        )
    }

    /// Calculate tick ratio (upticks / total)
    #[inline(always)]
    pub fn get_tick_ratio(&self) -> f64 {
        let total = self.trade_count.load(Ordering::Relaxed);
        if total == 0 {
            return 0.0;
        }
        self.uptick_count.load(Ordering::Relaxed) as f64 / total as f64
    }

    /// Reset processor state
    #[inline(always)]
    pub fn reset(&self) {
        self.prev_price.store(0, Ordering::Relaxed);
        self.prev_prev_price.store(0, Ordering::Relaxed);
        self.current_tick.store(TickType::Undefined as u64, Ordering::Relaxed);
        self.trade_count.store(0, Ordering::Relaxed);
        self.uptick_count.store(0, Ordering::Relaxed);
        self.downtick_count.store(0, Ordering::Relaxed);
    }
}

/// Lee-Ready algorithm implementation for trade classification
pub struct LeeReadyClassifier {
    /// Previous trade price
    prev_price: AtomicI64,
    /// Previous mid-price at time of trade
    prev_mid_price: AtomicI64,
    /// Classification count
    classification_count: AtomicU64,
    /// Buyer-initiated count
    buyer_count: AtomicU64,
    /// Seller-initiated count
    seller_count: AtomicU64,
}

impl LeeReadyClassifier {
    pub fn new() -> Self {
        Self {
            prev_price: AtomicI64::new(0),
            prev_mid_price: AtomicI64::new(0),
            classification_count: AtomicU64::new(0),
            buyer_count: AtomicU64::new(0),
            seller_count: AtomicU64::new(0),
        }
    }

    /// Classify a trade using Lee-Ready algorithm
    /// Compares trade price to prevailing mid-price
    #[inline(always)]
    pub fn classify_trade(&self, trade_price: i64, mid_price: i64) -> TradeClassification {
        let prev_mid = self.prev_mid_price.load(Ordering::Relaxed);
        
        let classification = if prev_mid == 0 {
            // First trade - cannot classify
            TradeClassification::Unclassified
        } else if trade_price > prev_mid {
            // Trade above mid-price = buyer initiated
            TradeClassification::BuyerInitiated
        } else if trade_price < prev_mid {
            // Trade below mid-price = seller initiated
            TradeClassification::SellerInitiated
        } else {
            // Trade at mid-price - use tick test
            let prev_price = self.prev_price.load(Ordering::Relaxed);
            if prev_price == 0 {
                TradeClassification::Unclassified
            } else if trade_price > prev_price {
                TradeClassification::BuyerInitiated
            } else if trade_price < prev_price {
                TradeClassification::SellerInitiated
            } else {
                TradeClassification::Unclassified
            }
        };

        // Update state
        self.prev_price.store(trade_price, Ordering::Relaxed);
        self.prev_mid_price.store(mid_price, Ordering::Relaxed);
        self.classification_count.fetch_add(1, Ordering::Relaxed);

        match classification {
            TradeClassification::BuyerInitiated => {
                self.buyer_count.fetch_add(1, Ordering::Relaxed);
            }
            TradeClassification::SellerInitiated => {
                self.seller_count.fetch_add(1, Ordering::Relaxed);
            }
            _ => {}
        }

        classification
    }

    /// Classify trade with explicit bid/ask comparison
    #[inline(always)]
    pub fn classify_trade_ba(&self, trade_price: i64, bid: i64, ask: i64) -> TradeClassification {
        if trade_price >= ask {
            self.buyer_count.fetch_add(1, Ordering::Relaxed);
            self.classification_count.fetch_add(1, Ordering::Relaxed);
            TradeClassification::BuyerInitiated
        } else if trade_price <= bid {
            self.seller_count.fetch_add(1, Ordering::Relaxed);
            self.classification_count.fetch_add(1, Ordering::Relaxed);
            TradeClassification::SellerInitiated
        } else {
            // Trade inside spread - use mid-price comparison
            let mid = (bid + ask) / 2;
            self.classification_count.fetch_add(1, Ordering::Relaxed);
            if trade_price > mid {
                self.buyer_count.fetch_add(1, Ordering::Relaxed);
                TradeClassification::BuyerInitiated
            } else if trade_price < mid {
                self.seller_count.fetch_add(1, Ordering::Relaxed);
                TradeClassification::SellerInitiated
            } else {
                TradeClassification::Unclassified
            }
        }
    }

    /// Get classification statistics
    #[inline(always)]
    pub fn get_stats(&self) -> (u64, u64, u64) {
        (
            self.classification_count.load(Ordering::Relaxed),
            self.buyer_count.load(Ordering::Relaxed),
            self.seller_count.load(Ordering::Relaxed),
        )
    }

    /// Calculate buyer initiation ratio
    #[inline(always)]
    pub fn get_buyer_ratio(&self) -> f64 {
        let total = self.classification_count.load(Ordering::Relaxed);
        if total == 0 {
            return 0.0;
        }
        self.buyer_count.load(Ordering::Relaxed) as f64 / total as f64
    }

    /// Get order flow imbalance from classifications
    #[inline(always)]
    pub fn get_flow_imbalance(&self) -> f64 {
        let buyer = self.buyer_count.load(Ordering::Relaxed) as f64;
        let seller = self.seller_count.load(Ordering::Relaxed) as f64;
        let total = buyer + seller;
        
        if total == 0.0 {
            return 0.0;
        }
        (buyer - seller) / total
    }

    /// Reset classifier state
    #[inline(always)]
    pub fn reset(&self) {
        self.prev_price.store(0, Ordering::Relaxed);
        self.prev_mid_price.store(0, Ordering::Relaxed);
        self.classification_count.store(0, Ordering::Relaxed);
        self.buyer_count.store(0, Ordering::Relaxed);
        self.seller_count.store(0, Ordering::Relaxed);
    }
}

/// Combined tick rule and Lee-Ready processor
pub struct TradeAnalyzer {
    tick_processor: TickRuleProcessor,
    lr_classifier: LeeReadyClassifier,
    /// Agreement count between methods
    agreement_count: AtomicU64,
}

impl TradeAnalyzer {
    pub fn new() -> Self {
        Self {
            tick_processor: TickRuleProcessor::new(),
            lr_classifier: LeeReadyClassifier::new(),
            agreement_count: AtomicU64::new(0),
        }
    }

    /// Process trade with both methods and compare results
    #[inline(always)]
    pub fn analyze_trade(&self, price: i64, mid_price: i64, bid: i64, ask: i64) -> (TickType, TradeClassification) {
        let tick_type = self.tick_processor.process_tick(price);
        let lr_class = self.lr_classifier.classify_trade_ba(price, bid, ask);

        // Check agreement between methods
        let tick_is_buy = matches!(tick_type, TickType::Uptick | TickType::ZeroUptick);
        let lr_is_buy = matches!(lr_class, TradeClassification::BuyerInitiated);

        if tick_is_buy == lr_is_buy {
            self.agreement_count.fetch_add(1, Ordering::Relaxed);
        }

        (tick_type, lr_class)
    }

    /// Get combined statistics
    #[inline(always)]
    pub fn get_combined_stats(&self) -> (f64, f64, f64) {
        let tick_ratio = self.tick_processor.get_tick_ratio();
        let buyer_ratio = self.lr_classifier.get_buyer_ratio();
        let total = self.tick_processor.get_stats().0;
        let agreement = self.agreement_count.load(Ordering::Relaxed);
        let agreement_ratio = if total == 0 { 0.0 } else { agreement as f64 / total as f64 };

        (tick_ratio, buyer_ratio, agreement_ratio)
    }

    /// Access to individual components
    #[inline(always)]
    pub fn get_tick_processor(&self) -> &TickRuleProcessor {
        &self.tick_processor
    }

    #[inline(always)]
    pub fn get_lr_classifier(&self) -> &LeeReadyClassifier {
        &self.lr_classifier
    }

    /// Reset all components
    #[inline(always)]
    pub fn reset(&self) {
        self.tick_processor.reset();
        self.lr_classifier.reset();
        self.agreement_count.store(0, Ordering::Relaxed);
    }
}

impl Default for TickRuleProcessor {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for LeeReadyClassifier {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for TradeAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tick_rule_uptick() {
        let processor = TickRuleProcessor::new();
        processor.process_tick(10000);
        processor.process_tick(10010);
        
        assert_eq!(processor.get_current_tick(), TickType::Uptick);
    }

    #[test]
    fn test_lee_ready_classification() {
        let classifier = LeeReadyClassifier::new();
        
        // Set initial mid-price
        classifier.classify_trade(10000, 9995);
        // Trade above mid-price should be buyer initiated
        let result = classifier.classify_trade(10005, 10000);
        assert_eq!(result, TradeClassification::BuyerInitiated);
    }

    #[test]
    fn test_combined_analysis() {
        let analyzer = TradeAnalyzer::new();
        
        let (tick, lr) = analyzer.analyze_trade(10000, 9995, 9990, 10000);
        
        // Both should indicate buying pressure
        assert!(matches!(tick, TickType::Undefined | TickType::Uptick));
        assert_eq!(lr, TradeClassification::BuyerInitiated);
    }
}
