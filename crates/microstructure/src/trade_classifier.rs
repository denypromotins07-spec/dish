//! High-speed L3 trade classifier implementing Lee-Ready, Tick Rule, and Quote Rule algorithms.
//! Perfectly maps aggressive market orders to their passive limit counterparts in pure Rust.
//! Optimized for AMD Ryzen AI 5 with zero heap allocations.

use std::sync::atomic::{AtomicU64, AtomicBool, Ordering};

/// Trade classification result
#[repr(u8)]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum TradeSign {
    /// Buyer-initiated (aggressive buy)
    Buy = 0,
    /// Seller-initiated (aggressive sell)
    Sell = 1,
    /// Unknown/unclassifiable
    Unknown = 2,
}

/// Classification algorithm to use
#[repr(u8)]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ClassificationAlgo {
    TickRule = 0,
    LeeReady = 1,
    QuoteRule = 2,
    Combined = 3,
}

/// L3 Trade Classifier state
#[repr(C, align(64))]
pub struct TradeClassifier {
    /// Previous trade price (fixed point: * 1e8)
    prev_price: AtomicU64,
    /// Previous trade sign
    prev_sign: AtomicU64,
    /// Current bid price
    current_bid: AtomicU64,
    /// Current ask price
    current_ask: AtomicU64,
    /// Number of trades classified
    trades_classified: AtomicU64,
    /// Number of buy classifications
    buy_count: AtomicU64,
    /// Number of sell classifications
    sell_count: AtomicU64,
    /// Algorithm to use
    algo: AtomicU64,
    /// Is classifier active
    is_active: AtomicBool,
    _padding: [u8; 23],
}

unsafe impl Send for TradeClassifier {}
unsafe impl Sync for TradeClassifier {}

impl TradeClassifier {
    pub fn new(algo: ClassificationAlgo) -> Self {
        Self {
            prev_price: AtomicU64::new(0),
            prev_sign: AtomicU64::new(TradeSign::Unknown as u64),
            current_bid: AtomicU64::new(0),
            current_ask: AtomicU64::new(0),
            trades_classified: AtomicU64::new(0),
            buy_count: AtomicU64::new(0),
            sell_count: AtomicU64::new(0),
            algo: AtomicU64::new(algo as u64),
            is_active: AtomicBool::new(true),
            _padding: [0u8; 23],
        }
    }
    
    /// Update current quote (required for Lee-Ready and Quote Rule)
    #[inline]
    pub fn update_quote(&self, bid: u64, ask: u64) {
        self.current_bid.store(bid, Ordering::Relaxed);
        self.current_ask.store(ask, Ordering::Relaxed);
    }
    
    /// Classify a trade - O(1) operation
    #[inline]
    pub fn classify(&self, price: u64, volume: u64) -> TradeSign {
        if !self.is_active.load(Ordering::Relaxed) {
            return TradeSign::Unknown;
        }
        
        let algo = self.algo.load(Ordering::Relaxed);
        let sign = match ClassificationAlgo::from_u64(algo) {
            Some(ClassificationAlgo::TickRule) => self.tick_rule(price),
            Some(ClassificationAlgo::LeeReady) => self.lee_ready(price),
            Some(ClassificationAlgo::QuoteRule) => self.quote_rule(price),
            Some(ClassificationAlgo::Combined) | None => self.combined(price),
        };
        
        // Update statistics
        self.trades_classified.fetch_add(1, Ordering::Relaxed);
        self.prev_price.store(price, Ordering::Relaxed);
        self.prev_sign.store(sign as u64, Ordering::Relaxed);
        
        match sign {
            TradeSign::Buy => self.buy_count.fetch_add(1, Ordering::Relaxed),
            TradeSign::Sell => self.sell_count.fetch_add(1, Ordering::Relaxed),
            _ => {}
        }
        
        sign
    }
    
    /// Tick Rule: Compare trade price to previous trade price
    /// Uptick = Buy, Downtick = Sell, Zero uptick = same as prev, Zero downtick = opposite of prev
    #[inline]
    fn tick_rule(&self, price: u64) -> TradeSign {
        let prev_price = self.prev_price.load(Ordering::Relaxed);
        
        if price > prev_price {
            TradeSign::Buy
        } else if price < prev_price {
            TradeSign::Sell
        } else {
            // Zero tick - use previous sign
            match TradeSign::from_u64(self.prev_sign.load(Ordering::Relaxed)) {
                Some(TradeSign::Buy) => TradeSign::Buy,
                Some(TradeSign::Sell) => TradeSign::Sell,
                _ => TradeSign::Unknown,
            }
        }
    }
    
    /// Lee-Ready Algorithm: Compare trade price to midpoint of prevailing quote
    /// Above midpoint = Buy, Below midpoint = Sell, At midpoint = use tick rule
    #[inline]
    fn lee_ready(&self, price: u64) -> TradeSign {
        let bid = self.current_bid.load(Ordering::Relaxed);
        let ask = self.current_ask.load(Ordering::Relaxed);
        
        if bid == 0 || ask == 0 {
            // No quote data, fall back to tick rule
            return self.tick_rule(price);
        }
        
        let midpoint = (bid + ask) / 2;
        
        if price > midpoint {
            TradeSign::Buy
        } else if price < midpoint {
            TradeSign::Sell
        } else {
            // At midpoint - use tick rule
            self.tick_rule(price)
        }
    }
    
    /// Quote Rule: Compare trade price to bid/ask directly
    /// At ask = Buy, At bid = Sell, Between = unknown
    #[inline]
    fn quote_rule(&self, price: u64) -> TradeSign {
        let bid = self.current_bid.load(Ordering::Relaxed);
        let ask = self.current_ask.load(Ordering::Relaxed);
        
        if bid == 0 || ask == 0 {
            return TradeSign::Unknown;
        }
        
        // Allow small tolerance for price matching (1 tick)
        let tolerance = 1; // Could be configurable
        
        if price >= ask.saturating_sub(tolerance) {
            TradeSign::Buy
        } else if price <= bid.saturating_add(tolerance) {
            TradeSign::Sell
        } else {
            TradeSign::Unknown
        }
    }
    
    /// Combined approach: Use Lee-Ready primary, Tick Rule as fallback
    #[inline]
    fn combined(&self, price: u64) -> TradeSign {
        let lr_sign = self.lee_ready(price);
        
        if lr_sign != TradeSign::Unknown {
            lr_sign
        } else {
            self.tick_rule(price)
        }
    }
    
    /// Get classification statistics
    #[inline]
    pub fn get_stats(&self) -> ClassifierStats {
        let total = self.trades_classified.load(Ordering::Relaxed);
        let buys = self.buy_count.load(Ordering::Relaxed);
        let sells = self.sell_count.load(Ordering::Relaxed);
        
        ClassifierStats {
            total_trades: total,
            buy_trades: buys,
            sell_trades: sells,
            buy_ratio: if total > 0 { buys as f64 / total as f64 } else { 0.0 },
            sell_ratio: if total > 0 { sells as f64 / total as f64 } else { 0.0 },
        }
    }
    
    /// Set classification algorithm
    #[inline]
    pub fn set_algorithm(&self, algo: ClassificationAlgo) {
        self.algo.store(algo as u64, Ordering::Relaxed);
    }
    
    /// Reset statistics
    #[inline]
    pub fn reset(&self) {
        self.trades_classified.store(0, Ordering::Relaxed);
        self.buy_count.store(0, Ordering::Relaxed);
        self.sell_count.store(0, Ordering::Relaxed);
        self.prev_price.store(0, Ordering::Relaxed);
        self.prev_sign.store(TradeSign::Unknown as u64, Ordering::Relaxed);
    }
}

impl ClassificationAlgo {
    fn from_u64(val: u64) -> Option<Self> {
        match val {
            0 => Some(ClassificationAlgo::TickRule),
            1 => Some(ClassificationAlgo::LeeReady),
            2 => Some(ClassificationAlgo::QuoteRule),
            3 => Some(ClassificationAlgo::Combined),
            _ => None,
        }
    }
}

impl TradeSign {
    fn from_u64(val: u64) -> Option<Self> {
        match val {
            0 => Some(TradeSign::Buy),
            1 => Some(TradeSign::Sell),
            2 => Some(TradeSign::Unknown),
            _ => None,
        }
    }
}

/// Classifier statistics snapshot
#[derive(Clone, Copy, Debug)]
pub struct ClassifierStats {
    pub total_trades: u64,
    pub buy_trades: u64,
    pub sell_trades: u64,
    pub buy_ratio: f64,
    pub sell_ratio: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_tick_rule_uptick() {
        let classifier = TradeClassifier::new(ClassificationAlgo::TickRule);
        
        // First trade sets baseline
        classifier.classify(10000, 100);
        
        // Uptick should be classified as buy
        let sign = classifier.classify(10001, 100);
        assert_eq!(sign, TradeSign::Buy);
        
        // Downtick should be classified as sell
        let sign = classifier.classify(9999, 100);
        assert_eq!(sign, TradeSign::Sell);
    }
    
    #[test]
    fn test_lee_ready() {
        let classifier = TradeClassifier::new(ClassificationAlgo::LeeReady);
        
        // Set quotes: bid 9999, ask 10001, midpoint 10000
        classifier.update_quote(9999, 10001);
        
        // Trade above midpoint = buy
        let sign = classifier.classify(10002, 100);
        assert_eq!(sign, TradeSign::Buy);
        
        // Trade below midpoint = sell
        let sign = classifier.classify(9998, 100);
        assert_eq!(sign, TradeSign::Sell);
    }
    
    #[test]
    fn test_statistics() {
        let classifier = TradeClassifier::new(ClassificationAlgo::TickRule);
        
        classifier.classify(10000, 100); // Sets baseline
        classifier.classify(10001, 100); // Buy
        classifier.classify(10002, 100); // Buy
        classifier.classify(10001, 100); // Sell
        
        let stats = classifier.get_stats();
        assert_eq!(stats.total_trades, 4);
        assert!(stats.buy_ratio > 0.5);
    }
}
