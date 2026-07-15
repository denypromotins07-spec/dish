//! Implementation Shortfall and adaptive execution algorithms.
//! Dynamically adjust order aggression (limit vs. market) based on real-time order book imbalance, CVD, and alpha decay.

use std::sync::atomic::{AtomicF64, AtomicBool, AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

/// Order book imbalance calculation
#[derive(Debug, Clone)]
pub struct OrderBookImbalance {
    pub bid_volume: f64,
    pub ask_volume: f64,
    pub imbalance_ratio: f64, // -1.0 to 1.0 (positive = more bids)
    pub weighted_imbalance: f64, // Distance-weighted
}

/// Cumulative Volume Delta (CVD) tracker
pub struct CvdTracker {
    /// Cumulative buy volume
    buy_volume: AtomicF64,
    /// Cumulative sell volume
    sell_volume: AtomicF64,
    /// CVD value (buy - sell)
    cvd_value: AtomicF64,
    /// CVD change rate (derivative)
    cvd_rate: AtomicF64,
    /// Last update timestamp
    last_update_ts: AtomicU64,
}

/// Alpha decay model for urgency calculation
pub struct AlphaDecayModel {
    /// Current alpha estimate (expected return)
    alpha: AtomicF64,
    /// Alpha half-life in seconds
    half_life_sec: f64,
    /// Decay constant
    decay_constant: f64,
    /// Time since signal generation
    signal_age_sec: AtomicF64,
}

/// Adaptive execution strategy result
#[derive(Debug, Clone)]
pub struct ExecutionStrategy {
    /// Aggression level: 0.0 (passive) to 1.0 (aggressive)
    pub aggression_level: f64,
    /// Recommended order type
    pub order_type: ExecutionOrderType,
    /// Price offset from mid (negative = better price for limit)
    pub price_offset_bps: i32,
    /// Urgency score
    pub urgency_score: f64,
    /// Expected slippage bps
    pub expected_slippage_bps: f64,
    /// Confidence in execution
    pub confidence: f64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ExecutionOrderType {
    LimitPassive,    // Post-only, far from spread
    LimitNormal,     // At best bid/ask
    LimitAggressive, // Crossing spread slightly
    MarketImmediate, // Immediate market order
}

/// Main adaptive execution engine
pub struct AdaptiveExecutor {
    /// Target quantity to execute
    target_quantity: AtomicF64,
    /// Executed quantity
    executed_quantity: AtomicF64,
    /// Average execution price
    avg_price: AtomicF64,
    /// Benchmark price (arrival price)
    benchmark_price: AtomicF64,
    /// Implementation shortfall (bps)
    impl_shortfall_bps: AtomicF64,
    /// Is buy order
    is_buy: AtomicBool,
    /// Active flag
    is_active: AtomicBool,
    /// CVD tracker
    cvd_tracker: CvdTracker,
    /// Alpha decay model
    alpha_model: AlphaDecayModel,
    /// Minimum urgency threshold
    min_urgency_threshold: f64,
    /// Maximum urgency threshold
    max_urgency_threshold: f64,
}

impl CvdTracker {
    /// Create new CVD tracker
    pub fn new() -> Self {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();
        
        Self {
            buy_volume: AtomicF64::new(0.0),
            sell_volume: AtomicF64::new(0.0),
            cvd_value: AtomicF64::new(0.0),
            cvd_rate: AtomicF64::new(0.0),
            last_update_ts: AtomicU64::new(now),
        }
    }

    /// Record a trade
    #[inline(always)]
    pub fn record_trade(&self, quantity: f64, is_buy_aggressor: bool) {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();
        
        let last_ts = self.last_update_ts.load(Ordering::Relaxed);
        let elapsed = (now - last_ts) as f64;
        
        if is_buy_aggressor {
            self.buy_volume.fetch_add(quantity, Ordering::Relaxed);
        } else {
            self.sell_volume.fetch_add(quantity, Ordering::Relaxed);
        }
        
        // Update CVD
        let buy_vol = self.buy_volume.load(Ordering::Relaxed);
        let sell_vol = self.sell_volume.load(Ordering::Relaxed);
        let new_cvd = buy_vol - sell_vol;
        self.cvd_value.store(new_cvd, Ordering::Relaxed);
        
        // Update CVD rate (change per second)
        if elapsed > 0.0 && elapsed < 60.0 {
            let old_cvd = self.cvd_value.load(Ordering::Relaxed);
            let rate = (new_cvd - old_cvd) / elapsed;
            self.cvd_rate.store(rate, Ordering::Relaxed);
        }
        
        self.last_update_ts.store(now, Ordering::Relaxed);
    }

    /// Get current CVD value
    #[inline(always)]
    pub fn get_cvd(&self) -> f64 {
        self.cvd_value.load(Ordering::Relaxed)
    }

    /// Get CVD rate of change
    #[inline(always)]
    pub fn get_cvd_rate(&self) -> f64 {
        self.cvd_rate.load(Ordering::Relaxed)
    }

    /// Get normalized CVD (-1 to 1)
    pub fn get_normalized_cvd(&self) -> f64 {
        let buy = self.buy_volume.load(Ordering::Relaxed);
        let sell = self.sell_volume.load(Ordering::Relaxed);
        let total = buy + sell;
        
        if total < 0.001 {
            return 0.0;
        }
        
        (buy - sell) / total
    }

    /// Reset CVD counters
    #[inline(always)]
    pub fn reset(&self) {
        self.buy_volume.store(0.0, Ordering::Relaxed);
        self.sell_volume.store(0.0, Ordering::Relaxed);
        self.cvd_value.store(0.0, Ordering::Relaxed);
        self.cvd_rate.store(0.0, Ordering::Relaxed);
    }
}

impl AlphaDecayModel {
    /// Create new alpha decay model
    /// 
    /// # Arguments
    /// * `initial_alpha` - Initial expected return (bps)
    /// * `half_life_sec` - Time for alpha to decay by half
    pub fn new(initial_alpha: f64, half_life_sec: f64) -> Self {
        let decay_constant = std::f64::consts::LN_2 / half_life_sec.max(1.0);
        
        Self {
            alpha: AtomicF64::new(initial_alpha),
            half_life_sec,
            decay_constant,
            signal_age_sec: AtomicF64::new(0.0),
        }
    }

    /// Update signal age
    #[inline(always)]
    pub fn update_age(&self, age_seconds: f64) {
        self.signal_age_sec.store(age_seconds, Ordering::Relaxed);
    }

    /// Get current decayed alpha
    pub fn get_current_alpha(&self) -> f64 {
        let age = self.signal_age_sec.load(Ordering::Relaxed);
        let initial = self.alpha.load(Ordering::Relaxed);
        
        // Exponential decay: alpha(t) = alpha_0 * e^(-lambda * t)
        initial * (-self.decay_constant * age).exp()
    }

    /// Set new alpha signal
    #[inline(always)]
    pub fn set_new_alpha(&self, new_alpha: f64) {
        self.alpha.store(new_alpha, Ordering::Relaxed);
        self.signal_age_sec.store(0.0, Ordering::Relaxed);
    }

    /// Get remaining alpha as percentage
    pub fn get_alpha_remaining_pct(&self) -> f64 {
        let age = self.signal_age_sec.load(Ordering::Relaxed);
        100.0 * (-self.decay_constant * age).exp()
    }
}

impl AdaptiveExecutor {
    /// Create new adaptive executor
    pub fn new(
        target_quantity: f64,
        benchmark_price: f64,
        is_buy: bool,
        alpha_half_life_sec: f64,
    ) -> Self {
        Self {
            target_quantity: AtomicF64::new(target_quantity),
            executed_quantity: AtomicF64::new(0.0),
            avg_price: AtomicF64::new(0.0),
            benchmark_price: AtomicF64::new(benchmark_price),
            impl_shortfall_bps: AtomicF64::new(0.0),
            is_buy: AtomicBool::new(is_buy),
            is_active: AtomicBool::new(false),
            cvd_tracker: CvdTracker::new(),
            alpha_model: AlphaDecayModel::new(50.0, alpha_half_life_sec), // 50 bps initial alpha
            min_urgency_threshold: 0.2,
            max_urgency_threshold: 0.9,
        }
    }

    /// Start execution
    #[inline(always)]
    pub fn start(&self) {
        self.is_active.store(true, Ordering::Relaxed);
    }

    /// Stop execution
    #[inline(always)]
    pub fn stop(&self) {
        self.is_active.store(false, Ordering::Relaxed);
    }

    /// Calculate optimal execution strategy based on market conditions
    pub fn calculate_strategy(
        &self,
        order_book_imbalance: &OrderBookImbalance,
        current_price: f64,
        spread_bps: f64,
        volatility: f64,
    ) -> ExecutionStrategy {
        let is_buy = self.is_buy.load(Ordering::Relaxed);
        let remaining = self.get_remaining();
        let total = self.target_quantity.load(Ordering::Relaxed);
        let progress = if total > 0.0 { self.executed_quantity.load(Ordering::Relaxed) / total } else { 0.0 };
        
        // Factor 1: Alpha decay urgency
        let current_alpha = self.alpha_model.get_current_alpha();
        let alpha_urgency = (current_alpha.abs() / 100.0).min(1.0); // Normalize
        
        // Factor 2: Order book imbalance
        let ob_urgency = if is_buy {
            // Buying into strong bids = less urgency, weak bids = more urgency
            (1.0 - order_book_imbalance.imbalance_ratio) / 2.0
        } else {
            // Selling into strong asks = less urgency
            (1.0 + order_book_imbalance.imbalance_ratio) / 2.0
        };
        
        // Factor 3: CVD momentum
        let cvd_normalized = self.cvd_tracker.get_normalized_cvd();
        let cvd_urgency = if is_buy {
            // Positive CVD (buying pressure) = more urgency to buy
            (cvd_normalized + 1.0) / 2.0
        } else {
            // Negative CVD (selling pressure) = more urgency to sell
            (1.0 - cvd_normalized) / 2.0
        };
        
        // Factor 4: Progress-based urgency (increase as we fall behind)
        let time_urgency = progress; // Simple linear increase
        
        // Factor 5: Volatility urgency (higher vol = more urgency to complete)
        let vol_urgency = (volatility * 10.0).min(1.0);
        
        // Weighted combination
        let weights = [0.30, 0.25, 0.20, 0.15, 0.10]; // Alpha, OB, CVD, Progress, Vol
        let urgencies = [alpha_urgency, ob_urgency, cvd_urgency, time_urgency, vol_urgency];
        
        let mut urgency_score = 0.0;
        for (w, u) in weights.iter().zip(urgencies.iter()) {
            urgency_score += w * u;
        }
        
        // Clamp urgency
        urgency_score = urgency_score.clamp(self.min_urgency_threshold, self.max_urgency_threshold);
        
        // Determine aggression level based on urgency
        let aggression_level = urgency_score;
        
        // Select order type based on aggression
        let order_type = match aggression_level {
            x if x < 0.25 => ExecutionOrderType::LimitPassive,
            x if x < 0.50 => ExecutionOrderType::LimitNormal,
            x if x < 0.75 => ExecutionOrderType::LimitAggressive,
            _ => ExecutionOrderType::MarketImmediate,
        };
        
        // Calculate price offset
        let price_offset_bps = match order_type {
            ExecutionOrderType::LimitPassive => -(spread_bps as i32) / 2,
            ExecutionOrderType::LimitNormal => 0,
            ExecutionOrderType::LimitAggressive => (spread_bps as i32) / 4,
            ExecutionOrderType::MarketImmediate => (spread_bps as i32) / 2,
        };
        
        // Estimate slippage
        let base_slippage = spread_bps / 2.0;
        let impact_slippage = (remaining / total).max(0.0) * volatility * 5.0;
        let expected_slippage_bps = base_slippage + impact_slippage;
        
        // Confidence based on signal strength and market conditions
        let confidence = (1.0 - urgency_score) * (current_alpha.abs() / 100.0).min(1.0);
        
        ExecutionStrategy {
            aggression_level,
            order_type,
            price_offset_bps,
            urgency_score,
            expected_slippage_bps,
            confidence,
        }
    }

    /// Record execution fill
    #[inline(always)]
    pub fn record_fill(&self, quantity: f64, price: f64) {
        let executed = self.executed_quantity.load(Ordering::Relaxed);
        let avg = self.avg_price.load(Ordering::Relaxed);
        
        // Update average price
        let new_executed = executed + quantity;
        let new_avg = if new_executed > 0.0 {
            ((avg * executed) + (price * quantity)) / new_executed
        } else {
            price
        };
        
        self.executed_quantity.store(new_executed, Ordering::Relaxed);
        self.avg_price.store(new_avg, Ordering::Relaxed);
        
        // Update implementation shortfall
        let benchmark = self.benchmark_price.load(Ordering::Relaxed);
        let is_buy = self.is_buy.load(Ordering::Relaxed);
        
        let shortfall_bps = if is_buy {
            ((new_avg - benchmark) / benchmark) * 10000.0
        } else {
            ((benchmark - new_avg) / benchmark) * 10000.0
        };
        
        self.impl_shortfall_bps.store(shortfall_bps, Ordering::Relaxed);
    }

    /// Get remaining quantity
    #[inline(always)]
    pub fn get_remaining(&self) -> f64 {
        self.target_quantity.load(Ordering::Relaxed) - self.executed_quantity.load(Ordering::Relaxed)
    }

    /// Get implementation shortfall in bps
    #[inline(always)]
    pub fn get_impl_shortfall_bps(&self) -> f64 {
        self.impl_shortfall_bps.load(Ordering::Relaxed)
    }

    /// Get execution status
    pub fn get_status(&self) -> AdaptiveExecutionStatus {
        let total = self.target_quantity.load(Ordering::Relaxed);
        let executed = self.executed_quantity.load(Ordering::Relaxed);
        
        AdaptiveExecutionStatus {
            target_quantity: total,
            executed_quantity: executed,
            remaining_quantity: total - executed,
            average_price: self.avg_price.load(Ordering::Relaxed),
            benchmark_price: self.benchmark_price.load(Ordering::Relaxed),
            implementation_shortfall_bps: self.impl_shortfall_bps.load(Ordering::Relaxed),
            progress_pct: if total > 0.0 { executed / total * 100.0 } else { 0.0 },
            current_alpha: self.alpha_model.get_current_alpha(),
            alpha_remaining_pct: self.alpha_model.get_alpha_remaining_pct(),
        }
    }

    /// Record CVD trade
    #[inline(always)]
    pub fn record_cvd_trade(&self, quantity: f64, is_buy_aggressor: bool) {
        self.cvd_tracker.record_trade(quantity, is_buy_aggressor);
    }

    /// Update alpha signal
    #[inline(always)]
    pub fn update_alpha(&self, new_alpha_bps: f64) {
        self.alpha_model.set_new_alpha(new_alpha_bps);
    }

    /// Update signal age
    #[inline(always)]
    pub fn update_signal_age(&self, age_seconds: f64) {
        self.alpha_model.update_age(age_seconds);
    }
}

/// Execution status
#[derive(Debug, Clone)]
pub struct AdaptiveExecutionStatus {
    pub target_quantity: f64,
    pub executed_quantity: f64,
    pub remaining_quantity: f64,
    pub average_price: f64,
    pub benchmark_price: f64,
    pub implementation_shortfall_bps: f64,
    pub progress_pct: f64,
    pub current_alpha: f64,
    pub alpha_remaining_pct: f64,
}

/// Calculate order book imbalance
pub fn calculate_orderbook_imbalance(bids: &[(f64, f64)], asks: &[(f64, f64)], levels: usize) -> OrderBookImbalance {
    let mut bid_volume = 0.0;
    let mut ask_volume = 0.0;
    let mut weighted_bid = 0.0;
    let mut weighted_ask = 0.0;
    
    for (i, (_, qty)) in bids.iter().enumerate().take(levels) {
        let weight = 1.0 / (i + 1) as f64;
        bid_volume += qty;
        weighted_bid += qty * weight;
    }
    
    for (i, (_, qty)) in asks.iter().enumerate().take(levels) {
        let weight = 1.0 / (i + 1) as f64;
        ask_volume += qty;
        weighted_ask += qty * weight;
    }
    
    let total = bid_volume + ask_volume;
    let imbalance_ratio = if total > 0.0 {
        (bid_volume - ask_volume) / total
    } else {
        0.0
    };
    
    let weighted_total = weighted_bid + weighted_ask;
    let weighted_imbalance = if weighted_total > 0.0 {
        (weighted_bid - weighted_ask) / weighted_total
    } else {
        0.0
    };
    
    OrderBookImbalance {
        bid_volume,
        ask_volume,
        imbalance_ratio,
        weighted_imbalance,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_adaptive_executor() {
        let executor = AdaptiveExecutor::new(10.0, 50000.0, true, 60.0); // Buy 10 BTC, 60s half-life
        executor.start();
        
        let imbalance = OrderBookImbalance {
            bid_volume: 100.0,
            ask_volume: 50.0,
            imbalance_ratio: 0.33,
            weighted_imbalance: 0.25,
        };
        
        let strategy = executor.calculate_strategy(&imbalance, 50000.0, 10.0, 0.02);
        
        assert!(strategy.aggression_level >= 0.0 && strategy.aggression_level <= 1.0);
        assert!(strategy.urgency_score >= 0.0 && strategy.urgency_score <= 1.0);
    }

    #[test]
    fn test_alpha_decay() {
        let alpha_model = AlphaDecayModel::new(100.0, 60.0); // 100 bps, 60s half-life
        
        alpha_model.update_age(60.0);
        let decayed = alpha_model.get_current_alpha();
        
        assert!((decayed - 50.0).abs() < 1.0); // Should be ~50 bps after one half-life
    }

    #[test]
    fn test_cvd_tracker() {
        let cvd = CvdTracker::new();
        
        cvd.record_trade(10.0, true);  // Buy
        cvd.record_trade(5.0, false);  // Sell
        cvd.record_trade(15.0, true);  // Buy
        
        assert!((cvd.get_cvd() - 20.0).abs() < 0.001); // 25 - 5 = 20
        assert!(cvd.get_normalized_cvd() > 0.0);
    }

    #[test]
    fn test_impl_shortfall_calculation() {
        let executor = AdaptiveExecutor::new(10.0, 50000.0, true, 60.0);
        executor.start();
        
        // Fill at worse price
        executor.record_fill(5.0, 50050.0);
        
        let shortfall = executor.get_impl_shortfall_bps();
        assert!(shortfall > 0.0); // Positive shortfall for buying above benchmark
    }
}
