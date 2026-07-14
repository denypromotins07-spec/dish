//! Liquidity Engineering: Pools, Equal Highs/Lows, Stop Hunts, and Sweeps
//! Analyzes order book imbalances and aggressive execution footprints

use crossbeam::atomic::AtomicCell;
use std::collections::VecDeque;

/// Liquidity Pool type
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LiquidityType {
    BuySideLiquidity,
    SellSideLiquidity,
    EqualHighs,
    EqualLows,
    StopCluster,
}

/// Liquidity Pool representation
#[repr(C, align(64))]
#[derive(Clone, Copy)]
pub struct LiquidityPool {
    pub pool_type: LiquidityType,
    pub price_level: f64,
    pub estimated_size: f64, // Estimated stop/liquidity size
    pub touch_count: u32,
    pub last_touched_ns: u64,
    pub created_at_ns: u64,
    pub swept: bool,
}

impl LiquidityPool {
    #[inline]
    pub fn new(pool_type: LiquidityType, price: f64, size: f64, timestamp_ns: u64) -> Self {
        Self {
            pool_type,
            price_level: price,
            estimated_size: size,
            touch_count: 0,
            last_touched_ns: 0,
            created_at_ns: timestamp_ns,
            swept: false,
        }
    }

    #[inline]
    pub fn touch(&mut self, timestamp_ns: u64) {
        self.touch_count += 1;
        self.last_touched_ns = timestamp_ns;
    }

    #[inline]
    pub fn mark_swept(&mut self) {
        self.swept = true;
    }

    #[inline]
    pub fn proximity(&self, price: f64) -> f64 {
        (self.price_level - price).abs()
    }
}

/// Equal Highs/Lows detector
#[repr(C, align(64))]
pub struct EqualLevelsDetector {
    highs_history: VecDeque<f64>,
    lows_history: VecDeque<f64>,
    max_history: usize,
    tolerance_bps: f64, // Tolerance in basis points
    min_touches: u32,
}

impl EqualLevelsDetector {
    pub fn new(max_history: usize, tolerance_bps: f64, min_touches: u32) -> Self {
        Self {
            highs_history: VecDeque::with_capacity(max_history),
            lows_history: VecDeque::with_capacity(max_history),
            max_history,
            tolerance_bps,
            min_touches,
        }
    }

    #[inline]
    pub fn update(&mut self, high: f64, low: f64) -> (Option<f64>, Option<f64>) {
        // Add to history
        if self.highs_history.len() >= self.max_history {
            self.highs_history.pop_front();
        }
        self.highs_history.push_back(high);

        if self.lows_history.len() >= self.max_history {
            self.lows_history.pop_front();
        }
        self.lows_history.push_back(low);

        // Check for equal highs
        let equal_high = self.detect_equal_levels(&self.highs_history, true);
        
        // Check for equal lows
        let equal_low = self.detect_equal_levels(&self.lows_history, false);

        (equal_high, equal_low)
    }

    #[inline]
    fn detect_equal_levels(&self, levels: &VecDeque<f64>, is_high: bool) -> Option<f64> {
        if levels.len() < self.min_touches as usize {
            return None;
        }

        let latest = *levels.back().unwrap();
        let tolerance = latest * self.tolerance_bps / 10000.0;

        let mut matching_count = 0;
        let mut avg_price = 0.0;

        for &level in levels.iter().rev() {
            if (level - latest).abs() <= tolerance {
                matching_count += 1;
                avg_price += level;
            }
        }

        if matching_count >= self.min_touches {
            Some(avg_price / matching_count as f64)
        } else {
            None
        }
    }

    #[inline]
    pub fn clear(&mut self) {
        self.highs_history.clear();
        self.lows_history.clear();
    }
}

/// Stop Hunt detector
#[repr(C, align(64))]
pub struct StopHuntDetector {
    recent_highs: VecDeque<(f64, u64)>,
    recent_lows: VecDeque<(f64, u64)>,
    hunt_threshold_bps: f64,
    reversal_threshold_bps: f64,
    window_ns: u64,
}

impl StopHuntDetector {
    pub fn new(hunt_threshold_bps: f64, reversal_threshold_bps: f64, window_ns: u64) -> Self {
        Self {
            recent_highs: VecDeque::with_capacity(50),
            recent_lows: VecDeque::with_capacity(50),
            hunt_threshold_bps,
            reversal_threshold_bps,
            window_ns,
        }
    }

    /// Detect stop hunt pattern
    #[inline]
    pub fn detect(&mut self, high: f64, low: f64, close: f64, timestamp_ns: u64) -> Option<LiquidityPool> {
        // Clean old entries
        self.cleanup(timestamp_ns);

        // Check for bullish stop hunt (sweep of lows then reversal)
        if let Some(&(_, prev_low_ts)) = self.recent_lows.front() {
            if let Some(&(prev_low, _)) = self.recent_lows.front() {
                let sweep_threshold = prev_low * self.hunt_threshold_bps / 10000.0;
                
                if low < prev_low - sweep_threshold {
                    // Price swept below previous low
                    let reversal_threshold = (prev_low - low) * self.reversal_threshold_bps / 10000.0;
                    
                    if close > prev_low + reversal_threshold {
                        // Strong reversal - likely stop hunt
                        let pool = LiquidityPool::new(
                            LiquidityType::StopCluster,
                            low,
                            (prev_low - low).abs(),
                            timestamp_ns,
                        );
                        return Some(pool);
                    }
                }
            }
        }

        // Check for bearish stop hunt (sweep of highs then reversal)
        if let Some(&(prev_high, _)) = self.recent_highs.front() {
            let sweep_threshold = prev_high * self.hunt_threshold_bps / 10000.0;
            
            if high > prev_high + sweep_threshold {
                let reversal_threshold = (high - prev_high) * self.reversal_threshold_bps / 10000.0;
                
                if close < prev_high - reversal_threshold {
                    let pool = LiquidityPool::new(
                        LiquidityType::StopCluster,
                        high,
                        (high - prev_high).abs(),
                        timestamp_ns,
                    );
                    return Some(pool);
                }
            }
        }

        // Record new high/low
        self.recent_highs.push_front((high, timestamp_ns));
        self.recent_lows.push_front((low, timestamp_ns));

        None
    }

    #[inline]
    fn cleanup(&mut self, current_ns: u64) {
        while let Some(&(_, ts)) = self.recent_highs.back() {
            if current_ns.saturating_sub(ts) > self.window_ns {
                self.recent_highs.pop_back();
            } else {
                break;
            }
        }
        while let Some(&(_, ts)) = self.recent_lows.back() {
            if current_ns.saturating_sub(ts) > self.window_ns {
                self.recent_lows.pop_back();
            } else {
                break;
            }
        }
    }
}

/// Liquidity Sweep detector using order book data
#[repr(C, align(64))]
pub struct LiquiditySweepDetector {
    bid_liquidity: AtomicCell<f64>,
    ask_liquidity: AtomicCell<f64>,
    imbalance_threshold: f64,
    sweep_detected: AtomicCell<bool>,
    last_sweep_price: AtomicCell<f64>,
    last_sweep_time: AtomicCell<u64>,
}

impl LiquiditySweepDetector {
    pub fn new(imbalance_threshold: f64) -> Self {
        Self {
            bid_liquidity: AtomicCell::new(0.0),
            ask_liquidity: AtomicCell::new(0.0),
            imbalance_threshold,
            sweep_detected: AtomicCell::new(false),
            last_sweep_price: AtomicCell::new(0.0),
            last_sweep_time: AtomicCell::new(0),
        }
    }

    #[inline]
    pub fn update_orderbook(&self, bid_liquidity: f64, ask_liquidity: f64) {
        self.bid_liquidity.store(bid_liquidity);
        self.ask_liquidity.store(ask_liquidity);
    }

    /// Detect sweep from aggressive trades
    #[inline]
    pub fn detect_sweep(&self, trade_volume: f64, is_buy: bool, price: f64, timestamp_ns: u64) -> Option<LiquidityPool> {
        let bid_liq = self.bid_liquidity.load();
        let ask_liq = self.ask_liquidity.load();

        if bid_liq == 0.0 || ask_liq == 0.0 {
            return None;
        }

        let imbalance = (bid_liq - ask_liq) / (bid_liq + ask_liq);

        // Large aggressive trade against significant liquidity
        if trade_volume > (bid_liq + ask_liq) * 0.1 {
            if is_buy && imbalance < -self.imbalance_threshold {
                // Buying into heavy ask liquidity - potential sweep
                self.sweep_detected.store(true);
                self.last_sweep_price.store(price);
                self.last_sweep_time.store(timestamp_ns);
                
                return Some(LiquidityPool::new(
                    LiquidityType::SellSideLiquidity,
                    price,
                    trade_volume,
                    timestamp_ns,
                ));
            }
            
            if !is_buy && imbalance > self.imbalance_threshold {
                // Selling into heavy bid liquidity - potential sweep
                self.sweep_detected.store(true);
                self.last_sweep_price.store(price);
                self.last_sweep_time.store(timestamp_ns);
                
                return Some(LiquidityPool::new(
                    LiquidityType::BuySideLiquidity,
                    price,
                    trade_volume,
                    timestamp_ns,
                ));
            }
        }

        None
    }

    #[inline]
    pub fn last_sweep(&self) -> (f64, u64) {
        (self.last_sweep_price.load(), self.last_sweep_time.load())
    }
}

/// Combined Liquidity Engine
#[repr(C, align(64))]
pub struct LiquidityEngine {
    pools: Vec<AtomicCell<Option<LiquidityPool>>>,
    head: AtomicCell<usize>,
    count: AtomicCell<usize>,
    max_pools: usize,
    equal_detector: EqualLevelsDetector,
    stop_hunt_detector: StopHuntDetector,
    sweep_detector: LiquiditySweepDetector,
}

impl LiquidityEngine {
    pub fn new(max_pools: usize, eq_tolerance_bps: f64, hunt_threshold_bps: f64) -> Self {
        Self {
            pools: (0..max_pools).map(|_| AtomicCell::new(None)).collect(),
            head: AtomicCell::new(0),
            count: AtomicCell::new(0),
            max_pools,
            equal_detector: EqualLevelsDetector::new(20, eq_tolerance_bps, 3),
            stop_hunt_detector: StopHuntDetector::new(hunt_threshold_bps, 50.0, 300_000_000_000),
            sweep_detector: LiquiditySweepDetector::new(0.3),
        }
    }

    /// Main update function
    #[inline]
    pub fn update(&mut self, 
                  high: f64, low: f64, close: f64,
                  bid_liq: f64, ask_liq: f64,
                  timestamp_ns: u64) -> Vec<LiquidityPool> {
        
        let mut detected = Vec::with_capacity(4);

        // Update order book liquidity
        self.sweep_detector.update_orderbook(bid_liq, ask_liq);

        // Check for equal highs/lows
        if let Some(eq_high) = self.equal_detector.update(high, low).0 {
            let pool = LiquidityPool::new(LiquidityType::EqualHighs, eq_high, 0.0, timestamp_ns);
            self.add_pool(pool);
            detected.push(pool);
        }
        
        if let Some(eq_low) = self.equal_detector.update(high, low).1 {
            let pool = LiquidityPool::new(LiquidityType::EqualLows, eq_low, 0.0, timestamp_ns);
            self.add_pool(pool);
            detected.push(pool);
        }

        // Check for stop hunts
        if let Some(hunt) = self.stop_hunt_detector.detect(high, low, close, timestamp_ns) {
            self.add_pool(hunt);
            detected.push(hunt);
        }

        detected
    }

    /// Process a trade for sweep detection
    #[inline]
    pub fn process_trade(&self, volume: f64, is_buy: bool, price: f64, timestamp_ns: u64) -> Option<LiquidityPool> {
        self.sweep_detector.detect_sweep(volume, is_buy, price, timestamp_ns)
    }

    #[inline]
    fn add_pool(&self, pool: LiquidityPool) {
        let idx = self.head.fetch_add(1) % self.max_pools;
        self.pools[idx].store(Some(pool));
        
        let mut count = self.count.load();
        if count < self.max_pools {
            self.count.store(count + 1);
        }
    }

    /// Get active liquidity pools near current price
    #[inline]
    pub fn pools_near_price(&self, price: f64, threshold_bps: f64) -> Vec<LiquidityPool> {
        let mut result = Vec::with_capacity(8);
        let threshold = price * threshold_bps / 10000.0;
        let count = self.count.load();

        for i in 0..count.min(self.max_pools) {
            let idx = (self.head.load().wrapping_sub(i + 1)) % self.max_pools;
            if let Some(pool) = self.pools[idx].load() {
                if pool.proximity(price) <= threshold && !pool.swept {
                    result.push(pool);
                }
            }
        }

        result
    }

    /// Mark pool as swept
    #[inline]
    pub fn mark_swept(&self, price: f64) {
        let count = self.count.load();
        for i in 0..count.min(self.max_pools) {
            let idx = (self.head.load().wrapping_sub(i + 1)) % self.max_pools;
            if let Some(mut pool) = self.pools[idx].load() {
                if pool.proximity(price) < pool.price_level * 0.001 {
                    pool.mark_swept();
                    self.pools[idx].store(Some(pool));
                }
            }
        }
    }

    /// Get total estimated liquidity at levels
    #[inline]
    pub fn total_liquidity(&self) -> (f64, f64) {
        let mut buy_side = 0.0;
        let mut sell_side = 0.0;
        let count = self.count.load();

        for i in 0..count.min(self.max_pools) {
            let idx = (self.head.load().wrapping_sub(i + 1)) % self.max_pools;
            if let Some(pool) = self.pools[idx].load() {
                if !pool.swept {
                    match pool.pool_type {
                        LiquidityType::BuySideLiquidity | LiquidityType::EqualLows => {
                            buy_side += pool.estimated_size;
                        }
                        LiquidityType::SellSideLiquidity | LiquidityType::EqualHighs => {
                            sell_side += pool.estimated_size;
                        }
                        LiquidityType::StopCluster => {
                            // Could be either side
                            buy_side += pool.estimated_size / 2.0;
                            sell_side += pool.estimated_size / 2.0;
                        }
                    }
                }
            }
        }

        (buy_side, sell_side)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_equal_levels() {
        let mut detector = EqualLevelsDetector::new(20, 5.0, 3);
        
        // Simulate touches at similar levels
        detector.update(100.0, 90.0);
        detector.update(100.05, 90.02);
        detector.update(99.98, 90.01);
        
        let (eq_high, eq_low) = detector.update(100.02, 90.03);
        
        assert!(eq_high.is_some() || eq_low.is_some());
    }

    #[test]
    fn test_liquidity_pool() {
        let pool = LiquidityPool::new(LiquidityType::EqualHighs, 100.0, 50.0, 1000);
        assert_eq!(pool.price_level, 100.0);
        assert!(!pool.swept);
    }
}
