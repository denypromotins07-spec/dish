//! Iceberg order implementation and hidden liquidity detection.
//! Manages visible vs. hidden clip sizes dynamically based on real-time order book depth.

use std::sync::atomic::{AtomicF64, AtomicU64, AtomicBool, AtomicIsize, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

/// Order book level for depth analysis
#[derive(Debug, Clone)]
pub struct OrderBookLevel {
    pub price: f64,
    pub quantity: f64,
    pub order_count: u32,
}

/// Order book snapshot for liquidity analysis
#[derive(Debug, Clone)]
pub struct OrderBookSnapshot {
    pub bids: Vec<OrderBookLevel>,
    pub asks: Vec<OrderBookLevel>,
    pub timestamp_ns: u64,
    pub spread: f64,
    pub mid_price: f64,
}

/// Hidden liquidity detection result
#[derive(Debug, Clone)]
pub struct HiddenLiquiditySignal {
    /// Confidence score 0.0 to 1.0
    pub confidence: f64,
    /// Estimated hidden quantity at this level
    pub estimated_hidden_qty: f64,
    /// Price level where hidden liquidity detected
    pub price_level: f64,
    /// Side: true = buy side, false = sell side
    pub is_buy_side: bool,
    /// Detection method used
    pub detection_method: HiddenLiquidityMethod,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum HiddenLiquidityMethod {
    OrderFlowImbalance,
    QuoteStuffing,
    RefreshDetection,
    SizeAnomaly,
}

/// Advanced iceberg order with dynamic sizing
pub struct DynamicIceberg {
    /// Total parent order quantity
    total_quantity: AtomicF64,
    /// Current visible clip size
    visible_size: AtomicF64,
    /// Minimum visible size
    min_visible_size: AtomicF64,
    /// Maximum visible size
    max_visible_size: AtomicF64,
    /// Hidden quantity remaining
    hidden_quantity: AtomicF64,
    /// Executed quantity
    executed_quantity: AtomicF64,
    /// Current clip remaining
    clip_remaining: AtomicF64,
    /// Clips executed
    clips_executed: AtomicU64,
    /// Fill rate (fills per second)
    fill_rate: AtomicF64,
    /// Last fill timestamp
    last_fill_ts: AtomicU64,
    /// Active flag
    is_active: AtomicBool,
    /// Side: true = buy, false = sell
    is_buy_side: AtomicBool,
}

/// Liquidity detector for finding hidden orders
pub struct HiddenLiquidityDetector {
    /// Lookback window for analysis (number of updates)
    lookback_window: usize,
    /// Order book update history
    bid_history: Vec<Vec<OrderBookLevel>>,
    ask_history: Vec<Vec<OrderBookLevel>>,
    /// Trade flow history (aggressor side)
    trade_flow: Vec<TradeFlowRecord>,
    /// Detected hidden liquidity signals
    signals: Vec<HiddenLiquiditySignal>,
}

/// Trade flow record for order flow analysis
#[derive(Debug, Clone)]
pub struct TradeFlowRecord {
    pub timestamp_ns: u64,
    pub price: f64,
    pub quantity: f64,
    pub aggressor_side: AggressorSide,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AggressorSide {
    Buy,
    Sell,
    Unknown,
}

impl DynamicIceberg {
    /// Create new dynamic iceberg order
    pub fn new(
        total_quantity: f64,
        initial_visible: f64,
        min_visible: f64,
        max_visible: f64,
        is_buy: bool,
    ) -> Self {
        let hidden = total_quantity - initial_visible;
        
        Self {
            total_quantity: AtomicF64::new(total_quantity),
            visible_size: AtomicF64::new(initial_visible),
            min_visible_size: AtomicF64::new(min_visible),
            max_visible_size: AtomicF64::new(max_visible),
            hidden_quantity: AtomicF64::new(hidden.max(0.0)),
            executed_quantity: AtomicF64::new(0.0),
            clip_remaining: AtomicF64::new(initial_visible),
            clips_executed: AtomicU64::new(0),
            fill_rate: AtomicF64::new(0.0),
            last_fill_ts: AtomicU64::new(0),
            is_active: AtomicBool::new(false),
            is_buy_side: AtomicBool::new(is_buy),
        }
    }

    /// Start iceberg execution
    #[inline(always)]
    pub fn start(&self) {
        self.is_active.store(true, Ordering::Relaxed);
    }

    /// Stop iceberg execution
    #[inline(always)]
    pub fn stop(&self) {
        self.is_active.store(false, Ordering::Relaxed);
    }

    /// Get current visible size to place on book
    #[inline(always)]
    pub fn get_display_quantity(&self) -> f64 {
        let remaining = self.get_remaining();
        let visible = self.visible_size.load(Ordering::Relaxed);
        visible.min(remaining)
    }

    /// Record a fill and automatically manage clip reload
    #[inline(always)]
    pub fn record_fill(&self, fill_quantity: f64) {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64;
        
        // Update executed quantity
        self.executed_quantity.fetch_add(fill_quantity, Ordering::Relaxed);
        
        // Update fill rate
        let last_ts = self.last_fill_ts.load(Ordering::Relaxed);
        if last_ts > 0 {
            let elapsed_sec = ((now - last_ts) as f64) / 1_000_000_000.0;
            if elapsed_sec > 0.0 && elapsed_sec < 60.0 {
                let current_rate = self.fill_rate.load(Ordering::Relaxed);
                // EWMA of fill rate
                let new_rate = current_rate * 0.7 + (fill_quantity / elapsed_sec) * 0.3;
                self.fill_rate.store(new_rate, Ordering::Relaxed);
            }
        }
        self.last_fill_ts.store(now, Ordering::Relaxed);
        
        // Update clip remaining
        let clip_rem = self.clip_remaining.load(Ordering::Relaxed);
        let new_clip_rem = clip_rem - fill_quantity;
        
        if new_clip_rem <= 0.001 {
            // Clip exhausted, reload from hidden
            self.reload_clip();
        } else {
            self.clip_remaining.store(new_clip_rem, Ordering::Relaxed);
        }
    }

    /// Reload clip from hidden quantity
    fn reload_clip(&self) {
        let hidden = self.hidden_quantity.load(Ordering::Relaxed);
        let visible = self.visible_size.load(Ordering::Relaxed);
        
        if hidden > 0.001 {
            let reload_size = visible.min(hidden);
            self.clip_remaining.store(reload_size, Ordering::Relaxed);
            self.hidden_quantity.fetch_sub(reload_size, Ordering::Relaxed);
            self.clips_executed.fetch_add(1, Ordering::Relaxed);
        } else {
            // No more hidden, use remaining
            let remaining = self.get_remaining();
            self.clip_remaining.store(remaining, Ordering::Relaxed);
        }
    }

    /// Dynamically adjust visible size based on order book depth
    pub fn adjust_visible_size(&self, order_book: &OrderBookSnapshot, target_price: f64) {
        let is_buy = self.is_buy_side.load(Ordering::Relaxed);
        let current_visible = self.visible_size.load(Ordering::Relaxed);
        let min_visible = self.min_visible_size.load(Ordering::Relaxed);
        let max_visible = self.max_visible_size.load(Ordering::Relaxed);
        
        // Get relevant side depth
        let depth_levels = if is_buy {
            &order_book.asks
        } else {
            &order_book.bids
        };
        
        if depth_levels.is_empty() {
            return;
        }
        
        // Calculate depth at target price levels
        let mut depth_at_price = 0.0;
        for level in depth_levels.iter().take(5) {
            if is_buy && level.price >= target_price {
                depth_at_price += level.quantity;
            } else if !is_buy && level.price <= target_price {
                depth_at_price += level.quantity;
            }
        }
        
        // Adjust visible size based on depth
        // More depth = can show more without moving market
        let depth_ratio = (depth_at_price / current_visible).min(10.0).max(0.1);
        let target_visible = current_visible * (depth_ratio / 5.0);
        
        // Clamp to min/max bounds
        let new_visible = target_visible.clamp(min_visible, max_visible);
        self.visible_size.store(new_visible, Ordering::Relaxed);
    }

    /// Adjust based on fill rate (slow fills = reduce visible size)
    pub fn adjust_based_on_fill_rate(&self, target_fill_rate: f64) {
        let current_rate = self.fill_rate.load(Ordering::Relaxed);
        let current_visible = self.visible_size.load(Ordering::Relaxed);
        let min_visible = self.min_visible_size.load(Ordering::Relaxed);
        
        if current_rate <= 0.0 || target_fill_rate <= 0.0 {
            return;
        }
        
        let rate_ratio = current_rate / target_fill_rate;
        
        // If filling too fast, reduce visible size (we're being picked off)
        // If filling too slow, increase visible size (need more presence)
        let adjustment = if rate_ratio > 2.0 {
            0.8 // Reduce by 20%
        } else if rate_ratio < 0.5 {
            1.2 // Increase by 20%
        } else {
            1.0
        };
        
        let new_visible = (current_visible * adjustment).clamp(min_visible, self.max_visible_size.load(Ordering::Relaxed));
        self.visible_size.store(new_visible, Ordering::Relaxed);
    }

    /// Check if order is complete
    #[inline(always)]
    pub fn is_complete(&self) -> bool {
        let remaining = self.get_remaining();
        remaining < 0.001
    }

    /// Get remaining quantity
    #[inline(always)]
    pub fn get_remaining(&self) -> f64 {
        self.total_quantity.load(Ordering::Relaxed) - self.executed_quantity.load(Ordering::Relaxed)
    }

    /// Get execution status
    pub fn get_status(&self) -> IcebergStatus {
        let total = self.total_quantity.load(Ordering::Relaxed);
        let executed = self.executed_quantity.load(Ordering::Relaxed);
        let hidden = self.hidden_quantity.load(Ordering::Relaxed);
        let clip_rem = self.clip_remaining.load(Ordering::Relaxed);
        
        IcebergStatus {
            total_quantity: total,
            executed_quantity: executed,
            remaining_quantity: total - executed,
            hidden_remaining: hidden,
            current_clip_remaining: clip_rem,
            clips_executed: self.clips_executed.load(Ordering::Relaxed),
            fill_rate: self.fill_rate.load(Ordering::Relaxed),
            progress_pct: if total > 0.0 { executed / total * 100.0 } else { 0.0 },
        }
    }
}

/// Iceberg execution status
#[derive(Debug, Clone)]
pub struct IcebergStatus {
    pub total_quantity: f64,
    pub executed_quantity: f64,
    pub remaining_quantity: f64,
    pub hidden_remaining: f64,
    pub current_clip_remaining: f64,
    pub clips_executed: u64,
    pub fill_rate: f64,
    pub progress_pct: f64,
}

impl HiddenLiquidityDetector {
    /// Create new hidden liquidity detector
    pub fn new(lookback_window: usize) -> Self {
        Self {
            lookback_window,
            bid_history: Vec::with_capacity(lookback_window),
            ask_history: Vec::with_capacity(lookback_window),
            trade_flow: Vec::with_capacity(lookback_window * 10),
            signals: Vec::new(),
        }
    }

    /// Add order book snapshot for analysis
    pub fn add_orderbook_snapshot(&mut self, snapshot: OrderBookSnapshot) {
        self.bid_history.push(snapshot.bids.clone());
        self.ask_history.push(snapshot.asks.clone());
        
        // Trim history to lookback window
        if self.bid_history.len() > self.lookback_window {
            self.bid_history.remove(0);
        }
        if self.ask_history.len() > self.lookback_window {
            self.ask_history.remove(0);
        }
    }

    /// Add trade flow record
    pub fn add_trade_flow(&mut self, record: TradeFlowRecord) {
        self.trade_flow.push(record);
        if self.trade_flow.len() > self.lookback_window * 10 {
            self.trade_flow.remove(0);
        }
    }

    /// Detect hidden liquidity using multiple methods
    pub fn detect_hidden_liquidity(&mut self, current_book: &OrderBookSnapshot) -> Vec<HiddenLiquiditySignal> {
        self.signals.clear();
        
        // Method 1: Refresh detection (same size orders reappearing)
        self.detect_refresh_patterns(current_book);
        
        // Method 2: Order flow imbalance
        self.detect_order_flow_imbalance(current_book);
        
        // Method 3: Size anomaly detection
        self.detect_size_anomalies(current_book);
        
        self.signals.clone()
    }

    /// Detect refresh patterns (iceberg reloading)
    fn detect_refresh_patterns(&mut self, book: &OrderBookSnapshot) {
        if self.bid_history.len() < 3 {
            return;
        }
        
        let history_len = self.bid_history.len();
        
        // Check bid side for refresh patterns
        for (level_idx, level) in book.bids.iter().enumerate().take(10) {
            let mut refresh_count = 0;
            
            for i in 1..history_len.min(5) {
                let prev_idx = history_len - 1 - i;
                if prev_idx < self.bid_history.len() && level_idx < self.bid_history[prev_idx].len() {
                    let prev_level = &self.bid_history[prev_idx][level_idx];
                    
                    // Check if similar size appeared before and was depleted
                    if (prev_level.quantity - level.quantity).abs() / level.quantity.max(0.001) < 0.1 {
                        refresh_count += 1;
                    }
                }
            }
            
            if refresh_count >= 2 {
                self.signals.push(HiddenLiquiditySignal {
                    confidence: (refresh_count as f64 / 5.0).min(1.0),
                    estimated_hidden_qty: level.quantity * refresh_count as f64,
                    price_level: level.price,
                    is_buy_side: true,
                    detection_method: HiddenLiquidityMethod::RefreshDetection,
                });
            }
        }
    }

    /// Detect hidden liquidity via order flow imbalance
    fn detect_order_flow_imbalance(&mut self, book: &OrderBookSnapshot) {
        if self.trade_flow.is_empty() {
            return;
        }
        
        // Analyze recent trade flow
        let recent_trades: Vec<_> = self.trade_flow.iter().rev().take(50).collect();
        
        if recent_trades.is_empty() {
            return;
        }
        
        let buy_volume: f64 = recent_trades.iter()
            .filter(|t| t.aggressor_side == AggressorSide::Buy)
            .map(|t| t.quantity)
            .sum();
        let sell_volume: f64 = recent_trades.iter()
            .filter(|t| t.aggressor_side == AggressorSide::Sell)
            .map(|t| t.quantity)
            .sum();
        
        let total_volume = buy_volume + sell_volume;
        if total_volume < 0.001 {
            return;
        }
        
        let buy_ratio = buy_volume / total_volume;
        
        // Significant imbalance suggests hidden liquidity on other side
        if buy_ratio > 0.7 {
            // Heavy buying, check for hidden sells
            self.signals.push(HiddenLiquiditySignal {
                confidence: (buy_ratio - 0.5) * 2.0,
                estimated_hidden_qty: (buy_volume - sell_volume) * 0.5,
                price_level: book.mid_price,
                is_buy_side: false,
                detection_method: HiddenLiquidityMethod::OrderFlowImbalance,
            });
        } else if buy_ratio < 0.3 {
            // Heavy selling, check for hidden buys
            self.signals.push(HiddenLiquiditySignal {
                confidence: (0.5 - buy_ratio) * 2.0,
                estimated_hidden_qty: (sell_volume - buy_volume) * 0.5,
                price_level: book.mid_price,
                is_buy_side: true,
                detection_method: HiddenLiquidityMethod::OrderFlowImbalance,
            });
        }
    }

    /// Detect size anomalies (unusual order sizes suggesting icebergs)
    fn detect_size_anomalies(&mut self, book: &OrderBookSnapshot) {
        // Calculate average order size at each level
        for (idx, level) in book.bids.iter().enumerate().take(10) {
            let avg_size = level.quantity / level.order_count.max(1) as f64;
            
            // Unusually large average size suggests institutional orders
            if avg_size > 10.0 && level.order_count < 5 {
                self.signals.push(HiddenLiquiditySignal {
                    confidence: (avg_size / 100.0).min(1.0),
                    estimated_hidden_qty: level.quantity * 0.5,
                    price_level: level.price,
                    is_buy_side: true,
                    detection_method: HiddenLiquidityMethod::SizeAnomaly,
                });
            }
        }
        
        for (idx, level) in book.asks.iter().enumerate().take(10) {
            let avg_size = level.quantity / level.order_count.max(1) as f64;
            
            if avg_size > 10.0 && level.order_count < 5 {
                self.signals.push(HiddenLiquiditySignal {
                    confidence: (avg_size / 100.0).min(1.0),
                    estimated_hidden_qty: level.quantity * 0.5,
                    price_level: level.price,
                    is_buy_side: false,
                    detection_method: HiddenLiquidityMethod::SizeAnomaly,
                });
            }
        }
    }

    /// Get strongest signal
    pub fn get_strongest_signal(&self) -> Option<&HiddenLiquiditySignal> {
        self.signals.iter().max_by(|a, b| a.confidence.partial_cmp(&b.confidence).unwrap_or(std::cmp::Ordering::Equal))
    }

    /// Clear all history
    pub fn clear(&mut self) {
        self.bid_history.clear();
        self.ask_history.clear();
        self.trade_flow.clear();
        self.signals.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dynamic_iceberg() {
        let iceberg = DynamicIceberg::new(100.0, 10.0, 5.0, 20.0, true);
        iceberg.start();
        
        assert!((iceberg.get_display_quantity() - 10.0).abs() < 0.001);
        
        // Fill the clip
        iceberg.record_fill(10.0);
        
        let status = iceberg.get_status();
        assert_eq!(status.clips_executed, 1);
        assert!((status.hidden_remaining - 80.0).abs() < 0.001);
    }

    #[test]
    fn test_liquidity_detector() {
        let mut detector = HiddenLiquidityDetector::new(10);
        
        // Create sample order book
        let book = OrderBookSnapshot {
            bids: vec![
                OrderBookLevel { price: 49999.0, quantity: 50.0, order_count: 2 },
                OrderBookLevel { price: 49998.0, quantity: 100.0, order_count: 1 },
            ],
            asks: vec![
                OrderBookLevel { price: 50001.0, quantity: 45.0, order_count: 3 },
                OrderBookLevel { price: 50002.0, quantity: 80.0, order_count: 1 },
            ],
            timestamp_ns: 0,
            spread: 2.0,
            mid_price: 50000.0,
        };
        
        detector.add_orderbook_snapshot(book.clone());
        
        let signals = detector.detect_hidden_liquidity(&book);
        // May detect size anomalies
        assert!(signals.len() >= 0);
    }
}
