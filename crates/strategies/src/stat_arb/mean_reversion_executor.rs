//! Mean Reversion Execution Logic for Stat Arb Positions
//! Enters/exits using limit orders at spread distribution extremes with dynamic TP/SL

use std::sync::atomic::{AtomicF64, AtomicBool, AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

/// Configuration for mean reversion executor
#[derive(Clone, Copy, Debug)]
pub struct MeanReversionConfig {
    /// Z-score threshold for entry (long spread)
    pub entry_z_long: f64,
    /// Z-score threshold for entry (short spread)
    pub entry_z_short: f64,
    /// Z-score threshold for take profit
    pub exit_z_profit: f64,
    /// Z-score threshold for stop loss
    pub exit_z_loss: f64,
    /// Order timeout in milliseconds
    pub order_timeout_ms: u64,
    /// Max position size per leg
    pub max_position: f64,
    /// Limit order price offset (ticks from theoretical)
    pub limit_offset_ticks: f64,
}

impl Default for MeanReversionConfig {
    fn default() -> Self {
        Self {
            entry_z_long: -2.0,
            entry_z_short: 2.0,
            exit_z_profit: 0.0,
            exit_z_loss: 3.0,
            order_timeout_ms: 5000,
            max_position: 100.0,
            limit_offset_ticks: 1.0,
        }
    }
}

/// Position state
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum PositionState {
    Flat,
    LongSpread,   // Long A, Short B
    ShortSpread,  // Short A, Long B
    Closing,      // In process of closing
}

/// Active position tracking
pub struct ActivePosition {
    pub state: PositionState,
    pub entry_z: f64,
    pub entry_time_ns: u64,
    pub quantity_a: f64,
    pub quantity_b: f64,
    pub entry_spread: f64,
    pub current_pnl: f64,
}

impl ActivePosition {
    pub fn new(state: PositionState, z: f64, qty_a: f64, qty_b: f64, spread: f64) -> Self {
        Self {
            state,
            entry_z: z,
            entry_time_ns: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos() as u64,
            quantity_a: qty_a,
            quantity_b: qty_b,
            entry_spread: spread,
            current_pnl: 0.0,
        }
    }
}

/// Mean reversion execution engine
pub struct MeanReversionExecutor {
    /// Current z-score
    pub z_score: AtomicF64,
    /// Current spread value
    pub spread: AtomicF64,
    /// Spread standard deviation
    pub spread_std: AtomicF64,
    /// Spread mean
    pub spread_mean: AtomicF64,
    /// Configuration
    pub config: MeanReversionConfig,
    /// Active position
    pub position: Option<ActivePosition>,
    /// Order pending flag
    pub order_pending: AtomicBool,
    /// Last signal timestamp
    pub last_signal_ns: AtomicU64,
    /// Enabled flag
    pub enabled: AtomicBool,
}

impl MeanReversionExecutor {
    pub fn new(config: MeanReversionConfig) -> Self {
        Self {
            z_score: AtomicF64::new(0.0),
            spread: AtomicF64::new(0.0),
            spread_std: AtomicF64::new(1.0),
            spread_mean: AtomicF64::new(0.0),
            config,
            position: None,
            order_pending: AtomicBool::new(false),
            last_signal_ns: AtomicU64::new(0),
            enabled: AtomicBool::new(true),
        }
    }

    /// Update market data
    #[inline]
    pub fn update_market(&self, spread: f64, mean: f64, std: f64) {
        self.spread.store(spread, Ordering::Relaxed);
        self.spread_mean.store(mean, Ordering::Relaxed);
        self.spread_std.store(std.max(1e-10), Ordering::Relaxed);
        
        let z = (spread - mean) / std;
        self.z_score.store(z, Ordering::Relaxed);
    }

    /// Get current signal based on z-score and position state
    #[inline]
    pub fn get_signal(&self) -> ExecutionSignal {
        if !self.enabled.load(Ordering::Relaxed) {
            return ExecutionSignal::Hold;
        }

        let z = self.z_score.load(Ordering::Relaxed);
        let pos_state = match &self.position {
            Some(p) => p.state,
            None => PositionState::Flat,
        };

        match pos_state {
            PositionState::Flat => {
                if z < self.config.entry_z_long {
                    ExecutionSignal::EnterLong
                } else if z > self.config.entry_z_short {
                    ExecutionSignal::EnterShort
                } else {
                    ExecutionSignal::Hold
                }
            }
            PositionState::LongSpread => {
                // Currently long spread, check for exit
                if z > self.config.exit_z_profit {
                    ExecutionSignal::ExitLongProfit
                } else if z < -self.config.exit_z_loss {
                    ExecutionSignal::ExitLongLoss
                } else {
                    ExecutionSignal::Hold
                }
            }
            PositionState::ShortSpread => {
                // Currently short spread, check for exit
                if z < self.config.exit_z_profit {
                    ExecutionSignal::ExitShortProfit
                } else if z > self.config.exit_z_loss {
                    ExecutionSignal::ExitShortLoss
                } else {
                    ExecutionSignal::Hold
                }
            }
            PositionState::Closing => ExecutionSignal::Hold,
        }
    }

    /// Calculate limit order prices for entry/exit
    #[inline]
    pub fn calculate_limit_prices(
        &self,
        signal: ExecutionSignal,
        price_a: f64,
        price_b: f64,
        hedge_ratio: f64,
    ) -> LimitOrderPair {
        let tick_a = (price_a * 0.0001).max(0.01); // Assume 0.01% tick
        let tick_b = (price_b * 0.0001).max(0.01);
        let offset_a = self.config.limit_offset_ticks * tick_a;
        let offset_b = self.config.limit_offset_ticks * tick_b;

        match signal {
            ExecutionSignal::EnterLong => {
                // Long spread: buy A at bid, sell B at ask
                LimitOrderPair {
                    order_a_side: Side::Buy,
                    order_a_price: (price_a - offset_a).max(tick_a),
                    order_a_qty: self.calculate_quantity(price_a),
                    order_b_side: Side::Sell,
                    order_b_price: (price_b + offset_b),
                    order_b_qty: self.calculate_quantity(price_b) * hedge_ratio,
                }
            }
            ExecutionSignal::EnterShort => {
                // Short spread: sell A at ask, buy B at bid
                LimitOrderPair {
                    order_a_side: Side::Sell,
                    order_a_price: (price_a + offset_a),
                    order_a_qty: self.calculate_quantity(price_a),
                    order_b_side: Side::Buy,
                    order_b_price: (price_b - offset_b).max(tick_b),
                    order_b_qty: self.calculate_quantity(price_b) * hedge_ratio,
                }
            }
            ExecutionSignal::ExitLongProfit | ExecutionSignal::ExitLongLoss => {
                // Close long: sell A, buy B
                LimitOrderPair {
                    order_a_side: Side::Sell,
                    order_a_price: (price_a - offset_a).max(tick_a),
                    order_a_qty: self.position.as_ref().map(|p| p.quantity_a).unwrap_or(0.0),
                    order_b_side: Side::Buy,
                    order_b_price: (price_b + offset_b),
                    order_b_qty: self.position.as_ref().map(|p| p.quantity_b).unwrap_or(0.0),
                }
            }
            ExecutionSignal::ExitShortProfit | ExecutionSignal::ExitShortLoss => {
                // Close short: buy A, sell B
                LimitOrderPair {
                    order_a_side: Side::Buy,
                    order_a_price: (price_a + offset_a),
                    order_a_qty: self.position.as_ref().map(|p| p.quantity_a).unwrap_or(0.0),
                    order_b_side: Side::Sell,
                    order_b_price: (price_b - offset_b).max(tick_b),
                    order_b_qty: self.position.as_ref().map(|p| p.quantity_b).unwrap_or(0.0),
                }
            }
            ExecutionSignal::Hold => LimitOrderPair::empty(),
        }
    }

    /// Calculate position quantity based on volatility targeting
    #[inline]
    fn calculate_quantity(&self, price: f64) -> f64 {
        let std = self.spread_std.load(Ordering::Relaxed);
        let max_pos = self.config.max_position;
        
        // Reduce size when volatility is high
        let vol_factor = (1.0 / std).min(2.0).max(0.5);
        max_pos * vol_factor
    }

    /// Execute entry for long spread
    #[inline]
    pub fn enter_long(&mut self, qty_a: f64, qty_b: f64, spread: f64) {
        if self.order_pending.load(Ordering::Relaxed) {
            return;
        }
        
        let z = self.z_score.load(Ordering::Relaxed);
        self.position = Some(ActivePosition::new(
            PositionState::LongSpread,
            z,
            qty_a,
            qty_b,
            spread,
        ));
        self.last_signal_ns.store(
            SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos() as u64,
            Ordering::Relaxed,
        );
    }

    /// Execute entry for short spread
    #[inline]
    pub fn enter_short(&mut self, qty_a: f64, qty_b: f64, spread: f64) {
        if self.order_pending.load(Ordering::Relaxed) {
            return;
        }
        
        let z = self.z_score.load(Ordering::Relaxed);
        self.position = Some(ActivePosition::new(
            PositionState::ShortSpread,
            z,
            qty_a,
            qty_b,
            spread,
        ));
        self.last_signal_ns.store(
            SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos() as u64,
            Ordering::Relaxed,
        );
    }

    /// Close current position
    #[inline]
    pub fn close_position(&mut self) {
        self.position = None;
        self.order_pending.store(false, Ordering::Relaxed);
    }

    /// Update PnL on position
    #[inline]
    pub fn update_pnl(&mut self, current_spread: f64) {
        if let Some(ref mut pos) = self.position {
            let pnl_per_unit = current_spread - pos.entry_spread;
            match pos.state {
                PositionState::LongSpread => {
                    pos.current_pnl = pnl_per_unit * pos.quantity_a;
                }
                PositionState::ShortSpread => {
                    pos.current_pnl = -pnl_per_unit * pos.quantity_a;
                }
                _ => {}
            }
        }
    }

    /// Check for order timeout
    #[inline]
    pub fn check_timeout(&self) -> bool {
        if !self.order_pending.load(Ordering::Relaxed) {
            return false;
        }
        
        let now_ns = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64;
        let last_ns = self.last_signal_ns.load(Ordering::Relaxed);
        let elapsed_ms = (now_ns - last_ns) / 1_000_000;
        
        elapsed_ms > self.config.order_timeout_ms
    }

    /// Enable/disable executor
    #[inline]
    pub fn set_enabled(&self, enabled: bool) {
        self.enabled.store(enabled, Ordering::Relaxed);
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ExecutionSignal {
    EnterLong,
    EnterShort,
    ExitLongProfit,
    ExitLongLoss,
    ExitShortProfit,
    ExitShortLoss,
    Hold,
}

#[derive(Clone, Copy, Debug)]
pub struct LimitOrderPair {
    pub order_a_side: Side,
    pub order_a_price: f64,
    pub order_a_qty: f64,
    pub order_b_side: Side,
    pub order_b_price: f64,
    pub order_b_qty: f64,
}

impl LimitOrderPair {
    pub fn empty() -> Self {
        Self {
            order_a_side: Side::Buy,
            order_a_price: 0.0,
            order_a_qty: 0.0,
            order_b_side: Side::Buy,
            order_b_price: 0.0,
            order_b_qty: 0.0,
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Side {
    Buy,
    Sell,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_entry_signals() {
        let config = MeanReversionConfig {
            entry_z_long: -2.0,
            entry_z_short: 2.0,
            exit_z_profit: 0.0,
            exit_z_loss: 3.0,
            ..Default::default()
        };
        let executor = MeanReversionExecutor::new(config);

        // Test long entry signal
        executor.update_market(-5.0, 0.0, 1.0); // z = -5
        let signal = executor.get_signal();
        assert_eq!(signal, ExecutionSignal::EnterLong);

        // Test short entry signal
        executor.update_market(5.0, 0.0, 1.0); // z = 5
        let signal = executor.get_signal();
        assert_eq!(signal, ExecutionSignal::EnterShort);
    }

    #[test]
    fn test_exit_signals() {
        let config = MeanReversionConfig::default();
        let mut executor = MeanReversionExecutor::new(config);

        // Enter long position
        executor.update_market(-3.0, 0.0, 1.0);
        executor.enter_long(1.0, 1.0, -3.0);

        // Move to profit zone
        executor.update_market(1.0, 0.0, 1.0); // z = 1 > 0 (profit target)
        let signal = executor.get_signal();
        assert_eq!(signal, ExecutionSignal::ExitLongProfit);
    }

    #[test]
    fn test_limit_price_calculation() {
        let config = MeanReversionConfig::default();
        let executor = MeanReversionExecutor::new(config);

        executor.update_market(-3.0, 0.0, 1.0);
        let signal = executor.get_signal();
        
        let prices = executor.calculate_limit_prices(signal, 100.0, 100.0, 1.0);
        
        assert_eq!(prices.order_a_side, Side::Buy);
        assert!(prices.order_a_price < 100.0); // Bid below mid
        assert!(prices.order_b_price > 100.0); // Ask above mid
    }
}
