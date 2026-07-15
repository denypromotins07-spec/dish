//! Microsecond state tracker for active positions, unrealized PnL, realized PnL, and margin requirements.
//! Syncs seamlessly with Nautilus Portfolio component via zero-copy PyO3 bindings.

use std::sync::atomic::{AtomicF64, AtomicBool, AtomicU64, Ordering};
use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};

/// Position state representation
#[derive(Debug, Clone)]
pub struct PositionState {
    pub symbol: String,
    pub side: PositionSide,
    pub quantity: f64,
    pub entry_price: f64,
    pub current_price: f64,
    pub unrealized_pnl: f64,
    pub realized_pnl: f64,
    pub notional_value: f64,
    pub margin_required: f64,
    pub leverage: f64,
    pub timestamp_ns: u64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PositionSide {
    Long,
    Short,
    Flat,
}

/// Position update event
#[derive(Debug, Clone)]
pub struct PositionUpdate {
    pub symbol: String,
    pub old_quantity: f64,
    pub new_quantity: f64,
    pub old_entry_price: f64,
    pub new_entry_price: f64,
    pub realized_pnl_delta: f64,
}

/// Main position tracker
pub struct PositionTracker {
    /// Active positions by symbol
    positions: HashMap<String, PositionData>,
    /// Total portfolio value
    total_equity: AtomicF64,
    /// Total unrealized PnL
    total_unrealized_pnl: AtomicF64,
    /// Total realized PnL (session)
    total_realized_pnl: AtomicF64,
    /// Total margin used
    total_margin_used: AtomicF64,
    /// Margin limit
    max_margin_used: AtomicF64,
    /// Default leverage
    default_leverage: AtomicF64,
    /// Last update timestamp
    last_update_ts: AtomicU64,
}

/// Internal position data (lock-free via atomics where possible)
struct PositionData {
    side: AtomicI8, // -1 = short, 0 = flat, 1 = long
    quantity: AtomicF64,
    entry_price: AtomicF64,
    current_price: AtomicF64,
    realized_pnl: AtomicF64,
    last_update_ns: AtomicU64,
}

impl PositionData {
    fn new() -> Self {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64;
        
        Self {
            side: AtomicI8::new(0),
            quantity: AtomicF64::new(0.0),
            entry_price: AtomicF64::new(0.0),
            current_price: AtomicF64::new(0.0),
            realized_pnl: AtomicF64::new(0.0),
            last_update_ns: AtomicU64::new(now),
        }
    }
    
    fn get_side(&self) -> PositionSide {
        match self.side.load(Ordering::Relaxed) {
            1 => PositionSide::Long,
            -1 => PositionSide::Short,
            _ => PositionSide::Flat,
        }
    }
    
    fn set_side(&self, side: PositionSide) {
        let val = match side {
            PositionSide::Long => 1,
            PositionSide::Short => -1,
            PositionSide::Flat => 0,
        };
        self.side.store(val, Ordering::Relaxed);
    }
}

impl PositionTracker {
    /// Create new position tracker
    pub fn new(initial_equity: f64, max_margin: f64) -> Self {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64;
        
        Self {
            positions: HashMap::new(),
            total_equity: AtomicF64::new(initial_equity),
            total_unrealized_pnl: AtomicF64::new(0.0),
            total_realized_pnl: AtomicF64::new(0.0),
            total_margin_used: AtomicF64::new(0.0),
            max_margin_used: AtomicF64::new(max_margin),
            default_leverage: AtomicF64::new(1.0),
            last_update_ts: AtomicU64::new(now),
        }
    }

    /// Update or create a position after a fill
    pub fn update_position(
        &mut self,
        symbol: &str,
        side: PositionSide,
        fill_quantity: f64,
        fill_price: f64,
        is_closing: bool,
    ) -> PositionUpdate {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64;
        
        let pos = self.positions.entry(symbol.to_string())
            .or_insert_with(PositionData::new);
        
        let old_quantity = pos.quantity.load(Ordering::Relaxed);
        let old_entry_price = pos.entry_price.load(Ordering::Relaxed);
        let current_side = pos.get_side();
        
        let mut realized_pnl_delta = 0.0;
        let mut new_quantity = old_quantity;
        let mut new_entry_price = old_entry_price;
        let mut new_side = current_side;
        
        if side == PositionSide::Flat {
            // Closing position
            realized_pnl_delta = self.calculate_realized_pnl(
                current_side,
                old_quantity,
                old_entry_price,
                fill_price,
            );
            new_quantity = 0.0;
            new_side = PositionSide::Flat;
        } else if current_side == PositionSide::Flat {
            // Opening new position
            new_quantity = fill_quantity;
            new_entry_price = fill_price;
            new_side = side;
        } else if current_side == side {
            // Adding to existing position
            let total_notional = (old_quantity * old_entry_price) + (fill_quantity * fill_price);
            new_quantity = old_quantity + fill_quantity;
            new_entry_price = total_notional / new_quantity;
        } else {
            // Reducing or reversing position
            if fill_quantity >= old_quantity {
                // Fully close and reverse
                realized_pnl_delta = self.calculate_realized_pnl(
                    current_side,
                    old_quantity,
                    old_entry_price,
                    fill_price,
                );
                new_quantity = fill_quantity - old_quantity;
                new_entry_price = fill_price;
                new_side = side;
            } else {
                // Partially reduce
                realized_pnl_delta = self.calculate_realized_pnl(
                    current_side,
                    fill_quantity,
                    old_entry_price,
                    fill_price,
                );
                new_quantity = old_quantity - fill_quantity;
            }
        }
        
        // Update position data
        pos.set_side(new_side);
        pos.quantity.store(new_quantity, Ordering::Relaxed);
        pos.entry_price.store(new_entry_price, Ordering::Relaxed);
        pos.current_price.store(fill_price, Ordering::Relaxed);
        pos.realized_pnl.fetch_add(realized_pnl_delta, Ordering::Relaxed);
        pos.last_update_ns.store(now, Ordering::Relaxed);
        
        // Update totals
        self.total_realized_pnl.fetch_add(realized_pnl_delta, Ordering::Relaxed);
        self.update_margin_used();
        self.last_update_ts.store(now, Ordering::Relaxed);
        
        PositionUpdate {
            symbol: symbol.to_string(),
            old_quantity,
            new_quantity,
            old_entry_price,
            new_entry_price,
            realized_pnl_delta,
        }
    }

    /// Calculate realized PnL for a partial/full close
    fn calculate_realized_pnl(
        &self,
        side: PositionSide,
        quantity: f64,
        entry_price: f64,
        exit_price: f64,
    ) -> f64 {
        match side {
            PositionSide::Long => (exit_price - entry_price) * quantity,
            PositionSide::Short => (entry_price - exit_price) * quantity,
            PositionSide::Flat => 0.0,
        }
    }

    /// Update current price for a symbol (for unrealized PnL calculation)
    #[inline(always)]
    pub fn update_price(&self, symbol: &str, price: f64) {
        if let Some(pos) = self.positions.get(symbol) {
            pos.current_price.store(price, Ordering::Relaxed);
        }
    }

    /// Get position state for a symbol
    pub fn get_position(&self, symbol: &str) -> Option<PositionState> {
        let pos = self.positions.get(symbol)?;
        
        let quantity = pos.quantity.load(Ordering::Relaxed);
        let entry_price = pos.entry_price.load(Ordering::Relaxed);
        let current_price = pos.current_price.load(Ordering::Relaxed);
        let side = pos.get_side();
        let realized_pnl = pos.realized_pnl.load(Ordering::Relaxed);
        
        let unrealized_pnl = self.calculate_unrealized_pnl(side, quantity, entry_price, current_price);
        let notional = quantity * current_price;
        let leverage = self.default_leverage.load(Ordering::Relaxed);
        let margin_required = notional / leverage;
        
        Some(PositionState {
            symbol: symbol.to_string(),
            side,
            quantity,
            entry_price,
            current_price,
            unrealized_pnl,
            realized_pnl,
            notional_value: notional,
            margin_required,
            leverage,
            timestamp_ns: pos.last_update_ns.load(Ordering::Relaxed),
        })
    }

    /// Calculate unrealized PnL
    fn calculate_unrealized_pnl(
        &self,
        side: PositionSide,
        quantity: f64,
        entry_price: f64,
        current_price: f64,
    ) -> f64 {
        if quantity <= 0.0 || entry_price <= 0.0 {
            return 0.0;
        }
        
        match side {
            PositionSide::Long => (current_price - entry_price) * quantity,
            PositionSide::Short => (entry_price - current_price) * quantity,
            PositionSide::Flat => 0.0,
        }
    }

    /// Update total margin used across all positions
    fn update_margin_used(&self) {
        let mut total_margin = 0.0;
        let leverage = self.default_leverage.load(Ordering::Relaxed);
        
        for pos in self.positions.values() {
            let quantity = pos.quantity.load(Ordering::Relaxed);
            let price = pos.current_price.load(Ordering::Relaxed);
            let notional = quantity * price;
            total_margin += notional / leverage;
        }
        
        self.total_margin_used.store(total_margin, Ordering::Relaxed);
    }

    /// Recalculate total unrealized PnL
    pub fn recalculate_unrealized_pnl(&self) -> f64 {
        let mut total_unrealized = 0.0;
        
        for pos in self.positions.values() {
            let quantity = pos.quantity.load(Ordering::Relaxed);
            let entry_price = pos.entry_price.load(Ordering::Relaxed);
            let current_price = pos.current_price.load(Ordering::Relaxed);
            let side = pos.get_side();
            
            total_unrealized += self.calculate_unrealized_pnl(side, quantity, entry_price, current_price);
        }
        
        self.total_unrealized_pnl.store(total_unrealized, Ordering::Relaxed);
        total_unrealized
    }

    /// Get all active positions
    pub fn get_all_positions(&self) -> Vec<PositionState> {
        let mut positions = Vec::new();
        
        for symbol in self.positions.keys() {
            if let Some(state) = self.get_position(symbol) {
                if state.side != PositionSide::Flat {
                    positions.push(state);
                }
            }
        }
        
        positions
    }

    /// Get total portfolio value (equity + unrealized PnL)
    pub fn get_portfolio_value(&self) -> f64 {
        let equity = self.total_equity.load(Ordering::Relaxed);
        let unrealized = self.recalculate_unrealized_pnl();
        equity + unrealized
    }

    /// Set default leverage for margin calculations
    #[inline(always)]
    pub fn set_default_leverage(&self, leverage: f64) {
        self.default_leverage.store(leverage.max(1.0), Ordering::Relaxed);
        self.update_margin_used();
    }

    /// Check if margin limit would be exceeded
    pub fn check_margin_limit(&self, additional_margin: f64) -> bool {
        let current_margin = self.total_margin_used.load(Ordering::Relaxed);
        let max_margin = self.max_margin_used.load(Ordering::Relaxed);
        (current_margin + additional_margin) <= max_margin
    }

    /// Get margin utilization percentage
    pub fn get_margin_utilization(&self) -> f64 {
        let used = self.total_margin_used.load(Ordering::Relaxed);
        let max = self.max_margin_used.load(Ordering::Relaxed);
        if max > 0.0 { used / max } else { 0.0 }
    }

    /// Close a position (market exit)
    pub fn close_position(&mut self, symbol: &str, current_price: f64) -> Option<f64> {
        let pos = self.positions.get(symbol)?;
        let quantity = pos.quantity.load(Ordering::Relaxed);
        
        if quantity <= 0.0 {
            return None;
        }
        
        let side = pos.get_side();
        let exit_side = match side {
            PositionSide::Long => PositionSide::Short,
            PositionSide::Short => PositionSide::Long,
            PositionSide::Flat => return None,
        };
        
        let update = self.update_position(symbol, exit_side, quantity, current_price, true);
        Some(update.realized_pnl_delta)
    }

    /// Close all positions (emergency liquidation)
    pub fn close_all_positions(&mut self, prices: &HashMap<String, f64>) -> f64 {
        let mut total_realized = 0.0;
        let symbols: Vec<String> = self.positions.keys().cloned().collect();
        
        for symbol in symbols {
            if let Some(price) = prices.get(&symbol) {
                if let Some(pnl) = self.close_position(&symbol, *price) {
                    total_realized += pnl;
                }
            }
        }
        
        total_realized
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_position_open_and_close() {
        let mut tracker = PositionTracker::new(100_000.0, 200_000.0);
        
        // Open long position
        tracker.update_position("BTCUSDT", PositionSide::Long, 1.0, 50_000.0, false);
        
        let pos = tracker.get_position("BTCUSDT").unwrap();
        assert_eq!(pos.side, PositionSide::Long);
        assert!((pos.quantity - 1.0).abs() < 0.001);
        assert!((pos.entry_price - 50_000.0).abs() < 0.01);
        
        // Update price
        tracker.update_price("BTCUSDT", 51_000.0);
        
        let pos = tracker.get_position("BTCUSDT").unwrap();
        assert!((pos.unrealized_pnl - 1_000.0).abs() < 0.01);
        
        // Close position
        let pnl = tracker.close_position("BTCUSDT", 51_000.0).unwrap();
        assert!((pnl - 1_000.0).abs() < 0.01);
    }

    #[test]
    fn test_position_add_to_existing() {
        let mut tracker = PositionTracker::new(100_000.0, 200_000.0);
        
        tracker.update_position("BTCUSDT", PositionSide::Long, 1.0, 50_000.0, false);
        tracker.update_position("BTCUSDT", PositionSide::Long, 1.0, 52_000.0, false);
        
        let pos = tracker.get_position("BTCUSDT").unwrap();
        assert!((pos.quantity - 2.0).abs() < 0.001);
        assert!((pos.entry_price - 51_000.0).abs() < 0.01); // Average
    }

    #[test]
    fn test_margin_check() {
        let mut tracker = PositionTracker::new(100_000.0, 50_000.0);
        
        tracker.update_position("BTCUSDT", PositionSide::Long, 1.0, 50_000.0, false);
        
        assert!(!tracker.check_margin_limit(10_000.0)); // Would exceed
        assert!(tracker.check_margin_limit(0.0)); // OK
    }
}
