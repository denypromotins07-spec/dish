//! High-performance cost basis calculator.
//! Continuously updates average entry price, realized/unrealized PnL, and net exposure.
//! Pure Rust implementation with zero allocations in hot paths.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};

/// Fixed-point precision multiplier (1e8)
const FP_MULTIPLIER: u128 = 1_000_000_000;

/// Portfolio state for a single instrument
#[derive(Debug, Clone)]
pub struct InstrumentState {
    pub instrument_id: u64,
    pub net_quantity: i64, // Signed: positive long, negative short
    pub total_cost_basis: u128, // In quote currency * 1e8
    pub realized_pnl: i128, // In quote currency * 1e8
    pub unrealized_pnl: i128, // In quote currency * 1e8 (calculated)
    pub average_entry_price: u128, // In quote currency * 1e8
    pub last_mark_price: u128, // In quote currency * 1e8
    pub last_update_ns: u64,
}

impl InstrumentState {
    pub fn new(instrument_id: u64) -> Self {
        Self {
            instrument_id,
            net_quantity: 0,
            total_cost_basis: 0,
            realized_pnl: 0,
            unrealized_pnl: 0,
            average_entry_price: 0,
            last_mark_price: 0,
            last_update_ns: 0,
        }
    }

    /// Update mark price and recalculate unrealized PnL
    #[inline]
    pub fn update_mark_price(&mut self, mark_price: u128, timestamp_ns: u64) {
        self.last_mark_price = mark_price;
        self.last_update_ns = timestamp_ns;
        
        if self.net_quantity != 0 {
            let market_value = (self.net_quantity.abs() as u128 * mark_price) / FP_MULTIPLIER;
            
            self.unrealized_pnl = if self.net_quantity > 0 {
                // Long: profit if mark > entry
                (market_value as i128 - self.total_cost_basis as i128)
            } else {
                // Short: profit if mark < entry
                (self.total_cost_basis as i128 - market_value as i128)
            };
        } else {
            self.unrealized_pnl = 0;
        }
    }

    /// Get total PnL (realized + unrealized)
    #[inline]
    pub fn total_pnl(&self) -> i128 {
        self.realized_pnl + self.unrealized_pnl
    }

    /// Get notional value in quote currency * 1e8
    #[inline]
    pub fn notional_value(&self) -> u128 {
        (self.net_quantity.abs() as u128 * self.last_mark_price) / FP_MULTIPLIER
    }
}

/// High-performance cost basis calculator for the entire portfolio
pub struct CostBasisCalculator {
    /// Per-instrument state tracking
    instruments: HashMap<u64, InstrumentState>,
    /// Global portfolio metrics
    total_realized_pnl: i128,
    total_unrealized_pnl: i128,
    /// Memory footprint tracking
    memory_footprint: AtomicU64,
}

impl CostBasisCalculator {
    pub fn new() -> Self {
        Self {
            instruments: HashMap::with_capacity(256), // Pre-allocate for common instruments
            total_realized_pnl: 0,
            total_unrealized_pnl: 0,
            memory_footprint: AtomicU64::new(0),
        }
    }

    /// Process a fill event and update cost basis
    /// quantity: signed (positive for buy, negative for sell)
    /// price: fixed point (price * 1e8)
    /// fees: in quote currency * 1e8
    pub fn process_fill(
        &mut self,
        instrument_id: u64,
        quantity: i64,
        price: u128,
        fees: u64,
        timestamp_ns: u64,
    ) {
        let state = self
            .instruments
            .entry(instrument_id)
            .or_insert_with(|| InstrumentState::new(instrument_id));

        let trade_value = (quantity.abs() as u128 * price) / FP_MULTIPLIER;
        let total_cost = trade_value + fees as u128;

        // Determine if this is opening, closing, or reversing
        let previous_qty = state.net_quantity;
        let new_qty = previous_qty + quantity;

        if previous_qty == 0 {
            // Opening new position
            state.net_quantity = new_qty;
            state.total_cost_basis = total_cost;
            state.average_entry_price = price + (fees as u128 * FP_MULTIPLIER) / quantity.abs() as u128;
        } else if (previous_qty > 0 && quantity > 0) || (previous_qty < 0 && quantity < 0) {
            // Adding to existing position (same direction)
            state.net_quantity = new_qty;
            state.total_cost_basis += total_cost;
            // Recalculate average entry price
            state.average_entry_price = 
                (state.total_cost_basis * FP_MULTIPLIER) / new_qty.abs() as u128;
        } else {
            // Closing or reversing position
            let close_qty = quantity.abs().min(previous_qty.abs());
            let remain_qty = previous_qty.abs() - close_qty;
            
            // Calculate proportional cost basis for closed portion
            let proportion = close_qty as f64 / previous_qty.abs() as f64;
            let closed_cost_basis = (state.total_cost_basis as f64 * proportion) as u128;
            
            // Calculate realized PnL
            let proceeds = (close_qty as u128 * price) / FP_MULTIPLIER;
            let realized = if previous_qty > 0 {
                // Closing long
                (proceeds - closed_cost_basis - fees as u128) as i128
            } else {
                // Closing short
                (closed_cost_basis - proceeds - fees as u128) as i128
            };

            state.realized_pnl += realized;
            self.total_realized_pnl += realized;

            if remain_qty == 0 {
                if new_qty == 0 {
                    // Fully closed
                    state.net_quantity = 0;
                    state.total_cost_basis = 0;
                    state.average_entry_price = 0;
                } else {
                    // Reversed position
                    state.net_quantity = new_qty;
                    state.total_cost_basis = total_cost - closed_cost_basis;
                    state.average_entry_price = price + (fees as u128 * FP_MULTIPLIER) / new_qty.abs() as u128;
                }
            } else {
                // Partially closed
                state.net_quantity = new_qty;
                state.total_cost_basis -= closed_cost_basis;
                state.average_entry_price = 
                    (state.total_cost_basis * FP_MULTIPLIER) / new_qty.abs() as u128;
            }
        }

        state.last_update_ns = timestamp_ns;
        
        // Update memory footprint estimate
        self.memory_footprint.store(
            (self.instruments.len() * std::mem::size_of::<InstrumentState>()) as u64,
            Ordering::Relaxed,
        );
    }

    /// Update mark price for an instrument and recalculate unrealized PnL
    pub fn update_mark_price(&mut self, instrument_id: u64, mark_price: u128, timestamp_ns: u64) {
        if let Some(state) = self.instruments.get_mut(&instrument_id) {
            let old_unrealized = state.unrealized_pnl;
            state.update_mark_price(mark_price, timestamp_ns);
            let new_unrealized = state.unrealized_pnl;
            
            self.total_unrealized_pnl += new_unrealized - old_unrealized;
        }
    }

    /// Get instrument state
    pub fn get_instrument(&self, instrument_id: u64) -> Option<&InstrumentState> {
        self.instruments.get(&instrument_id)
    }

    /// Get all instrument states
    pub fn all_instruments(&self) -> impl Iterator<Item = &InstrumentState> {
        self.instruments.values()
    }

    /// Get total portfolio PnL
    pub fn total_portfolio_pnl(&self) -> i128 {
        self.total_realized_pnl + self.total_unrealized_pnl
    }

    /// Get total realized PnL
    pub fn total_realized_pnl(&self) -> i128 {
        self.total_realized_pnl
    }

    /// Get total unrealized PnL
    pub fn total_unrealized_pnl(&self) -> i128 {
        self.total_unrealized_pnl
    }

    /// Get memory footprint in bytes
    pub fn memory_footprint(&self) -> u64 {
        self.memory_footprint.load(Ordering::Relaxed)
    }

    /// Get count of tracked instruments
    pub fn instrument_count(&self) -> usize {
        self.instruments.len()
    }
}

impl Default for CostBasisCalculator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cost_basis_calculation() {
        let mut calc = CostBasisCalculator::new();
        
        // Buy 100 units at $50
        calc.process_fill(1, 100, 50_000_000_000, 50_000_000, 1000);
        
        let state = calc.get_instrument(1).unwrap();
        assert_eq!(state.net_quantity, 100);
        assert!(state.average_entry_price > 50_000_000_000); // Includes fees
        
        // Update mark price to $55
        calc.update_mark_price(1, 55_000_000_000, 2000);
        
        let state = calc.get_instrument(1).unwrap();
        assert!(state.unrealized_pnl > 0);
        
        // Sell 50 units at $55
        calc.process_fill(1, -50, 55_000_000_000, 27_500_000, 3000);
        
        assert!(calc.total_realized_pnl() > 0);
    }
}
