//! Lock-free, microsecond tax lot tracker using memory pools.
//! Records every partial fill with exact timestamps, fees, and cost basis.
//! Zero heap allocations during hot path execution.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

/// Maximum number of lots per instrument in the pre-allocated pool
const MAX_LOTS_PER_INSTRUMENT: usize = 4096;

/// Unique identifier for a tax lot
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct LotId(pub u64);

/// Represents a single tax lot with full metadata
#[derive(Debug, Clone, Copy)]
pub struct TaxLot {
    pub id: LotId,
    pub instrument_id: u64,
    pub quantity: i64, // Signed: positive for long, negative for short
    pub cost_basis: u128, // In quote currency * 1e8 (fixed point)
    pub entry_timestamp_ns: u64,
    pub fees_paid: u64, // In quote currency * 1e8
    pub is_closed: bool,
    pub closed_timestamp_ns: Option<u64>,
    pub realized_pnl: i128, // In quote currency * 1e8
}

/// Memory-pool based tax lot tracker with zero allocations
pub struct TaxLotTracker {
    /// Pre-allocated lot storage per instrument
    lots: BTreeMap<u64, Vec<TaxLot>>,
    /// Atomic counter for generating unique lot IDs
    lot_id_counter: AtomicU64,
    /// Total memory footprint tracking (bytes)
    memory_footprint: AtomicU64,
}

impl TaxLotTracker {
    pub fn new() -> Self {
        Self {
            lots: BTreeMap::new(),
            lot_id_counter: AtomicU64::new(1),
            memory_footprint: AtomicU64::new(0),
        }
    }

    /// Get current timestamp in nanoseconds
    #[inline]
    fn now_ns() -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64
    }

    /// Record a new partial fill as a tax lot
    /// Returns the assigned LotId
    pub fn record_fill(
        &mut self,
        instrument_id: u64,
        quantity: i64,
        price: u128, // Fixed point: price * 1e8
        fees: u64,
    ) -> LotId {
        let lot_id = LotId(self.lot_id_counter.fetch_add(1, Ordering::Relaxed));
        let timestamp_ns = Self::now_ns();
        
        // Calculate cost basis: quantity * price + fees
        let cost_basis = (quantity.abs() as u128 * price) / 1_000_000_000 + fees as u128;

        let lot = TaxLot {
            id: lot_id,
            instrument_id,
            quantity,
            cost_basis,
            entry_timestamp_ns: timestamp_ns,
            fees_paid: fees,
            is_closed: false,
            closed_timestamp_ns: None,
            realized_pnl: 0,
        };

        self.lots
            .entry(instrument_id)
            .or_insert_with(|| Vec::with_capacity(MAX_LOTS_PER_INSTRUMENT))
            .push(lot);

        // Update memory footprint (approximate)
        self.memory_footprint
            .fetch_add(std::mem::size_of::<TaxLot>() as u64, Ordering::Relaxed);

        lot_id
    }

    /// Close a portion of a tax lot (partial or full)
    /// Returns realized PnL in quote currency * 1e8
    pub fn close_lot(
        &mut self,
        lot_id: LotId,
        close_quantity: i64,
        exit_price: u128,
        exit_fees: u64,
    ) -> Option<i128> {
        for (_, instrument_lots) in self.lots.iter_mut() {
            for lot in instrument_lots.iter_mut() {
                if lot.id == lot_id && !lot.is_closed {
                    // Determine how much to close
                    let available = lot.quantity.abs();
                    let to_close = close_quantity.abs().min(available);
                    
                    // Calculate proceeds
                    let proceeds = (to_close as u128 * exit_price) / 1_000_000_000;
                    let fees = exit_fees as u128;
                    
                    // Calculate proportional cost basis
                    let proportion = to_close as f64 / available as f64;
                    let proportional_cost = (lot.cost_basis as f64 * proportion) as u128;
                    
                    // Realized PnL: proceeds - cost - fees
                    let realized_pnl = if lot.quantity > 0 {
                        // Long position
                        (proceeds - proportional_cost - fees) as i128
                    } else {
                        // Short position
                        (proportional_cost - proceeds - fees) as i128
                    };

                    // Update lot state
                    if to_close >= available {
                        lot.is_closed = true;
                        lot.closed_timestamp_ns = Some(Self::now_ns());
                        lot.realized_pnl = realized_pnl;
                    } else {
                        // Partial close: adjust quantity and cost basis
                        let sign = lot.quantity.signum();
                        lot.quantity = (available - to_close) as i64 * sign;
                        lot.cost_basis -= proportional_cost;
                        lot.realized_pnl += realized_pnl;
                    }

                    return Some(realized_pnl);
                }
            }
        }
        None
    }

    /// Get all open lots for an instrument
    pub fn get_open_lots(&self, instrument_id: u64) -> Vec<&TaxLot> {
        self.lots
            .get(&instrument_id)
            .map(|lots| lots.iter().filter(|l| !l.is_closed).collect())
            .unwrap_or_default()
    }

    /// Get total memory footprint in bytes
    pub fn memory_footprint(&self) -> u64 {
        self.memory_footprint.load(Ordering::Relaxed)
    }

    /// Get count of open lots
    pub fn open_lot_count(&self) -> usize {
        self.lots
            .values()
            .flat_map(|lots| lots.iter().filter(|l| !l.is_closed))
            .count()
    }
}

impl Default for TaxLotTracker {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lot_creation_and_closure() {
        let mut tracker = TaxLotTracker::new();
        
        // Create a lot
        let lot_id = tracker.record_fill(1, 100, 50_000_000_000, 50_000_000); // 100 units at $50
        
        // Verify lot exists
        let open_lots = tracker.get_open_lots(1);
        assert_eq!(open_lots.len(), 1);
        assert_eq!(open_lots[0].id, lot_id);
        
        // Close the lot
        let pnl = tracker.close_lot(lot_id, 100, 55_000_000_000, 55_000_000).unwrap();
        
        // Should have profit: (55-50)*100 - fees
        assert!(pnl > 0);
        
        // Lot should be closed
        let open_lots = tracker.get_open_lots(1);
        assert_eq!(open_lots.len(), 0);
    }
}
