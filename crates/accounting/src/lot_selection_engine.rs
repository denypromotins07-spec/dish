//! Pluggable lot selection engine with O(log n) retrieval via indexed B-Trees.
//! Supports FIFO, LIFO, HIFO, and custom selection strategies.

use std::collections::{BTreeMap, BTreeSet};
use std::cmp::Ordering;
use crate::tax_lot_tracker::{LotId, TaxLot};

/// Selection strategy for tax lot optimization
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LotSelectionStrategy {
    /// First-In, First-Out (default for most jurisdictions)
    Fifo,
    /// Last-In, First-Out
    Lifo,
    /// Highest-In, First-Out (tax-efficient for gains)
    Hifo,
    /// Lowest-In, First-Out (tax-efficient for losses)
    Lifo,
    /// Specific lot ID selection
    Specific(LotId),
}

/// Indexed entry for B-Tree ordering
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct IndexedLot {
    lot_id: LotId,
    timestamp_ns: u64,
    cost_basis: u128,
    quantity: i64,
}

impl IndexedLot {
    fn from_tax_lot(lot: &TaxLot) -> Self {
        Self {
            lot_id: lot.id,
            timestamp_ns: lot.entry_timestamp_ns,
            cost_basis: lot.cost_basis,
            quantity: lot.quantity,
        }
    }
}

/// Comparison helpers for different strategies
impl IndexedLot {
    /// Compare by timestamp (ascending for FIFO)
    fn cmp_by_timestamp(&self, other: &Self) -> Ordering {
        self.timestamp_ns.cmp(&other.timestamp_ns)
    }

    /// Compare by timestamp (descending for LIFO)
    fn cmp_by_timestamp_desc(&self, other: &Self) -> Ordering {
        other.timestamp_ns.cmp(&self.timestamp_ns)
    }

    /// Compare by cost basis (descending for HIFO)
    fn cmp_by_cost_basis_desc(&self, other: &Self) -> Ordering {
        other.cost_basis.cmp(&self.cost_basis)
    }

    /// Compare by cost basis (ascending for LOFO)
    fn cmp_by_cost_basis_asc(&self, other: &Self) -> Ordering {
        self.cost_basis.cmp(&other.cost_basis)
    }
}

/// High-performance lot selection engine
pub struct LotSelectionEngine {
    /// B-Tree index by timestamp (for FIFO/LIFO)
    timestamp_index: BTreeMap<(u64, LotId), u64>, // (timestamp, lot_id) -> instrument_id
    /// B-Tree index by cost basis (for HIFO/LOFO)
    cost_basis_index: BTreeMap<(u128, LotId), u64>, // (cost_basis, lot_id) -> instrument_id
    /// Set of closed lot IDs for quick lookup
    closed_lots: BTreeSet<LotId>,
    /// Mapping from lot_id to instrument_id
    lot_to_instrument: BTreeMap<LotId, u64>,
}

impl LotSelectionEngine {
    pub fn new() -> Self {
        Self {
            timestamp_index: BTreeMap::new(),
            cost_basis_index: BTreeMap::new(),
            closed_lots: BTreeSet::new(),
            lot_to_instrument: BTreeMap::new(),
        }
    }

    /// Register a new lot in all indices
    pub fn register_lot(&mut self, lot: &TaxLot) {
        if self.closed_lots.contains(&lot.id) {
            return; // Already closed
        }

        let timestamp_key = (lot.entry_timestamp_ns, lot.id);
        let cost_basis_key = (lot.cost_basis, lot.id);

        self.timestamp_index.insert(timestamp_key, lot.instrument_id);
        self.cost_basis_index.insert(cost_basis_key, lot.instrument_id);
        self.lot_to_instrument.insert(lot.id, lot.instrument_id);
    }

    /// Mark a lot as closed and remove from indices
    pub fn mark_lot_closed(&mut self, lot_id: LotId) {
        if let Some(&instrument_id) = self.lot_to_instrument.get(&lot_id) {
            // Find and remove from timestamp index
            if let Some((key, _)) = self.timestamp_index.iter().find(|(_, &v)| v == instrument_id && key.1 == lot_id) {
                let key = *key;
                self.timestamp_index.remove(&key);
            }

            // Find and remove from cost basis index
            if let Some((key, _)) = self.cost_basis_index.iter().find(|(_, &v)| v == instrument_id && key.1 == lot_id) {
                let key = *key;
                self.cost_basis_index.remove(&key);
            }

            self.lot_to_instrument.remove(&lot_id);
            self.closed_lots.insert(lot_id);
        }
    }

    /// Select lots to close based on strategy
    /// Returns vector of (lot_id, available_quantity) up to the required quantity
    pub fn select_lots(
        &self,
        instrument_id: u64,
        quantity_to_close: i64,
        strategy: LotSelectionStrategy,
    ) -> Vec<(LotId, i64)> {
        let mut result = Vec::with_capacity(16);
        let mut remaining = quantity_to_close.abs();

        match strategy {
            LotSelectionStrategy::Fifo => {
                // Select oldest lots first (ascending timestamp)
                for ((timestamp, lot_id), &inst_id) in &self.timestamp_index {
                    if inst_id != instrument_id || self.closed_lots.contains(lot_id) {
                        continue;
                    }
                    
                    // In real implementation, would look up actual quantity
                    // Here we assume full lot closure for simplicity
                    if remaining <= 0 {
                        break;
                    }
                    
                    result.push((*lot_id, remaining)); // Simplified
                    remaining = 0;
                }
            }
            LotSelectionStrategy::Lifo => {
                // Select newest lots first (descending timestamp)
                for ((timestamp, lot_id), &inst_id) in self.timestamp_index.iter().rev() {
                    if inst_id != instrument_id || self.closed_lots.contains(lot_id) {
                        continue;
                    }
                    
                    if remaining <= 0 {
                        break;
                    }
                    
                    result.push((*lot_id, remaining));
                    remaining = 0;
                }
            }
            LotSelectionStrategy::Hifo => {
                // Select highest cost basis lots first (descending cost)
                for ((cost_basis, lot_id), &inst_id) in self.cost_basis_index.iter().rev() {
                    if inst_id != instrument_id || self.closed_lots.contains(lot_id) {
                        continue;
                    }
                    
                    if remaining <= 0 {
                        break;
                    }
                    
                    result.push((*lot_id, remaining));
                    remaining = 0;
                }
            }
            LotSelectionStrategy::Lifo => {
                // Select lowest cost basis lots first (ascending cost)
                for ((cost_basis, lot_id), &inst_id) in &self.cost_basis_index {
                    if inst_id != instrument_id || self.closed_lots.contains(lot_id) {
                        continue;
                    }
                    
                    if remaining <= 0 {
                        break;
                    }
                    
                    result.push((*lot_id, remaining));
                    remaining = 0;
                }
            }
            LotSelectionStrategy::Specific(target_lot_id) => {
                if let Some(&inst_id) = self.lot_to_instrument.get(&target_lot_id) {
                    if inst_id == instrument_id && !self.closed_lots.contains(&target_lot_id) {
                        result.push((target_lot_id, remaining));
                    }
                }
            }
        }

        result
    }

    /// Get the optimal lot for closing a single unit based on strategy
    pub fn select_optimal_lot(
        &self,
        instrument_id: u64,
        strategy: LotSelectionStrategy,
    ) -> Option<LotId> {
        let selected = self.select_lots(instrument_id, 1, strategy);
        selected.first().map(|(lot_id, _)| *lot_id)
    }

    /// Get count of open lots for an instrument
    pub fn open_lot_count(&self, instrument_id: u64) -> usize {
        self.timestamp_index
            .iter()
            .filter(|((_, _), &inst_id)| inst_id == instrument_id)
            .count()
    }

    /// Get total open lots across all instruments
    pub fn total_open_lots(&self) -> usize {
        self.timestamp_index.len() - self.closed_lots.len()
    }

    /// Clear all indices (for reset)
    pub fn clear(&mut self) {
        self.timestamp_index.clear();
        self.cost_basis_index.clear();
        self.closed_lots.clear();
        self.lot_to_instrument.clear();
    }
}

impl Default for LotSelectionEngine {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lot_selection_strategies() {
        let mut engine = LotSelectionEngine::new();
        
        // Create mock lots with different timestamps and cost bases
        let lot1 = TaxLot {
            id: LotId(1),
            instrument_id: 1,
            quantity: 100,
            cost_basis: 50_000_000_000,
            entry_timestamp_ns: 1000,
            fees_paid: 0,
            is_closed: false,
            closed_timestamp_ns: None,
            realized_pnl: 0,
        };
        
        let lot2 = TaxLot {
            id: LotId(2),
            instrument_id: 1,
            quantity: 100,
            cost_basis: 60_000_000_000,
            entry_timestamp_ns: 2000,
            fees_paid: 0,
            is_closed: false,
            closed_timestamp_ns: None,
            realized_pnl: 0,
        };
        
        engine.register_lot(&lot1);
        engine.register_lot(&lot2);
        
        // FIFO should select lot1 (oldest)
        let fifo_lot = engine.select_optimal_lot(1, LotSelectionStrategy::Fifo);
        assert_eq!(fifo_lot, Some(LotId(1)));
        
        // LIFO should select lot2 (newest)
        let lifo_lot = engine.select_optimal_lot(1, LotSelectionStrategy::Lifo);
        assert_eq!(lifo_lot, Some(LotId(2)));
        
        // HIFO should select lot2 (highest cost)
        let hifo_lot = engine.select_optimal_lot(1, LotSelectionStrategy::Hifo);
        assert_eq!(hifo_lot, Some(LotId(2)));
    }
}
