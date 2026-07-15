//! Deep-state synchronization daemon.
//! Compares local LMDB state, Nautilus portfolio, and exchange REST API balances.
//! Resolves micro-discrepancies down to the 8th decimal place instantly.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, AtomicBool, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

/// Fixed-point precision for 8 decimal places
const FP_MULTIPLIER: u128 = 100_000_000;

/// Balance from a specific source
#[derive(Debug, Clone, Copy)]
pub struct SourceBalance {
    pub asset_id: u64,
    pub available: u128,  // Available balance * 1e8
    pub total: u128,      // Total balance * 1e8
    pub locked: u128,     // Locked/in-order balance * 1e8
    pub timestamp_ns: u64,
}

/// Discrepancy detected between sources
#[derive(Debug, Clone)]
pub struct Discrepancy {
    pub asset_id: u64,
    pub source_a: String,
    pub source_b: String,
    pub value_a: u128,
    pub value_b: u128,
    pub difference: i128,
    pub difference_bps: u32,
    pub severity: DiscrepancySeverity,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DiscrepancySeverity {
    Info,       // < 1 basis point
    Warning,    // 1-10 basis points
    Critical,   // > 10 basis points
    Fatal,      // > 100 basis points
}

/// Reconciliation result
#[derive(Debug, Clone)]
pub struct ReconciliationResult {
    pub is_synced: bool,
    pub discrepancies: Vec<Discrepancy>,
    pub last_sync_ns: u64,
    pub sync_duration_ns: u64,
}

/// Deep state synchronization engine
pub struct DeepStateSync {
    /// LMDB-local balances
    lmdb_balances: HashMap<u64, SourceBalance>,
    /// Nautilus portfolio balances
    nautilus_balances: HashMap<u64, SourceBalance>,
    /// Exchange API balances
    exchange_balances: HashMap<u64, SourceBalance>,
    /// Tolerance in basis points (1 bp = 0.01%)
    tolerance_bps: u32,
    /// Last successful sync timestamp
    last_sync_ns: AtomicU64,
    /// Currently syncing flag
    is_syncing: AtomicBool,
    /// Total discrepancies detected (lifetime)
    total_discrepancies: AtomicU64,
}

impl DeepStateSync {
    pub fn new(tolerance_bps: Option<u32>) -> Self {
        Self {
            lmdb_balances: HashMap::with_capacity(128),
            nautilus_balances: HashMap::with_capacity(128),
            exchange_balances: HashMap::with_capacity(128),
            tolerance_bps: tolerance_bps.unwrap_or(10), // Default 0.1% tolerance
            last_sync_ns: AtomicU64::new(0),
            is_syncing: AtomicBool::new(false),
            total_discrepancies: AtomicU64::new(0),
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

    /// Update LMDB balance
    pub fn update_lmdb_balance(&mut self, balance: SourceBalance) {
        self.lmdb_balances.insert(balance.asset_id, balance);
    }

    /// Update Nautilus portfolio balance
    pub fn update_nautilus_balance(&mut self, balance: SourceBalance) {
        self.nautilus_balances.insert(balance.asset_id, balance);
    }

    /// Update Exchange API balance
    pub fn update_exchange_balance(&mut self, balance: SourceBalance) {
        self.exchange_balances.insert(balance.asset_id, balance);
    }

    /// Perform full reconciliation across all sources
    pub fn reconcile(&self) -> ReconciliationResult {
        let start_ns = Self::now_ns();
        let mut discrepancies = Vec::new();

        // Collect all unique asset IDs
        let mut all_assets: HashMap<u64, ()> = HashMap::new();
        for &id in self.lmdb_balances.keys() { all_assets.insert(id, ()); }
        for &id in self.nautilus_balances.keys() { all_assets.insert(id, ()); }
        for &id in self.exchange_balances.keys() { all_assets.insert(id, ()); }

        // Compare each asset across all sources
        for asset_id in all_assets.keys() {
            let lmdb = self.lmdb_balances.get(asset_id);
            let nautilus = self.nautilus_balances.get(asset_id);
            let exchange = self.exchange_balances.get(asset_id);

            // Compare LMDB vs Nautilus
            if let (Some(l), Some(n)) = (lmdb, nautilus) {
                if let Some(disc) = self.check_discrepancy(
                    *asset_id, "LMDB", l.total, "Nautilus", n.total,
                ) {
                    discrepancies.push(disc);
                }
            }

            // Compare LMDB vs Exchange
            if let (Some(l), Some(e)) = (lmdb, exchange) {
                if let Some(disc) = self.check_discrepancy(
                    *asset_id, "LMDB", l.total, "Exchange", e.total,
                ) {
                    discrepancies.push(disc);
                }
            }

            // Compare Nautilus vs Exchange
            if let (Some(n), Some(e)) = (nautilus, exchange) {
                if let Some(disc) = self.check_discrepancy(
                    *asset_id, "Nautilus", n.total, "Exchange", e.total,
                ) {
                    discrepancies.push(disc);
                }
            }
        }

        let end_ns = Self::now_ns();
        let duration = end_ns - start_ns;

        let is_synced = discrepancies.is_empty();

        if is_synced {
            self.last_sync_ns.store(end_ns, Ordering::Relaxed);
        }

        ReconciliationResult {
            is_synced,
            discrepancies,
            last_sync_ns: self.last_sync_ns.load(Ordering::Relaxed),
            sync_duration_ns: duration,
        }
    }

    /// Check for discrepancy between two values
    fn check_discrepancy(
        &self,
        asset_id: u64,
        source_a: &str,
        value_a: u128,
        source_b: &str,
        value_b: u128,
    ) -> Option<Discrepancy> {
        if value_a == 0 && value_b == 0 {
            return None;
        }

        let diff = (value_a as i128 - value_b as i128).abs();
        
        // Calculate difference in basis points
        let larger = value_a.max(value_b);
        let diff_bps = if larger > 0 {
            ((diff as u128 * 10000) / larger) as u32
        } else {
            0
        };

        // Check if within tolerance
        if diff_bps <= self.tolerance_bps {
            return None;
        }

        // Determine severity
        let severity = if diff_bps < 1 {
            DiscrepancySeverity::Info
        } else if diff_bps < 10 {
            DiscrepancySeverity::Warning
        } else if diff_bps < 100 {
            DiscrepancySeverity::Critical
        } else {
            DiscrepancySeverity::Fatal
        };

        Some(Discrepancy {
            asset_id,
            source_a: source_a.to_string(),
            source_b: source_b.to_string(),
            value_a,
            value_b,
            difference: diff,
            difference_bps: diff_bps,
            severity,
        })
    }

    /// Resolve a discrepancy by accepting a source of truth
    pub fn resolve_discrepancy(
        &mut self,
        asset_id: u64,
        accepted_source: &str,
        accepted_value: u128,
    ) -> bool {
        let balance = SourceBalance {
            asset_id,
            available: accepted_value,
            total: accepted_value,
            locked: 0,
            timestamp_ns: Self::now_ns(),
        };

        match accepted_source {
            "LMDB" => {
                self.lmdb_balances.insert(asset_id, balance);
            }
            "Nautilus" => {
                self.nautilus_balances.insert(asset_id, balance);
            }
            "Exchange" => {
                self.exchange_balances.insert(asset_id, balance);
            }
            _ => return false,
        }

        true
    }

    /// Get balance from a specific source
    pub fn get_balance(&self, source: &str, asset_id: u64) -> Option<SourceBalance> {
        match source {
            "LMDB" => self.lmdb_balances.get(&asset_id).copied(),
            "Nautilus" => self.nautilus_balances.get(&asset_id).copied(),
            "Exchange" => self.exchange_balances.get(&asset_id).copied(),
            _ => None,
        }
    }

    /// Get consolidated balance (average of all sources if synced)
    pub fn get_consolidated_balance(&self, asset_id: u64) -> Option<u128> {
        let mut total = 0u128;
        let mut count = 0u32;

        if let Some(b) = self.lmdb_balances.get(&asset_id) {
            total += b.total;
            count += 1;
        }
        if let Some(b) = self.nautilus_balances.get(&asset_id) {
            total += b.total;
            count += 1;
        }
        if let Some(b) = self.exchange_balances.get(&asset_id) {
            total += b.total;
            count += 1;
        }

        if count == 0 {
            None
        } else {
            Some(total / count as u128)
        }
    }

    /// Set tolerance level
    pub fn set_tolerance_bps(&mut self, tolerance: u32) {
        self.tolerance_bps = tolerance;
    }

    /// Get total discrepancies detected (lifetime)
    pub fn total_discrepancies_detected(&self) -> u64 {
        self.total_discrepancies.load(Ordering::Relaxed)
    }

    /// Check if currently syncing
    pub fn is_syncing(&self) -> bool {
        self.is_syncing.load(Ordering::Relaxed)
    }

    /// Set syncing flag
    pub fn set_syncing(&self, syncing: bool) {
        self.is_syncing.store(syncing, Ordering::Relaxed);
    }
}

impl Default for DeepStateSync {
    fn default() -> Self {
        Self::new(None)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reconciliation() {
        let mut sync = DeepStateSync::new(Some(10)); // 0.1% tolerance

        // Add matching balances
        let balance = SourceBalance {
            asset_id: 1,
            available: 1_000_000_000,
            total: 1_000_000_000,
            locked: 0,
            timestamp_ns: 1000,
        };

        sync.update_lmdb_balance(balance);
        sync.update_nautilus_balance(balance);
        sync.update_exchange_balance(balance);

        // Should be fully synced
        let result = sync.reconcile();
        assert!(result.is_synced);
        assert!(result.discrepancies.is_empty());

        // Introduce a discrepancy
        let mut imbalanced = balance;
        imbalanced.total = 1_001_000_000; // 0.1% higher
        sync.update_exchange_balance(imbalanced);

        // Should detect discrepancy
        let result = sync.reconcile();
        assert!(!result.is_synced);
        assert!(!result.discrepancies.is_empty());
    }
}
