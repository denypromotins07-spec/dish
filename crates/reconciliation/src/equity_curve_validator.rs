//! Cryptographic equity curve validator.
//! Hashes daily PnL and trade logs to ensure data integrity.
//! Detects corruption or tampering before generating end-of-day reports.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

/// Simple hash function for financial data (using FNV-1a variant)
/// For production, consider using a cryptographic hash like SHA-256
fn fnv1a_hash(data: &[u8]) -> u64 {
    const FNV_OFFSET: u64 = 0xcbf29ce484222325;
    const FNV_PRIME: u64 = 0x100000001b3;
    
    let mut hash = FNV_OFFSET;
    for byte in data {
        hash ^= *byte as u64;
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash
}

/// Daily PnL record with hash
#[derive(Debug, Clone)]
pub struct DailyPnLRecord {
    pub date: u64, // YYYYMMDD format
    pub realized_pnl: i128,
    pub unrealized_pnl: i128,
    pub trade_count: u64,
    pub volume: u128,
    pub hash: u64,
    pub previous_hash: u64,
    pub chain_hash: u64, // Cumulative hash including all previous days
}

/// Trade log entry for hashing
#[derive(Debug, Clone, Copy)]
pub struct TradeLogEntry {
    pub trade_id: u64,
    pub timestamp_ns: u64,
    pub instrument_id: u64,
    pub quantity: i64,
    pub price: u128,
    pub fees: u64,
    pub realized_pnl: i128,
}

/// Validation result
#[derive(Debug, Clone)]
pub struct ValidationResult {
    pub is_valid: bool,
    pub validated_days: u64,
    pub first_invalid_day: Option<u64>,
    pub validation_error: Option<String>,
    pub validation_duration_ns: u64,
}

/// Cryptographic equity curve validator
pub struct EquityCurveValidator {
    /// Daily PnL records (date -> record)
    daily_records: BTreeMap<u64, DailyPnLRecord>,
    /// Trade logs by date
    trade_logs: BTreeMap<u64, Vec<TradeLogEntry>>,
    /// Genesis hash (fixed starting point)
    genesis_hash: u64,
    /// Total validations performed
    total_validations: AtomicU64,
    /// Memory footprint tracking
    memory_footprint: AtomicU64,
}

impl EquityCurveValidator {
    pub fn new() -> Self {
        Self {
            daily_records: BTreeMap::new(),
            trade_logs: BTreeMap::new(),
            genesis_hash: 0xdeadbeef_cafebabe, // Fixed genesis hash
            total_validations: AtomicU64::new(0),
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

    /// Record a daily PnL entry with hash chaining
    pub fn record_daily_pnl(
        &mut self,
        date: u64, // YYYYMMDD
        realized_pnl: i128,
        unrealized_pnl: i128,
        trade_count: u64,
        volume: u128,
    ) -> DailyPnLRecord {
        let previous_hash = self.daily_records
            .last_key_value()
            .map(|(_, r)| r.chain_hash)
            .unwrap_or(self.genesis_hash);

        // Create hash of this day's data
        let mut hasher_data = Vec::with_capacity(64);
        hasher_data.extend_from_slice(&date.to_le_bytes());
        hasher_data.extend_from_slice(&realized_pnl.to_le_bytes());
        hasher_data.extend_from_slice(&unrealized_pnl.to_le_bytes());
        hasher_data.extend_from_slice(&trade_count.to_le_bytes());
        hasher_data.extend_from_slice(&volume.to_le_bytes());
        hasher_data.extend_from_slice(&previous_hash.to_le_bytes());

        let hash = fnv1a_hash(&hasher_data);
        
        // Chain hash includes all previous days
        let chain_hash = fnv1a_hash(&[hash.to_le_bytes(), previous_hash.to_le_bytes()].concat());

        let record = DailyPnLRecord {
            date,
            realized_pnl,
            unrealized_pnl,
            trade_count,
            volume,
            hash,
            previous_hash,
            chain_hash,
        };

        self.daily_records.insert(date, record.clone());
        
        self.memory_footprint.store(
            (self.daily_records.len() * std::mem::size_of::<DailyPnLRecord>()) as u64,
            Ordering::Relaxed,
        );

        record
    }

    /// Add trade log entries for a specific date
    pub fn add_trade_logs(&mut self, date: u64, trades: Vec<TradeLogEntry>) {
        self.trade_logs.insert(date, trades);
    }

    /// Calculate hash for a specific date's trade logs
    pub fn calculate_trade_log_hash(&self, date: u64) -> Option<u64> {
        let trades = self.trade_logs.get(&date)?;
        
        let mut combined_hash = self.genesis_hash;
        for trade in trades {
            let mut trade_data = Vec::with_capacity(64);
            trade_data.extend_from_slice(&trade.trade_id.to_le_bytes());
            trade_data.extend_from_slice(&trade.timestamp_ns.to_le_bytes());
            trade_data.extend_from_slice(&trade.instrument_id.to_le_bytes());
            trade_data.extend_from_slice(&trade.quantity.to_le_bytes());
            trade_data.extend_from_slice(&trade.price.to_le_bytes());
            trade_data.extend_from_slice(&trade.fees.to_le_bytes());
            trade_data.extend_from_slice(&trade.realized_pnl.to_le_bytes());
            
            combined_hash = fnv1a_hash(&trade_data) ^ combined_hash;
        }
        
        Some(combined_hash)
    }

    /// Validate the entire equity curve chain
    pub fn validate_chain(&self) -> ValidationResult {
        let start_ns = Self::now_ns();
        let mut expected_previous_hash = self.genesis_hash;
        
        for (&date, record) in &self.daily_records {
            // Recalculate hash
            let mut hasher_data = Vec::with_capacity(64);
            hasher_data.extend_from_slice(&date.to_le_bytes());
            hasher_data.extend_from_slice(&record.realized_pnl.to_le_bytes());
            hasher_data.extend_from_slice(&record.unrealized_pnl.to_le_bytes());
            hasher_data.extend_from_slice(&record.trade_count.to_le_bytes());
            hasher_data.extend_from_slice(&record.volume.to_le_bytes());
            hasher_data.extend_from_slice(&expected_previous_hash.to_le_bytes());

            let recalculated_hash = fnv1a_hash(&hasher_data);
            
            if recalculated_hash != record.hash {
                return ValidationResult {
                    is_valid: false,
                    validated_days: self.daily_records.keys().filter(|&&d| d < date).count() as u64,
                    first_invalid_day: Some(date),
                    validation_error: Some(format!(
                        "Hash mismatch on {}: expected {}, got {}",
                        date, record.hash, recalculated_hash
                    )),
                    validation_duration_ns: Self::now_ns() - start_ns,
                };
            }

            // Verify chain hash
            let expected_chain = fnv1a_hash(&[record.hash.to_le_bytes(), expected_previous_hash.to_le_bytes()].concat());
            if expected_chain != record.chain_hash {
                return ValidationResult {
                    is_valid: false,
                    validated_days: self.daily_records.keys().filter(|&&d| d < date).count() as u64,
                    first_invalid_day: Some(date),
                    validation_error: Some(format!("Chain hash broken on {}", date)),
                    validation_duration_ns: Self::now_ns() - start_ns,
                };
            }

            expected_previous_hash = record.chain_hash;
        }

        self.total_validations.fetch_add(1, Ordering::Relaxed);

        ValidationResult {
            is_valid: true,
            validated_days: self.daily_records.len() as u64,
            first_invalid_day: None,
            validation_error: None,
            validation_duration_ns: Self::now_ns() - start_ns,
        }
    }

    /// Validate a single day's PnL against trade logs
    pub fn validate_day_against_trades(&self, date: u64) -> Option<bool> {
        let record = self.daily_records.get(&date)?;
        let trades = self.trade_logs.get(&date)?;

        // Sum up realized PnL from trades
        let trade_sum_pnl: i128 = trades.iter().map(|t| t.realized_pnl).sum();
        
        // Allow small tolerance for rounding
        let tolerance = 1000; // 0.00001 in fixed point
        let diff = (record.realized_pnl - trade_sum_pnl).abs();

        Some(diff <= tolerance)
    }

    /// Get the latest chain hash (for external verification)
    pub fn get_latest_chain_hash(&self) -> Option<u64> {
        self.daily_records.last_key_value().map(|(_, r)| r.chain_hash)
    }

    /// Get genesis hash
    pub fn get_genesis_hash(&self) -> u64 {
        self.genesis_hash
    }

    /// Export daily records for external audit
    pub fn export_records(&self) -> Vec<DailyPnLRecord> {
        self.daily_records.values().cloned().collect()
    }

    /// Get total validations performed
    pub fn total_validations(&self) -> u64 {
        self.total_validations.load(Ordering::Relaxed)
    }

    /// Get memory footprint in bytes
    pub fn memory_footprint(&self) -> u64 {
        self.memory_footprint.load(Ordering::Relaxed)
    }

    /// Get record count
    pub fn record_count(&self) -> usize {
        self.daily_records.len()
    }

    /// Clear all data
    pub fn clear(&mut self) {
        self.daily_records.clear();
        self.trade_logs.clear();
    }
}

impl Default for EquityCurveValidator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chain_validation() {
        let mut validator = EquityCurveValidator::new();
        
        // Add some daily records
        validator.record_daily_pnl(20240101, 100_000_000, 50_000_000, 10, 1_000_000_000);
        validator.record_daily_pnl(20240102, 150_000_000, 75_000_000, 15, 1_500_000_000);
        validator.record_daily_pnl(20240103, -50_000_000, 25_000_000, 8, 800_000_000);
        
        // Should validate successfully
        let result = validator.validate_chain();
        assert!(result.is_valid);
        assert_eq!(result.validated_days, 3);
        
        // Get chain hash for external verification
        let chain_hash = validator.get_latest_chain_hash();
        assert!(chain_hash.is_some());
    }

    #[test]
    fn test_tampering_detection() {
        let mut validator = EquityCurveValidator::new();
        
        validator.record_daily_pnl(20240101, 100_000_000, 50_000_000, 10, 1_000_000_000);
        validator.record_daily_pnl(20240102, 150_000_000, 75_000_000, 15, 1_500_000_000);
        
        // Tamper with the stored data (simulate corruption)
        if let Some(record) = validator.daily_records.get_mut(&20240102) {
            record.realized_pnl = 999_999_999; // Corrupted!
        }
        
        // Should detect tampering
        let result = validator.validate_chain();
        assert!(!result.is_valid);
        assert!(result.validation_error.is_some());
    }
}
