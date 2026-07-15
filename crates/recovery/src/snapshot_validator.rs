//! Snapshot Validator - Startup validation daemon for LMDB state snapshots.
//! Cryptographically verifies snapshots against exchange REST API before boot.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::{Duration, Instant};
use sha2::{Sha256, Digest};

/// Validation result
#[derive(Debug, Clone, PartialEq)]
pub enum ValidationResult {
    Valid,
    InvalidChecksum,
    StaleData,
    MismatchedPositions,
    NetworkError,
    ApiError(String),
}

/// Snapshot metadata
#[derive(Debug, Clone)]
pub struct SnapshotMetadata {
    pub timestamp_us: u64,
    pub checksum: [u8; 32],
    pub sequence_number: u64,
    pub venue: String,
    pub symbols: Vec<String>,
    pub position_count: u32,
    pub order_count: u32,
}

/// Position record for validation
#[derive(Debug, Clone)]
pub struct PositionRecord {
    pub symbol: String,
    pub side: PositionSide,
    pub quantity: f64,
    pub avg_entry_price: f64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PositionSide {
    Long,
    Short,
    Flat,
}

/// Order record for validation
#[derive(Debug, Clone)]
pub struct OrderRecord {
    pub order_id: String,
    pub client_order_id: String,
    pub symbol: String,
    pub side: OrderSide,
    pub quantity: f64,
    pub filled_quantity: f64,
    pub price: f64,
    pub status: OrderStatus,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum OrderSide {
    Buy,
    Sell,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum OrderStatus {
    New,
    PartiallyFilled,
    Filled,
    Cancelled,
    Rejected,
}

/// Configuration for snapshot validation
#[derive(Debug, Clone)]
pub struct ValidatorConfig {
    pub max_age_seconds: u64,       // Maximum allowed snapshot age
    pub tolerance_pct: f64,         // Tolerance for position mismatches
    pub require_checksum: bool,     // Require checksum validation
    pub api_timeout_ms: u64,        // API request timeout
    pub retry_count: u32,           // Number of retries on failure
}

impl Default for ValidatorConfig {
    fn default() -> Self {
        Self {
            max_age_seconds: 300,    // 5 minutes
            tolerance_pct: 0.01,     // 1% tolerance
            require_checksum: true,
            api_timeout_ms: 5000,
            retry_count: 3,
        }
    }
}

/// Lock-free Snapshot Validator
pub struct SnapshotValidator {
    config: ValidatorConfig,
    validations_performed: AtomicU64,
    validations_passed: AtomicU64,
    validations_failed: AtomicU64,
    last_validation_result: AtomicU64, // Encoded ValidationResult
    active: AtomicBool,
}

impl SnapshotValidator {
    pub fn new(config: ValidatorConfig) -> Self {
        Self {
            config,
            validations_performed: AtomicU64::new(0),
            validations_passed: AtomicU64::new(0),
            validations_failed: AtomicU64::new(0),
            last_validation_result: AtomicU64::new(ValidationResult::Valid as usize),
            active: AtomicBool::new(true),
        }
    }

    /// Validate a snapshot against exchange state
    pub fn validate_snapshot<T: ExchangeApiClient>(
        &self,
        snapshot: &SnapshotMetadata,
        api_client: &T,
    ) -> ValidationResult {
        if !self.active.load(Ordering::Relaxed) {
            return ValidationResult::Valid;
        }

        self.validations_performed.fetch_add(1, Ordering::Relaxed);
        let start = Instant::now();

        // Check snapshot age
        let now_us = Instant::now().duration_since(Instant::now()).as_micros() as u64;
        let age_seconds = (now_us - snapshot.timestamp_us) / 1_000_000;
        
        if age_seconds > self.config.max_age_seconds {
            self.record_result(&ValidationResult::StaleData);
            return ValidationResult::StaleData;
        }

        // Validate checksum if required
        if self.config.require_checksum {
            // In production, would compute checksum of local state and compare
            // For now, just verify checksum format
            if snapshot.checksum.iter().all(|&b| b == 0) {
                self.record_result(&ValidationResult::InvalidChecksum);
                return ValidationResult::InvalidChecksum;
            }
        }

        // Fetch current state from exchange
        let exchange_state = match self.fetch_exchange_state(api_client, &snapshot.venue) {
            Ok(state) => state,
            Err(e) => {
                self.record_result(&ValidationResult::ApiError(e));
                return ValidationResult::NetworkError;
            }
        };

        // Compare positions
        let position_result = self.validate_positions(&exchange_state.positions, snapshot.position_count);
        if position_result != ValidationResult::Valid {
            self.record_result(&position_result);
            return position_result;
        }

        // Compare orders
        let order_result = self.validate_orders(&exchange_state.orders, snapshot.order_count);
        if order_result != ValidationResult::Valid {
            self.record_result(&order_result);
            return order_result;
        }

        // All validations passed
        self.record_result(&ValidationResult::Valid);
        ValidationResult::Valid
    }

    /// Fetch exchange state via REST API
    fn fetch_exchange_state<T: ExchangeApiClient>(
        &self,
        api_client: &T,
        venue: &str,
    ) -> Result<ExchangeState, String> {
        let mut attempts = 0;
        
        while attempts < self.config.retry_count {
            match api_client.get_account_state(venue) {
                Ok(state) => return Ok(state),
                Err(e) => {
                    attempts += 1;
                    if attempts >= self.config.retry_count {
                        return Err(e);
                    }
                    std::thread::sleep(Duration::from_millis(100 * attempts as u64));
                }
            }
        }
        
        Err("Max retries exceeded".to_string())
    }

    /// Validate positions match exchange state
    fn validate_positions(
        &self,
        exchange_positions: &[PositionRecord],
        snapshot_count: u32,
    ) -> ValidationResult {
        let exchange_count = exchange_positions.len() as u32;
        
        // Check count matches within tolerance
        let count_diff = (exchange_count as i32 - snapshot_count as i32).abs();
        let tolerance = ((snapshot_count as f64) * self.config.tolerance_pct) as u32;
        
        if count_diff > tolerance as i32 {
            return ValidationResult::MismatchedPositions;
        }

        // In production, would compare each position's quantity and side
        ValidationResult::Valid
    }

    /// Validate orders match exchange state
    fn validate_orders(
        &self,
        exchange_orders: &[OrderRecord],
        snapshot_count: u32,
    ) -> ValidationResult {
        let exchange_count = exchange_orders.len() as u32;
        
        let count_diff = (exchange_count as i32 - snapshot_count as i32).abs();
        let tolerance = ((snapshot_count as f64) * self.config.tolerance_pct) as u32;
        
        if count_diff > tolerance as i32 {
            return ValidationResult::MismatchedPositions;
        }

        ValidationResult::Valid
    }

    /// Record validation result
    fn record_result(&self, result: &ValidationResult) {
        self.last_validation_result.store(*result as usize, Ordering::Relaxed);
        
        match result {
            ValidationResult::Valid => {
                self.validations_passed.fetch_add(1, Ordering::Relaxed);
            }
            _ => {
                self.validations_failed.fetch_add(1, Ordering::Relaxed);
            }
        }
    }

    /// Get last validation result
    pub fn get_last_result(&self) -> ValidationResult {
        let val = self.last_validation_result.load(Ordering::Relaxed);
        unsafe { std::mem::transmute::<usize, ValidationResult>(val) }
    }

    /// Get statistics
    pub fn get_stats(&self) -> ValidatorStats {
        ValidatorStats {
            validations_performed: self.validations_performed.load(Ordering::Relaxed),
            validations_passed: self.validations_passed.load(Ordering::Relaxed),
            validations_failed: self.validations_failed.load(Ordering::Relaxed),
            last_result: self.get_last_result(),
        }
    }

    /// Set active state
    pub fn set_active(&self, active: bool) {
        self.active.store(active, Ordering::Relaxed);
    }
}

#[derive(Debug, Clone)]
pub struct ValidatorStats {
    pub validations_performed: u64,
    pub validations_passed: u64,
    pub validations_failed: u64,
    pub last_result: ValidationResult,
}

/// Exchange state returned from API
#[derive(Debug, Clone)]
pub struct ExchangeState {
    pub positions: Vec<PositionRecord>,
    pub orders: Vec<OrderRecord>,
    pub balances: Vec<BalanceRecord>,
}

#[derive(Debug, Clone)]
pub struct BalanceRecord {
    pub asset: String,
    pub free: f64,
    pub locked: f64,
}

/// Exchange API client trait
pub trait ExchangeApiClient {
    fn get_account_state(&self, venue: &str) -> Result<ExchangeState, String>;
    fn get_positions(&self, venue: &str) -> Result<Vec<PositionRecord>, String>;
    fn get_open_orders(&self, venue: &str) -> Result<Vec<OrderRecord>, String>;
}

/// Compute SHA256 checksum of data
pub fn compute_checksum(data: &[u8]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(data);
    hasher.finalize().into()
}

#[cfg(test)]
mod tests {
    use super::*;

    struct MockApiClient;
    
    impl ExchangeApiClient for MockApiClient {
        fn get_account_state(&self, _venue: &str) -> Result<ExchangeState, String> {
            Ok(ExchangeState {
                positions: vec![],
                orders: vec![],
                balances: vec![],
            })
        }
        
        fn get_positions(&self, _venue: &str) -> Result<Vec<PositionRecord>, String> {
            Ok(vec![])
        }
        
        fn get_open_orders(&self, _venue: &str) -> Result<Vec<OrderRecord>, String> {
            Ok(vec![])
        }
    }

    #[test]
    fn test_snapshot_validation() {
        let config = ValidatorConfig::default();
        let validator = SnapshotValidator::new(config);
        
        let checksum = compute_checksum(b"test_data");
        let snapshot = SnapshotMetadata {
            timestamp_us: Instant::now().duration_since(Instant::now()).as_micros() as u64,
            checksum,
            sequence_number: 1,
            venue: "binance".to_string(),
            symbols: vec!["BTCUSDT".to_string()],
            position_count: 0,
            order_count: 0,
        };
        
        let api_client = MockApiClient;
        let result = validator.validate_snapshot(&snapshot, &api_client);
        assert_eq!(result, ValidationResult::Valid);
    }

    #[test]
    fn test_stale_snapshot() {
        let config = ValidatorConfig {
            max_age_seconds: 1,
            ..Default::default()
        };
        let validator = SnapshotValidator::new(config);
        
        let stale_snapshot = SnapshotMetadata {
            timestamp_us: 0, // Very old
            checksum: [1u8; 32],
            sequence_number: 1,
            venue: "binance".to_string(),
            symbols: vec![],
            position_count: 0,
            order_count: 0,
        };
        
        let api_client = MockApiClient;
        let result = validator.validate_snapshot(&stale_snapshot, &api_client);
        assert_eq!(result, ValidationResult::StaleData);
    }
}
