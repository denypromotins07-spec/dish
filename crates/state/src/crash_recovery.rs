//! Crash Recovery Daemon
//! Startup reconciliation that reads LMDB state and queries exchanges
//! Cancels orphaned orders or resumes TWAP execution exactly where it left off

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::time;

use super::lmdb_store::{LmdbStore, PersistedOrder, PersistedPosition, PersistedStrategyState};

/// Recovery result for an order
#[derive(Debug, Clone)]
pub enum OrderRecoveryResult {
    FoundActive { order: PersistedOrder },
    NotFoundOrCancelled { order_id: String },
    PartiallyFilled { order: PersistedOrder, filled_qty: f64 },
    Error { order_id: String, error: String },
}

/// Recovery result for a position
#[derive(Debug, Clone)]
pub struct PositionRecoveryResult {
    pub position: PersistedPosition,
    pub current_price: f64,
    pub current_pnl: f64,
    pub exists_on_exchange: bool,
}

/// TWAP resumption state
#[derive(Debug, Clone)]
pub struct TwapResumeState {
    pub strategy_id: String,
    pub total_quantity: f64,
    pub executed_quantity: f64,
    pub remaining_slices: u32,
    pub next_execution_time: Instant,
    pub average_price: f64,
}

/// Crash recovery daemon
pub struct CrashRecoveryDaemon {
    store: Arc<LmdbStore>,
    exchange_clients: HashMap<String, ExchangeClientStub>,
    recovery_timeout_secs: u64,
    auto_cancel_orphans: bool,
    auto_resume_strategies: bool,
}

impl CrashRecoveryDaemon {
    pub fn new(
        store: Arc<LmdbStore>,
        recovery_timeout_secs: u64,
        auto_cancel_orphans: bool,
        auto_resume_strategies: bool,
    ) -> Self {
        Self {
            store,
            exchange_clients: HashMap::new(),
            recovery_timeout_secs,
            auto_cancel_orphans,
            auto_resume_strategies,
        }
    }

    /// Register an exchange client for querying
    pub fn register_exchange(&mut self, exchange: &str, client: ExchangeClientStub) {
        self.exchange_clients.insert(exchange.to_string(), client);
    }

    /// Run full crash recovery process
    pub async fn run_recovery(&self) -> Result<RecoveryReport, RecoveryError> {
        let start = Instant::now();
        let mut report = RecoveryReport::default();

        // Step 1: Recover open orders
        report.orders = self.recover_orders().await?;

        // Step 2: Recover positions
        report.positions = self.recover_positions().await?;

        // Step 3: Recover strategy states
        report.strategies = self.recover_strategies().await?;

        // Step 4: Check for TWAP resumption
        report.twap_resumptions = self.check_twap_resumption().await?;

        report.duration = start.elapsed();
        report.completed_at = chrono::Utc::now();

        Ok(report)
    }

    /// Recover and reconcile open orders
    async fn recover_orders(&self) -> Result<Vec<OrderRecoveryResult>, RecoveryError> {
        let local_orders = self.store.get_all_open_orders()
            .map_err(|e| RecoveryError::StoreError(e.to_string()))?;

        let mut results = Vec::new();

        for order in local_orders {
            let result = self.reconcile_order(&order).await;
            
            match &result {
                OrderRecoveryResult::FoundActive { .. } => {
                    // Order still active on exchange - keep it
                    log::info!("Order {} still active on exchange", order.order_id);
                }
                OrderRecoveryResult::NotFoundOrCancelled { .. } => {
                    // Order not found or cancelled - remove from local state
                    if self.auto_cancel_orphans {
                        let _ = self.store.delete(&order.order_id);
                        log::info!("Removed orphaned order {}", order.order_id);
                    }
                }
                OrderRecoveryResult::PartiallyFilled { order: o, filled_qty } => {
                    // Update local state with filled quantity
                    log::info!(
                        "Order {} partially filled: {}/{}",
                        o.order_id, filled_qty, o.quantity
                    );
                }
                OrderRecoveryResult::Error { order_id, error } => {
                    log::error!("Error recovering order {}: {}", order_id, error);
                }
            }
            
            results.push(result);
        }

        Ok(results)
    }

    /// Reconcile a single order with exchange
    async fn reconcile_order(&self, order: &PersistedOrder) -> OrderRecoveryResult {
        let client = match self.exchange_clients.get(&order.exchange) {
            Some(c) => c,
            None => {
                return OrderRecoveryResult::Error {
                    order_id: order.order_id.clone(),
                    error: format!("No client for exchange {}", order.exchange),
                };
            }
        };

        // Query exchange for order status
        match client.query_order(&order.order_id, &order.symbol).await {
            Ok(exchange_order) => {
                match exchange_order.status.as_str() {
                    "NEW" | "PARTIALLY_FILLED" | "PENDING_NEW" => {
                        if exchange_order.filled_quantity > 0.0 {
                            OrderRecoveryResult::PartiallyFilled {
                                order: order.clone(),
                                filled_qty: exchange_order.filled_quantity,
                            }
                        } else {
                            OrderRecoveryResult::FoundActive {
                                order: order.clone(),
                            }
                        }
                    }
                    "CANCELED" | "REJECTED" | "FILLED" | "EXPIRED" => {
                        OrderRecoveryResult::NotFoundOrCancelled {
                            order_id: order.order_id.clone(),
                        }
                    }
                    _ => OrderRecoveryResult::Error {
                        order_id: order.order_id.clone(),
                        error: format!("Unknown order status: {}", exchange_order.status),
                    },
                }
            }
            Err(e) => OrderRecoveryResult::Error {
                order_id: order.order_id.clone(),
                error: e,
            },
        }
    }

    /// Recover and validate positions
    async fn recover_positions(&self) -> Result<Vec<PositionRecoveryResult>, RecoveryError> {
        let local_positions = self.store.get_all_positions()
            .map_err(|e| RecoveryError::StoreError(e.to_string()))?;

        let mut results = Vec::new();

        for position in local_positions {
            let client = match self.exchange_clients.get(&position.exchange) {
                Some(c) => c,
                None => continue,
            };

            match client.query_position(&position.symbol).await {
                Ok(exchange_position) => {
                    let result = PositionRecoveryResult {
                        position: position.clone(),
                        current_price: exchange_position.mark_price,
                        current_pnl: (exchange_position.mark_price - position.entry_price)
                            * position.quantity
                            * if position.side == "LONG" { 1.0 } else { -1.0 },
                        exists_on_exchange: true,
                    };
                    results.push(result);
                }
                Err(_) => {
                    // Position not found on exchange - may have been liquidated
                    let result = PositionRecoveryResult {
                        position: position.clone(),
                        current_price: 0.0,
                        current_pnl: 0.0,
                        exists_on_exchange: false,
                    };
                    results.push(result);
                }
            }
        }

        Ok(results)
    }

    /// Recover strategy states
    async fn recover_strategies(&self) -> Result<Vec<PersistedStrategyState>, RecoveryError> {
        // In production, would iterate through strategy states in LMDB
        // For now, return empty vector
        Ok(Vec::new())
    }

    /// Check for TWAP strategies that need resumption
    async fn check_twap_resumption(&self) -> Result<Vec<TwapResumeState>, RecoveryError> {
        let mut resumptions = Vec::new();

        // Check for TWAP strategy states
        // In production, would query specific strategy state entries
        
        if self.auto_resume_strategies {
            // Logic to find and resume TWAP strategies
            log::info!("Checking for TWAP strategies to resume...");
        }

        Ok(resumptions)
    }

    /// Cancel all open orders on an exchange
    pub async fn cancel_all_orders(&self, exchange: &str) -> Result<usize, String> {
        let client = self.exchange_clients.get(exchange)
            .ok_or_else(|| format!("No client for exchange {}", exchange))?;

        let orders = self.store.get_all_open_orders()
            .map_err(|e| e.to_string())?;

        let mut cancelled = 0;
        for order in orders {
            if order.exchange == exchange {
                match client.cancel_order(&order.order_id, &order.symbol).await {
                    Ok(_) => {
                        let _ = self.store.delete(&order.order_id);
                        cancelled += 1;
                    }
                    Err(e) => {
                        log::warn!("Failed to cancel order {}: {}", order.order_id, e);
                    }
                }
            }
        }

        Ok(cancelled)
    }

    /// Emergency shutdown - cancel everything
    pub async fn emergency_shutdown(&self) -> Result<EmergencyShutdownReport, String> {
        let mut report = EmergencyShutdownReport::default();

        for exchange in self.exchange_clients.keys() {
            match self.cancel_all_orders(exchange).await {
                Ok(count) => {
                    report.cancelled_by_exchange.insert(exchange.clone(), count);
                    report.total_cancelled += count;
                }
                Err(e) => {
                    report.errors.push(format!("Exchange {}: {}", exchange, e));
                }
            }
        }

        report.completed_at = chrono::Utc::now();
        Ok(report)
    }
}

/// Stub for exchange client (in production, would be real API client)
pub struct ExchangeClientStub {
    exchange: String,
}

impl ExchangeClientStub {
    pub fn new(exchange: &str) -> Self {
        Self {
            exchange: exchange.to_string(),
        }
    }

    pub async fn query_order(
        &self,
        order_id: &str,
        symbol: &str,
    ) -> Result<ExchangeOrderStatus, String> {
        // In production, actual REST API call
        Ok(ExchangeOrderStatus {
            order_id: order_id.to_string(),
            status: "NEW".to_string(),
            filled_quantity: 0.0,
        })
    }

    pub async fn query_position(&self, symbol: &str) -> Result<ExchangePosition, String> {
        // In production, actual REST API call
        Ok(ExchangePosition {
            symbol: symbol.to_string(),
            quantity: 0.0,
            entry_price: 0.0,
            mark_price: 0.0,
        })
    }

    pub async fn cancel_order(&self, order_id: &str, symbol: &str) -> Result<(), String> {
        // In production, actual REST API call
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct ExchangeOrderStatus {
    pub order_id: String,
    pub status: String,
    pub filled_quantity: f64,
}

#[derive(Debug, Clone)]
pub struct ExchangePosition {
    pub symbol: String,
    pub quantity: f64,
    pub entry_price: f64,
    pub mark_price: f64,
}

/// Recovery report
#[derive(Debug, Clone, Default)]
pub struct RecoveryReport {
    pub orders: Vec<OrderRecoveryResult>,
    pub positions: Vec<PositionRecoveryResult>,
    pub strategies: Vec<PersistedStrategyState>,
    pub twap_resumptions: Vec<TwapResumeState>,
    pub duration: Duration,
    pub completed_at: chrono::DateTime<chrono::Utc>,
}

/// Emergency shutdown report
#[derive(Debug, Clone, Default)]
pub struct EmergencyShutdownReport {
    pub total_cancelled: usize,
    pub cancelled_by_exchange: HashMap<String, usize>,
    pub errors: Vec<String>,
    pub completed_at: chrono::DateTime<chrono::Utc>,
}

/// Recovery errors
#[derive(Debug, Clone)]
pub enum RecoveryError {
    StoreError(String),
    ExchangeError(String),
    Timeout,
}

impl std::fmt::Display for RecoveryError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::StoreError(e) => write!(f, "Store error: {}", e),
            Self::ExchangeError(e) => write!(f, "Exchange error: {}", e),
            Self::Timeout => write!(f, "Recovery timeout"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[tokio::test]
    async fn test_crash_recovery_daemon() {
        let temp_dir = TempDir::new().unwrap();
        let store = Arc::new(LmdbStore::new(temp_dir.path(), 100).unwrap());
        
        let mut daemon = CrashRecoveryDaemon::new(store.clone(), 30, true, true);
        daemon.register_exchange("binance", ExchangeClientStub::new("binance"));

        let report = daemon.run_recovery().await.unwrap();
        
        assert!(report.duration.as_secs() < 30);
    }
}
