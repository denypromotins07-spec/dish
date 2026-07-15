//! Background reconciliation daemon for syncing local state with Binance REST API.
//! Periodically queries to fix missed WebSocket packets or network jitter discrepancies.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH, Duration};
use std::collections::HashMap;

/// Reconciliation status for a single symbol
#[derive(Debug, Clone)]
pub struct ReconciliationStatus {
    pub symbol: String,
    pub local_orders_count: u32,
    pub exchange_orders_count: u32,
    pub matched_orders: u32,
    pub discrepancies_found: u32,
    pub discrepancies_fixed: u32,
    pub last_sync_timestamp_ms: u64,
    pub sync_status: SyncStatus,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SyncStatus {
    InSync,
    MinorDrift,
    MajorDrift,
    Critical,
    Syncing,
}

/// Discrepancy record
#[derive(Debug, Clone)]
pub struct DiscrepancyRecord {
    pub discrepancy_id: String,
    pub symbol: String,
    pub client_order_id: Option<String>,
    pub discrepancy_type: DiscrepancyType,
    pub local_value: String,
    pub exchange_value: String,
    pub detected_at_ms: u64,
    pub resolved: bool,
    pub resolution_action: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DiscrepancyType {
    MissingOrder,
    StatusMismatch,
    QuantityMismatch,
    PriceMismatch,
    FillCountMismatch,
    PhantomOrder,
}

/// Reconciliation result summary
#[derive(Debug, Clone)]
pub struct ReconciliationSummary {
    pub total_symbols_checked: u32,
    pub total_discrepancies_found: u32,
    pub total_discrepancies_fixed: u32,
    pub sync_duration_ms: u64,
    pub critical_issues: u32,
    pub recommendations: Vec<String>,
}

/// Main reconciliation daemon
pub struct ReconciliationDaemon {
    /// Symbols to monitor
    monitored_symbols: Vec<String>,
    /// Discrepancy history
    discrepancy_history: Vec<DiscrepancyRecord>,
    /// Last sync timestamp per symbol
    last_sync_per_symbol: HashMap<String, u64>,
    /// Sync interval in seconds
    sync_interval_sec: u64,
    /// Maximum discrepancy history size
    max_history_size: usize,
    /// Daemon active flag
    is_active: AtomicBool,
    /// Total reconciliations performed
    total_reconciliations: AtomicU64,
    /// Total discrepancies found
    total_discrepancies_found: AtomicU64,
    /// Total discrepancies fixed
    total_discrepancies_fixed: AtomicU64,
    /// Consecutive failures
    consecutive_failures: AtomicU64,
}

impl ReconciliationDaemon {
    /// Create new reconciliation daemon
    pub fn new(sync_interval_sec: u64) -> Self {
        Self {
            monitored_symbols: Vec::new(),
            discrepancy_history: Vec::new(),
            last_sync_per_symbol: HashMap::new(),
            sync_interval_sec,
            max_history_size: 1000,
            is_active: AtomicBool::new(true),
            total_reconciliations: AtomicU64::new(0),
            total_discrepancies_found: AtomicU64::new(0),
            total_discrepancies_fixed: AtomicU64::new(0),
            consecutive_failures: AtomicU64::new(0),
        }
    }

    /// Add symbol to monitoring list
    #[inline(always)]
    pub fn add_symbol(&mut self, symbol: &str) {
        if !self.monitored_symbols.contains(&symbol.to_string()) {
            self.monitored_symbols.push(symbol.to_string());
        }
    }

    /// Remove symbol from monitoring
    #[inline(always)]
    pub fn remove_symbol(&mut self, symbol: &str) {
        self.monitored_symbols.retain(|s| s != symbol);
    }

    /// Check if sync is due for a symbol
    pub fn should_sync(&self, symbol: &str) -> bool {
        if !self.is_active.load(Ordering::Relaxed) {
            return false;
        }
        
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();
        
        let last_sync = self.last_sync_per_symbol.get(symbol)
            .copied()
            .unwrap_or(0);
        
        (now - last_sync) >= self.sync_interval_sec
    }

    /// Perform reconciliation for a single symbol
    pub fn reconcile_symbol(
        &mut self,
        symbol: &str,
        local_orders: &[LocalOrderSnapshot],
        exchange_orders: &[ExchangeOrderSnapshot],
    ) -> ReconciliationStatus {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64;
        
        let mut matched = 0u32;
        let mut discrepancies = 0u32;
        let mut fixed = 0u32;
        
        // Build lookup maps
        let local_map: HashMap<&str, &LocalOrderSnapshot> = local_orders
            .iter()
            .map(|o| (o.client_order_id.as_str(), o))
            .collect();
        
        let exchange_map: HashMap<&str, &ExchangeOrderSnapshot> = exchange_orders
            .iter()
            .map(|o| (o.client_order_id.as_str(), o))
            .collect();
        
        // Check for missing/mismatched orders
        for (client_id, local) in &local_map {
            if let Some(exchange) = exchange_map.get(client_id) {
                // Compare order states
                if let Some(discrepancy) = self.compare_orders(symbol, local, exchange) {
                    discrepancies += 1;
                    self.record_discrepancy(discrepancy);
                    
                    // Auto-fix minor discrepancies
                    if self.can_auto_fix(&discrepancy) {
                        fixed += 1;
                    }
                } else {
                    matched += 1;
                }
            } else {
                // Order exists locally but not on exchange (phantom or filled/canceled)
                if local.status != "FILLED" && local.status != "CANCELED" {
                    let discrepancy = DiscrepancyRecord {
                        discrepancy_id: format!("PHANTOM_{}_{}", symbol, client_id),
                        symbol: symbol.to_string(),
                        client_order_id: Some(client_id.to_string()),
                        discrepancy_type: DiscrepancyType::PhantomOrder,
                        local_value: format!("{:?}", local),
                        exchange_value: "NOT_FOUND".to_string(),
                        detected_at_ms: now,
                        resolved: false,
                        resolution_action: None,
                    };
                    discrepancies += 1;
                    self.record_discrepancy(discrepancy);
                }
            }
        }
        
        // Check for orders on exchange but not locally (missing)
        for (client_id, exchange) in &exchange_map {
            if !local_map.contains_key(client_id) {
                let discrepancy = DiscrepancyRecord {
                    discrepancy_id: format!("MISSING_{}_{}", symbol, client_id),
                    symbol: symbol.to_string(),
                    client_order_id: Some(client_id.to_string()),
                    discrepancy_type: DiscrepancyType::MissingOrder,
                    local_value: "NOT_FOUND".to_string(),
                    exchange_value: format!("{:?}", exchange),
                    detected_at_ms: now,
                    resolved: false,
                    resolution_action: None,
                };
                discrepancies += 1;
                self.record_discrepancy(discrepancy);
            }
        }
        
        // Determine sync status
        let sync_status = if discrepancies == 0 {
            SyncStatus::InSync
        } else if discrepancies <= 2 {
            SyncStatus::MinorDrift
        } else if discrepancies <= 5 {
            SyncStatus::MajorDrift
        } else {
            SyncStatus::Critical
        };
        
        // Update counters
        self.total_reconciliations.fetch_add(1, Ordering::Relaxed);
        self.total_discrepancies_found.fetch_add(discrepancies as u64, Ordering::Relaxed);
        self.total_discrepancies_fixed.fetch_add(fixed as u64, Ordering::Relaxed);
        self.consecutive_failures.store(0, Ordering::Relaxed);
        
        // Update last sync time
        self.last_sync_per_symbol.insert(symbol.to_string(), now / 1000);
        
        ReconciliationStatus {
            symbol: symbol.to_string(),
            local_orders_count: local_orders.len() as u32,
            exchange_orders_count: exchange_orders.len() as u32,
            matched_orders: matched,
            discrepancies_found: discrepancies,
            discrepancies_fixed: fixed,
            last_sync_timestamp_ms: now,
            sync_status,
        }
    }

    /// Compare two orders and return discrepancy if found
    fn compare_orders(
        &self,
        symbol: &str,
        local: &LocalOrderSnapshot,
        exchange: &ExchangeOrderSnapshot,
    ) -> Option<DiscrepancyRecord> {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64;
        
        // Check status mismatch
        if local.status != exchange.status {
            return Some(DiscrepancyRecord {
                discrepancy_id: format!("STATUS_{}_{}", symbol, local.client_order_id),
                symbol: symbol.to_string(),
                client_order_id: Some(local.client_order_id.clone()),
                discrepancy_type: DiscrepancyType::StatusMismatch,
                local_value: local.status.clone(),
                exchange_value: exchange.status.clone(),
                detected_at_ms: now,
                resolved: false,
                resolution_action: None,
            });
        }
        
        // Check quantity mismatch
        if (local.filled_qty - exchange.filled_qty).abs() > 0.0001 {
            return Some(DiscrepancyRecord {
                discrepancy_id: format!("QTY_{}_{}", symbol, local.client_order_id),
                symbol: symbol.to_string(),
                client_order_id: Some(local.client_order_id.clone()),
                discrepancy_type: DiscrepancyType::QuantityMismatch,
                local_value: local.filled_qty.to_string(),
                exchange_value: exchange.filled_qty.to_string(),
                detected_at_ms: now,
                resolved: false,
                resolution_action: None,
            });
        }
        
        None
    }

    /// Record a discrepancy
    fn record_discrepancy(&mut self, discrepancy: DiscrepancyRecord) {
        self.discrepancy_history.push(discrepancy);
        
        // Trim history if too large
        if self.discrepancy_history.len() > self.max_history_size {
            self.discrepancy_history.remove(0);
        }
    }

    /// Check if discrepancy can be auto-fixed
    fn can_auto_fix(&self, discrepancy: &DiscrepancyRecord) -> bool {
        matches!(
            discrepancy.discrepancy_type,
            DiscrepancyType::StatusMismatch | DiscrepancyType::FillCountMismatch
        )
    }

    /// Get recent discrepancies
    pub fn get_recent_discrepancies(&self, limit: usize) -> Vec<&DiscrepancyRecord> {
        self.discrepancy_history
            .iter()
            .rev()
            .take(limit)
            .collect()
    }

    /// Get all unreconciled discrepancies
    pub fn get_unresolved_discrepancies(&self) -> Vec<&DiscrepancyRecord> {
        self.discrepancy_history
            .iter()
            .filter(|d| !d.resolved)
            .collect()
    }

    /// Mark discrepancy as resolved
    pub fn mark_resolved(&mut self, discrepancy_id: &str, action: &str) {
        for discrepancy in &mut self.discrepancy_history {
            if discrepancy.discrepancy_id == discrepancy_id {
                discrepancy.resolved = true;
                discrepancy.resolution_action = Some(action.to_string());
                break;
            }
        }
    }

    /// Perform full reconciliation across all symbols
    pub fn full_reconciliation(
        &mut self,
        all_local_orders: &HashMap<String, Vec<LocalOrderSnapshot>>,
        all_exchange_orders: &HashMap<String, Vec<ExchangeOrderSnapshot>>,
    ) -> ReconciliationSummary {
        let start = SystemTime::now();
        
        let mut total_discrepancies = 0u32;
        let mut total_fixed = 0u32;
        let mut critical_issues = 0u32;
        let mut recommendations = Vec::new();
        
        for symbol in &self.monitored_symbols {
            let local = all_local_orders.get(symbol)
                .map(|v| v.as_slice())
                .unwrap_or(&[]);
            
            let exchange = all_exchange_orders.get(symbol)
                .map(|v| v.as_slice())
                .unwrap_or(&[]);
            
            let status = self.reconcile_symbol(symbol, local, exchange);
            
            total_discrepancies += status.discrepancies_found;
            total_fixed += status.discrepancies_fixed;
            
            if status.sync_status == SyncStatus::Critical {
                critical_issues += 1;
                recommendations.push(format!(
                    "Critical drift on {}: {} discrepancies detected",
                    symbol, status.discrepancies_found
                ));
            }
        }
        
        let duration_ms = SystemTime::now()
            .duration_since(start)
            .unwrap()
            .as_millis() as u64;
        
        // Add general recommendations
        if total_discrepancies > 10 {
            recommendations.push("High discrepancy rate detected - check WebSocket connection stability".to_string());
        }
        
        ReconciliationSummary {
            total_symbols_checked: self.monitored_symbols.len() as u32,
            total_discrepancies_found: total_discrepancies,
            total_discrepancies_fixed: total_fixed,
            sync_duration_ms: duration_ms,
            critical_issues,
            recommendations,
        }
    }

    /// Get daemon statistics
    pub fn get_stats(&self) -> ReconciliationStats {
        ReconciliationStats {
            is_active: self.is_active.load(Ordering::Relaxed),
            monitored_symbols_count: self.monitored_symbols.len() as u32,
            total_reconciliations: self.total_reconciliations.load(Ordering::Relaxed),
            total_discrepancies_found: self.total_discrepancies_found.load(Ordering::Relaxed),
            total_discrepancies_fixed: self.total_discrepancies_fixed.load(Ordering::Relaxed),
            consecutive_failures: self.consecutive_failures.load(Ordering::Relaxed),
            history_size: self.discrepancy_history.len() as u32,
        }
    }

    /// Set sync interval
    #[inline(always)]
    pub fn set_sync_interval(&mut self, interval_sec: u64) {
        self.sync_interval_sec = interval_sec.max(5); // Minimum 5 seconds
    }

    /// Activate daemon
    #[inline(always)]
    pub fn activate(&self) {
        self.is_active.store(true, Ordering::Relaxed);
    }

    /// Deactivate daemon
    #[inline(always)]
    pub fn deactivate(&self) {
        self.is_active.store(false, Ordering::Relaxed);
    }
}

/// Local order snapshot for comparison
#[derive(Debug, Clone)]
pub struct LocalOrderSnapshot {
    pub client_order_id: String,
    pub symbol: String,
    pub side: String,
    pub quantity: f64,
    pub filled_qty: f64,
    pub price: f64,
    pub status: String,
}

/// Exchange order snapshot for comparison
#[derive(Debug, Clone)]
pub struct ExchangeOrderSnapshot {
    pub client_order_id: String,
    pub symbol: String,
    pub side: String,
    pub quantity: f64,
    pub filled_qty: f64,
    pub price: f64,
    pub status: String,
}

/// Reconciliation statistics
#[derive(Debug, Clone)]
pub struct ReconciliationStats {
    pub is_active: bool,
    pub monitored_symbols_count: u32,
    pub total_reconciliations: u64,
    pub total_discrepancies_found: u64,
    pub total_discrepancies_fixed: u64,
    pub consecutive_failures: u64,
    pub history_size: u32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reconciliation_daemon() {
        let mut daemon = ReconciliationDaemon::new(60);
        daemon.add_symbol("BTCUSDT");
        
        let local_orders = vec![
            LocalOrderSnapshot {
                client_order_id: "ORDER_001".to_string(),
                symbol: "BTCUSDT".to_string(),
                side: "BUY".to_string(),
                quantity: 1.0,
                filled_qty: 0.5,
                price: 50_000.0,
                status: "PARTIALLY_FILLED".to_string(),
            },
        ];
        
        let exchange_orders = vec![
            ExchangeOrderSnapshot {
                client_order_id: "ORDER_001".to_string(),
                symbol: "BTCUSDT".to_string(),
                side: "BUY".to_string(),
                quantity: 1.0,
                filled_qty: 0.5,
                price: 50_000.0,
                status: "PARTIALLY_FILLED".to_string(),
            },
        ];
        
        let status = daemon.reconcile_symbol("BTCUSDT", &local_orders, &exchange_orders);
        
        assert_eq!(status.sync_status, SyncStatus::InSync);
        assert_eq!(status.discrepancies_found, 0);
    }

    #[test]
    fn test_discrepancy_detection() {
        let mut daemon = ReconciliationDaemon::new(60);
        
        let local_orders = vec![
            LocalOrderSnapshot {
                client_order_id: "ORDER_002".to_string(),
                symbol: "ETHUSDT".to_string(),
                side: "SELL".to_string(),
                quantity: 5.0,
                filled_qty: 2.0,
                price: 3_000.0,
                status: "PARTIALLY_FILLED".to_string(),
            },
        ];
        
        // Exchange shows different fill quantity
        let exchange_orders = vec![
            ExchangeOrderSnapshot {
                client_order_id: "ORDER_002".to_string(),
                symbol: "ETHUSDT".to_string(),
                side: "SELL".to_string(),
                quantity: 5.0,
                filled_qty: 3.0, // Different!
                price: 3_000.0,
                status: "PARTIALLY_FILLED".to_string(),
            },
        ];
        
        let status = daemon.reconcile_symbol("ETHUSDT", &local_orders, &exchange_orders);
        
        assert!(status.discrepancies_found > 0);
        assert_eq!(status.sync_status, SyncStatus::MinorDrift);
    }
}
