//! Order lifecycle manager handling Binance WebSocket order updates.
//! Reconciles local Nautilus state with exchange state to prevent orphaned/duplicate orders.

use std::sync::atomic::{AtomicF64, AtomicU64, AtomicBool, Ordering};
use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};

/// Binance order status
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BinanceOrderStatus {
    New,
    PartiallyFilled,
    Filled,
    Canceled,
    PendingCancel,
    Rejected,
    Expired,
}

/// Binance order side
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BinanceOrderSide {
    Buy,
    Sell,
}

/// Binance order type
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BinanceOrderType {
    Limit,
    Market,
    StopLoss,
    StopLossLimit,
    TakeProfit,
    TakeProfitLimit,
}

/// Binance order time in force
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BinanceTimeInForce {
    GTC, // Good Till Cancel
    IOC, // Immediate or Cancel
    FOK, // Fill or Kill
    GTX, // Post-only (Good Till Crossing)
}

/// Order data from Binance executionReport
#[derive(Debug, Clone)]
pub struct BinanceOrderUpdate {
    pub symbol: String,
    pub order_id: u64,
    pub client_order_id: String,
    pub side: BinanceOrderSide,
    pub order_type: BinanceOrderType,
    pub time_in_force: BinanceTimeInForce,
    pub quantity: f64,
    pub price: f64,
    pub stop_price: Option<f64>,
    pub executed_qty: f64,
    pub cummulative_quote_qty: f64,
    pub average_price: f64,
    pub status: BinanceOrderStatus,
    pub update_reason: OrderUpdateReason,
    pub timestamp_ms: u64,
    pub last_executed_qty: f64,
    pub last_executed_price: f64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum OrderUpdateReason {
    PlacementNew,
    Canceled,
    Replaced,
    Triggered,
    Expired,
    Trade,
}

/// Local order state for tracking
#[derive(Debug, Clone)]
pub struct LocalOrderState {
    pub client_order_id: String,
    pub symbol: String,
    pub side: BinanceOrderSide,
    pub quantity: f64,
    pub filled_quantity: f64,
    pub average_price: f64,
    pub status: BinanceOrderStatus,
    pub created_at_ns: u64,
    pub updated_at_ns: u64,
    pub fill_count: u32,
}

/// Order reconciliation result
#[derive(Debug, Clone)]
pub struct ReconciliationResult {
    pub matched: bool,
    pub discrepancy_type: Option<DiscrepancyType>,
    pub local_state: Option<LocalOrderState>,
    pub exchange_state: Option<BinanceOrderUpdate>,
    pub action_required: ReconciliationAction,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DiscrepancyType {
    QuantityMismatch,
    StatusMismatch,
    PriceMismatch,
    MissingLocal,
    MissingExchange,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ReconciliationAction {
    None,
    UpdateLocal,
    CancelOrphan,
    Resubmit,
    Alert,
}

/// Main Binance order manager
pub struct BinanceOrderManager {
    /// Local order states by client order ID
    local_orders: HashMap<String, LocalOrderState>,
    /// Order IDs by symbol
    orders_by_symbol: HashMap<String, Vec<String>>,
    /// Pending orders awaiting confirmation
    pending_orders: HashMap<String, u64>, // client_order_id -> creation_timestamp
    /// Filled orders (recent history)
    filled_orders: HashMap<String, BinanceOrderUpdate>,
    /// Max pending order age (ms)
    max_pending_age_ms: u64,
    /// Total orders placed today
    orders_today: AtomicU64,
    /// Total fills today
    fills_today: AtomicU64,
    /// Active flag
    is_active: AtomicBool,
}

impl BinanceOrderManager {
    /// Create new order manager
    pub fn new() -> Self {
        Self {
            local_orders: HashMap::new(),
            orders_by_symbol: HashMap::new(),
            pending_orders: HashMap::new(),
            filled_orders: HashMap::new(),
            max_pending_age_ms: 10_000, // 10 seconds
            orders_today: AtomicU64::new(0),
            fills_today: AtomicU64::new(0),
            is_active: AtomicBool::new(true),
        }
    }

    /// Register a new outgoing order
    pub fn register_outgoing_order(
        &mut self,
        client_order_id: &str,
        symbol: &str,
        side: BinanceOrderSide,
        quantity: f64,
        price: f64,
    ) {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64;
        
        let order_state = LocalOrderState {
            client_order_id: client_order_id.to_string(),
            symbol: symbol.to_string(),
            side,
            quantity,
            filled_quantity: 0.0,
            average_price: 0.0,
            status: BinanceOrderStatus::New,
            created_at_ns: now,
            updated_at_ns: now,
            fill_count: 0,
        };
        
        self.local_orders.insert(client_order_id.to_string(), order_state);
        self.pending_orders.insert(client_order_id.to_string(), now / 1_000_000); // Convert to ms
        
        // Index by symbol
        self.orders_by_symbol
            .entry(symbol.to_string())
            .or_insert_with(Vec::new)
            .push(client_order_id.to_string());
        
        self.orders_today.fetch_add(1, Ordering::Relaxed);
    }

    /// Process incoming executionReport from Binance WebSocket
    pub fn process_execution_report(&mut self, update: BinanceOrderUpdate) -> OrderProcessingResult {
        if !self.is_active.load(Ordering::Relaxed) {
            return OrderProcessingResult {
                success: false,
                error: Some("Order manager inactive".to_string()),
            };
        }
        
        let client_order_id = &update.client_order_id;
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64;
        
        // Remove from pending
        self.pending_orders.remove(client_order_id);
        
        // Get or create local state
        let local_state = self.local_orders.entry(client_order_id.clone())
            .or_insert_with(|| {
                LocalOrderState {
                    client_order_id: client_order_id.clone(),
                    symbol: update.symbol.clone(),
                    side: update.side,
                    quantity: update.quantity,
                    filled_quantity: 0.0,
                    average_price: 0.0,
                    status: BinanceOrderStatus::New,
                    created_at_ns: now,
                    updated_at_ns: now,
                    fill_count: 0,
                }
            });
        
        // Check for discrepancies
        let discrepancy = self.check_discrepancy(local_state, &update);
        
        // Update local state
        local_state.filled_quantity = update.executed_qty;
        local_state.average_price = update.average_price;
        local_state.status = update.status;
        local_state.updated_at_ns = now;
        
        if update.last_executed_qty > 0.0 {
            local_state.fill_count += 1;
            self.fills_today.fetch_add(1, Ordering::Relaxed);
        }
        
        // Handle filled orders
        if update.status == BinanceOrderStatus::Filled 
            || update.status == BinanceOrderStatus::Canceled 
            || update.status == BinanceOrderStatus::Expired 
            || update.status == BinanceOrderStatus::Rejected 
        {
            self.filled_orders.insert(client_order_id.clone(), update.clone());
        }
        
        OrderProcessingResult {
            success: true,
            error: None,
            discrepancy,
        }
    }

    /// Check for discrepancies between local and exchange state
    fn check_discrepancy(
        &self,
        local: &LocalOrderState,
        exchange: &BinanceOrderUpdate,
    ) -> Option<DiscrepancyType> {
        // Quantity mismatch
        if (local.filled_quantity - exchange.executed_qty).abs() > 0.0001 {
            return Some(DiscrepancyType::QuantityMismatch);
        }
        
        // Status mismatch (significant ones)
        if local.status != exchange.status {
            match (local.status, exchange.status) {
                (BinanceOrderStatus::New, BinanceOrderStatus::Canceled) => {
                    // Possible race condition
                    return Some(DiscrepancyType::StatusMismatch);
                }
                (BinanceOrderStatus::Filled, _) => {
                    // Already filled locally but different on exchange
                    return Some(DiscrepancyType::StatusMismatch);
                }
                _ => {}
            }
        }
        
        None
    }

    /// Get order status by client order ID
    pub fn get_order_status(&self, client_order_id: &str) -> Option<&LocalOrderState> {
        self.local_orders.get(client_order_id)
    }

    /// Get all active orders for a symbol
    pub fn get_active_orders(&self, symbol: &str) -> Vec<&LocalOrderState> {
        let mut active = Vec::new();
        
        if let Some(order_ids) = self.orders_by_symbol.get(symbol) {
            for order_id in order_ids {
                if let Some(state) = self.local_orders.get(order_id) {
                    match state.status {
                        BinanceOrderStatus::New | BinanceOrderStatus::PartiallyFilled => {
                            active.push(state);
                        }
                        _ => {}
                    }
                }
            }
        }
        
        active
    }

    /// Check for stale pending orders
    pub fn check_stale_pending_orders(&self) -> Vec<String> {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64;
        
        let mut stale = Vec::new();
        
        for (client_order_id, created_at) in &self.pending_orders {
            if now - created_at > self.max_pending_age_ms {
                stale.push(client_order_id.clone());
            }
        }
        
        stale
    }

    /// Remove filled order from active tracking
    pub fn cleanup_filled_orders(&mut self, older_than_hours: u64) {
        let cutoff_ns = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64 - (older_than_hours * 3_600_000_000_000);
        
        let mut to_remove = Vec::new();
        
        for (client_order_id, state) in &self.local_orders {
            if state.status == BinanceOrderStatus::Filled 
                && state.updated_at_ns < cutoff_ns 
            {
                to_remove.push(client_order_id.clone());
            }
        }
        
        for order_id in to_remove {
            self.local_orders.remove(&order_id);
        }
    }

    /// Get daily statistics
    pub fn get_daily_stats(&self) -> DailyOrderStats {
        DailyOrderStats {
            orders_placed: self.orders_today.load(Ordering::Relaxed),
            fills_received: self.fills_today.load(Ordering::Relaxed),
            pending_orders: self.pending_orders.len() as u64,
            active_orders: self.local_orders.values()
                .filter(|s| matches!(s.status, BinanceOrderStatus::New | BinanceOrderStatus::PartiallyFilled))
                .count() as u64,
        }
    }

    /// Reset daily counters
    #[inline(always)]
    pub fn reset_daily_counters(&self) {
        self.orders_today.store(0, Ordering::Relaxed);
        self.fills_today.store(0, Ordering::Relaxed);
    }

    /// Set maximum pending order age
    #[inline(always)]
    pub fn set_max_pending_age_ms(&mut self, age_ms: u64) {
        self.max_pending_age_ms = age_ms;
    }

    /// Deactivate order manager
    #[inline(always)]
    pub fn deactivate(&self) {
        self.is_active.store(false, Ordering::Relaxed);
    }

    /// Activate order manager
    #[inline(always)]
    pub fn activate(&self) {
        self.is_active.store(true, Ordering::Relaxed);
    }
}

impl Default for BinanceOrderManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Order processing result
#[derive(Debug, Clone)]
pub struct OrderProcessingResult {
    pub success: bool,
    pub error: Option<String>,
    pub discrepancy: Option<DiscrepancyType>,
}

/// Daily order statistics
#[derive(Debug, Clone)]
pub struct DailyOrderStats {
    pub orders_placed: u64,
    pub fills_received: u64,
    pub pending_orders: u64,
    pub active_orders: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_order_lifecycle() {
        let mut manager = BinanceOrderManager::new();
        
        // Register outgoing order
        manager.register_outgoing_order(
            "ORDER_001",
            "BTCUSDT",
            BinanceOrderSide::Buy,
            1.0,
            50_000.0,
        );
        
        assert_eq!(manager.pending_orders.len(), 1);
        
        // Simulate execution report
        let update = BinanceOrderUpdate {
            symbol: "BTCUSDT".to_string(),
            order_id: 12345,
            client_order_id: "ORDER_001".to_string(),
            side: BinanceOrderSide::Buy,
            order_type: BinanceOrderType::Limit,
            time_in_force: BinanceTimeInForce::GTC,
            quantity: 1.0,
            price: 50_000.0,
            stop_price: None,
            executed_qty: 0.0,
            cummulative_quote_qty: 0.0,
            average_price: 0.0,
            status: BinanceOrderStatus::New,
            update_reason: OrderUpdateReason::PlacementNew,
            timestamp_ms: 0,
            last_executed_qty: 0.0,
            last_executed_price: 0.0,
        };
        
        let result = manager.process_execution_report(update);
        assert!(result.success);
        assert_eq!(manager.pending_orders.len(), 0);
    }

    #[test]
    fn test_fill_tracking() {
        let mut manager = BinanceOrderManager::new();
        
        manager.register_outgoing_order("ORDER_002", "ETHUSDT", BinanceOrderSide::Sell, 5.0, 3_000.0);
        
        // Partial fill
        let partial = BinanceOrderUpdate {
            symbol: "ETHUSDT".to_string(),
            order_id: 12346,
            client_order_id: "ORDER_002".to_string(),
            side: BinanceOrderSide::Sell,
            order_type: BinanceOrderType::Limit,
            time_in_force: BinanceTimeInForce::GTC,
            quantity: 5.0,
            price: 3_000.0,
            stop_price: None,
            executed_qty: 2.0,
            cummulative_quote_qty: 6_000.0,
            average_price: 3_000.0,
            status: BinanceOrderStatus::PartiallyFilled,
            update_reason: OrderUpdateReason::Trade,
            timestamp_ms: 0,
            last_executed_qty: 2.0,
            last_executed_price: 3_000.0,
        };
        
        manager.process_execution_report(partial);
        
        let stats = manager.get_daily_stats();
        assert_eq!(stats.fills_received, 1);
    }
}
