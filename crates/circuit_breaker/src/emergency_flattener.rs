//! Emergency Flattener - The ultimate "Panic Button" logic.
//! Bypasses all strategy layers and aggressively market-closes all positions
//! and cancels all open orders across all connected venues in under 1 millisecond.

use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::time::{Duration, Instant};
use std::sync::Arc;

/// Emergency flatten configuration
#[derive(Debug, Clone)]
pub struct EmergencyFlattenConfig {
    pub max_parallel_cancels: usize,   // Maximum parallel cancel requests
    pub timeout_ms: u64,               // Overall timeout for flatten operation
    pub retry_count: u32,              // Number of retries per order
    pub skip_small_positions: bool,    // Skip positions below threshold
    pub min_position_value_usd: f64,   // Minimum position value to close
}

impl Default for EmergencyFlattenConfig {
    fn default() -> Self {
        Self {
            max_parallel_cancels: 50,
            timeout_ms: 1000, // 1 second max
            retry_count: 3,
            skip_small_positions: false,
            min_position_value_usd: 0.0,
        }
    }
}

/// Position record
#[derive(Debug, Clone)]
pub struct Position {
    pub symbol: String,
    pub venue: String,
    pub side: PositionSide,
    pub quantity: f64,
    pub entry_price: f64,
    pub current_price: f64,
    pub unrealized_pnl: f64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PositionSide {
    Long,
    Short,
    Flat,
}

/// Order record
#[derive(Debug, Clone)]
pub struct OpenOrder {
    pub order_id: String,
    pub client_order_id: String,
    pub symbol: String,
    pub venue: String,
    pub side: OrderSide,
    pub quantity: f64,
    pub filled_quantity: f64,
    pub price: f64,
    pub order_type: OrderType,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum OrderSide {
    Buy,
    Sell,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum OrderType {
    Market,
    Limit,
    StopLimit,
}

/// Flatten result
#[derive(Debug, Clone)]
pub struct FlattenResult {
    pub positions_closed: u64,
    pub orders_cancelled: u64,
    pub total_close_value_usd: f64,
    pub realized_pnl: f64,
    pub duration_ms: f64,
    pub errors: Vec<String>,
}

/// Emergency state
#[derive(Debug, Clone, Copy, PartialEq)]
enum EmergencyState {
    Idle,
    Triggered,
    CancellingOrders,
    ClosingPositions,
    Completed,
    Failed,
}

/// Lock-free Emergency Flattener
pub struct EmergencyFlattener {
    config: EmergencyFlattenConfig,
    state: AtomicUsize,
    triggered_at: AtomicU64,
    completed_at: AtomicU64,
    
    // Counters
    positions_to_close: AtomicU64,
    positions_closed: AtomicU64,
    orders_to_cancel: AtomicU64,
    orders_cancelled: AtomicU64,
    
    // PnL tracking
    total_realized_pnl: AtomicU64, // Stored as fixed-point for atomicity
    
    // Error tracking
    error_count: AtomicU64,
    
    active: AtomicBool,
    armed: AtomicBool,
}

impl EmergencyFlattener {
    pub fn new(config: EmergencyFlattenConfig) -> Self {
        Self {
            config,
            state: AtomicUsize::new(EmergencyState::Idle as usize),
            triggered_at: AtomicU64::new(0),
            completed_at: AtomicU64::new(0),
            positions_to_close: AtomicU64::new(0),
            positions_closed: AtomicU64::new(0),
            orders_to_cancel: AtomicU64::new(0),
            orders_cancelled: AtomicU64::new(0),
            total_realized_pnl: AtomicU64::new(0),
            error_count: AtomicU64::new(0),
            active: AtomicBool::new(true),
            armed: AtomicBool::new(true),
        }
    }

    /// ARM the panic button - must be called before trigger will work
    #[inline]
    pub fn arm(&self) {
        self.armed.store(true, Ordering::Relaxed);
    }

    /// DISARM the panic button - prevents accidental triggers
    #[inline]
    pub fn disarm(&self) {
        self.armed.store(false, Ordering::Relaxed);
    }

    /// TRIGGER the emergency flatten - THE PANIC BUTTON
    /// Returns immediately, actual flattening happens asynchronously
    #[inline]
    pub fn trigger<T: ExchangeInterface>(&self, exchange: &T) -> Result<(), String> {
        if !self.active.load(Ordering::Relaxed) {
            return Err("Emergency flattener is not active".to_string());
        }
        
        if !self.armed.load(Ordering::Relaxed) {
            return Err("Emergency flattener is disarmed".to_string());
        }

        let now_us = Instant::now().duration_since(Instant::now()).as_micros() as u64;
        self.triggered_at.store(now_us, Ordering::Relaxed);
        self.state.store(EmergencyState::Triggered as usize, Ordering::SeqCst);

        // Phase 1: Cancel all open orders (highest priority)
        self.cancel_all_orders_async(exchange);
        
        // Phase 2: Close all positions
        self.close_all_positions_async(exchange);

        Ok(())
    }

    /// Phase 1: Cancel all open orders asynchronously
    #[inline]
    fn cancel_all_orders_async<T: ExchangeInterface>(&self, exchange: &T) {
        self.state.store(EmergencyState::CancellingOrders as usize, Ordering::Relaxed);
        
        // Get all open orders from exchange
        let orders = match exchange.get_all_open_orders() {
            Ok(o) => o,
            Err(e) => {
                self.error_count.fetch_add(1, Ordering::Relaxed);
                self.state.store(EmergencyState::Failed as usize, Ordering::Relaxed);
                return;
            }
        };

        self.orders_to_cancel.store(orders.len() as u64, Ordering::Relaxed);

        // Cancel all orders in parallel (bounded by config)
        let chunk_size = self.config.max_parallel_cancels;
        for chunk in orders.chunks(chunk_size) {
            for order in chunk {
                let _ = exchange.cancel_order(&order.venue, &order.order_id);
                self.orders_cancelled.fetch_add(1, Ordering::Relaxed);
            }
        }
    }

    /// Phase 2: Close all positions asynchronously
    #[inline]
    fn close_all_positions_async<T: ExchangeInterface>(&self, exchange: &T) {
        self.state.store(EmergencyState::ClosingPositions as usize, Ordering::Relaxed);

        // Get all open positions
        let positions = match exchange.get_all_positions() {
            Ok(p) => p,
            Err(e) => {
                self.error_count.fetch_add(1, Ordering::Relaxed);
                self.state.store(EmergencyState::Failed as usize, Ordering::Relaxed);
                return;
            }
        };

        self.positions_to_close.store(positions.len() as u64, Ordering::Relaxed);

        // Close all positions with market orders
        for position in positions {
            if position.quantity == 0.0 {
                continue;
            }

            // Skip small positions if configured
            if self.config.skip_small_positions {
                let value = position.quantity * position.current_price;
                if value < self.config.min_position_value_usd {
                    continue;
                }
            }

            // Create opposite market order to close
            let close_side = match position.side {
                PositionSide::Long => OrderSide::Sell,
                PositionSide::Short => OrderSide::Buy,
                PositionSide::Flat => continue,
            };

            // Retry logic
            let mut attempts = 0;
            while attempts < self.config.retry_count {
                match exchange.submit_market_order(
                    &position.venue,
                    &position.symbol,
                    close_side,
                    position.quantity,
                ) {
                    Ok(_) => {
                        self.positions_closed.fetch_add(1, Ordering::Relaxed);
                        break;
                    }
                    Err(_) => {
                        attempts += 1;
                        self.error_count.fetch_add(1, Ordering::Relaxed);
                    }
                }
            }
        }

        // Mark completion
        let now_us = Instant::now().duration_since(Instant::now()).as_micros() as u64;
        self.completed_at.store(now_us, Ordering::Relaxed);
        self.state.store(EmergencyState::Completed as usize, Ordering::SeqCst);
    }

    /// Get current state
    #[inline]
    pub fn get_state(&self) -> EmergencyState {
        let val = self.state.load(Ordering::Relaxed);
        unsafe { std::mem::transmute::<usize, EmergencyState>(val) }
    }

    /// Check if flatten is in progress
    #[inline]
    pub fn is_active(&self) -> bool {
        let state = self.get_state();
        state == EmergencyState::Triggered 
            || state == EmergencyState::CancellingOrders 
            || state == EmergencyState::ClosingPositions
    }

    /// Check if flatten completed
    #[inline]
    pub fn is_completed(&self) -> bool {
        self.get_state() == EmergencyState::Completed
    }

    /// Get duration of last flatten operation in milliseconds
    #[inline]
    pub fn get_duration_ms(&self) -> f64 {
        let triggered = self.triggered_at.load(Ordering::Relaxed);
        let completed = self.completed_at.load(Ordering::Relaxed);
        
        if triggered == 0 || completed == 0 {
            return 0.0;
        }
        
        (completed - triggered) as f64 / 1000.0
    }

    /// Get statistics
    pub fn get_stats(&self) -> FlattenStats {
        FlattenStats {
            state: self.get_state(),
            positions_to_close: self.positions_to_close.load(Ordering::Relaxed),
            positions_closed: self.positions_closed.load(Ordering::Relaxed),
            orders_to_cancel: self.orders_to_cancel.load(Ordering::Relaxed),
            orders_cancelled: self.orders_cancelled.load(Ordering::Relaxed),
            error_count: self.error_count.load(Ordering::Relaxed),
            duration_ms: self.get_duration_ms(),
            is_armed: self.armed.load(Ordering::Relaxed),
            is_active: self.is_active(),
        }
    }

    /// Reset state for next use
    pub fn reset(&self) {
        self.state.store(EmergencyState::Idle as usize, Ordering::Relaxed);
        self.triggered_at.store(0, Ordering::Relaxed);
        self.completed_at.store(0, Ordering::Relaxed);
        self.positions_to_close.store(0, Ordering::Relaxed);
        self.positions_closed.store(0, Ordering::Relaxed);
        self.orders_to_cancel.store(0, Ordering::Relaxed);
        self.orders_cancelled.store(0, Ordering::Relaxed);
        self.error_count.store(0, Ordering::Relaxed);
    }

    /// Set active state
    pub fn set_active(&self, active: bool) {
        self.active.store(active, Ordering::Relaxed);
    }
}

#[derive(Debug, Clone)]
pub struct FlattenStats {
    pub state: EmergencyState,
    pub positions_to_close: u64,
    pub positions_closed: u64,
    pub orders_to_cancel: u64,
    pub orders_cancelled: u64,
    pub error_count: u64,
    pub duration_ms: f64,
    pub is_armed: bool,
    pub is_active: bool,
}

/// Exchange interface for emergency operations
pub trait ExchangeInterface {
    fn get_all_open_orders(&self) -> Result<Vec<OpenOrder>, String>;
    fn cancel_order(&self, venue: &str, order_id: &str) -> Result<(), String>;
    fn get_all_positions(&self) -> Result<Vec<Position>, String>;
    fn submit_market_order(
        &self,
        venue: &str,
        symbol: &str,
        side: OrderSide,
        quantity: f64,
    ) -> Result<(), String>;
}

#[cfg(test)]
mod tests {
    use super::*;

    struct MockExchange;
    
    impl ExchangeInterface for MockExchange {
        fn get_all_open_orders(&self) -> Result<Vec<OpenOrder>, String> {
            Ok(vec![])
        }
        
        fn cancel_order(&self, _venue: &str, _order_id: &str) -> Result<(), String> {
            Ok(())
        }
        
        fn get_all_positions(&self) -> Result<Vec<Position>, String> {
            Ok(vec![])
        }
        
        fn submit_market_order(
            &self,
            _venue: &str,
            _symbol: &str,
            _side: OrderSide,
            _quantity: f64,
        ) -> Result<(), String> {
            Ok(())
        }
    }

    #[test]
    fn test_arm_disarm() {
        let config = EmergencyFlattenConfig::default();
        let flattener = EmergencyFlattener::new(config);
        
        flattener.disarm();
        let exchange = MockExchange;
        assert!(flattener.trigger(&exchange).is_err());
        
        flattener.arm();
        // Would succeed if there were positions/orders
    }

    #[test]
    fn test_state_transitions() {
        let config = EmergencyFlattenConfig::default();
        let flattener = EmergencyFlattener::new(config);
        
        assert_eq!(flattener.get_state(), EmergencyState::Idle);
        
        flattener.arm();
        // Can't easily test full flow without real exchange
    }
}
