//! Orphan Order Sweeper - Lock-free daemon for detecting and canceling orphaned orders.
//! Detects orders sent but not acknowledged due to network drops and aggressively cancels them.

use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::time::{Duration, Instant};
use std::collections::HashMap;
use crossbeam_queue::SegQueue;

/// Order record in the pending state
#[derive(Debug, Clone)]
pub struct PendingOrder {
    pub client_order_id: String,
    pub order_id: Option<u64>,
    pub symbol: String,
    pub side: OrderSide,
    pub quantity: f64,
    pub price: Option<f64>,
    pub sent_at: Instant,
    pub timeout_ms: u64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum OrderSide {
    Buy,
    Sell,
}

/// Orphan detection result
#[derive(Debug, Clone)]
pub struct OrphanedOrder {
    pub client_order_id: String,
    pub symbol: String,
    pub side: OrderSide,
    pub quantity: f64,
    pub age_ms: u64,
}

/// Configuration for orphan detection
#[derive(Debug, Clone)]
pub struct OrphanSweeperConfig {
    pub default_timeout_ms: u64,
    pub scan_interval_ms: u64,
    pub aggressive_mode: bool, // Cancel immediately on network drop detection
}

impl Default for OrphanSweeperConfig {
    fn default() -> Self {
        Self {
            default_timeout_ms: 5000,
            scan_interval_ms: 100,
            aggressive_mode: true,
        }
    }
}

/// Lock-free Orphan Order Sweeper
pub struct OrphanOrderSweeper {
    config: OrphanSweeperConfig,
    pending_orders: SegQueue<PendingOrder>,
    confirmed_orders: SegQueue<String>, // client_order_id
    orphaned_count: AtomicU64,
    cancelled_count: AtomicU64,
    false_positives: AtomicU64,
    network_drops_detected: AtomicU64,
    active: AtomicBool,
    network_connected: AtomicBool,
}

impl OrphanOrderSweeper {
    pub fn new(config: OrphanSweeperConfig) -> Self {
        Self {
            config,
            pending_orders: SegQueue::new(),
            confirmed_orders: SegQueue::new(),
            orphaned_count: AtomicU64::new(0),
            cancelled_count: AtomicU64::new(0),
            false_positives: AtomicU64::new(0),
            network_drops_detected: AtomicU64::new(0),
            active: AtomicBool::new(true),
            network_connected: AtomicBool::new(true),
        }
    }

    /// Register a newly sent order as pending acknowledgment
    #[inline]
    pub fn register_pending(&self, order: PendingOrder) {
        self.pending_orders.push(order);
    }

    /// Confirm an order was acknowledged by the exchange
    #[inline]
    pub fn confirm_order(&self, client_order_id: &str) {
        self.confirmed_orders.push(client_order_id.to_string());
    }

    /// Detect network disconnection event
    #[inline]
    pub fn on_network_drop(&self) {
        self.network_drops_detected.fetch_add(1, Ordering::Relaxed);
        self.network_connected.store(false, Ordering::Relaxed);
        
        if self.config.aggressive_mode {
            // Will trigger immediate cancellation on next sweep
        }
    }

    /// Detect network reconnection event
    #[inline]
    pub fn on_network_restore(&self) {
        self.network_connected.store(true, Ordering::Relaxed);
    }

    /// Sweep for orphaned orders and return list of orders to cancel
    #[inline]
    pub fn sweep(&self) -> Vec<OrphanedOrder> {
        if !self.active.load(Ordering::Relaxed) {
            return Vec::new();
        }

        let now = Instant::now();
        let mut orphans = Vec::new();
        let mut still_pending = Vec::new();

        // Collect all pending orders
        while let Some(order) = self.pending_orders.pop() {
            // Check if this order was confirmed
            if self.is_confirmed(&order.client_order_id) {
                continue; // Order was confirmed, remove from pending
            }

            let age_ms = now.duration_since(order.sent_at).as_millis() as u64;
            
            // Check for timeout or network drop in aggressive mode
            let is_orphan = if age_ms > order.timeout_ms {
                true
            } else if self.config.aggressive_mode && !self.network_connected.load(Ordering::Relaxed) {
                true
            } else {
                false
            };

            if is_orphan {
                self.orphaned_count.fetch_add(1, Ordering::Relaxed);
                orphans.push(OrphanedOrder {
                    client_order_id: order.client_order_id.clone(),
                    symbol: order.symbol,
                    side: order.side,
                    quantity: order.quantity,
                    age_ms,
                });
            } else {
                still_pending.push(order);
            }
        }

        // Re-queue non-orphaned orders
        for order in still_pending {
            self.pending_orders.push(order);
        }

        orphans
    }

    /// Check if an order has been confirmed (non-blocking check)
    #[inline]
    fn is_confirmed(&self, client_order_id: &str) -> bool {
        // This is a simplified check - in production would use a concurrent hash set
        // For lock-free operation, we just check if it's in the confirmed queue
        // Note: This creates a memory leak in this simplified version
        // Production would need epoch-based reclamation or similar
        false // Placeholder - real implementation would check a concurrent set
    }

    /// Mark an order as successfully cancelled
    #[inline]
    pub fn mark_cancelled(&self, _client_order_id: &str) {
        self.cancelled_count.fetch_add(1, Ordering::Relaxed);
    }

    /// Mark a false positive (order was actually filled after being marked orphan)
    #[inline]
    pub fn mark_false_positive(&self) {
        self.false_positives.fetch_add(1, Ordering::Relaxed);
    }

    /// Get statistics
    pub fn get_stats(&self) -> SweeperStats {
        SweeperStats {
            pending_count: self.pending_orders.len() as u64,
            orphaned_count: self.orphaned_count.load(Ordering::Relaxed),
            cancelled_count: self.cancelled_count.load(Ordering::Relaxed),
            false_positives: self.false_positives.load(Ordering::Relaxed),
            network_drops_detected: self.network_drops_detected.load(Ordering::Relaxed),
            network_connected: self.network_connected.load(Ordering::Relaxed),
        }
    }

    /// Set active state
    pub fn set_active(&self, active: bool) {
        self.active.store(active, Ordering::Relaxed);
    }

    /// Clear all state (use with caution)
    pub fn clear(&self) {
        while self.pending_orders.pop().is_some() {}
        while self.confirmed_orders.pop().is_some() {}
    }
}

#[derive(Debug, Clone, Copy)]
pub struct SweeperStats {
    pub pending_count: u64,
    pub orphaned_count: u64,
    pub cancelled_count: u64,
    pub false_positives: u64,
    pub network_drops_detected: u64,
    pub network_connected: bool,
}

/// REST API client interface for cancel operations
pub trait CancelClient {
    fn cancel_order(&self, symbol: &str, client_order_id: &str) -> Result<(), String>;
    fn cancel_all_orders(&self, symbol: &str) -> Result<u64, String>;
}

/// Aggressive canceller for emergency situations
pub struct EmergencyCanceller<T: CancelClient> {
    client: T,
    cancellation_count: AtomicU64,
    last_cancellation: AtomicU64, // Timestamp in ms
}

impl<T: CancelClient> EmergencyCanceller<T> {
    pub fn new(client: T) -> Self {
        Self {
            client,
            cancellation_count: AtomicU64::new(0),
            last_cancellation: AtomicU64::new(0),
        }
    }

    /// Cancel a single order
    pub fn cancel(&self, symbol: &str, client_order_id: &str) -> Result<(), String> {
        let result = self.client.cancel_order(symbol, client_order_id);
        if result.is_ok() {
            let now_ms = Instant::now().duration_since(Instant::now()).as_millis() as u64;
            self.cancellation_count.fetch_add(1, Ordering::Relaxed);
            self.last_cancellation.store(now_ms, Ordering::Relaxed);
        }
        result
    }

    /// Cancel all orders for a symbol
    pub fn cancel_all(&self, symbol: &str) -> Result<u64, String> {
        let result = self.client.cancel_all_orders(symbol);
        if let Ok(count) = result {
            let now_ms = Instant::now().duration_since(Instant::now()).as_millis() as u64;
            self.cancellation_count.fetch_add(count, Ordering::Relaxed);
            self.last_cancellation.store(now_ms, Ordering::Relaxed);
        }
        result
    }

    pub fn get_stats(&self) -> CancellerStats {
        CancellerStats {
            total_cancellations: self.cancellation_count.load(Ordering::Relaxed),
            last_cancellation_ms: self.last_cancellation.load(Ordering::Relaxed),
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct CancellerStats {
    pub total_cancellations: u64,
    pub last_cancellation_ms: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_orphan_detection() {
        let config = OrphanSweeperConfig {
            default_timeout_ms: 100,
            scan_interval_ms: 50,
            aggressive_mode: false,
        };
        let sweeper = OrphanOrderSweeper::new(config);

        let order = PendingOrder {
            client_order_id: "test_1".to_string(),
            order_id: None,
            symbol: "BTCUSDT".to_string(),
            side: OrderSide::Buy,
            quantity: 1.0,
            price: Some(50000.0),
            sent_at: Instant::now() - Duration::from_millis(200),
            timeout_ms: 100,
        };

        sweeper.register_pending(order);
        
        let orphans = sweeper.sweep();
        assert_eq!(orphans.len(), 1);
        assert_eq!(sweeper.get_stats().orphaned_count, 1);
    }

    #[test]
    fn test_aggressive_mode_on_network_drop() {
        let config = OrphanSweeperConfig {
            default_timeout_ms: 5000,
            scan_interval_ms: 50,
            aggressive_mode: true,
        };
        let sweeper = OrphanOrderSweeper::new(config);

        let order = PendingOrder {
            client_order_id: "test_2".to_string(),
            order_id: None,
            symbol: "BTCUSDT".to_string(),
            side: OrderSide::Sell,
            quantity: 0.5,
            price: None,
            sent_at: Instant::now(),
            timeout_ms: 5000,
        };

        sweeper.register_pending(order);
        sweeper.on_network_drop();
        
        let orphans = sweeper.sweep();
        assert_eq!(orphans.len(), 1);
    }
}
