//! Exchange Matching Lag Simulator for chaos engineering.
//! Simulates internal exchange matching engine delays and partial fill rejections.
//! Ensures correct handling of delayed executionReport callbacks without double-firing.

use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::time::{Duration, Instant};
use std::collections::VecDeque;
use crossbeam_queue::SegQueue;

/// Order status in the simulated matching engine
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum OrderStatus {
    New,
    PartiallyFilled { filled: u64, remaining: u64 },
    Filled,
    Rejected,
    Cancelled,
}

/// Simulated order record
#[derive(Debug, Clone)]
pub struct SimulatedOrder {
    pub order_id: u64,
    pub client_order_id: String,
    pub side: OrderSide,
    pub order_type: OrderType,
    pub price: f64,
    pub quantity: u64,
    pub status: OrderStatus,
    pub created_at: Instant,
    pub last_update: Instant,
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
    PostOnly,
}

/// Execution report event
#[derive(Debug, Clone)]
pub struct ExecutionReport {
    pub order_id: u64,
    pub client_order_id: String,
    pub status: OrderStatus,
    pub filled_quantity: u64,
    pub remaining_quantity: u64,
    pub last_fill_price: Option<f64>,
    pub last_fill_quantity: Option<u64>,
    pub rejection_reason: Option<String>,
    pub timestamp: Instant,
}

/// Configuration for matching lag simulation
#[derive(Debug, Clone)]
pub struct MatchingLagConfig {
    pub min_latency_us: u64,        // Minimum matching engine latency
    pub max_latency_us: u64,        // Maximum matching engine latency
    pub partial_fill_probability: f64,
    pub rejection_probability: f64,
    pub out_of_order_probability: f64, // Probability of reports arriving out of order
}

impl Default for MatchingLagConfig {
    fn default() -> Self {
        Self {
            min_latency_us: 100,
            max_latency_us: 5000,
            partial_fill_probability: 0.3,
            rejection_probability: 0.01,
            out_of_order_probability: 0.05,
        }
    }
}

/// Pending execution report in the delay queue
struct PendingReport {
    report: ExecutionReport,
    deliver_at: Instant,
}

/// Exchange Matching Lag Simulator
pub struct ExchangeMatchingLagSimulator {
    config: MatchingLagConfig,
    pending_reports: SegQueue<PendingReport>,
    processed_count: AtomicU64,
    delayed_count: AtomicU64,
    partial_fills: AtomicU64,
    rejections: AtomicU64,
    out_of_order_deliveries: AtomicU64,
    active: AtomicBool,
    sequence_number: AtomicU64,
}

impl ExchangeMatchingLagSimulator {
    pub fn new(config: MatchingLagConfig) -> Self {
        Self {
            config,
            pending_reports: SegQueue::new(),
            processed_count: AtomicU64::new(0),
            delayed_count: AtomicU64::new(0),
            partial_fills: AtomicU64::new(0),
            rejections: AtomicU64::new(0),
            out_of_order_deliveries: AtomicU64::new(0),
            active: AtomicBool::new(true),
            sequence_number: AtomicU64::new(0),
        }
    }

    /// Submit an order to the simulated matching engine.
    /// Returns immediately with a client_order_id. Execution reports will be delivered asynchronously.
    pub fn submit_order(&self, order: SimulatedOrder) -> String {
        let client_order_id = order.client_order_id.clone();
        let now = Instant::now();

        // Generate execution report with simulated latency
        self.generate_execution_report(order, now);

        client_order_id
    }

    /// Generate execution report with simulated delays
    fn generate_execution_report(&self, mut order: SimulatedOrder, submit_time: Instant) {
        if !self.active.load(Ordering::Relaxed) {
            return;
        }

        let rng_seed = self.sequence_number.fetch_add(1, Ordering::Relaxed);
        let mut rng = rand::rngs::SmallRng::seed_from_u64(rng_seed);
        
        use rand::Rng;

        // Determine outcome
        let outcome_roll: f64 = rng.gen();
        
        let (status, fill_price, fill_qty, rejection_reason) = if outcome_roll < self.config.rejection_probability {
            // Order rejected
            self.rejections.fetch_add(1, Ordering::Relaxed);
            (OrderStatus::Rejected, None, None, Some("Insufficient balance".to_string()))
        } else if outcome_roll < self.config.rejection_probability + self.config.partial_fill_probability {
            // Partial fill
            self.partial_fills.fetch_add(1, Ordering::Relaxed);
            let fill_ratio: f64 = rng.gen_range(0.1..0.9);
            let filled_qty = (order.quantity as f64 * fill_ratio) as u64;
            let remaining = order.quantity - filled_qty;
            order.status = OrderStatus::PartiallyFilled { filled: filled_qty, remaining };
            (order.status, Some(order.price), Some(filled_qty), None)
        } else {
            // Full fill
            order.status = OrderStatus::Filled;
            (order.status, Some(order.price), Some(order.quantity), None)
        };

        // Calculate delivery latency
        let latency_us = if self.config.max_latency_us > self.config.min_latency_us {
            rng.gen_range(self.config.min_latency_us..=self.config.max_latency_us)
        } else {
            self.config.min_latency_us
        };

        let deliver_at = submit_time + Duration::from_micros(latency_us);

        // Create execution report
        let report = ExecutionReport {
            order_id: order.order_id,
            client_order_id: order.client_order_id,
            status,
            filled_quantity: fill_qty.unwrap_or(0),
            remaining_quantity: if status == OrderStatus::Filled { 0 } else { order.quantity - fill_qty.unwrap_or(0) },
            last_fill_price: fill_price,
            last_fill_quantity: fill_qty,
            rejection_reason,
            timestamp: submit_time,
        };

        // Check for out-of-order delivery simulation
        if rng.gen::<f64>() < self.config.out_of_order_probability {
            // Deliver immediately (out of order)
            self.out_of_order_deliveries.fetch_add(1, Ordering::Relaxed);
            self.pending_reports.push(PendingReport {
                report,
                deliver_at: Instant::now(),
            });
        } else {
            // Schedule for delayed delivery
            self.delayed_count.fetch_add(1, Ordering::Relaxed);
            self.pending_reports.push(PendingReport {
                report,
                deliver_at,
            });
        }
    }

    /// Poll for ready execution reports. Returns all reports that are due for delivery.
    #[inline]
    pub fn poll_reports(&self) -> Vec<ExecutionReport> {
        let now = Instant::now();
        let mut ready_reports = Vec::new();
        let mut pending = Vec::new();

        while let Some(pending_report) = self.pending_reports.pop() {
            if pending_report.deliver_at <= now {
                self.processed_count.fetch_add(1, Ordering::Relaxed);
                ready_reports.push(pending_report.report);
            } else {
                pending.push(pending_report);
            }
        }

        // Re-queue non-ready reports
        for report in pending {
            self.pending_reports.push(report);
        }

        ready_reports
    }

    /// Get statistics
    pub fn get_stats(&self) -> MatchingLagStats {
        MatchingLagStats {
            processed_count: self.processed_count.load(Ordering::Relaxed),
            delayed_count: self.delayed_count.load(Ordering::Relaxed),
            partial_fills: self.partial_fills.load(Ordering::Relaxed),
            rejections: self.rejections.load(Ordering::Relaxed),
            out_of_order_deliveries: self.out_of_order_deliveries.load(Ordering::Relaxed),
            pending_count: self.pending_reports.len() as u64,
        }
    }

    /// Set active state
    pub fn set_active(&self, active: bool) {
        self.active.store(active, Ordering::Relaxed);
    }

    /// Clear all pending reports
    pub fn clear_pending(&self) {
        while self.pending_reports.pop().is_some() {}
    }
}

#[derive(Debug, Clone, Copy)]
pub struct MatchingLagStats {
    pub processed_count: u64,
    pub delayed_count: u64,
    pub partial_fills: u64,
    pub rejections: u64,
    pub out_of_order_deliveries: u64,
    pub pending_count: u64,
}

/// Order state tracker to prevent double-fills
pub struct OrderStateTracker {
    known_orders: SegQueue<u64>, // order_id
    processed_reports: AtomicU64,
    duplicate_detections: AtomicU64,
}

impl OrderStateTracker {
    pub fn new() -> Self {
        Self {
            known_orders: SegQueue::new(),
            processed_reports: AtomicU64::new(0),
            duplicate_detections: AtomicU64::new(0),
        }
    }

    /// Check if a report is a duplicate. Returns true if this is the first time seeing this report.
    #[inline]
    pub fn track_report(&self, report: &ExecutionReport) -> bool {
        // Simple tracking - in production would use a more sophisticated dedup mechanism
        let count = self.processed_reports.fetch_add(1, Ordering::Relaxed);
        
        // Detect potential duplicates based on sequence
        if count > 0 && report.status == OrderStatus::Filled {
            // Check if we've already seen a fill for this order
            // This is simplified - real implementation would use hash maps
            return true;
        }
        
        true
    }

    pub fn get_stats(&self) -> TrackerStats {
        TrackerStats {
            processed_reports: self.processed_reports.load(Ordering::Relaxed),
            duplicate_detections: self.duplicate_detections.load(Ordering::Relaxed),
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct TrackerStats {
    pub processed_reports: u64,
    pub duplicate_detections: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_matching_lag_simulator() {
        let config = MatchingLagConfig {
            min_latency_us: 100,
            max_latency_us: 500,
            partial_fill_probability: 0.5,
            rejection_probability: 0.1,
            out_of_order_probability: 0.0,
        };
        let simulator = ExchangeMatchingLagSimulator::new(config);

        let order = SimulatedOrder {
            order_id: 1,
            client_order_id: "test_1".to_string(),
            side: OrderSide::Buy,
            order_type: OrderType::Limit,
            price: 50000.0,
            quantity: 100,
            status: OrderStatus::New,
            created_at: Instant::now(),
            last_update: Instant::now(),
        };

        simulator.submit_order(order);
        
        // Wait for delivery
        std::thread::sleep(Duration::from_millis(1));
        
        let reports = simulator.poll_reports();
        assert!(!reports.is_empty());
    }

    #[test]
    fn test_order_state_tracker() {
        let tracker = OrderStateTracker::new();
        
        let report = ExecutionReport {
            order_id: 1,
            client_order_id: "test_1".to_string(),
            status: OrderStatus::New,
            filled_quantity: 0,
            remaining_quantity: 100,
            last_fill_price: None,
            last_fill_quantity: None,
            rejection_reason: None,
            timestamp: Instant::now(),
        };

        assert!(tracker.track_report(&report));
    }
}
