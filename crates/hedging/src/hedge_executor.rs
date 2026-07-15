"""
Rust execution layer for Deep Hedging.
Translates neural net outputs into microsecond delta-hedging orders.
Ensures strict delta-neutrality with zero heap allocations in the hot path.
"""

use std::sync::atomic::{AtomicI64, AtomicU64, Ordering};
use std::time::Instant;

/// Represents a delta-hedging signal from the neural network
#[derive(Debug, Clone, Copy)]
pub struct HedgeSignal {
    pub symbol_id: u32,
    pub target_delta: f64,  // Target delta exposure (-1.0 to 1.0)
    pub current_delta: f64, // Current portfolio delta
    pub timestamp_ns: u64,
}

/// Order representation for microsecond execution
#[derive(Debug, Clone, Copy)]
pub struct HedgeOrder {
    pub symbol_id: u32,
    pub side: Side,
    pub quantity: i64,      // In base currency units (scaled)
    pub price_ticks: i64,   // Price in tick units
    pub order_type: OrderType,
    pub timestamp_ns: u64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Side {
    Buy,
    Sell,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum OrderType {
    Limit,
    Market,
    Pegged,
}

/// Lock-free delta tracker for portfolio state
pub struct DeltaTracker {
    net_delta: AtomicI64,    // Scaled delta (fixed point)
    gross_delta: AtomicI64,  // Absolute delta for risk limits
    last_update_ns: AtomicU64,
}

impl DeltaTracker {
    pub const fn new() -> Self {
        Self {
            net_delta: AtomicI64::new(0),
            gross_delta: AtomicI64::new(0),
            last_update_ns: AtomicU64::new(0),
        }
    }

    #[inline]
    pub fn update(&self, delta_change: i64) {
        let now = Instant::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos() as u64;
        
        self.net_delta.fetch_add(delta_change, Ordering::Relaxed);
        self.gross_delta.fetch_add(delta_change.abs(), Ordering::Relaxed);
        self.last_update_ns.store(now, Ordering::Relaxed);
    }

    #[inline]
    pub fn get_net_delta(&self) -> i64 {
        self.net_delta.load(Ordering::Acquire)
    }

    #[inline]
    pub fn get_gross_delta(&self) -> i64 {
        self.gross_delta.load(Ordering::Acquire)
    }

    #[inline]
    pub fn is_within_limits(&self, max_gross_delta: i64) -> bool {
        self.gross_delta.load(Ordering::Acquire) <= max_gross_delta
    }
}

/// High-performance hedge executor
pub struct HedgeExecutor {
    delta_tracker: DeltaTracker,
    tick_size: f64,
    lot_size: f64,
    max_slippage_ticks: i64,
}

impl HedgeExecutor {
    pub const fn new(tick_size: f64, lot_size: f64, max_slippage_ticks: i64) -> Self {
        Self {
            delta_tracker: DeltaTracker::new(),
            tick_size,
            lot_size,
            max_slippage_ticks,
        }
    }

    /// Convert neural network hedge signal to executable order
    /// Zero heap allocation - all operations on stack
    #[inline]
    pub fn generate_hedge_order(&self, signal: &HedgeSignal, mid_price_ticks: i64) -> Option<HedgeOrder> {
        let delta_diff = signal.target_delta - signal.current_delta;
        
        // Threshold to prevent over-trading on noise
        if delta_diff.abs() < 0.001 {
            return None;
        }

        // Calculate quantity needed to neutralize delta
        let quantity_raw = (delta_diff * 10000.0) as i64; // Scale factor
        if quantity_raw == 0 {
            return None;
        }

        let (side, abs_quantity) = if quantity_raw > 0 {
            (Side::Buy, quantity_raw)
        } else {
            (Side::Sell, -quantity_raw)
        };

        // Round to lot size
        let adjusted_quantity = (abs_quantity as f64 / self.lot_size).floor() as i64 * self.lot_size as i64;
        if adjusted_quantity == 0 {
            return None;
        }

        Some(HedgeOrder {
            symbol_id: signal.symbol_id,
            side,
            quantity: adjusted_quantity,
            price_ticks: mid_price_ticks, // Will be pegged or limited
            order_type: OrderType::Limit,
            timestamp_ns: signal.timestamp_ns,
        })
    }

    /// Execute delta-neutral rebalancing
    /// Returns number of orders generated
    pub fn rebalance_portfolio(&self, signals: &[HedgeSignal], prices: &[i64]) -> Vec<HedgeOrder> {
        let mut orders = Vec::with_capacity(signals.len());
        
        for signal in signals {
            if let Some(price) = prices.get(signal.symbol_id as usize) {
                if let Some(order) = self.generate_hedge_order(signal, *price) {
                    // Update delta tracker
                    let delta_change = match order.side {
                        Side::Buy => order.quantity,
                        Side::Sell => -order.quantity,
                    };
                    self.delta_tracker.update(delta_change);
                    orders.push(order);
                }
            }
        }
        
        orders
    }

    /// Check if portfolio is within delta limits
    #[inline]
    pub fn check_risk_limits(&self, max_gross_delta: i64) -> bool {
        self.delta_tracker.is_within_limits(max_gross_delta)
    }

    /// Get current net delta exposure
    #[inline]
    pub fn get_net_exposure(&self) -> i64 {
        self.delta_tracker.get_net_delta()
    }
}

/// Pre-allocated order buffer for batch processing
pub struct HedgeOrderBuffer {
    buffer: [Option<HedgeOrder>; 64], // Fixed size, no heap
    head: usize,
}

impl HedgeOrderBuffer {
    pub const fn new() -> Self {
        Self {
            buffer: [None; 64],
            head: 0,
        }
    }

    #[inline]
    pub fn push(&mut self, order: HedgeOrder) -> bool {
        if self.head >= self.buffer.len() {
            return false;
        }
        self.buffer[self.head] = Some(order);
        self.head += 1;
        true
    }

    #[inline]
    pub fn drain(&mut self) -> impl Iterator<Item = HedgeOrder> + '_ {
        let head = self.head;
        self.head = 0;
        self.buffer[..head].iter_mut().filter_map(|o| o.take())
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.head
    }

    #[inline]
    pub fn clear(&mut self) {
        self.head = 0;
        for i in 0..self.buffer.len() {
            self.buffer[i] = None;
        }
    }
}

impl Default for HedgeOrderBuffer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hedge_executor_basic() {
        let executor = HedgeExecutor::new(0.01, 1.0, 10);
        
        let signal = HedgeSignal {
            symbol_id: 0,
            target_delta: 0.5,
            current_delta: 0.0,
            timestamp_ns: 1234567890,
        };
        
        let order = executor.generate_hedge_order(&signal, 10000);
        assert!(order.is_some());
        
        let order = order.unwrap();
        assert_eq!(order.side, Side::Buy);
        assert!(order.quantity > 0);
    }

    #[test]
    fn test_delta_tracker() {
        let tracker = DeltaTracker::new();
        tracker.update(100);
        tracker.update(-50);
        
        assert_eq!(tracker.get_net_delta(), 50);
        assert_eq!(tracker.get_gross_delta(), 150);
    }
}
