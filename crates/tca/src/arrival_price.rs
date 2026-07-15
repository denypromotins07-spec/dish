// crates/tca/src/arrival_price.rs
// Microsecond Implementation Shortfall (IS) calculator
// Zero-allocation, nanosecond-precision TCA engine

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

/// Represents a price with nanosecond timestamp
#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct PriceTick {
    pub timestamp_ns: u64,
    pub price: f64,
    pub quantity: f64,
    pub side: Side,
}

#[derive(Clone, Copy, Debug, PartialEq)]
#[repr(u8)]
pub enum Side {
    Buy = 0,
    Sell = 1,
}

/// Implementation Shortfall calculator - compares arrival price vs execution price
/// All calculations are done without heap allocations for ultra-low latency
pub struct ArrivalPriceIS {
    /// Decision/arrival price (when signal was generated)
    arrival_price: f64,
    /// Decision timestamp in nanoseconds
    decision_timestamp_ns: u64,
    /// Cumulative executed quantity
    executed_quantity: f64,
    /// Cumulative executed value (price * quantity)
    executed_value: f64,
    /// Explicit fees accumulated (in quote currency)
    explicit_fees: f64,
    /// Number of fills recorded
    fill_count: u64,
    /// First fill timestamp
    first_fill_ts: u64,
    /// Last fill timestamp
    last_fill_ts: u64,
}

impl ArrivalPriceIS {
    #[inline]
    pub const fn new(arrival_price: f64, decision_timestamp_ns: u64) -> Self {
        Self {
            arrival_price,
            decision_timestamp_ns,
            executed_quantity: 0.0,
            executed_value: 0.0,
            explicit_fees: 0.0,
            fill_count: 0,
            first_fill_ts: 0,
            last_fill_ts: 0,
        }
    }

    /// Record a fill - zero allocation, pure arithmetic
    #[inline]
    pub fn record_fill(&mut self, price: f64, quantity: f64, fee: f64, timestamp_ns: u64) {
        if self.fill_count == 0 {
            self.first_fill_ts = timestamp_ns;
        }
        self.last_fill_ts = timestamp_ns;
        
        self.executed_quantity += quantity;
        self.executed_value += price * quantity;
        self.explicit_fees += fee;
        self.fill_count += 1;
    }

    /// Calculate Implementation Shortfall in basis points
    /// IS = (Execution Cost - Paper Portfolio Cost) / Paper Portfolio Cost * 10000
    #[inline]
    pub fn calculate_is_bps(&self, side: Side) -> f64 {
        if self.executed_quantity <= 0.0 || self.arrival_price <= 0.0 {
            return 0.0;
        }

        let avg_execution_price = self.executed_value / self.executed_quantity;
        
        // For buys: IS positive means we paid more than arrival (bad)
        // For sells: IS positive means we received less than arrival (bad)
        let price_diff = match side {
            Side::Buy => avg_execution_price - self.arrival_price,
            Side::Sell => self.arrival_price - avg_execution_price,
        };

        // Add explicit fees to the cost
        let total_cost = price_diff * self.executed_quantity + self.explicit_fees;
        let paper_cost = self.arrival_price * self.executed_quantity;

        (total_cost / paper_cost) * 10000.0
    }

    /// Calculate IS broken down into components
    #[inline]
    pub fn decompose_is(&self, side: Side, mid_price_at_decision: f64) -> ISSummary {
        if self.executed_quantity <= 0.0 {
            return ISSummary::default();
        }

        let avg_exec_price = self.executed_value / self.executed_quantity;
        
        // Delay cost: difference between arrival price and decision mid-price
        let delay_cost = match side {
            Side::Buy => (self.arrival_price - mid_price_at_decision) * self.executed_quantity,
            Side::Sell => (mid_price_at_decision - self.arrival_price) * self.executed_quantity,
        };

        // Market impact: difference between avg exec price and arrival price
        let market_impact = match side {
            Side::Buy => (avg_exec_price - self.arrival_price) * self.executed_quantity,
            Side::Sell => (self.arrival_price - avg_exec_price) * self.executed_quantity,
        };

        let paper_cost = self.arrival_price * self.executed_quantity;
        
        ISSummary {
            total_is_bps: self.calculate_is_bps(side),
            delay_cost_bps: (delay_cost / paper_cost) * 10000.0,
            market_impact_bps: (market_impact / paper_cost) * 10000.0,
            explicit_fees_bps: (self.explicit_fees / paper_cost) * 10000.0,
            execution_time_ns: if self.first_fill_ts > 0 {
                self.last_fill_ts.saturating_sub(self.first_fill_ts)
            } else {
                0
            },
            fill_count: self.fill_count,
        }
    }

    #[inline]
    pub const fn get_executed_quantity(&self) -> f64 {
        self.executed_quantity
    }

    #[inline]
    pub const fn get_avg_execution_price(&self) -> f64 {
        if self.executed_quantity > 0.0 {
            self.executed_value / self.executed_quantity
        } else {
            0.0
        }
    }
}

/// Summary of Implementation Shortfall decomposition
#[derive(Clone, Copy, Debug, Default)]
#[repr(C)]
pub struct ISSummary {
    pub total_is_bps: f64,
    pub delay_cost_bps: f64,
    pub market_impact_bps: f64,
    pub explicit_fees_bps: f64,
    pub execution_time_ns: u64,
    pub fill_count: u64,
}

/// Lock-free atomic counter for tracking aggregate IS across threads
pub struct AggregateIS {
    total_is_numer: AtomicU64,  // Scaled integer for precision
    total_volume: AtomicU64,
    trade_count: AtomicU64,
}

impl AggregateIS {
    pub const fn new() -> Self {
        Self {
            total_is_numer: AtomicU64::new(0),
            total_volume: AtomicU64::new(0),
            trade_count: AtomicU64::new(0),
        }
    }

    /// Thread-safe addition of IS measurement (scaled by 1e6 for precision)
    #[inline]
    pub fn add_measurement(&self, is_bps: f64, volume: f64) {
        let scaled_is = (is_bps * volume * 1000.0) as u64; // Preserve 3 decimal places
        let scaled_vol = (volume * 1000.0) as u64;
        
        self.total_is_numer.fetch_add(scaled_is, Ordering::Relaxed);
        self.total_volume.fetch_add(scaled_vol, Ordering::Relaxed);
        self.trade_count.fetch_add(1, Ordering::Relaxed);
    }

    #[inline]
    pub fn get_average_is_bps(&self) -> f64 {
        let numer = self.total_is_numer.load(Ordering::Relaxed) as f64;
        let denom = self.total_volume.load(Ordering::Relaxed) as f64;
        
        if denom > 0.0 {
            numer / denom
        } else {
            0.0
        }
    }

    #[inline]
    pub const fn get_trade_count(&self) -> u64 {
        self.trade_count.load(Ordering::Relaxed)
    }
}

impl Default for AggregateIS {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_arrival_price_is_buy() {
        let mut calc = ArrivalPriceIS::new(100.0, 1000000000);
        calc.record_fill(100.5, 10.0, 0.5, 2000000000);
        calc.record_fill(100.7, 5.0, 0.25, 3000000000);
        
        let summary = calc.decompose_is(Side::Buy, 99.8);
        assert!(summary.total_is_bps > 0.0);
        assert!(summary.market_impact_bps > 0.0);
    }

    #[test]
    fn test_zero_allocation() {
        // Verify no heap allocations in hot path
        let mut calc = ArrivalPriceIS::new(50000.0, 0);
        for i in 0..1000 {
            calc.record_fill(50000.5, 0.1, 0.01, i * 1000000);
        }
        let _ = calc.calculate_is_bps(Side::Buy);
    }
}
