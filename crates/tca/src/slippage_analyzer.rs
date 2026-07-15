// crates/tca/src/slippage_analyzer.rs
// Real-time slippage and market impact model
// Tracks delay slippage vs market impact slippage without heap allocations

use std::sync::atomic::{AtomicU64, Ordering};
use crate::arrival_price::Side;

/// Market snapshot at signal generation time
#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct SignalSnapshot {
    pub timestamp_ns: u64,
    pub bid: f64,
    pub ask: f64,
    pub mid: f64,
    pub spread_bps: f64,
}

impl SignalSnapshot {
    #[inline]
    pub const fn new(timestamp_ns: u64, bid: f64, ask: f64) -> Self {
        let mid = (bid + ask) * 0.5;
        let spread_bps = if mid > 0.0 { (ask - bid) / mid * 10000.0 } else { 0.0 };
        Self { timestamp_ns, bid, ask, mid, spread_bps }
    }
}

/// Fill event with precise timing
#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct FillEvent {
    pub timestamp_ns: u64,
    pub price: f64,
    pub quantity: f64,
    pub liquidity_taken: bool, // true if taker, false if maker
}

/// Slippage decomposition result
#[derive(Clone, Copy, Debug, Default)]
#[repr(C)]
pub struct SlippageBreakdown {
    /// Total slippage in basis points
    pub total_slippage_bps: f64,
    /// Delay slippage: price movement between signal and order arrival
    pub delay_slippage_bps: f64,
    /// Market impact slippage: price moved due to our own trading
    pub impact_slippage_bps: f64,
    /// Spread cost: half-spread paid for immediate execution
    pub spread_cost_bps: f64,
    /// Timing luck: residual unexplained slippage
    pub timing_luck_bps: f64,
    /// Signal to fill latency in nanoseconds
    pub signal_to_fill_ns: u64,
}

/// Real-time slippage analyzer - zero allocation design
pub struct SlippageAnalyzer {
    /// Signal snapshot when alpha was generated
    signal: Option<SignalSnapshot>,
    /// Order arrival snapshot (when order hit the exchange)
    order_arrival_mid: f64,
    order_arrival_ts_ns: u64,
    /// Cumulative fills
    total_filled_qty: f64,
    total_filled_value: f64,
    /// First fill after signal
    first_fill_ts_ns: u64,
    last_fill_ts_ns: u64,
    fill_count: u32,
    /// Order side
    side: Side,
}

impl SlippageAnalyzer {
    #[inline]
    pub const fn new(side: Side) -> Self {
        Self {
            signal: None,
            order_arrival_mid: 0.0,
            order_arrival_ts_ns: 0,
            total_filled_qty: 0.0,
            total_filled_value: 0.0,
            first_fill_ts_ns: 0,
            last_fill_ts_ns: 0,
            fill_count: 0,
            side,
        }
    }

    /// Record the signal generation snapshot
    #[inline]
    pub fn record_signal(&mut self, snapshot: SignalSnapshot) {
        self.signal = Some(snapshot);
    }

    /// Record order arrival at exchange (for delay calculation)
    #[inline]
    pub fn record_order_arrival(&mut self, mid_price: f64, timestamp_ns: u64) {
        self.order_arrival_mid = mid_price;
        self.order_arrival_ts_ns = timestamp_ns;
    }

    /// Record a fill event
    #[inline]
    pub fn record_fill(&mut self, fill: FillEvent) {
        if self.fill_count == 0 {
            self.first_fill_ts_ns = fill.timestamp_ns;
        }
        self.last_fill_ts_ns = fill.timestamp_ns;
        
        self.total_filled_qty += fill.quantity;
        self.total_filled_value += fill.price * fill.quantity;
        self.fill_count += 1;
    }

    /// Calculate complete slippage breakdown
    #[inline]
    pub fn analyze_slippage(&self) -> SlippageBreakdown {
        let mut breakdown = SlippageBreakdown::default();

        let signal_snap = match self.signal {
            Some(s) => s,
            None => return breakdown,
        };

        if self.total_filled_qty <= 0.0 || signal_snap.mid <= 0.0 {
            return breakdown;
        }

        let avg_fill_price = self.total_filled_value / self.total_filled_qty;
        
        // Total slippage: difference between avg fill and signal mid
        breakdown.total_slippage_bps = match self.side {
            Side::Buy => (avg_fill_price - signal_snap.mid) / signal_snap.mid * 10000.0,
            Side::Sell => (signal_snap.mid - avg_fill_price) / signal_snap.mid * 10000.0,
        };

        // Half-spread cost (always positive cost)
        breakdown.spread_cost_bps = signal_snap.spread_bps * 0.5;

        // Delay slippage: price movement from signal to order arrival
        if self.order_arrival_ts_ns > signal_snap.timestamp_ns && self.order_arrival_mid > 0.0 {
            breakdown.delay_slippage_bps = match self.side {
                Side::Buy => (self.order_arrival_mid - signal_snap.mid) / signal_snap.mid * 10000.0,
                Side::Sell => (signal_snap.mid - self.order_arrival_mid) / signal_snap.mid * 10000.0,
            };
        }

        // Market impact: price movement from order arrival to average fill
        // This isolates the impact of our own trading
        if self.order_arrival_mid > 0.0 {
            breakdown.impact_slippage_bps = match self.side {
                Side::Buy => (avg_fill_price - self.order_arrival_mid) / self.order_arrival_mid * 10000.0,
                Side::Sell => (self.order_arrival_mid - avg_fill_price) / self.order_arrival_mid * 10000.0,
            };
        }

        // Ensure impact doesn't include spread (adjust for maker/taker if needed)
        // For now, we keep it raw

        // Timing luck: residual (total - delay - impact - spread)
        breakdown.timing_luck_bps = breakdown.total_slippage_bps 
            - breakdown.delay_slippage_bps 
            - breakdown.impact_slippage_bps 
            - breakdown.spread_cost_bps;

        // Latency metrics
        if self.first_fill_ts_ns > signal_snap.timestamp_ns {
            breakdown.signal_to_fill_ns = self.first_fill_ts_ns - signal_snap.timestamp_ns;
        }

        breakdown
    }

    /// Get signal-to-first-fill latency in microseconds
    #[inline]
    pub fn get_latency_us(&self) -> u64 {
        if let Some(signal) = self.signal {
            if self.first_fill_ts_ns > signal.timestamp_ns {
                return (self.first_fill_ts_ns - signal.timestamp_ns) / 1000;
            }
        }
        0
    }

    #[inline]
    pub const fn get_fill_count(&self) -> u32 {
        self.fill_count
    }

    #[inline]
    pub fn reset(&mut self, new_side: Side) {
        self.signal = None;
        self.order_arrival_mid = 0.0;
        self.order_arrival_ts_ns = 0;
        self.total_filled_qty = 0.0;
        self.total_filled_value = 0.0;
        self.first_fill_ts_ns = 0;
        self.last_fill_ts_ns = 0;
        self.fill_count = 0;
        self.side = new_side;
    }
}

/// Aggregate slippage statistics across multiple orders
pub struct SlippageAggregate {
    count: AtomicU64,
    total_slippage_sum: AtomicU64, // scaled by 1e6
    total_delay_sum: AtomicU64,
    total_impact_sum: AtomicU64,
    total_volume: AtomicU64,
    max_slippage: AtomicU64,
}

impl SlippageAggregate {
    pub const fn new() -> Self {
        Self {
            count: AtomicU64::new(0),
            total_slippage_sum: AtomicU64::new(0),
            total_delay_sum: AtomicU64::new(0),
            total_impact_sum: AtomicU64::new(0),
            total_volume: AtomicU64::new(0),
            max_slippage: AtomicU64::new(0),
        }
    }

    #[inline]
    pub fn record(&self, breakdown: &SlippageBreakdown, volume: f64) {
        let scale = 1_000_000.0;
        
        self.count.fetch_add(1, Ordering::Relaxed);
        self.total_slippage_sum.fetch_add(
            (breakdown.total_slippage_bps * scale).abs() as u64, Ordering::Relaxed);
        self.total_delay_sum.fetch_add(
            (breakdown.delay_slippage_bps * scale).abs() as u64, Ordering::Relaxed);
        self.total_impact_sum.fetch_add(
            (breakdown.impact_slippage_bps * scale).abs() as u64, Ordering::Relaxed);
        self.total_volume.fetch_add((volume * 1000.0) as u64, Ordering::Relaxed);
        
        // Update max using CAS loop
        let scaled_slippage = (breakdown.total_slippage_bps.abs() * scale) as u64;
        let mut current_max = self.max_slippage.load(Ordering::Relaxed);
        while scaled_slippage > current_max {
            match self.max_slippage.compare_exchange_weak(
                current_max, scaled_slippage, Ordering::Relaxed, Ordering::Relaxed) {
                Ok(_) => break,
                Err(x) => current_max = x,
            }
        }
    }

    #[inline]
    pub fn get_avg_slippage_bps(&self) -> f64 {
        let count = self.count.load(Ordering::Relaxed);
        if count == 0 { return 0.0; }
        self.total_slippage_sum.load(Ordering::Relaxed) as f64 / count as f64 / 1_000_000.0
    }

    #[inline]
    pub fn get_avg_delay_bps(&self) -> f64 {
        let count = self.count.load(Ordering::Relaxed);
        if count == 0 { return 0.0; }
        self.total_delay_sum.load(Ordering::Relaxed) as f64 / count as f64 / 1_000_000.0
    }

    #[inline]
    pub fn get_avg_impact_bps(&self) -> f64 {
        let count = self.count.load(Ordering::Relaxed);
        if count == 0 { return 0.0; }
        self.total_impact_sum.load(Ordering::Relaxed) as f64 / count as f64 / 1_000_000.0
    }

    #[inline]
    pub fn get_max_slippage_bps(&self) -> f64 {
        self.max_slippage.load(Ordering::Relaxed) as f64 / 1_000_000.0
    }

    #[inline]
    pub const fn get_sample_count(&self) -> u64 {
        self.count.load(Ordering::Relaxed)
    }
}

impl Default for SlippageAggregate {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_slippage_breakdown_buy() {
        let mut analyzer = SlippageAnalyzer::new(Side::Buy);
        
        // Signal at mid=100, spread=10bps
        analyzer.record_signal(SignalSnapshot::new(1000, 99.95, 100.05));
        
        // Order arrives at 100.10 (price moved against us)
        analyzer.record_order_arrival(100.10, 2000);
        
        // Fill at 100.15
        analyzer.record_fill(FillEvent {
            timestamp_ns: 3000,
            price: 100.15,
            quantity: 1.0,
            liquidity_taken: true,
        });

        let breakdown = analyzer.analyze_slippage();
        assert!(breakdown.total_slippage_bps > 0.0);
        assert!(breakdown.delay_slippage_bps > 0.0);
    }

    #[test]
    fn test_zero_allocation_path() {
        let mut analyzer = SlippageAnalyzer::new(Side::Sell);
        for i in 0..100 {
            analyzer.record_signal(SignalSnapshot::new(i * 1000, 50.0 - 0.01, 50.0 + 0.01));
            analyzer.record_order_arrival(50.0, i * 1000 + 500);
            analyzer.record_fill(FillEvent {
                timestamp_ns: i * 1000 + 1000,
                price: 50.0,
                quantity: 0.1,
                liquidity_taken: false,
            });
            let _ = analyzer.analyze_slippage();
        }
    }
}
