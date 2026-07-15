// crates/tca/src/vwap_twap_benchmark.rs
// Lock-free benchmark engine for VWAP/TWAP calculation
// Grades execution algorithm performance against standard benchmarks

use std::sync::atomic::{AtomicU64, AtomicBool, Ordering};
use crate::arrival_price::Side;

/// Single price-volume tick for VWAP calculation
#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct MarketTick {
    pub timestamp_ns: u64,
    pub price: f64,
    pub volume: f64,
}

/// Execution window definition
#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct ExecutionWindow {
    pub start_ns: u64,
    pub end_ns: u64,
    pub target_quantity: f64,
}

/// VWAP/TWAP benchmark result
#[derive(Clone, Copy, Debug, Default)]
#[repr(C)]
pub struct BenchmarkResult {
    /// Actual average execution price achieved
    pub actual_avg_price: f64,
    /// Market VWAP over the execution window
    pub market_vwap: f64,
    /// Market TWAP over the execution window
    pub market_twap: f64,
    /// Performance vs VWAP in basis points (positive = beat benchmark)
    pub vs_vwap_bps: f64,
    /// Performance vs TWAP in basis points
    pub vs_twap_bps: f64,
    /// Percentage of market volume captured
    pub participation_rate: f64,
    /// Execution quality score (0-100)
    pub quality_score: f64,
    /// Total market volume in window
    pub market_volume: f64,
    /// Number of ticks in window
    pub tick_count: u32,
}

/// Lock-free ring buffer for market ticks (fixed size, no allocation)
pub struct TickRingBuffer<const N: usize> {
    buffer: [Option<MarketTick>; N],
    head: AtomicU64,
    tail: AtomicU64,
    count: AtomicU64,
}

impl<const N: usize> TickRingBuffer<N> {
    pub const fn new() -> Self {
        const INIT: Option<MarketTick> = None;
        Self {
            buffer: [INIT; N],
            head: AtomicU64::new(0),
            tail: AtomicU64::new(0),
            count: AtomicU64::new(0),
        }
    }

    #[inline]
    pub fn push(&self, tick: MarketTick) -> bool {
        let tail = self.tail.load(Ordering::Relaxed);
        let idx = (tail as usize) % N;
        
        // Check if buffer is full
        let count = self.count.load(Ordering::Relaxed);
        if count >= N as u64 {
            // Buffer full, overwrite oldest (move head)
            let head = self.head.load(Ordering::Relaxed);
            self.head.store(head + 1, Ordering::Relaxed);
        } else {
            self.count.fetch_add(1, Ordering::Relaxed);
        }

        unsafe {
            let slot = &mut *(self.buffer.as_ptr().add(idx) as *mut Option<MarketTick>);
            *slot = Some(tick);
        }
        
        self.tail.store(tail + 1, Ordering::Relaxed);
        true
    }

    #[inline]
    pub fn get_ticks_in_window(&self, start_ns: u64, end_ns: u64, result: &mut [MarketTick]) -> usize {
        let count = self.count.load(Ordering::Relaxed) as usize;
        let head = self.head.load(Ordering::Relaxed) as usize;
        let mut written = 0;

        for i in 0..count.min(N) {
            let idx = (head + i) % N;
            unsafe {
                if let Some(tick) = *self.buffer.as_ptr().add(idx) as *const Option<MarketTick> {
                    if let Some(t) = &*(&tick as *const _) {
                        if t.timestamp_ns >= start_ns && t.timestamp_ns <= end_ns {
                            if written < result.len() {
                                result[written] = *t;
                                written += 1;
                            }
                        }
                    }
                }
            }
        }
        written
    }

    #[inline]
    pub const fn len(&self) -> usize {
        self.count.load(Ordering::Relaxed) as usize
    }
}

impl<const N: usize> Default for TickRingBuffer<N> {
    fn default() -> Self {
        Self::new()
    }
}

/// VWAP/TWAP calculator - zero allocation
pub struct VwapTwapCalculator {
    /// Accumulated sum of price * volume for VWAP
    pv_sum: f64,
    /// Accumulated volume for VWAP
    v_sum: f64,
    /// Accumulated price for TWAP
    p_sum: f64,
    /// Tick count for TWAP
    tick_count: u32,
    /// Window start time
    window_start_ns: u64,
    /// Window end time
    window_end_ns: u64,
    /// First tick timestamp
    first_tick_ns: u64,
    /// Last tick timestamp
    last_tick_ns: u64,
}

impl VwapTwapCalculator {
    #[inline]
    pub const fn new() -> Self {
        Self {
            pv_sum: 0.0,
            v_sum: 0.0,
            p_sum: 0.0,
            tick_count: 0,
            window_start_ns: 0,
            window_end_ns: 0,
            first_tick_ns: 0,
            last_tick_ns: 0,
        }
    }

    #[inline]
    pub fn reset(&mut self, start_ns: u64, end_ns: u64) {
        self.pv_sum = 0.0;
        self.v_sum = 0.0;
        self.p_sum = 0.0;
        self.tick_count = 0;
        self.window_start_ns = start_ns;
        self.window_end_ns = end_ns;
        self.first_tick_ns = 0;
        self.last_tick_ns = 0;
    }

    #[inline]
    pub fn add_tick(&mut self, tick: MarketTick) {
        if tick.timestamp_ns < self.window_start_ns || tick.timestamp_ns > self.window_end_ns {
            return;
        }

        if self.tick_count == 0 {
            self.first_tick_ns = tick.timestamp_ns;
        }
        self.last_tick_ns = tick.timestamp_ns;

        self.pv_sum += tick.price * tick.volume;
        self.v_sum += tick.volume;
        self.p_sum += tick.price;
        self.tick_count += 1;
    }

    #[inline]
    pub fn calculate_vwap(&self) -> f64 {
        if self.v_sum > 0.0 {
            self.pv_sum / self.v_sum
        } else {
            0.0
        }
    }

    #[inline]
    pub fn calculate_twap(&self) -> f64 {
        if self.tick_count > 0 {
            self.p_sum / self.tick_count as f64
        } else {
            0.0
        }
    }

    #[inline]
    pub const fn get_market_volume(&self) -> f64 {
        self.v_sum
    }

    #[inline]
    pub const fn get_tick_count(&self) -> u32 {
        self.tick_count
    }
}

impl Default for VwapTwapCalculator {
    fn default() -> Self {
        Self::new()
    }
}

/// Main benchmark engine - compares execution against VWAP/TWAP
pub struct BenchmarkEngine {
    calculator: VwapTwapCalculator,
    side: Side,
    executed_quantity: f64,
    executed_value: f64,
    is_active: AtomicBool,
}

impl BenchmarkEngine {
    #[inline]
    pub const fn new(side: Side) -> Self {
        Self {
            calculator: VwapTwapCalculator::new(),
            side,
            executed_quantity: 0.0,
            executed_value: 0.0,
            is_active: AtomicBool::new(false),
        }
    }

    #[inline]
    pub fn start_window(&mut self, start_ns: u64, end_ns: u64) {
        self.calculator.reset(start_ns, end_ns);
        self.executed_quantity = 0.0;
        self.executed_value = 0.0;
        self.is_active.store(true, Ordering::Relaxed);
    }

    #[inline]
    pub fn add_market_tick(&mut self, tick: MarketTick) {
        if self.is_active.load(Ordering::Relaxed) {
            self.calculator.add_tick(tick);
        }
    }

    #[inline]
    pub fn record_execution(&mut self, price: f64, quantity: f64) {
        self.executed_quantity += quantity;
        self.executed_value += price * quantity;
    }

    #[inline]
    pub fn finish_and_evaluate(&mut self) -> BenchmarkResult {
        self.is_active.store(false, Ordering::Relaxed);

        let mut result = BenchmarkResult::default();
        
        result.market_vwap = self.calculator.calculate_vwap();
        result.market_twap = self.calculator.calculate_twap();
        result.market_volume = self.calculator.get_market_volume();
        result.tick_count = self.calculator.get_tick_count();

        if self.executed_quantity > 0.0 {
            result.actual_avg_price = self.executed_value / self.executed_quantity;
            
            // Calculate performance vs benchmarks
            // For buys: lower price is better (negative diff = good)
            // For sells: higher price is better (positive diff = good)
            match self.side {
                Side::Buy => {
                    if result.market_vwap > 0.0 {
                        result.vs_vwap_bps = (result.market_vwap - result.actual_avg_price) 
                            / result.market_vwap * 10000.0;
                    }
                    if result.market_twap > 0.0 {
                        result.vs_twap_bps = (result.market_twap - result.actual_avg_price) 
                            / result.market_twap * 10000.0;
                    }
                }
                Side::Sell => {
                    if result.market_vwap > 0.0 {
                        result.vs_vwap_bps = (result.actual_avg_price - result.market_vwap) 
                            / result.market_vwap * 10000.0;
                    }
                    if result.market_twap > 0.0 {
                        result.vs_twap_bps = (result.actual_avg_price - result.market_twap) 
                            / result.market_twap * 10000.0;
                    }
                }
            }

            // Participation rate
            if result.market_volume > 0.0 {
                result.participation_rate = self.executed_quantity / result.market_volume * 100.0;
            }

            // Quality score: weighted combination of VWAP and TWAP performance
            // Higher is better, capped at 100
            let vwap_score = if result.vs_vwap_bps > 0.0 { 
                (50.0 + result.vs_vwap_bps).min(100.0) 
            } else { 
                (50.0 + result.vs_vwap_bps).max(0.0) 
            };
            let twap_score = if result.vs_twap_bps > 0.0 { 
                (50.0 + result.vs_twap_bps).min(100.0) 
            } else { 
                (50.0 + result.vs_twap_bps).max(0.0) 
            };
            result.quality_score = (vwap_score * 0.7 + twap_score * 0.3).clamp(0.0, 100.0);
        }

        result
    }

    #[inline]
    pub const fn is_active(&self) -> bool {
        self.is_active.load(Ordering::Relaxed)
    }
}

/// Aggregate statistics across multiple benchmark periods
pub struct BenchmarkAggregate {
    trade_count: AtomicU64,
    total_vs_vwap: AtomicU64, // scaled by 1e6
    total_vs_twap: AtomicU64,
    beats_vwap_count: AtomicU64,
    beats_twap_count: AtomicU64,
    avg_quality_score: AtomicU64,
}

impl BenchmarkAggregate {
    pub const fn new() -> Self {
        Self {
            trade_count: AtomicU64::new(0),
            total_vs_vwap: AtomicU64::new(0),
            total_vs_twap: AtomicU64::new(0),
            beats_vwap_count: AtomicU64::new(0),
            beats_twap_count: AtomicU64::new(0),
            avg_quality_score: AtomicU64::new(0),
        }
    }

    #[inline]
    pub fn record(&self, result: &BenchmarkResult) {
        let scale = 1_000_000.0;
        
        self.trade_count.fetch_add(1, Ordering::Relaxed);
        self.total_vs_vwap.fetch_add(
            (result.vs_vwap_bps * scale) as i64 as u64, Ordering::Relaxed);
        self.total_vs_twap.fetch_add(
            (result.vs_twap_bps * scale) as i64 as u64, Ordering::Relaxed);
        
        if result.vs_vwap_bps > 0.0 {
            self.beats_vwap_count.fetch_add(1, Ordering::Relaxed);
        }
        if result.vs_twap_bps > 0.0 {
            self.beats_twap_count.fetch_add(1, Ordering::Relaxed);
        }

        // Running average for quality score using integer math
        let current_avg = self.avg_quality_score.load(Ordering::Relaxed);
        let count = self.trade_count.load(Ordering::Relaxed);
        let new_avg = ((current_avg * (count - 1)) as f64 + result.quality_score * 100.0) / count as f64;
        self.avg_quality_score.store(new_avg as u64, Ordering::Relaxed);
    }

    #[inline]
    pub fn get_avg_vs_vwap_bps(&self) -> f64 {
        let count = self.trade_count.load(Ordering::Relaxed);
        if count == 0 { return 0.0; }
        self.total_vs_vwap.load(Ordering::Relaxed) as f64 / count as f64 / 1_000_000.0
    }

    #[inline]
    pub fn get_beat_vwap_rate(&self) -> f64 {
        let count = self.trade_count.load(Ordering::Relaxed);
        if count == 0 { return 0.0; }
        self.beats_vwap_count.load(Ordering::Relaxed) as f64 / count as f64 * 100.0
    }

    #[inline]
    pub fn get_beat_twap_rate(&self) -> f64 {
        let count = self.trade_count.load(Ordering::Relaxed);
        if count == 0 { return 0.0; }
        self.beats_twap_count.load(Ordering::Relaxed) as f64 / count as f64 * 100.0
    }

    #[inline]
    pub fn get_avg_quality_score(&self) -> f64 {
        self.avg_quality_score.load(Ordering::Relaxed) as f64 / 100.0
    }
}

impl Default for BenchmarkAggregate {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vwap_calculation() {
        let mut calc = VwapTwapCalculator::new();
        calc.reset(0, 10000);
        
        calc.add_tick(MarketTick { timestamp_ns: 1000, price: 100.0, volume: 10.0 });
        calc.add_tick(MarketTick { timestamp_ns: 2000, price: 102.0, volume: 20.0 });
        calc.add_tick(MarketTick { timestamp_ns: 3000, price: 98.0, volume: 10.0 });

        // VWAP = (100*10 + 102*20 + 98*10) / (10+20+10) = (1000+2040+980)/40 = 100.5
        let vwap = calc.calculate_vwap();
        assert!((vwap - 100.5).abs() < 0.001);

        // TWAP = (100 + 102 + 98) / 3 = 100
        let twap = calc.calculate_twap();
        assert!((twap - 100.0).abs() < 0.001);
    }

    #[test]
    fn test_benchmark_engine() {
        let mut engine = BenchmarkEngine::new(Side::Buy);
        engine.start_window(0, 10000);

        // Add market ticks
        engine.add_market_tick(MarketTick { timestamp_ns: 1000, price: 100.0, volume: 50.0 });
        engine.add_market_tick(MarketTick { timestamp_ns: 5000, price: 101.0, volume: 50.0 });
        engine.add_market_tick(MarketTick { timestamp_ns: 10000, price: 102.0, volume: 50.0 });

        // Our execution: bought at 100.5 (better than VWAP of ~101)
        engine.record_execution(100.5, 10.0);

        let result = engine.finish_and_evaluate();
        assert!(result.vs_vwap_bps > 0.0); // Beat VWAP
    }

    #[test]
    fn test_ring_buffer() {
        let buffer = TickRingBuffer::<100>::new();
        for i in 0..150 {
            buffer.push(MarketTick {
                timestamp_ns: i * 1000,
                price: 100.0 + (i as f64 * 0.01),
                volume: 1.0,
            });
        }
        // Should have exactly 100 items (buffer size)
        assert_eq!(buffer.len(), 100);
    }
}
