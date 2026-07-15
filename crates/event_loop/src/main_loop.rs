//! Hyper-optimized Main Event Loop
//! Zero-allocation, cache-friendly tick processing with integrated risk checks.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use log::{debug, error, info, warn};

/// Core event types processed by the main loop
#[derive(Debug, Clone, Copy)]
pub enum EventType {
    Tick,
    Signal,
    OrderFill,
    RiskUpdate,
    Heartbeat,
}

/// Event structure (fixed size, no heap allocations)
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct Event {
    pub event_type: u8,
    pub symbol_id: u32,
    pub timestamp_ns: u64,
    pub price_ticks: i64,  // Price in ticks (avoid float)
    pub quantity: i64,     // Quantity in base units
    pub flags: u32,
}

impl Event {
    pub const SIZE: usize = std::mem::size_of::<Self>();

    pub fn new(event_type: EventType, symbol_id: u32, timestamp_ns: u64) -> Self {
        Self {
            event_type: event_type as u8,
            symbol_id,
            timestamp_ns,
            price_ticks: 0,
            quantity: 0,
            flags: 0,
        }
    }
}

/// Pre-trade risk check result
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RiskCheckResult {
    Pass,
    FailExposureLimit,
    FailPositionLimit,
    FailRateLimit,
    FailMarginRequirement,
}

/// Strategy signal output
#[derive(Debug, Clone, Copy)]
pub struct StrategySignal {
    pub symbol_id: u32,
    pub side: i8,  // 1 = buy, -1 = sell, 0 = flat
    pub quantity: i64,
    pub confidence: u16,  // 0-10000 basis points
    pub timestamp_ns: u64,
}

/// Order dispatch result
#[derive(Debug, Clone, Copy)]
pub struct OrderDispatch {
    pub order_id: u64,
    pub symbol_id: u32,
    pub filled: bool,
    pub fill_price_ticks: i64,
    pub fill_quantity: i64,
    pub latency_ns: u64,
}

/// Main event loop state
pub struct EventLoopState {
    pub tick_count: Arc<AtomicU64>,
    pub signal_count: Arc<AtomicU64>,
    pub order_count: Arc<AtomicU64>,
    pub last_heartbeat: Arc<AtomicU64>,
    pub is_running: Arc<AtomicBool>,
    pub total_latency_ns: Arc<AtomicU64>,
}

impl EventLoopState {
    pub fn new() -> Self {
        Self {
            tick_count: Arc::new(AtomicU64::new(0)),
            signal_count: Arc::new(AtomicU64::new(0)),
            order_count: Arc::new(AtomicU64::new(0)),
            last_heartbeat: Arc::new(AtomicU64::new(0)),
            is_running: Arc::new(AtomicBool::new(false)),
            total_latency_ns: Arc::new(AtomicU64::new(0)),
        }
    }

    pub fn record_tick(&self) {
        self.tick_count.fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_signal(&self) {
        self.signal_count.fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_order(&self) {
        self.order_count.fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_latency(&self, latency_ns: u64) {
        self.total_latency_ns.fetch_add(latency_ns, Ordering::Relaxed);
    }

    pub fn avg_latency_ns(&self) -> f64 {
        let total = self.total_latency_ns.load(Ordering::Relaxed);
        let count = self.tick_count.load(Ordering::Relaxed);
        if count == 0 {
            return 0.0;
        }
        total as f64 / count as f64
    }
}

impl Default for EventLoopState {
    fn default() -> Self {
        Self::new()
    }
}

/// Main event loop configuration
pub struct EventLoopConfig {
    pub max_ticks_per_batch: usize,
    pub risk_check_interval_ticks: u64,
    pub heartbeat_interval_ms: u64,
    pub max_order_latency_ns: u64,
}

impl Default for EventLoopConfig {
    fn default() -> Self {
        Self {
            max_ticks_per_batch: 1024,
            risk_check_interval_ticks: 100,
            heartbeat_interval_ms: 1000,
            max_order_latency_ns: 1_000_000, // 1ms
        }
    }
}

/// The hyper-optimized main event loop
pub struct MainEventLoop {
    pub config: EventLoopConfig,
    pub state: EventLoopState,
    event_buffer: Vec<Event>,  // Pre-allocated, reused
}

impl MainEventLoop {
    pub fn new(config: EventLoopConfig) -> Self {
        let mut event_buffer = Vec::with_capacity(config.max_ticks_per_batch);
        // Pre-fill to avoid reallocations
        unsafe {
            event_buffer.set_len(config.max_ticks_per_batch);
        }

        Self {
            config,
            state: EventLoopState::new(),
            event_buffer,
        }
    }

    /// Run the main event loop (zero-allocation inner loop)
    pub fn run<F, G, H, I>(
        &mut self,
        mut poll_events: F,      // Fn() -> Option<Event>
        mut process_strategy: G, // Fn(&Event) -> Option<StrategySignal>
        mut check_risk: H,       // Fn(&StrategySignal) -> RiskCheckResult
        mut dispatch_order: I,   // Fn(&StrategySignal) -> OrderDispatch
    ) where
        F: FnMut() -> Option<Event>,
        G: FnMut(&Event) -> Option<StrategySignal>,
        H: FnMut(&StrategySignal) -> RiskCheckResult,
        I: FnMut(&StrategySignal) -> OrderDispatch,
    {
        self.state.is_running.store(true, Ordering::SeqCst);
        info!("Main event loop started");

        let start_time = Instant::now();
        let mut heartbeat_counter = 0u64;
        let mut batch_idx = 0usize;

        while self.state.is_running.load(Ordering::SeqCst) {
            let loop_start = Instant::now();

            // Poll for events (non-blocking)
            if let Some(event) = poll_events() {
                let event_start = Instant::now();

                // Process tick
                self.state.record_tick();
                batch_idx += 1;

                // Run strategy logic
                if let Some(signal) = process_strategy(&event) {
                    self.state.record_signal();

                    // Periodic risk check
                    if self.state.tick_count.load(Ordering::Relaxed) % self.config.risk_check_interval_ticks == 0 {
                        let risk_result = check_risk(&signal);
                        
                        if risk_result == RiskCheckResult::Pass {
                            // Dispatch order
                            let order_start = Instant::now();
                            let dispatch = dispatch_order(&signal);
                            let order_latency = order_start.elapsed().as_nanos() as u64;

                            if dispatch.filled {
                                self.state.record_order();
                                
                                if order_latency > self.config.max_order_latency_ns {
                                    warn!("Order latency exceeded threshold: {}ns", order_latency);
                                }
                            }

                            self.state.record_latency(order_latency);
                        } else {
                            warn!("Risk check failed: {:?}", risk_result);
                        }
                    }
                }

                let event_latency = event_start.elapsed().as_nanos() as u64;
                self.state.record_latency(event_latency);
            }

            // Heartbeat
            heartbeat_counter += 1;
            if heartbeat_counter >= self.config.heartbeat_interval_ms {
                let now = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_micros() as u64;
                self.state.last_heartbeat.store(now, Ordering::SeqCst);
                heartbeat_counter = 0;
                debug!("Heartbeat - Ticks: {}, Signals: {}, Orders: {}", 
                    self.state.tick_count.load(Ordering::Relaxed),
                    self.state.signal_count.load(Ordering::Relaxed),
                    self.state.order_count.load(Ordering::Relaxed)
                );
            }

            // Cache-friendly spin wait if no events (avoid syscalls)
            let elapsed = loop_start.elapsed();
            if elapsed.as_nanos() < 100 {
                // Busy-wait for sub-100ns precision
                std::hint::spin_loop();
            } else {
                // Yield for longer gaps
                std::thread::yield_now();
            }
        }

        let total_duration = start_time.elapsed();
        let tick_count = self.state.tick_count.load(Ordering::Relaxed);
        info!(
            "Main event loop stopped. Total ticks: {}, Duration: {:?}, Avg latency: {:.2}ns",
            tick_count,
            total_duration,
            self.state.avg_latency_ns()
        );
    }

    /// Stop the event loop
    pub fn stop(&mut self) {
        self.state.is_running.store(false, Ordering::SeqCst);
        info!("Stopping main event loop...");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_event_loop_basic() {
        let config = EventLoopConfig::default();
        let mut loop_instance = MainEventLoop::new(config);

        let mut tick_count = 0u64;
        let max_ticks = 100u64;

        loop_instance.run(
            || {
                if tick_count < max_ticks {
                    tick_count += 1;
                    Some(Event::new(EventType::Tick, 1, 0))
                } else {
                    None
                }
            },
            |_event| Some(StrategySignal {
                symbol_id: 1,
                side: 1,
                quantity: 100,
                confidence: 9000,
                timestamp_ns: 0,
            }),
            |_signal| RiskCheckResult::Pass,
            |_signal| OrderDispatch {
                order_id: 1,
                symbol_id: 1,
                filled: true,
                fill_price_ticks: 50000,
                fill_quantity: 100,
                latency_ns: 100,
            },
        );

        assert_eq!(loop_instance.state.tick_count.load(Ordering::Relaxed), max_ticks);
        assert_eq!(loop_instance.state.signal_count.load(Ordering::Relaxed), max_ticks);
        assert_eq!(loop_instance.state.order_count.load(Ordering::Relaxed), max_ticks);
    }
}
