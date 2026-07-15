//! High-precision deterministic clock and event scheduler.
//! Synchronizes high-frequency market data with low-frequency alternative data.
//! Injects simulated network latencies for realistic backtesting.

use std::collections::BinaryHeap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};
use std::cmp::Ordering as CmpOrdering;

/// Event types that can be scheduled
#[derive(Debug, Clone)]
pub enum ScheduledEvent {
    MarketData { timestamp_ns: u64, data: Vec<u8> },
    AlternativeData { timestamp_ns: u64, source: String, payload: String },
    OrderSubmit { timestamp_ns: u64, order_id: u64 },
    OrderCancel { timestamp_ns: u64, order_id: u64 },
    LatencyInjection { timestamp_ns: u64, delay_ns: u64 },
    Custom { timestamp_ns: u64, event_type: u32, payload: Vec<u8> },
}

impl ScheduledEvent {
    pub fn timestamp(&self) -> u64 {
        match self {
            ScheduledEvent::MarketData { timestamp_ns, .. } => *timestamp_ns,
            ScheduledEvent::AlternativeData { timestamp_ns, .. } => *timestamp_ns,
            ScheduledEvent::OrderSubmit { timestamp_ns, .. } => *timestamp_ns,
            ScheduledEvent::OrderCancel { timestamp_ns, .. } => *timestamp_ns,
            ScheduledEvent::LatencyInjection { timestamp_ns, .. } => *timestamp_ns,
            ScheduledEvent::Custom { timestamp_ns, .. } => *timestamp_ns,
        }
    }
}

impl PartialEq for ScheduledEvent {
    fn eq(&self, other: &Self) -> bool {
        self.timestamp() == other.timestamp()
    }
}

impl Eq for ScheduledEvent {}

impl PartialOrd for ScheduledEvent {
    fn partial_cmp(&self, other: &Self) -> Option<CmpOrdering> {
        Some(self.cmp(other))
    }
}

impl Ord for ScheduledEvent {
    fn cmp(&self, other: &Self) -> CmpOrdering {
        // Reverse ordering for min-heap behavior (smallest timestamp first)
        other.timestamp().cmp(&self.timestamp())
    }
}

/// Network latency model
#[derive(Debug, Clone)]
pub struct LatencyModel {
    pub base_latency_ns: u64,
    pub jitter_ns: u64,
    pub distribution: LatencyDistribution,
}

#[derive(Debug, Clone, Copy)]
pub enum LatencyDistribution {
    Constant,
    Uniform,
    LogNormal { mean_ln: f64, std_ln: f64 },
    Exponential { rate: f64 },
}

impl LatencyModel {
    pub fn new(base_latency_ns: u64, jitter_ns: u64, distribution: LatencyDistribution) -> Self {
        Self {
            base_latency_ns,
            jitter_ns,
            distribution,
        }
    }
    
    /// Generate a latency sample in nanoseconds
    pub fn sample_latency(&self, rng_seed: u64) -> u64 {
        match self.distribution {
            LatencyDistribution::Constant => self.base_latency_ns,
            LatencyDistribution::Uniform => {
                let pseudo_random = ((rng_seed * 1103515245 + 12345) % 10000) as u64;
                self.base_latency_ns + (pseudo_random % (self.jitter_ns + 1))
            }
            LatencyDistribution::LogNormal { mean_ln, std_ln } => {
                // Box-Muller transform approximation
                let u1 = ((rng_seed % 10000) as f64) / 10000.0;
                let u2 = (((rng_seed * 7919) % 10000) as f64) / 10000.0;
                
                if u1 < 0.0001 {
                    return self.base_latency_ns;
                }
                
                let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
                let log_normal = (mean_ln + std_ln * z).exp();
                (self.base_latency_ns as f64 * log_normal) as u64
            }
            LatencyDistribution::Exponential { rate } => {
                let u = ((rng_seed % 10000) as f64) / 10000.0;
                if u < 0.0001 {
                    return self.base_latency_ns;
                }
                let exponential = -rate.ln() / rate;
                (self.base_latency_ns as f64 * (1.0 + exponential)) as u64
            }
        }
    }
}

/// Deterministic replay clock with event scheduling
pub struct ReplayClock {
    current_time_ns: AtomicU64,
    start_time_ns: u64,
    end_time_ns: u64,
    event_queue: BinaryHeap<ScheduledEvent>,
    latency_model: Option<LatencyModel>,
    latency_counter: AtomicU64,
    is_running: AtomicU64,
}

impl ReplayClock {
    pub fn new(start_time_ns: u64, end_time_ns: u64) -> Self {
        Self {
            current_time_ns: AtomicU64::new(start_time_ns),
            start_time_ns,
            end_time_ns,
            event_queue: BinaryHeap::new(),
            latency_model: None,
            latency_counter: AtomicU64::new(0),
            is_running: AtomicU64::new(0),
        }
    }
    
    /// Set the latency model for simulating network delays
    pub fn set_latency_model(&mut self, model: LatencyModel) {
        self.latency_model = Some(model);
    }
    
    /// Schedule an event
    pub fn schedule(&mut self, event: ScheduledEvent) {
        self.event_queue.push(event);
    }
    
    /// Schedule a market data event with optional latency injection
    pub fn schedule_market_data(&mut self, timestamp_ns: u64, data: Vec<u8>) {
        let final_timestamp = self.apply_latency(timestamp_ns);
        self.event_queue.push(ScheduledEvent::MarketData {
            timestamp_ns: final_timestamp,
            data,
        });
    }
    
    /// Schedule alternative data (macro/sentiment) events
    pub fn schedule_alternative_data(&mut self, timestamp_ns: u64, source: String, payload: String) {
        let final_timestamp = self.apply_latency(timestamp_ns);
        self.event_queue.push(ScheduledEvent::AlternativeData {
            timestamp_ns: final_timestamp,
            source,
            payload,
        });
    }
    
    /// Schedule an order submission with latency
    pub fn schedule_order_submit(&mut self, timestamp_ns: u64, order_id: u64) {
        let final_timestamp = self.apply_latency(timestamp_ns);
        self.event_queue.push(ScheduledEvent::OrderSubmit {
            timestamp_ns: final_timestamp,
            order_id,
        });
    }
    
    /// Schedule an order cancellation with latency
    pub fn schedule_order_cancel(&mut self, timestamp_ns: u64, order_id: u64) {
        let final_timestamp = self.apply_latency(timestamp_ns);
        self.event_queue.push(ScheduledEvent::OrderCancel {
            timestamp_ns: final_timestamp,
            order_id,
        });
    }
    
    fn apply_latency(&self, original_timestamp_ns: u64) -> u64 {
        if let Some(ref model) = self.latency_model {
            let counter = self.latency_counter.fetch_add(1, Ordering::Relaxed);
            let latency = model.sample_latency(counter);
            original_timestamp_ns + latency
        } else {
            original_timestamp_ns
        }
    }
    
    /// Get the next event from the queue
    pub fn next_event(&mut self) -> Option<ScheduledEvent> {
        if let Some(event) = self.event_queue.pop() {
            let event_ts = event.timestamp();
            
            // Don't process events beyond end time
            if event_ts > self.end_time_ns {
                return None;
            }
            
            // Update current time
            self.current_time_ns.store(event_ts, Ordering::Relaxed);
            Some(event)
        } else {
            None
        }
    }
    
    /// Process all events up to a specific timestamp
    pub fn process_until(&mut self, target_ns: u64, mut handler: impl FnMut(ScheduledEvent)) -> usize {
        let mut count = 0;
        
        while let Some(event) = self.event_queue.peek() {
            if event.timestamp() > target_ns || event.timestamp() > self.end_time_ns {
                break;
            }
            
            if let Some(event) = self.event_queue.pop() {
                self.current_time_ns.store(event.timestamp(), Ordering::Relaxed);
                handler(event);
                count += 1;
            }
        }
        
        count
    }
    
    /// Run the clock in real-time mode (for debugging/visualization)
    pub fn run_realtime<F>(&mut self, mut callback: F) -> usize
    where
        F: FnMut(ScheduledEvent) -> bool,
    {
        self.is_running.store(1, Ordering::Relaxed);
        let wall_start = Instant::now();
        let mut count = 0;
        
        while self.is_running.load(Ordering::Relaxed) == 1 {
            if let Some(event) = self.next_event() {
                let event_ts = event.timestamp();
                let elapsed_wall = wall_start.elapsed().as_nanos() as u64;
                let simulated_elapsed = event_ts.saturating_sub(self.start_time_ns);
                
                // Sleep to match simulated time with wall time
                if simulated_elapsed > elapsed_wall {
                    let sleep_ns = simulated_elapsed - elapsed_wall;
                    if sleep_ns < 1_000_000_000 { // Max 1 second sleep
                        std::thread::sleep(Duration::from_nanos(sleep_ns));
                    }
                }
                
                if !callback(event) {
                    break;
                }
                count += 1;
            } else {
                break;
            }
        }
        
        count
    }
    
    /// Stop the realtime clock
    pub fn stop(&self) {
        self.is_running.store(0, Ordering::Relaxed);
    }
    
    /// Get current simulated time
    #[inline]
    pub fn current_time_ns(&self) -> u64 {
        self.current_time_ns.load(Ordering::Relaxed)
    }
    
    /// Get current simulated time in microseconds
    #[inline]
    pub fn current_time_us(&self) -> u64 {
        self.current_time_ns() / 1000
    }
    
    /// Get current simulated time in milliseconds
    #[inline]
    pub fn current_time_ms(&self) -> u64 {
        self.current_time_ns() / 1_000_000
    }
    
    /// Check if clock has finished (no more events or reached end time)
    pub fn is_finished(&self) -> bool {
        self.event_queue.is_empty() || self.current_time_ns() >= self.end_time_ns
    }
    
    /// Get remaining events count
    pub fn pending_events(&self) -> usize {
        self.event_queue.len()
    }
    
    /// Reset clock to start time
    pub fn reset(&mut self) {
        self.current_time_ns.store(self.start_time_ns, Ordering::Relaxed);
        self.latency_counter.store(0, Ordering::Relaxed);
    }
    
    /// Fast-forward to a specific timestamp
    pub fn fast_forward_to(&mut self, target_ns: u64) {
        if target_ns <= self.end_time_ns {
            self.current_time_ns.store(target_ns, Ordering::Relaxed);
            
            // Remove events before target time
            while let Some(event) = self.event_queue.peek() {
                if event.timestamp() < target_ns {
                    self.event_queue.pop();
                } else {
                    break;
                }
            }
        }
    }
}

/// Multi-source event synchronizer for combining market data with alternative data
pub struct EventSynchronizer {
    clock: ReplayClock,
    sync_tolerance_ns: u64,
}

impl EventSynchronizer {
    pub fn new(start_time_ns: u64, end_time_ns: u64, sync_tolerance_ns: u64) -> Self {
        Self {
            clock: ReplayClock::new(start_time_ns, end_time_ns),
            sync_tolerance_ns,
        }
    }
    
    /// Add synchronized market data and alternative data events
    pub fn add_synchronized_events(
        &mut self,
        market_timestamp_ns: u64,
        market_data: Vec<u8>,
        alt_source: String,
        alt_payload: String,
    ) {
        // Schedule market data at exact timestamp
        self.clock.schedule_market_data(market_timestamp_ns, market_data);
        
        // Schedule alternative data within tolerance window
        let alt_timestamp = market_timestamp_ns + (self.sync_tolerance_ns / 2);
        self.clock.schedule_alternative_data(alt_timestamp, alt_source, alt_payload);
    }
    
    /// Process events, grouping those within tolerance window
    pub fn process_synchronized<F>(&mut self, mut handler: F) -> usize
    where
        F: FnMut(u64, Vec<ScheduledEvent>) -> bool,
    {
        let mut count = 0;
        let mut current_batch: Vec<ScheduledEvent> = Vec::new();
        let mut batch_start_ts: Option<u64> = None;
        
        while let Some(event) = self.clock.next_event() {
            let event_ts = event.timestamp();
            
            match batch_start_ts {
                None => {
                    batch_start_ts = Some(event_ts);
                    current_batch.push(event);
                }
                Some(start_ts) => {
                    if event_ts - start_ts <= self.sync_tolerance_ns {
                        current_batch.push(event);
                    } else {
                        // Process current batch
                        if !handler(start_ts, current_batch.clone()) {
                            break;
                        }
                        count += 1;
                        current_batch.clear();
                        current_batch.push(event);
                        batch_start_ts = Some(event_ts);
                    }
                }
            }
        }
        
        // Process final batch
        if !current_batch.is_empty() && let Some(start_ts) = batch_start_ts {
            handler(start_ts, current_batch);
            count += 1;
        }
        
        count
    }
    
    pub fn clock_mut(&mut self) -> &mut ReplayClock {
        &mut self.clock
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_replay_clock_basic() {
        let mut clock = ReplayClock::new(0, 1_000_000_000);
        
        clock.schedule_market_data(100_000, vec![1, 2, 3]);
        clock.schedule_market_data(200_000, vec![4, 5, 6]);
        clock.schedule_order_submit(150_000, 1);
        
        let mut events = Vec::new();
        while let Some(event) = clock.next_event() {
            events.push(event);
        }
        
        assert_eq!(events.len(), 3);
        assert_eq!(events[0].timestamp(), 100_000);
        assert_eq!(events[1].timestamp(), 150_000);
        assert_eq!(events[2].timestamp(), 200_000);
    }
    
    #[test]
    fn test_latency_injection() {
        let mut clock = ReplayClock::new(0, 1_000_000_000);
        clock.set_latency_model(LatencyModel::new(
            10_000, // 10 microseconds base
            5_000,  // 5 microseconds jitter
            LatencyDistribution::Uniform,
        ));
        
        clock.schedule_market_data(100_000, vec![1, 2, 3]);
        
        if let Some(event) = clock.next_event() {
            assert!(event.timestamp() >= 110_000); // At least base latency added
        }
    }
    
    #[test]
    fn test_process_until() {
        let mut clock = ReplayClock::new(0, 1_000_000_000);
        
        for i in 0..10 {
            clock.schedule_market_data(i * 100_000, vec![i as u8]);
        }
        
        let mut count = clock.process_until(500_000, |_| {});
        assert!(count >= 5);
    }
}
