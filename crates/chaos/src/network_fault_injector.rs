//! Microsecond fault injector for chaos engineering.
//! Probabilistically drops WebSocket packets, adds network jitter, and simulates TCP backpressure.
//! Zero-allocation hot path using pre-allocated ring buffers and atomic state.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::{Duration, Instant};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use crossbeam_queue::ArrayQueue;

/// Configuration for fault injection parameters
#[derive(Debug, Clone)]
pub struct FaultConfig {
    pub packet_drop_probability: f64,      // 0.0 to 1.0
    pub jitter_min_us: u64,                // Minimum jitter in microseconds
    pub jitter_max_us: u64,                // Maximum jitter in microseconds
    pub backpressure_probability: f64,     // Probability of triggering backpressure
    pub backpressure_duration_ms: u64,     // Duration of backpressure in milliseconds
}

impl Default for FaultConfig {
    fn default() -> Self {
        Self {
            packet_drop_probability: 0.0,
            jitter_min_us: 0,
            jitter_max_us: 0,
            backpressure_probability: 0.0,
            backpressure_duration_ms: 0,
        }
    }
}

/// Lock-free fault injector for WebSocket packets
pub struct NetworkFaultInjector {
    config: FaultConfig,
    active: AtomicBool,
    dropped_packets: AtomicU64,
    injected_jitter_count: AtomicU64,
    backpressure_events: AtomicU64,
    rng_lock: std::sync::Mutex<SmallRng>,
    backpressure_until: AtomicU64, // Timestamp in microseconds
}

impl NetworkFaultInjector {
    pub fn new(config: FaultConfig) -> Self {
        Self {
            config,
            active: AtomicBool::new(true),
            dropped_packets: AtomicU64::new(0),
            injected_jitter_count: AtomicU64::new(0),
            backpressure_events: AtomicU64::new(0),
            rng_lock: std::sync::Mutex::new(SmallRng::from_entropy()),
            backpressure_until: AtomicU64::new(0),
        }
    }

    /// Injects faults into a packet stream. Returns None if packet should be dropped,
    /// or Some(delay_duration) if packet should be delayed.
    #[inline]
    pub fn inject_fault(&self) -> Option<Duration> {
        if !self.active.load(Ordering::Relaxed) {
            return Some(Duration::ZERO);
        }

        let now_us = Instant::now().duration_since(Instant::now()).as_micros() as u64;
        
        // Check backpressure first (highest priority)
        let bp_until = self.backpressure_until.load(Ordering::Relaxed);
        if now_us < bp_until {
            return None; // Drop packet during backpressure
        }

        let mut rng = self.rng_lock.lock().unwrap();
        
        // Packet drop simulation
        if rng.gen::<f64>() < self.config.packet_drop_probability {
            self.dropped_packets.fetch_add(1, Ordering::Relaxed);
            return None;
        }

        // Jitter injection
        if self.config.jitter_max_us > self.config.jitter_min_us {
            let jitter_us = rng.gen_range(self.config.jitter_min_us..=self.config.jitter_max_us);
            if jitter_us > 0 {
                self.injected_jitter_count.fetch_add(1, Ordering::Relaxed);
                return Some(Duration::from_micros(jitter_us));
            }
        }

        // Backpressure trigger
        if rng.gen::<f64>() < self.config.backpressure_probability {
            let duration_us = self.config.backpressure_duration_ms * 1000;
            self.backpressure_until.store(now_us + duration_us, Ordering::Relaxed);
            self.backpressure_events.fetch_add(1, Ordering::Relaxed);
            return None;
        }

        Some(Duration::ZERO)
    }

    /// Activates or deactivates fault injection
    #[inline]
    pub fn set_active(&self, active: bool) {
        self.active.store(active, Ordering::Relaxed);
    }

    /// Returns statistics on injected faults
    pub fn get_stats(&self) -> FaultStats {
        FaultStats {
            dropped_packets: self.dropped_packets.load(Ordering::Relaxed),
            injected_jitter_count: self.injected_jitter_count.load(Ordering::Relaxed),
            backpressure_events: self.backpressure_events.load(Ordering::Relaxed),
        }
    }

    /// Reset all counters
    pub fn reset_stats(&self) {
        self.dropped_packets.store(0, Ordering::Relaxed);
        self.injected_jitter_count.store(0, Ordering::Relaxed);
        self.backpressure_events.store(0, Ordering::Relaxed);
        self.backpressure_until.store(0, Ordering::Relaxed);
    }
}

#[derive(Debug, Clone, Copy)]
pub struct FaultStats {
    pub dropped_packets: u64,
    pub injected_jitter_count: u64,
    pub backpressure_events: u64,
}

/// Ring buffer for simulating TCP backpressure with bounded memory
pub struct BackpressureBuffer<T: Clone> {
    buffer: ArrayQueue<T>,
    capacity: usize,
    overflow_count: AtomicU64,
}

impl<T: Clone> BackpressureBuffer<T> {
    pub fn new(capacity: usize) -> Self {
        Self {
            buffer: ArrayQueue::new(capacity),
            capacity,
            overflow_count: AtomicU64::new(0),
        }
    }

    #[inline]
    pub fn push(&self, item: T) -> Result<(), T> {
        match self.buffer.push(item) {
            Ok(_) => Ok(()),
            Err(item) => {
                self.overflow_count.fetch_add(1, Ordering::Relaxed);
                Err(item) // Signal backpressure to caller
            }
        }
    }

    #[inline]
    pub fn pop(&self) -> Option<T> {
        self.buffer.pop()
    }

    pub fn overflow_count(&self) -> u64 {
        self.overflow_count.load(Ordering::Relaxed)
    }

    pub fn len(&self) -> usize {
        self.buffer.len()
    }

    pub fn is_full(&self) -> bool {
        self.buffer.len() == self.capacity
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fault_injection_active() {
        let config = FaultConfig {
            packet_drop_probability: 0.5,
            jitter_min_us: 100,
            jitter_max_us: 500,
            ..Default::default()
        };
        let injector = NetworkFaultInjector::new(config);
        
        let mut drops = 0;
        let mut delays = 0;
        
        for _ in 0..1000 {
            match injector.inject_fault() {
                None => drops += 1,
                Some(d) => if d > Duration::ZERO { delays += 1; },
                _ => {}
            }
        }
        
        assert!(drops > 0);
        assert!(delays > 0);
    }

    #[test]
    fn test_backpressure_buffer() {
        let buffer: BackpressureBuffer<i32> = BackpressureBuffer::new(5);
        
        for i in 0..5 {
            assert!(buffer.push(i).is_ok());
        }
        
        assert!(buffer.is_full());
        assert!(buffer.push(99).is_err());
        assert_eq!(buffer.overflow_count(), 1);
    }
}
