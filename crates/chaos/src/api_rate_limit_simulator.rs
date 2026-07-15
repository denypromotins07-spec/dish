//! API Rate Limit Simulator for chaos engineering.
//! Simulates HTTP 429 (Too Many Requests) and 418 (IP Ban) errors.
//! Tests token-bucket rate limiters and exponential backoff logic.

use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::time::{Duration, Instant};
use std::collections::VecDeque;
use crossbeam_queue::SegQueue;

/// Rate limit error types
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RateLimitError {
    TooManyRequests(u64), // Retry-after in milliseconds
    IpBanned,
    Normal,
}

/// Token bucket configuration
#[derive(Debug, Clone)]
pub struct TokenBucketConfig {
    pub capacity: u64,
    pub refill_rate: f64, // Tokens per second
    pub initial_tokens: u64,
}

impl Default for TokenBucketConfig {
    fn default() -> Self {
        Self {
            capacity: 100,
            refill_rate: 10.0,
            initial_tokens: 100,
        }
    }
}

/// Lock-free token bucket for rate limiting simulation
pub struct TokenBucket {
    tokens: AtomicU64,
    capacity: u64,
    refill_rate: f64,
    last_refill: AtomicU64, // Timestamp in nanoseconds
}

impl TokenBucket {
    pub fn new(config: TokenBucketConfig) -> Self {
        let now_ns = Instant::now().duration_since(Instant::now()).as_nanos() as u64;
        Self {
            tokens: AtomicU64::new(config.initial_tokens),
            capacity: config.capacity,
            refill_rate: config.refill_rate,
            last_refill: AtomicU64::new(now_ns),
        }
    }

    /// Attempt to consume a token. Returns true if successful.
    #[inline]
    pub fn try_consume(&self) -> bool {
        self.refill();
        
        let mut current = self.tokens.load(Ordering::Relaxed);
        while current > 0 {
            match self.tokens.compare_exchange_weak(
                current,
                current - 1,
                Ordering::SeqCst,
                Ordering::Relaxed,
            ) {
                Ok(_) => return true,
                Err(new) => current = new,
            }
        }
        false
    }

    /// Refill tokens based on elapsed time
    #[inline]
    fn refill(&self) {
        let now_ns = Instant::now().duration_since(Instant::now()).as_nanos() as u64;
        let last = self.last_refill.load(Ordering::Relaxed);
        let elapsed_ns = now_ns.saturating_sub(last);
        
        if elapsed_ns == 0 {
            return;
        }

        let elapsed_sec = elapsed_ns as f64 / 1_000_000_000.0;
        let tokens_to_add = (elapsed_sec * self.refill_rate) as u64;
        
        if tokens_to_add == 0 {
            return;
        }

        let mut current = self.tokens.load(Ordering::Relaxed);
        loop {
            let new_tokens = (current + tokens_to_add).min(self.capacity);
            match self.tokens.compare_exchange_weak(
                current,
                new_tokens,
                Ordering::SeqCst,
                Ordering::Relaxed,
            ) {
                Ok(_) => {
                    self.last_refill.store(now_ns, Ordering::Relaxed);
                    break;
                }
                Err(new) => current = new,
            }
        }
    }

    pub fn available_tokens(&self) -> u64 {
        self.tokens.load(Ordering::Relaxed)
    }
}

/// API Rate Limit Simulator state
#[derive(Debug, Clone, Copy, PartialEq)]
enum SimulatorState {
    Normal,
    RateLimited(u64), // Until timestamp in ms
    IpBanned,
}

/// Simulates exchange-side rate limiting behavior
pub struct ApiRateLimitSimulator {
    bucket: TokenBucket,
    state: AtomicUsize, // Encoded SimulatorState
    request_count: AtomicU64,
    rate_limit_hits: AtomicU64,
    ip_ban_hits: AtomicU64,
    active: AtomicBool,
    
    // Configuration
    rate_limit_threshold: u64,      // Requests before triggering 429
    rate_limit_duration_ms: u64,    // Duration of 429
    ip_ban_threshold: u64,          // 429 hits before 418
    ip_ban_duration_ms: u64,        // Duration of 418
}

impl ApiRateLimitSimulator {
    pub fn new(bucket_config: TokenBucketConfig, rate_limit_threshold: u64, ip_ban_threshold: u64) -> Self {
        Self {
            bucket: TokenBucket::new(bucket_config),
            state: AtomicUsize::new(SimulatorState::Normal as usize),
            request_count: AtomicU64::new(0),
            rate_limit_hits: AtomicU64::new(0),
            ip_ban_hits: AtomicU64::new(0),
            active: AtomicBool::new(true),
            rate_limit_threshold,
            rate_limit_duration_ms: 5000,
            ip_ban_threshold,
            ip_ban_duration_ms: 60000,
        }
    }

    /// Simulate an API request. Returns the appropriate error type.
    #[inline]
    pub fn simulate_request(&self) -> RateLimitError {
        if !self.active.load(Ordering::Relaxed) {
            return RateLimitError::Normal;
        }

        let state_val = self.state.load(Ordering::Relaxed);
        let state = unsafe { std::mem::transmute::<usize, SimulatorState>(state_val) };

        match state {
            SimulatorState::IpBanned => {
                self.ip_ban_hits.fetch_add(1, Ordering::Relaxed);
                return RateLimitError::IpBanned;
            }
            SimulatorState::RateLimited(until_ms) => {
                let now_ms = Instant::now().duration_since(Instant::now()).as_millis() as u64;
                if now_ms < until_ms {
                    self.rate_limit_hits.fetch_add(1, Ordering::Relaxed);
                    return RateLimitError::TooManyRequests(until_ms - now_ms);
                } else {
                    // Rate limit expired, reset to normal
                    self.state.store(SimulatorState::Normal as usize, Ordering::Relaxed);
                }
            }
            SimulatorState::Normal => {}
        }

        // Count request
        let count = self.request_count.fetch_add(1, Ordering::Relaxed) + 1;

        // Check token bucket first
        if !self.bucket.try_consume() {
            // Trigger rate limit
            let now_ms = Instant::now().duration_since(Instant::now()).as_millis() as u64;
            let until_ms = now_ms + self.rate_limit_duration_ms;
            self.state.store(SimulatorState::RateLimited(until_ms) as usize, Ordering::Relaxed);
            
            let hits = self.rate_limit_hits.fetch_add(1, Ordering::Relaxed) + 1;
            
            // Check for IP ban
            if hits >= self.ip_ban_threshold {
                self.state.store(SimulatorState::IpBanned as usize, Ordering::Relaxed);
                return RateLimitError::IpBanned;
            }
            
            return RateLimitError::TooManyRequests(self.rate_limit_duration_ms);
        }

        RateLimitError::Normal
    }

    /// Reset simulator state
    pub fn reset(&self) {
        self.state.store(SimulatorState::Normal as usize, Ordering::Relaxed);
        self.request_count.store(0, Ordering::Relaxed);
        self.rate_limit_hits.store(0, Ordering::Relaxed);
        self.ip_ban_hits.store(0, Ordering::Relaxed);
    }

    /// Set active state
    pub fn set_active(&self, active: bool) {
        self.active.store(active, Ordering::Relaxed);
    }

    /// Get statistics
    pub fn get_stats(&self) -> RateLimitStats {
        RateLimitStats {
            request_count: self.request_count.load(Ordering::Relaxed),
            rate_limit_hits: self.rate_limit_hits.load(Ordering::Relaxed),
            ip_ban_hits: self.ip_ban_hits.load(Ordering::Relaxed),
            available_tokens: self.bucket.available_tokens(),
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct RateLimitStats {
    pub request_count: u64,
    pub rate_limit_hits: u64,
    pub ip_ban_hits: u64,
    pub available_tokens: u64,
}

/// Exponential backoff calculator
pub struct ExponentialBackoff {
    base_delay_ms: u64,
    max_delay_ms: u64,
    multiplier: f64,
    attempt: AtomicU32,
    jitter_factor: f64,
}

impl ExponentialBackoff {
    pub fn new(base_delay_ms: u64, max_delay_ms: u64, multiplier: f64, jitter_factor: f64) -> Self {
        Self {
            base_delay_ms,
            max_delay_ms,
            multiplier,
            attempt: AtomicU32::new(0),
            jitter_factor,
        }
    }

    /// Calculate next delay with jitter
    pub fn next_delay(&self) -> Duration {
        let attempt = self.attempt.fetch_add(1, Ordering::Relaxed) as f64;
        let delay = self.base_delay_ms as f64 * self.multiplier.powf(attempt);
        let delay = delay.min(self.max_delay_ms as f64);
        
        // Add jitter
        let jitter = delay * self.jitter_factor * (rand::random::<f64>() - 0.5);
        let final_delay = (delay + jitter).max(self.base_delay_ms as f64) as u64;
        
        Duration::from_millis(final_delay)
    }

    /// Reset attempt counter
    pub fn reset(&self) {
        self.attempt.store(0, Ordering::Relaxed);
    }

    /// Get current attempt number
    pub fn attempt(&self) -> u32 {
        self.attempt.load(Ordering::Relaxed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_token_bucket() {
        let config = TokenBucketConfig {
            capacity: 10,
            refill_rate: 1.0,
            initial_tokens: 10,
        };
        let bucket = TokenBucket::new(config);

        for _ in 0..10 {
            assert!(bucket.try_consume());
        }
        assert!(!bucket.try_consume());
    }

    #[test]
    fn test_rate_limit_simulator() {
        let bucket_config = TokenBucketConfig {
            capacity: 5,
            refill_rate: 0.0, // No refill for testing
            initial_tokens: 5,
        };
        let simulator = ApiRateLimitSimulator::new(bucket_config, 5, 3);

        // First 5 requests should succeed
        for _ in 0..5 {
            assert_eq!(simulator.simulate_request(), RateLimitError::Normal);
        }

        // Next request should trigger rate limit
        match simulator.simulate_request() {
            RateLimitError::TooManyRequests(_) => {},
            _ => panic!("Expected TooManyRequests"),
        }
    }

    #[test]
    fn test_exponential_backoff() {
        let backoff = ExponentialBackoff::new(100, 10000, 2.0, 0.1);
        
        let delay1 = backoff.next_delay();
        let delay2 = backoff.next_delay();
        
        assert!(delay2 > delay1);
        assert!(delay1.as_millis() >= 90); // With jitter
    }
}
