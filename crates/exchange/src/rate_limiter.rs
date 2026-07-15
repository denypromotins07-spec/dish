//! Strict token-bucket rate limiter for Binance REST/WebSocket APIs.
//! Handles Binance's 1200 req/min and 10 orders/sec limits without dropping packets.

use std::sync::atomic::{AtomicU64, AtomicBool, Ordering};
use std::time::{SystemTime, UNIX_EPOCH, Duration};

/// Token bucket for rate limiting
pub struct TokenBucket {
    /// Maximum tokens (capacity)
    capacity: u64,
    /// Current tokens available
    tokens: AtomicU64,
    /// Refill rate (tokens per second)
    refill_rate: u64,
    /// Last refill timestamp (nanoseconds)
    last_refill_ns: AtomicU64,
}

/// Rate limit configuration for Binance
#[derive(Debug, Clone)]
pub struct BinanceRateLimits {
    /// Weighted request limit per minute (default 1200)
    pub request_weight_per_min: u64,
    /// Order limit per second (default 10)
    pub orders_per_second: u64,
    /// Raw request limit per second (default varies by endpoint)
    pub raw_requests_per_sec: u64,
}

impl Default for BinanceRateLimits {
    fn default() -> Self {
        Self {
            request_weight_per_min: 1200,
            orders_per_second: 10,
            raw_requests_per_sec: 50,
        }
    }
}

/// Rate limit check result
#[derive(Debug, Clone)]
pub struct RateLimitResult {
    pub allowed: bool,
    pub wait_time_ms: u64,
    pub tokens_remaining: u64,
    pub retry_after_ms: Option<u64>,
}

/// Main rate limiter for Binance API
pub struct BinanceRateLimiter {
    /// Request weight bucket (for weighted endpoints)
    request_bucket: TokenBucket,
    /// Order placement bucket
    order_bucket: TokenBucket,
    /// Raw request bucket (for lightweight endpoints)
    raw_request_bucket: TokenBucket,
    /// Current limits
    limits: BinanceRateLimits,
    /// Enable strict mode (block when exceeded)
    strict_mode: AtomicBool,
    /// Total requests made today
    total_requests_today: AtomicU64,
    /// Total orders placed today
    total_orders_today: AtomicU64,
    /// Rate limit exceeded events
    rate_limit_exceeded_count: AtomicU64,
}

impl TokenBucket {
    /// Create new token bucket
    pub fn new(capacity: u64, refill_rate: u64) -> Self {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64;
        
        Self {
            capacity,
            tokens: AtomicU64::new(capacity),
            refill_rate,
            last_refill_ns: AtomicU64::new(now),
        }
    }

    /// Try to consume tokens
    #[inline(always)]
    pub fn try_consume(&self, tokens: u64) -> bool {
        self.refill();
        
        loop {
            let current = self.tokens.load(Ordering::Relaxed);
            if current < tokens {
                return false;
            }
            
            if self.tokens.compare_exchange_weak(
                current,
                current - tokens,
                Ordering::SeqCst,
                Ordering::Relaxed,
            ).is_ok() {
                return true;
            }
        }
    }

    /// Consume tokens with wait time calculation
    pub fn consume_with_wait(&self, tokens: u64) -> RateLimitResult {
        self.refill();
        
        let current = self.tokens.load(Ordering::Relaxed);
        
        if current >= tokens {
            if self.try_consume(tokens) {
                return RateLimitResult {
                    allowed: true,
                    wait_time_ms: 0,
                    tokens_remaining: current - tokens,
                    retry_after_ms: None,
                };
            }
        }
        
        // Calculate wait time
        let tokens_needed = tokens - current;
        let wait_seconds = tokens_needed as f64 / self.refill_rate as f64;
        let wait_ms = (wait_seconds * 1000.0).ceil() as u64;
        
        RateLimitResult {
            allowed: false,
            wait_time_ms: wait_ms,
            tokens_remaining: current,
            retry_after_ms: Some(wait_ms),
        }
    }

    /// Refill tokens based on elapsed time
    #[inline(always)]
    fn refill(&self) {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64;
        
        let last = self.last_refill_ns.load(Ordering::Relaxed);
        let elapsed_ns = now - last;
        
        if elapsed_ns < 1_000_000 {
            return; // Less than 1ms, skip refill
        }
        
        let elapsed_sec = elapsed_ns as f64 / 1_000_000_000.0;
        let tokens_to_add = (elapsed_sec * self.refill_rate as f64) as u64;
        
        if tokens_to_add == 0 {
            return;
        }
        
        loop {
            let current = self.tokens.load(Ordering::Relaxed);
            let new_tokens = (current + tokens_to_add).min(self.capacity);
            
            if self.tokens.compare_exchange_weak(
                current,
                new_tokens,
                Ordering::SeqCst,
                Ordering::Relaxed,
            ).is_ok() {
                self.last_refill_ns.store(now, Ordering::Relaxed);
                break;
            }
        }
    }

    /// Get current token count
    #[inline(always)]
    pub fn get_tokens(&self) -> u64 {
        self.refill();
        self.tokens.load(Ordering::Relaxed)
    }

    /// Reset bucket to full capacity
    #[inline(always)]
    pub fn reset(&self) {
        self.tokens.store(self.capacity, Ordering::Relaxed);
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64;
        self.last_refill_ns.store(now, Ordering::Relaxed);
    }
}

impl BinanceRateLimiter {
    /// Create new Binance rate limiter
    pub fn new(limits: BinanceRateLimits) -> Self {
        Self {
            request_bucket: TokenBucket::new(
                limits.request_weight_per_min,
                limits.request_weight_per_min / 60, // Per second refill
            ),
            order_bucket: TokenBucket::new(
                limits.orders_per_second,
                limits.orders_per_second,
            ),
            raw_request_bucket: TokenBucket::new(
                limits.raw_requests_per_sec,
                limits.raw_requests_per_sec,
            ),
            limits,
            strict_mode: AtomicBool::new(true),
            total_requests_today: AtomicU64::new(0),
            total_orders_today: AtomicU64::new(0),
            rate_limit_exceeded_count: AtomicU64::new(0),
        }
    }

    /// Check if a weighted request is allowed
    pub fn check_request(&self, weight: u64) -> RateLimitResult {
        let result = self.request_bucket.consume_with_wait(weight);
        
        if result.allowed {
            self.total_requests_today.fetch_add(1, Ordering::Relaxed);
        } else {
            self.rate_limit_exceeded_count.fetch_add(1, Ordering::Relaxed);
        }
        
        result
    }

    /// Check if an order placement is allowed
    pub fn check_order_placement(&self) -> RateLimitResult {
        let result = self.order_bucket.consume_with_wait(1);
        
        if result.allowed {
            self.total_orders_today.fetch_add(1, Ordering::Relaxed);
        } else {
            self.rate_limit_exceeded_count.fetch_add(1, Ordering::Relaxed);
        }
        
        result
    }

    /// Check if a raw (lightweight) request is allowed
    pub fn check_raw_request(&self) -> RateLimitResult {
        let result = self.raw_request_bucket.consume_with_wait(1);
        
        if result.allowed {
            self.total_requests_today.fetch_add(1, Ordering::Relaxed);
        }
        
        result
    }

    /// Check all limits before placing an order
    pub fn pre_order_check(&self, request_weight: u64) -> PreOrderCheckResult {
        let request_result = self.check_request(request_weight);
        let order_result = self.check_order_placement();
        
        PreOrderCheckResult {
            allowed: request_result.allowed && order_result.allowed,
            request_wait_ms: if request_result.allowed { 0 } else { request_result.wait_time_ms },
            order_wait_ms: if order_result.allowed { 0 } else { order_result.wait_time_ms },
            max_wait_ms: request_result.wait_time_ms.max(order_result.wait_time_ms),
        }
    }

    /// Wait until rate limit allows (blocking)
    pub fn wait_for_allowance(&self, weight: u64) {
        loop {
            let result = self.check_request(weight);
            if result.allowed {
                break;
            }
            
            if let Some(wait_ms) = result.retry_after_ms {
                std::thread::sleep(Duration::from_millis(wait_ms));
            }
        }
    }

    /// Non-blocking check with immediate result
    pub fn try_request(&self, weight: u64) -> bool {
        self.request_bucket.try_consume(weight)
    }

    /// Non-blocking order placement check
    pub fn try_order(&self) -> bool {
        self.order_bucket.try_consume(1)
    }

    /// Get current rate limit status
    pub fn get_status(&self) -> RateLimitStatus {
        RateLimitStatus {
            request_tokens: self.request_bucket.get_tokens(),
            request_capacity: self.limits.request_weight_per_min,
            order_tokens: self.order_bucket.get_tokens(),
            order_capacity: self.limits.orders_per_second,
            raw_tokens: self.raw_request_bucket.get_tokens(),
            raw_capacity: self.limits.raw_requests_per_sec,
            total_requests_today: self.total_requests_today.load(Ordering::Relaxed),
            total_orders_today: self.total_orders_today.load(Ordering::Relaxed),
            rate_limit_events: self.rate_limit_exceeded_count.load(Ordering::Relaxed),
        }
    }

    /// Set strict mode
    #[inline(always)]
    pub fn set_strict_mode(&self, strict: bool) {
        self.strict_mode.store(strict, Ordering::Relaxed);
    }

    /// Reset daily counters
    #[inline(always)]
    pub fn reset_daily_counters(&self) {
        self.total_requests_today.store(0, Ordering::Relaxed);
        self.total_orders_today.store(0, Ordering::Relaxed);
        self.rate_limit_exceeded_count.store(0, Ordering::Relaxed);
    }

    /// Update limits dynamically
    pub fn update_limits(&mut self, new_limits: BinanceRateLimits) {
        self.limits = new_limits.clone();
        
        // Recreate buckets with new limits
        self.request_bucket = TokenBucket::new(
            new_limits.request_weight_per_min,
            new_limits.request_weight_per_min / 60,
        );
        self.order_bucket = TokenBucket::new(
            new_limits.orders_per_second,
            new_limits.orders_per_second,
        );
        self.raw_request_bucket = TokenBucket::new(
            new_limits.raw_requests_per_sec,
            new_limits.raw_requests_per_sec,
        );
    }
}

impl Default for BinanceRateLimiter {
    fn default() -> Self {
        Self::new(BinanceRateLimits::default())
    }
}

/// Pre-order check result
#[derive(Debug, Clone)]
pub struct PreOrderCheckResult {
    pub allowed: bool,
    pub request_wait_ms: u64,
    pub order_wait_ms: u64,
    pub max_wait_ms: u64,
}

/// Rate limit status snapshot
#[derive(Debug, Clone)]
pub struct RateLimitStatus {
    pub request_tokens: u64,
    pub request_capacity: u64,
    pub order_tokens: u64,
    pub order_capacity: u64,
    pub raw_tokens: u64,
    pub raw_capacity: u64,
    pub total_requests_today: u64,
    pub total_orders_today: u64,
    pub rate_limit_events: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_token_bucket() {
        let bucket = TokenBucket::new(10, 5); // 10 capacity, 5/sec refill
        
        // Consume some tokens
        assert!(bucket.try_consume(5));
        assert_eq!(bucket.get_tokens(), 5);
        
        // Try to consume more than available
        assert!(!bucket.try_consume(6));
        
        // Should have exactly 5 remaining
        assert_eq!(bucket.get_tokens(), 5);
    }

    #[test]
    fn test_rate_limiter() {
        let limits = BinanceRateLimits {
            request_weight_per_min: 60,
            orders_per_second: 5,
            raw_requests_per_sec: 20,
        };
        
        let limiter = BinanceRateLimiter::new(limits);
        
        // Initial requests should succeed
        let result = limiter.check_request(10);
        assert!(result.allowed);
        
        // Check order placement
        let order_result = limiter.check_order_placement();
        assert!(order_result.allowed);
        
        let status = limiter.get_status();
        assert_eq!(status.total_orders_today, 1);
    }

    #[test]
    fn test_pre_order_check() {
        let limiter = BinanceRateLimiter::default();
        
        let result = limiter.pre_order_check(5);
        assert!(result.allowed);
        assert_eq!(result.max_wait_ms, 0);
    }
}
