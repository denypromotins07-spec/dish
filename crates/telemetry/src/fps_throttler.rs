//! Dynamic UI update throttler that automatically adjusts frontend push rate.
//! Drops from 60fps to 10fps during extreme market volatility to save resources.

use std::sync::atomic::{AtomicU64, AtomicU32, AtomicBool, Ordering};
use std::time::{Duration, Instant};

/// FPS throttler configuration
pub struct FpsThrottlerConfig {
    /// Normal target FPS
    pub normal_fps: u32,
    /// Minimum FPS during high volatility
    pub min_fps: u32,
    /// Volatility threshold to trigger throttling (percentage)
    pub volatility_threshold: f64,
    /// Cool-down period after throttling (ms)
    pub cooldown_ms: u64,
    /// Window size for volatility calculation (ms)
    pub volatility_window_ms: u64,
}

impl Default for FpsThrottlerConfig {
    fn default() -> Self {
        Self {
            normal_fps: 60,
            min_fps: 10,
            volatility_threshold: 5.0, // 5% price movement triggers throttling
            cooldown_ms: 5000,
            volatility_window_ms: 1000,
        }
    }
}

/// Current throttle state
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ThrottleState {
    Normal,
    Moderate,
    Aggressive,
}

impl ThrottleState {
    /// Get target FPS for this state
    pub fn target_fps(&self, config: &FpsThrottlerConfig) -> u32 {
        match self {
            ThrottleState::Normal => config.normal_fps,
            ThrottleState::Moderate => (config.normal_fps + config.min_fps) / 2,
            ThrottleState::Aggressive => config.min_fps,
        }
    }

    /// Get minimum interval between updates (ms)
    pub fn update_interval_ms(&self, config: &FpsThrottlerConfig) -> u64 {
        1000 / self.target_fps(config) as u64
    }
}

/// Dynamic FPS throttler for UI updates
pub struct FpsThrottler {
    config: FpsThrottlerConfig,
    current_state: AtomicU32, // ThrottleState as u32
    last_update_ns: AtomicU64,
    volatility_recent: AtomicU64, // Fixed-point * 1000
    is_throttling: AtomicBool,
    throttle_start_ns: AtomicU64,
    frames_dropped: AtomicU64,
    frames_sent: AtomicU64,
}

impl FpsThrottler {
    /// Create new FPS throttler
    pub fn new(config: FpsThrottlerConfig) -> Self {
        Self {
            config,
            current_state: AtomicU32::new(ThrottleState::Normal as u32),
            last_update_ns: AtomicU64::new(0),
            volatility_recent: AtomicU64::new(0),
            is_throttling: AtomicBool::new(false),
            throttle_start_ns: AtomicU64::new(0),
            frames_dropped: AtomicU64::new(0),
            frames_sent: AtomicU64::new(0),
        }
    }

    /// Check if an update should be sent based on current throttle state
    pub fn should_send_update(&self) -> bool {
        let now_ns = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64;

        let state = unsafe { std::mem::transmute::<u32, ThrottleState>(
            self.current_state.load(Ordering::Relaxed)
        )};

        let min_interval_ns = state.update_interval_ms(&self.config) * 1_000_000;
        let last_update = self.last_update_ns.load(Ordering::Relaxed);

        if now_ns - last_update < min_interval_ns {
            self.frames_dropped.fetch_add(1, Ordering::Relaxed);
            return false;
        }

        self.last_update_ns.store(now_ns, Ordering::Relaxed);
        self.frames_sent.fetch_add(1, Ordering::Relaxed);
        true
    }

    /// Update volatility measurement and adjust throttle state
    pub fn update_volatility(&self, price_change_pct: f64) {
        let volatility_fixed = (price_change_pct * 1000.0) as u64;
        self.volatility_recent.store(volatility_fixed, Ordering::Relaxed);

        let volatility = price_change_pct;
        let state = if volatility >= self.config.volatility_threshold * 2.0 {
            ThrottleState::Aggressive
        } else if volatility >= self.config.volatility_threshold {
            ThrottleState::Moderate
        } else {
            ThrottleState::Normal
        };

        let old_state = unsafe { std::mem::transmute::<u32, ThrottleState>(
            self.current_state.swap(state as u32, Ordering::Relaxed)
        )};

        // Track when throttling starts
        if state != ThrottleState::Normal && old_state == ThrottleState::Normal {
            self.is_throttling.store(true, Ordering::Relaxed);
            self.throttle_start_ns.store(
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_nanos() as u64,
                Ordering::Relaxed,
            );
        } else if state == ThrottleState::Normal && old_state != ThrottleState::Normal {
            self.is_throttling.store(false, Ordering::Relaxed);
        }

        // Auto-cooldown: force return to normal after cooldown period
        if state != ThrottleState::Normal {
            let throttle_duration_ns = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos() as u64
                - self.throttle_start_ns.load(Ordering::Relaxed);

            if throttle_duration_ns >= self.config.cooldown_ms * 1_000_000 {
                // Allow return to normal if volatility has decreased
                if volatility < self.config.volatility_threshold {
                    self.current_state.store(ThrottleState::Normal as u32, Ordering::Relaxed);
                    self.is_throttling.store(false, Ordering::Relaxed);
                }
            }
        }
    }

    /// Get current throttle state
    pub fn state(&self) -> ThrottleState {
        unsafe { std::mem::transmute::<u32, ThrottleState>(
            self.current_state.load(Ordering::Relaxed)
        )}
    }

    /// Get current target FPS
    pub fn current_target_fps(&self) -> u32 {
        self.state().target_fps(&self.config)
    }

    /// Get frame statistics
    pub fn get_stats(&self) -> FrameStats {
        FrameStats {
            frames_sent: self.frames_sent.load(Ordering::Relaxed),
            frames_dropped: self.frames_dropped.load(Ordering::Relaxed),
            drop_rate: {
                let total = self.frames_sent.load(Ordering::Relaxed) + self.frames_dropped.load(Ordering::Relaxed);
                if total > 0 {
                    self.frames_dropped.load(Ordering::Relaxed) as f64 / total as f64
                } else {
                    0.0
                }
            },
            current_fps: self.current_target_fps(),
            is_throttling: self.is_throttling.load(Ordering::Relaxed),
            volatility: self.volatility_recent.load(Ordering::Relaxed) as f64 / 1000.0,
        }
    }

    /// Reset statistics
    pub fn reset_stats(&self) {
        self.frames_dropped.store(0, Ordering::Relaxed);
        self.frames_sent.store(0, Ordering::Relaxed);
    }

    /// Force a specific throttle state (for manual override)
    pub fn force_state(&self, state: ThrottleState) {
        self.current_state.store(state as u32, Ordering::Relaxed);
        if state != ThrottleState::Normal {
            self.is_throttling.store(true, Ordering::Relaxed);
        }
    }
}

/// Frame statistics
#[derive(Debug, Clone)]
pub struct FrameStats {
    pub frames_sent: u64,
    pub frames_dropped: u64,
    pub drop_rate: f64,
    pub current_fps: u32,
    pub is_throttling: bool,
    pub volatility: f64,
}

/// Helper for calculating price volatility
pub mod volatility_helpers {
    /// Calculate percentage price change
    #[inline]
    pub fn price_change_pct(old_price: f64, new_price: f64) -> f64 {
        if old_price <= 0.0 {
            return 0.0;
        }
        ((new_price - old_price) / old_price) * 100.0
    }

    /// Calculate rolling volatility from a series of prices
    pub fn rolling_volatility(prices: &[f64], window_size: usize) -> f64 {
        if prices.len() < 2 || window_size < 2 {
            return 0.0;
        }

        let start = prices.len().saturating_sub(window_size);
        let window = &prices[start..];

        let returns: Vec<f64> = window
            .windows(2)
            .map(|w| (w[1] - w[0]) / w[0])
            .collect();

        if returns.is_empty() {
            return 0.0;
        }

        let mean = returns.iter().sum::<f64>() / returns.len() as f64;
        let variance = returns.iter()
            .map(|r| (r - mean).powi(2))
            .sum::<f64>() / returns.len() as f64;

        variance.sqrt() * 100.0 // Return as percentage
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_throttle_state_fps() {
        let config = FpsThrottlerConfig::default();

        assert_eq!(ThrottleState::Normal.target_fps(&config), 60);
        assert_eq!(ThrottleState::Moderate.target_fps(&config), 35);
        assert_eq!(ThrottleState::Aggressive.target_fps(&config), 10);
    }

    #[test]
    fn test_volatility_based_throttling() {
        let config = FpsThrottlerConfig::default();
        let throttler = FpsThrottler::new(config);

        // Start in normal state
        assert_eq!(throttler.state(), ThrottleState::Normal);

        // High volatility should trigger throttling
        throttler.update_volatility(10.0); // 10% change
        assert_eq!(throttler.state(), ThrottleState::Aggressive);

        // Moderate volatility
        throttler.update_volatility(3.0);
        assert_eq!(throttler.state(), ThrottleState::Moderate);

        // Low volatility returns to normal
        throttler.update_volatility(0.5);
        assert_eq!(throttler.state(), ThrottleState::Normal);
    }

    #[test]
    fn test_should_send_update() {
        let config = FpsThrottlerConfig {
            normal_fps: 100, // 10ms interval for testing
            ..Default::default()
        };
        let throttler = FpsThottler::new(config);

        // First call should always succeed
        assert!(throttler.should_send_update());

        // Immediate second call might be dropped depending on timing
        // (this test is timing-dependent)
    }

    #[test]
    fn test_frame_stats() {
        let config = FpsThrottlerConfig::default();
        let throttler = FpsThrottler::new(config);

        let stats = throttler.get_stats();
        
        assert_eq!(stats.frames_sent, 0);
        assert_eq!(stats.frames_dropped, 0);
        assert_eq!(stats.current_fps, 60);
        assert!(!stats.is_throttling);
    }

    #[test]
    fn test_force_state() {
        let config = FpsThrottlerConfig::default();
        let throttler = FpsThrottler::new(config);

        throttler.force_state(ThrottleState::Aggressive);
        
        assert_eq!(throttler.state(), ThrottleState::Aggressive);
        assert!(throttler.is_throttling.load(Ordering::Relaxed));
    }

    #[test]
    fn test_volatility_helpers() {
        use volatility_helpers::*;

        let change = price_change_pct(100.0, 105.0);
        assert!((change - 5.0).abs() < 0.001);

        let prices = vec![100.0, 101.0, 100.5, 102.0, 101.5];
        let vol = rolling_volatility(&prices, 5);
        assert!(vol > 0.0);
    }
}
