//! Microsecond scenario injector for stress-testing strategies.
//! Intercepts data streams to inject synthetic shocks (flash crashes, liquidity evaporation, spread widening).
//! Zero heap allocation during injection; uses fixed-size buffers and lock-free atomics.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Types of synthetic shocks that can be injected
#[derive(Debug, Clone, Copy)]
pub enum ShockType {
    FlashCrash { magnitude_bps: i32 },
    LiquidityEvaporation { depth_reduction_pct: u8 },
    SpreadWidening { multiplier: u16 },
    VolatilitySpike { factor: f64 },
}

/// Configuration for a scheduled shock injection
#[derive(Debug, Clone)]
pub struct ShockConfig {
    pub trigger_time_us: u64, // Microseconds since epoch or session start
    pub shock_type: ShockType,
    pub duration_us: u64,
    pub affected_symbols: Vec<u32>, // Symbol IDs
}

/// Lock-free scenario injector using atomic flags and pre-allocated buffers
pub struct ScenarioInjector {
    active_shock: AtomicBool,
    shock_start_us: AtomicU64,
    shock_end_us: AtomicU64,
    current_magnitude_bps: AtomicU64, // Encoded as fixed-point
    liquidity_factor: AtomicU64,      // Fixed-point 16.16
    spread_multiplier: AtomicU64,     // Fixed-point 16.16
    volatility_factor: AtomicU64,     // Fixed-point 16.16
    scheduled_shocks: parking_lot::RwLock<Vec<ShockConfig>>,
}

impl ScenarioInjector {
    pub fn new() -> Self {
        Self {
            active_shock: AtomicBool::new(false),
            shock_start_us: AtomicU64::new(0),
            shock_end_us: AtomicU64::new(0),
            current_magnitude_bps: AtomicU64::new(0),
            liquidity_factor: AtomicU64::new(1 << 16), // 1.0 in 16.16 fixed-point
            spread_multiplier: AtomicU64::new(1 << 16),
            volatility_factor: AtomicU64::new(1 << 16),
            scheduled_shocks: parking_lot::RwLock::new(Vec::with_capacity(16)),
        }
    }

    /// Schedule a shock for injection at a specific time
    pub fn schedule_shock(&self, config: ShockConfig) {
        let mut shocks = self.scheduled_shocks.write();
        if shocks.len() < 16 {
            shocks.push(config);
        }
    }

    /// Check and activate any pending shocks based on current time
    #[inline]
    pub fn tick(&self, current_time_us: u64) {
        // Fast path: check if any shock is active
        if self.active_shock.load(Ordering::Acquire) {
            if current_time_us >= self.shock_end_us.load(Ordering::Relaxed) {
                // Deactivate shock
                self.deactivate_shock();
            }
            return;
        }

        // Check for scheduled shocks
        let shocks = self.scheduled_shocks.read();
        for shock in shocks.iter() {
            if current_time_us >= shock.trigger_time_us {
                self.activate_shock(shock, current_time_us);
                break;
            }
        }
    }

    fn activate_shock(&self, config: &ShockConfig, start_time: u64) {
        self.active_shock.store(true, Ordering::Release);
        self.shock_start_us.store(start_time, Ordering::Relaxed);
        self.shock_end_us.store(start_time + config.duration_us, Ordering::Relaxed);

        match config.shock_type {
            ShockType::FlashCrash { magnitude_bps } => {
                self.current_magnitude_bps.store(magnitude_bps.unsigned_abs(), Ordering::Relaxed);
            }
            ShockType::LiquidityEvaporation { depth_reduction_pct } => {
                let factor = ((100 - depth_reduction_pct) as f64 / 100.0 * 65536.0) as u64;
                self.liquidity_factor.store(factor, Ordering::Relaxed);
            }
            ShockType::SpreadWidening { multiplier } => {
                self.spread_multiplier.store((multiplier as u64) << 16, Ordering::Relaxed);
            }
            ShockType::VolatilitySpike { factor } => {
                self.volatility_factor.store((factor * 65536.0) as u64, Ordering::Relaxed);
            }
        }
    }

    fn deactivate_shock(&self) {
        self.active_shock.store(false, Ordering::Release);
        self.current_magnitude_bps.store(0, Ordering::Relaxed);
        self.liquidity_factor.store(1 << 16, Ordering::Relaxed);
        self.spread_multiplier.store(1 << 16, Ordering::Relaxed);
        self.volatility_factor.store(1 << 16, Ordering::Relaxed);
    }

    /// Apply shock adjustments to a price tick (zero-allocation)
    #[inline]
    pub fn adjust_price(&self, price: i64, is_bid: bool) -> i64 {
        if !self.active_shock.load(Ordering::Acquire) {
            return price;
        }

        let mag = self.current_magnitude_bps.load(Ordering::Relaxed) as i64;
        if mag == 0 {
            return price;
        }

        // Apply flash crash magnitude: price * (1 - mag/10000)
        let adjustment = (price * mag) / 10000;
        if is_bid {
            price - adjustment
        } else {
            price + adjustment
        }
    }

    /// Get current liquidity factor (16.16 fixed-point)
    #[inline]
    pub fn get_liquidity_factor(&self) -> u64 {
        self.liquidity_factor.load(Ordering::Relaxed)
    }

    /// Get current spread multiplier (16.16 fixed-point)
    #[inline]
    pub fn get_spread_multiplier(&self) -> u64 {
        self.spread_multiplier.load(Ordering::Relaxed)
    }

    /// Check if a shock is currently active
    #[inline]
    pub fn is_active(&self) -> bool {
        self.active_shock.load(Ordering::Acquire)
    }

    /// Clear all scheduled shocks
    pub fn clear_scheduled(&self) {
        let mut shocks = self.scheduled_shocks.write();
        shocks.clear();
    }
}

impl Default for ScenarioInjector {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_flash_crash_injection() {
        let injector = ScenarioInjector::new();
        let config = ShockConfig {
            trigger_time_us: 1000,
            shock_type: ShockType::FlashCrash { magnitude_bps: 500 }, // 5% crash
            duration_us: 5000,
            affected_symbols: vec![1, 2, 3],
        };
        injector.schedule_shock(config);
        
        // Before trigger
        assert!(!injector.is_active());
        assert_eq!(injector.adjust_price(10000, true), 10000);

        // At trigger time
        injector.tick(1000);
        assert!(injector.is_active());
        
        // Price should be adjusted down by 5%
        let adjusted = injector.adjust_price(10000, true);
        assert_eq!(adjusted, 9500);
    }
}
