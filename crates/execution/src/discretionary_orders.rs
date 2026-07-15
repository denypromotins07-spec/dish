//! Discretionary and reserve order logic with hidden true size
//! Displays micro-clip while dynamically adjusting discretionary offset
//! Optimized for AMD Ryzen with zero heap allocations

use std::sync::atomic::{AtomicI64, AtomicU64, AtomicBool, AtomicF64, Ordering};

/// Reserve order state with hidden quantity
#[derive(Debug, Clone, Copy)]
pub struct ReserveOrder {
    /// Order ID
    pub order_id: u64,
    /// Total order quantity (scaled by 1e8)
    pub total_qty: i64,
    /// Display quantity (visible clip, scaled by 1e8)
    pub display_qty: i64,
    /// Hidden reserve quantity (scaled by 1e8)
    pub hidden_qty: i64,
    /// Filled quantity (scaled by 1e8)
    pub filled_qty: i64,
    /// Limit price (scaled by 1e8)
    pub limit_price: i64,
    /// Is buy order
    pub is_buy: bool,
    /// Is order active
    pub is_active: bool,
    /// Last fill timestamp
    pub last_fill_us: u64,
}

impl ReserveOrder {
    /// Create new reserve order
    #[inline(always)]
    pub fn new(order_id: u64, total_qty: i64, display_qty: i64, limit_price: i64, is_buy: bool) -> Self {
        let hidden_qty = (total_qty - display_qty).max(0);
        
        Self {
            order_id,
            total_qty,
            display_qty,
            hidden_qty,
            filled_qty: 0,
            limit_price,
            is_buy,
            is_active: true,
            last_fill_us: 0,
        }
    }

    /// Get remaining quantity
    #[inline(always)]
    pub fn remaining(&self) -> i64 {
        self.total_qty - self.filled_qty
    }

    /// Get current visible quantity
    #[inline(always)]
    pub fn current_visible(&self) -> i64 {
        let remaining = self.remaining();
        remaining.min(self.display_qty)
    }

    /// Check if hidden quantity remains
    #[inline(always)]
    pub fn has_hidden(&self) -> bool {
        self.hidden_qty > 0 || (self.remaining() > self.display_qty)
    }

    /// Process a fill
    #[inline(always)]
    pub fn fill(&mut self, qty: i64, timestamp_us: u64) -> i64 {
        let remaining = self.remaining();
        let actual_fill = qty.min(remaining);
        
        self.filled_qty += actual_fill;
        self.last_fill_us = timestamp_us;
        
        // Replenish display from hidden if needed
        if self.remaining() < self.display_qty && self.hidden_qty > 0 {
            let replenish = self.display_qty - self.remaining();
            let from_hidden = replenish.min(self.hidden_qty);
            self.hidden_qty -= from_hidden;
        }
        
        // Deactivate if fully filled
        if self.remaining() <= 0 {
            self.is_active = false;
        }
        
        actual_fill
    }

    /// Cancel order
    #[inline(always)]
    pub fn cancel(&mut self) {
        self.is_active = false;
    }
}

/// Discretionary offset manager for dynamic price adjustment
pub struct DiscretionaryManager {
    /// Base discretionary offset (scaled by 1e8)
    base_offset: AtomicI64,
    /// Current active offset (scaled by 1e8)
    current_offset: AtomicI64,
    /// Max allowed offset (scaled by 1e8)
    max_offset: AtomicI64,
    /// Volatility scaling factor (scaled by 1e6)
    vol_scale: AtomicF64,
    /// Pressure threshold for activation (scaled by 1e8)
    pressure_threshold: AtomicI64,
    /// Is discretion currently active
    is_active: AtomicBool,
    /// Activation count
    activation_count: AtomicU64,
}

impl DiscretionaryManager {
    pub fn new(base_offset: i64, max_offset: i64) -> Self {
        Self {
            base_offset: AtomicI64::new(base_offset),
            current_offset: AtomicI64::new(base_offset),
            max_offset: AtomicI64::new(max_offset),
            vol_scale: AtomicF64::new(1.0),
            pressure_threshold: AtomicI64::new(500_000), // Default threshold
            is_active: AtomicBool::new(false),
            activation_count: AtomicU64::new(0),
        }
    }

    /// Calculate discretionary price based on market pressure
    #[inline(always)]
    pub fn calculate_discretionary_price(&self, base_price: i64, is_buy: bool, market_pressure: i64) -> i64 {
        let threshold = self.pressure_threshold.load(Ordering::Relaxed);
        let vol = self.vol_scale.load(Ordering::Relaxed);
        
        // Check if pressure exceeds threshold
        if market_pressure.abs() > threshold {
            self.is_active.store(true, Ordering::Relaxed);
            self.activation_count.fetch_add(1, Ordering::Relaxed);
            
            // Calculate scaled offset
            let base = self.base_offset.load(Ordering::Relaxed);
            let pressure_ratio = (market_pressure.abs() as f64 / threshold as f64).min(3.0);
            let scaled_offset = (base as f64 * pressure_ratio * vol) as i64;
            let capped_offset = scaled_offset.min(self.max_offset.load(Ordering::Relaxed));
            
            self.current_offset.store(capped_offset, Ordering::Relaxed);
            
            // Apply offset in favorable direction for faster fill
            if is_buy {
                base_price + capped_offset // Pay more
            } else {
                base_price - capped_offset // Accept less
            }
        } else {
            self.is_active.store(false, Ordering::Relaxed);
            self.current_offset.store(self.base_offset.load(Ordering::Relaxed), Ordering::Relaxed);
            base_price
        }
    }

    /// Update volatility scaling factor
    #[inline(always)]
    pub fn update_volatility(&self, volatility: f64) {
        // Higher volatility = more aggressive discretion
        let scale = (1.0 + volatility * 0.5).min(3.0);
        self.vol_scale.store(scale, Ordering::Relaxed);
    }

    /// Set pressure threshold
    #[inline(always)]
    pub fn set_threshold(&self, threshold: i64) {
        self.pressure_threshold.store(threshold, Ordering::Relaxed);
    }

    /// Get current offset
    #[inline(always)]
    pub fn get_current_offset(&self) -> i64 {
        self.current_offset.load(Ordering::Relaxed)
    }

    /// Check if discretion is active
    #[inline(always)]
    pub fn is_discretion_active(&self) -> bool {
        self.is_active.load(Ordering::Relaxed)
    }

    /// Get statistics
    #[inline(always)]
    pub fn get_stats(&self) -> (u64, f64, bool) {
        (
            self.activation_count.load(Ordering::Relaxed),
            self.vol_scale.load(Ordering::Relaxed),
            self.is_active.load(Ordering::Relaxed),
        )
    }
}

/// Dynamic clip adjuster for reserve orders
pub struct DynamicClipAdjuster {
    /// Minimum clip size (scaled by 1e8)
    min_clip: AtomicI64,
    /// Maximum clip size (scaled by 1e8)
    max_clip: AtomicI64,
    /// Target execution time in microseconds
    target_time_us: AtomicU64,
    /// Current fill rate (per second, scaled by 1e4)
    fill_rate: AtomicF64,
    /// Adaptation speed (scaled by 1e6)
    adaptation_speed: AtomicF64,
}

impl DynamicClipAdjuster {
    pub fn new(min_clip: i64, max_clip: i64, target_time_ms: u64) -> Self {
        Self {
            min_clip: AtomicI64::new(min_clip),
            max_clip: AtomicI64::new(max_clip),
            target_time_us: AtomicU64::new(target_time_ms * 1000),
            fill_rate: AtomicF64::new(0.0),
            adaptation_speed: AtomicF64::new(0.1),
        }
    }

    /// Calculate optimal clip size based on execution progress
    #[inline(always)]
    pub fn calculate_optimal_clip(&self, remaining: i64, elapsed_us: u64, fill_rate_observed: f64) -> i64 {
        let min = self.min_clip.load(Ordering::Relaxed);
        let max = self.max_clip.load(Ordering::Relaxed);
        let target = self.target_time_us.load(Ordering::Relaxed);
        let speed = self.adaptation_speed.load(Ordering::Relaxed);

        // Update observed fill rate with exponential smoothing
        let current_rate = self.fill_rate.load(Ordering::Relaxed);
        let new_rate = current_rate * (1.0 - speed) + fill_rate_observed * speed;
        self.fill_rate.store(new_rate, Ordering::Relaxed);

        if elapsed_us == 0 || new_rate <= 0.0 {
            return min;
        }

        // Estimate time to completion at current rate
        let time_to_complete_sec = remaining as f64 / new_rate;
        let time_to_complete_us = (time_to_complete_sec * 1_000_000.0) as u64;

        // Adjust clip based on whether we're ahead or behind target
        if time_to_complete_us > target {
            // Behind target - increase clip
            let ratio = (time_to_complete_us as f64 / target as f64).min(3.0);
            ((min as f64 * ratio) as i64).min(max)
        } else if time_to_complete_us < target / 2 {
            // Ahead of target - can reduce clip to minimize market impact
            let ratio = (time_to_complete_us as f64 / (target as f64 / 2.0)).max(0.5);
            ((max as f64 * ratio) as i64).max(min)
        } else {
            // On track - maintain current strategy
            ((min + max) / 2).max(min).min(max)
        }
    }

    /// Update target execution time
    #[inline(always)]
    pub fn set_target_time(&self, target_time_ms: u64) {
        self.target_time_us.store(target_time_ms * 1000, Ordering::Relaxed);
    }

    /// Set adaptation speed
    #[inline(always)]
    pub fn set_adaptation_speed(&self, speed: f64) {
        let clamped = speed.max(0.01).min(0.5);
        self.adaptation_speed.store(clamped, Ordering::Relaxed);
    }

    /// Get current fill rate estimate
    #[inline(always)]
    pub fn get_fill_rate(&self) -> f64 {
        self.fill_rate.load(Ordering::Relaxed)
    }
}

/// Combined reserve order manager with discretion and dynamic clips
pub struct ReserveOrderManager {
    /// Active reserve orders count
    active_count: AtomicU64,
    /// Total fills processed
    total_fills: AtomicU64,
    /// Total volume filled (scaled by 1e8)
    total_volume: AtomicI64,
    /// Default discretion settings
    default_discretion_offset: AtomicI64,
    /// Default clip settings
    default_min_clip: AtomicI64,
    /// Enabled flag
    enabled: AtomicBool,
}

impl ReserveOrderManager {
    pub fn new(default_discretion: i64, min_clip: i64) -> Self {
        Self {
            active_count: AtomicU64::new(0),
            total_fills: AtomicU64::new(0),
            total_volume: AtomicI64::new(0),
            default_discretion_offset: AtomicI64::new(default_discretion),
            default_min_clip: AtomicI64::new(min_clip),
            enabled: AtomicBool::new(true),
        }
    }

    /// Create new reserve order with discretion
    #[inline(always)]
    pub fn create_reserve_order(
        &self,
        order_id: u64,
        total_qty: i64,
        display_pct: f64,
        limit_price: i64,
        is_buy: bool,
    ) -> ReserveOrder {
        let display_qty = ((total_qty as f64 * display_pct) as i64)
            .max(self.default_min_clip.load(Ordering::Relaxed))
            .min(total_qty);
        
        let order = ReserveOrder::new(order_id, total_qty, display_qty, limit_price, is_buy);
        self.active_count.fetch_add(1, Ordering::Relaxed);
        order
    }

    /// Process fill on reserve order
    #[inline(always)]
    pub fn process_fill(&self, order: &mut ReserveOrder, qty: i64, timestamp_us: u64) {
        if !order.is_active || !self.enabled.load(Ordering::Relaxed) {
            return;
        }

        let filled = order.fill(qty, timestamp_us);
        
        if filled > 0 {
            self.total_fills.fetch_add(1, Ordering::Relaxed);
            self.total_volume.fetch_add(filled, Ordering::Relaxed);
        }

        if !order.is_active {
            self.active_count.fetch_sub(1, Ordering::Relaxed);
        }
    }

    /// Get effective price with discretion applied
    #[inline(always)]
    pub fn get_discretionary_price(
        &self,
        base_price: i64,
        is_buy: bool,
        market_pressure: i64,
        custom_offset: Option<i64>,
    ) -> i64 {
        let offset = custom_offset.unwrap_or_else(|| self.default_discretion_offset.load(Ordering::Relaxed));
        
        // Simple discretion logic
        let threshold = 500_000;
        if market_pressure.abs() > threshold {
            let pressure_ratio = (market_pressure.abs() as f64 / threshold as f64).min(2.0);
            let adjusted_offset = (offset as f64 * pressure_ratio) as i64;
            
            if is_buy {
                base_price + adjusted_offset
            } else {
                base_price - adjusted_offset
            }
        } else {
            base_price
        }
    }

    /// Get statistics
    #[inline(always)]
    pub fn get_stats(&self) -> (u64, u64, i64, bool) {
        (
            self.active_count.load(Ordering::Relaxed),
            self.total_fills.load(Ordering::Relaxed),
            self.total_volume.load(Ordering::Relaxed),
            self.enabled.load(Ordering::Relaxed),
        )
    }

    /// Enable/disable manager
    #[inline(always)]
    pub fn set_enabled(&self, enabled: bool) {
        self.enabled.store(enabled, Ordering::Relaxed);
    }
}

impl Default for DiscretionaryManager {
    fn default() -> Self {
        Self::new(5, 50) // Default 5 cent base, 50 cent max
    }
}

impl Default for DynamicClipAdjuster {
    fn default() -> Self {
        Self::new(100_000, 1_000_000, 60000) // 100K min, 1M max, 60s target
    }
}

impl Default for ReserveOrderManager {
    fn default() -> Self {
        Self::new(5, 100_000)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reserve_order_fill() {
        let mut order = ReserveOrder::new(1, 1_000_000, 100_000, 50000, true);
        
        assert!(order.has_hidden());
        assert_eq!(order.current_visible(), 100_000);
        
        // Fill first clip
        order.fill(100_000, 1000);
        
        assert_eq!(order.filled_qty, 100_000);
        assert!(order.is_active);
        assert!(order.has_hidden());
    }

    #[test]
    fn test_discretionary_pricing() {
        let manager = DiscretionaryManager::new(5, 50);
        
        // Low pressure - no discretion
        let price1 = manager.calculate_discretionary_price(50000, true, 100_000);
        assert_eq!(price1, 50000);
        
        // High pressure - discretion active
        let price2 = manager.calculate_discretionary_price(50000, true, 1_000_000);
        assert!(price2 > 50000);
        assert!(manager.is_discretion_active());
    }

    #[test]
    fn test_dynamic_clip() {
        let adjuster = DynamicClipAdjuster::new(100_000, 1_000_000, 60000);
        
        // Slow fill rate - should increase clip
        let clip = adjuster.calculate_optimal_clip(500_000, 30_000_000, 1000.0);
        assert!(clip > 100_000);
    }

    #[test]
    fn test_reserve_manager() {
        let manager = ReserveOrderManager::new(5, 100_000);
        
        let mut order = manager.create_reserve_order(1, 1_000_000, 0.1, 50000, true);
        assert_eq!(order.display_qty, 100_000);
        
        manager.process_fill(&mut order, 100_000, 1000);
        
        let (_, fills, volume, _) = manager.get_stats();
        assert_eq!(fills, 1);
        assert_eq!(volume, 100_000);
    }
}
