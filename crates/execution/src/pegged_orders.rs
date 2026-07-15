//! Implementation of complex pegged orders (MidPrice Peg, Primary Peg)
//! Automatically reprices in microseconds as order book shifts
//! Zero heap allocations, optimized for AMD Ryzen architecture

use std::sync::atomic::{AtomicI64, AtomicU64, AtomicBool, Ordering};

/// Peg type for order pricing
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PegType {
    /// Peg to mid-price
    MidPrice,
    /// Peg to primary market (e.g., spot for futures)
    PrimaryPeg,
    /// Peg to best bid
    BestBid,
    /// Peg to best ask
    BestAsk,
    /// Fixed offset from reference
    FixedOffset,
}

/// Pegged order state
#[derive(Debug, Clone, Copy)]
pub struct PeggedOrder {
    /// Order ID
    pub order_id: u64,
    /// Peg type
    pub peg_type: PegType,
    /// Offset from reference price (scaled by 1e8, can be negative)
    pub offset: i64,
    /// Minimum price bound (scaled by 1e8)
    pub min_price: i64,
    /// Maximum price bound (scaled by 1e8)
    pub max_price: i64,
    /// Order size (scaled by 1e8)
    pub size: i64,
    /// Side: true = buy, false = sell
    pub is_buy: bool,
    /// Current calculated price
    pub current_price: i64,
    /// Last update timestamp
    pub last_update_us: u64,
    /// Is order active
    pub is_active: bool,
}

/// Lock-free pegged order manager
pub struct PeggedOrderManager {
    /// Active pegged orders count
    active_count: AtomicU64,
    /// Total repricing operations
    reprice_count: AtomicU64,
    /// Orders cancelled due to bounds
    bounds_cancel_count: AtomicU64,
    /// Enabled flag
    enabled: AtomicBool,
    /// Tick size for rounding (scaled by 1e8)
    tick_size: AtomicI64,
}

impl PeggedOrderManager {
    pub fn new(tick_size: i64) -> Self {
        Self {
            active_count: AtomicU64::new(0),
            reprice_count: AtomicU64::new(0),
            bounds_cancel_count: AtomicU64::new(0),
            enabled: AtomicBool::new(true),
            tick_size: AtomicI64::new(tick_size),
        }
    }

    /// Create a new pegged order
    #[inline(always)]
    pub fn create_pegged_order(
        &self,
        order_id: u64,
        peg_type: PegType,
        offset: i64,
        min_price: i64,
        max_price: i64,
        size: i64,
        is_buy: bool,
    ) -> PeggedOrder {
        let order = PeggedOrder {
            order_id,
            peg_type,
            offset,
            min_price,
            max_price,
            size,
            is_buy,
            current_price: 0,
            last_update_us: 0,
            is_active: true,
        };

        self.active_count.fetch_add(1, Ordering::Relaxed);
        order
    }

    /// Calculate pegged price based on reference and order book state
    #[inline(always)]
    pub fn calculate_pegged_price(
        &self,
        order: &mut PeggedOrder,
        mid_price: i64,
        best_bid: i64,
        best_ask: i64,
        primary_price: i64,
        timestamp_us: u64,
    ) -> i64 {
        if !order.is_active || !self.enabled.load(Ordering::Relaxed) {
            return 0;
        }

        // Get reference price based on peg type
        let reference = match order.peg_type {
            PegType::MidPrice => mid_price,
            PegType::PrimaryPeg => primary_price,
            PegType::BestBid => best_bid,
            PegType::BestAsk => best_ask,
            PegType::FixedOffset => mid_price, // Offset applied separately
        };

        // Apply offset
        let mut price = reference + order.offset;

        // Round to tick size
        let tick = self.tick_size.load(Ordering::Relaxed);
        if tick > 0 {
            price = (price / tick) * tick;
        }

        // Apply bounds
        if order.min_price > 0 && price < order.min_price {
            price = order.min_price;
        }
        if order.max_price > 0 && price > order.max_price {
            price = order.max_price;
        }

        // For buy orders, ensure price doesn't exceed best ask (avoid crossing)
        if order.is_buy && best_ask > 0 && price >= best_ask {
            price = best_ask - tick;
        }

        // For sell orders, ensure price doesn't go below best bid
        if !order.is_buy && best_bid > 0 && price <= best_bid {
            price = best_bid + tick;
        }

        // Check if price is outside bounds - cancel if so
        if (order.min_price > 0 && price < order.min_price) 
            || (order.max_price > 0 && price > order.max_price) {
            order.is_active = false;
            self.bounds_cancel_count.fetch_add(1, Ordering::Relaxed);
            return 0;
        }

        order.current_price = price;
        order.last_update_us = timestamp_us;
        self.reprice_count.fetch_add(1, Ordering::Relaxed);

        price
    }

    /// Update pegged order when market moves
    #[inline(always)]
    pub fn update_on_market_move(
        &self,
        order: &mut PeggedOrder,
        mid_price: i64,
        best_bid: i64,
        best_ask: i64,
        primary_price: i64,
        timestamp_us: u64,
    ) -> Option<i64> {
        if !order.is_active {
            return None;
        }

        let old_price = order.current_price;
        let new_price = self.calculate_pegged_price(
            order,
            mid_price,
            best_bid,
            best_ask,
            primary_price,
            timestamp_us,
        );

        if new_price == 0 {
            return None; // Order cancelled
        }

        // Return price change if significant
        if (new_price as i64 - old_price as i64).abs() > self.tick_size.load(Ordering::Relaxed) {
            Some(new_price)
        } else {
            None
        }
    }

    /// Cancel a pegged order
    #[inline(always)]
    pub fn cancel_order(&self, order: &mut PeggedOrder) {
        if order.is_active {
            order.is_active = false;
            self.active_count.fetch_sub(1, Ordering::Relaxed);
        }
    }

    /// Set tick size dynamically
    #[inline(always)]
    pub fn set_tick_size(&self, tick_size: i64) {
        self.tick_size.store(tick_size, Ordering::Relaxed);
    }

    /// Enable/disable pegged order processing
    #[inline(always)]
    pub fn set_enabled(&self, enabled: bool) {
        self.enabled.store(enabled, Ordering::Relaxed);
    }

    /// Get statistics
    #[inline(always)]
    pub fn get_stats(&self) -> (u64, u64, u64, bool) {
        (
            self.active_count.load(Ordering::Relaxed),
            self.reprice_count.load(Ordering::Relaxed),
            self.bounds_cancel_count.load(Ordering::Relaxed),
            self.enabled.load(Ordering::Relaxed),
        )
    }
}

/// Discretionary order logic with hidden reserve sizing
pub struct DiscretionaryOrder {
    /// Display quantity (visible to market)
    display_qty: AtomicI64,
    /// Reserve quantity (hidden)
    reserve_qty: AtomicI64,
    /// Total remaining quantity
    remaining_qty: AtomicI64,
    /// Discretionary offset (additional price improvement)
    discretionary_offset: AtomicI64,
    /// Trigger threshold for discretion
    trigger_threshold: AtomicI64,
    /// Is discretion currently active
    discretion_active: AtomicBool,
}

impl DiscretionaryOrder {
    pub fn new(display_qty: i64, reserve_qty: i64, discretionary_offset: i64) -> Self {
        Self {
            display_qty: AtomicI64::new(display_qty),
            reserve_qty: AtomicI64::new(reserve_qty),
            remaining_qty: AtomicI64::new(display_qty + reserve_qty),
            discretionary_offset: AtomicI64::new(discretionary_offset),
            trigger_threshold: AtomicI64::new(100_000), // Default threshold
            discretion_active: AtomicBool::new(false),
        }
    }

    /// Calculate effective price with discretion
    #[inline(always)]
    pub fn get_effective_price(&self, base_price: i64, is_buy: bool, market_pressure: i64) -> i64 {
        let offset = self.discretionary_offset.load(Ordering::Relaxed);
        let threshold = self.trigger_threshold.load(Ordering::Relaxed);

        // Activate discretion if market pressure exceeds threshold
        if market_pressure.abs() > threshold {
            self.discretion_active.store(true, Ordering::Relaxed);
            
            // Apply additional price improvement
            if is_buy {
                base_price + offset // Pay more to get filled
            } else {
                base_price - offset // Accept less to get filled
            }
        } else {
            self.discretion_active.store(false, Ordering::Relaxed);
            base_price
        }
    }

    /// Fill portion of the order
    #[inline(always)]
    pub fn fill(&self, qty: i64) -> i64 {
        let remaining = self.remaining_qty.load(Ordering::Relaxed);
        let filled = qty.min(remaining);
        
        self.remaining_qty.fetch_sub(filled, Ordering::Relaxed);
        
        // Replenish display qty from reserve if needed
        let display = self.display_qty.load(Ordering::Relaxed);
        let reserve = self.reserve_qty.load(Ordering::Relaxed);
        
        if display > remaining - filled {
            // Need to replenish from reserve
            let needed = display - (remaining - filled);
            let from_reserve = needed.min(reserve);
            self.reserve_qty.fetch_sub(from_reserve, Ordering::Relaxed);
        }
        
        filled
    }

    /// Get remaining quantity
    #[inline(always)]
    pub fn remaining(&self) -> i64 {
        self.remaining_qty.load(Ordering::Relaxed)
    }

    /// Get visible quantity
    #[inline(always)]
    pub fn visible_qty(&self) -> i64 {
        let remaining = self.remaining_qty.load(Ordering::Relaxed);
        let display = self.display_qty.load(Ordering::Relaxed);
        remaining.min(display)
    }

    /// Check if order is fully filled
    #[inline(always)]
    pub fn is_filled(&self) -> bool {
        self.remaining_qty.load(Ordering::Relaxed) <= 0
    }

    /// Update discretionary offset based on volatility
    #[inline(always)]
    pub fn update_discretion(&self, volatility: f64, base_offset: i64) {
        // Higher volatility = larger discretion for faster fills
        let vol_factor = 1.0 + volatility * 0.5;
        let new_offset = (base_offset as f64 * vol_factor) as i64;
        self.discretionary_offset.store(new_offset, Ordering::Relaxed);
    }

    /// Set trigger threshold
    #[inline(always)]
    pub fn set_trigger_threshold(&self, threshold: i64) {
        self.trigger_threshold.store(threshold, Ordering::Relaxed);
    }
}

/// Reserve order with dynamic clip adjustment
pub struct ReserveOrderManager {
    /// Default display clip size
    default_clip: AtomicI64,
    /// Min display clip
    min_clip: AtomicI64,
    /// Max display clip
    max_clip: AtomicI64,
    /// Volatility adjustment factor (scaled by 1e6)
    vol_adjustment: AtomicU32,
}

impl ReserveOrderManager {
    pub fn new(default_clip: i64, min_clip: i64, max_clip: i64) -> Self {
        Self {
            default_clip: AtomicI64::new(default_clip),
            min_clip: AtomicI64::new(min_clip),
            max_clip: AtomicI64::new(max_clip),
            vol_adjustment: AtomicU32::new(1_000_000), // 1.0 = no adjustment
        }
    }

    /// Calculate optimal display clip based on market conditions
    #[inline(always)]
    pub fn calculate_clip(&self, volatility: f64, spread_bps: f64, queue_size: i64) -> i64 {
        let base = self.default_clip.load(Ordering::Relaxed);
        let min = self.min_clip.load(Ordering::Relaxed);
        let max = self.max_clip.load(Ordering::Relaxed);

        // Adjust based on volatility
        let vol_factor = 1.0 + volatility * 0.3;
        
        // Adjust based on spread (wider spread = smaller clip)
        let spread_factor = 1.0 / (1.0 + spread_bps / 100.0);
        
        // Adjust based on queue position (larger queue = larger clip)
        let queue_factor = 1.0 + (queue_size as f64 / 1_000_000.0).min(2.0) * 0.2;

        let adjusted = base as f64 * vol_factor * spread_factor * queue_factor;
        let clipped = adjusted as i64;

        clipped.max(min).min(max)
    }

    /// Update volatility adjustment factor
    #[inline(always)]
    pub fn set_vol_adjustment(&self, factor: f64) {
        let scaled = ((factor * 1_000_000.0) as u32).max(100_000).min(2_000_000);
        self.vol_adjustment.store(scaled, Ordering::Relaxed);
    }

    /// Get current adjustment factor
    #[inline(always)]
    pub fn get_vol_adjustment(&self) -> f64 {
        self.vol_adjustment.load(Ordering::Relaxed) as f64 / 1_000_000.0
    }
}

impl Default for PeggedOrderManager {
    fn default() -> Self {
        Self::new(1) // Default tick size of 1
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_midprice_peg() {
        let manager = PeggedOrderManager::new(1);
        let mut order = manager.create_pegged_order(
            1,
            PegType::MidPrice,
            -5, // 5 cents below mid
            0,
            0,
            1000000,
            true,
        );

        let price = manager.calculate_pegged_price(
            &mut order,
            50000, // mid
            49990, // best_bid
            50010, // best_ask
            0,
            1000000,
        );

        assert_eq!(price, 49995); // mid - 5
        assert!(order.is_active);
    }

    #[test]
    fn test_discretionary_order() {
        let order = DiscretionaryOrder::new(100000, 900000, 2);
        
        // Low pressure - no discretion
        let price1 = order.get_effective_price(50000, true, 50000);
        assert_eq!(price1, 50000);

        // High pressure - discretion active
        let price2 = order.get_effective_price(50000, true, 200000);
        assert_eq!(price2, 50002); // base + offset
    }

    #[test]
    fn test_reserve_clip_calculation() {
        let manager = ReserveOrderManager::new(100000, 10000, 500000);
        
        let clip = manager.calculate_clip(0.02, 10.0, 5000000);
        
        assert!(clip >= 10000);
        assert!(clip <= 500000);
    }
}
