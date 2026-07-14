"""
Order Flow Analysis Metrics in Rust.

Defines structures for Order Flow analysis including CVD (Cumulative Volume Delta),
Delta, Footprint data, and Liquidity Sweeps. Mapped to Nautilus custom data types
to support "Order Flow" and "Smart Money Concepts" strategies.
"""

use bytemuck::{Pod, Zeroable};
use std::mem;

use super::market_data::{FixedPrice, FixedQuantity, TimestampNs};

/// Cache-aligned order flow metrics block
#[repr(C, align(64))]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct OrderFlowMetrics {
    /// Symbol identifier hash
    pub symbol_hash: u64,
    
    /// Timestamp in nanoseconds
    pub timestamp_ns: TimestampNs,
    
    /// Aggressive buy volume (fixed-point)
    pub aggressive_buy_volume: FixedQuantity,
    
    /// Aggressive sell volume (fixed-point)
    pub aggressive_sell_volume: FixedQuantity,
    
    /// Net delta (buy - sell, fixed-point)
    pub delta: FixedQuantity,
    
    /// Cumulative delta since session start (fixed-point)
    pub cvd: FixedQuantity,
    
    /// High price of the period (fixed-point)
    pub high: FixedPrice,
    
    /// Low price of the period (fixed-point)
    pub low: FixedPrice,
    
    /// VWAP (Volume Weighted Average Price, fixed-point)
    pub vwap: FixedPrice,
    
    /// Total volume traded (fixed-point)
    pub total_volume: FixedQuantity,
    
    /// Number of trades
    pub trade_count: u32,
    
    /// Large trade count (> threshold)
    pub large_trade_count: u32,
    
    /// Flags: bit 0 = imbalance detected, bit 1 = sweep detected
    pub flags: u32,
    
    _padding: [u8; 16],
}

impl OrderFlowMetrics {
    /// Create new empty metrics
    pub fn new(symbol_hash: u64, timestamp_ns: i64) -> Self {
        Self {
            symbol_hash,
            timestamp_ns,
            aggressive_buy_volume: 0,
            aggressive_sell_volume: 0,
            delta: 0,
            cvd: 0,
            high: 0,
            low: i64::MAX,
            vwap: 0,
            total_volume: 0,
            trade_count: 0,
            large_trade_count: 0,
            flags: 0,
            _padding: [0; 16],
        }
    }

    /// Update metrics with a trade
    #[inline]
    pub fn update(&mut self, price: f64, quantity: f64, is_buyer_maker: bool) {
        let price_fixed = (price * 1e8) as i64;
        let qty_fixed = (quantity * 1e8) as i64;

        // Track high/low
        if self.high == 0 || price_fixed > self.high {
            self.high = price_fixed;
        }
        if self.low == i64::MAX || price_fixed < self.low {
            self.low = price_fixed;
        }

        // Track aggressive buying/selling
        if is_buyer_maker {
            // Seller was aggressive (hit bid)
            self.aggressive_sell_volume += qty_fixed;
        } else {
            // Buyer was aggressive (lifted ask)
            self.aggressive_buy_volume += qty_fixed;
        }

        // Update delta
        self.delta = self.aggressive_buy_volume - self.aggressive_sell_volume;

        // Update volume totals
        self.total_volume += qty_fixed;

        // Update VWAP numerator (price * quantity sum)
        // Stored as accumulated value, actual VWAP calculated on demand
        self.vwap = self.vwap.wrapping_add(price_fixed.wrapping_mul(qty_fixed));

        self.trade_count += 1;

        // Check for large trades (> 1 BTC equivalent for crypto)
        if quantity > 1.0 {
            self.large_trade_count += 1;
        }
    }

    /// Update CVD (cumulative volume delta)
    #[inline]
    pub fn update_cvd(&mut self, delta_change: f64) {
        self.cvd += (delta_change * 1e8) as i64;
    }

    /// Get actual VWAP (call after all updates)
    #[inline]
    pub fn get_vwap(&self) -> Option<f64> {
        if self.total_volume > 0 {
            Some(self.vwap as f64 / self.total_volume as f64 / 1e8)
        } else {
            None
        }
    }

    /// Get delta as f64
    #[inline]
    pub fn delta_f64(&self) -> f64 {
        self.delta as f64 / 1e8
    }

    /// Get CVD as f64
    #[inline]
    pub fn cvd_f64(&self) -> f64 {
        self.cvd as f64 / 1e8
    }

    /// Check for buy/sell imbalance (> 70% one side)
    #[inline]
    pub fn check_imbalance(&mut self) -> bool {
        if self.total_volume == 0 {
            return false;
        }

        let buy_ratio = self.aggressive_buy_volume as f64 / self.total_volume as f64;
        
        if buy_ratio > 0.7 || buy_ratio < 0.3 {
            self.flags |= 1; // Set imbalance flag
            true
        } else {
            false
        }
    }

    /// Check if imbalance flag is set
    #[inline]
    pub fn has_imbalance(&self) -> bool {
        self.flags & 1 != 0
    }

    /// Mark sweep detected
    #[inline]
    pub fn mark_sweep(&mut self) {
        self.flags |= 2;
    }

    /// Check if sweep flag is set
    #[inline]
    pub fn has_sweep(&self) -> bool {
        self.flags & 2 != 0
    }
}

/// Footprint chart data for a single price level
#[repr(C, align(32))]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct FootprintLevel {
    /// Price (fixed-point)
    pub price: FixedPrice,
    
    /// Buy volume at this price (fixed-point)
    pub buy_volume: FixedQuantity,
    
    /// Sell volume at this price (fixed-point)
    pub sell_volume: FixedQuantity,
    
    /// Trade count at this price
    pub trade_count: u32,
    
    /// Imbalance ratio (buy/sell, scaled by 1000)
    pub imbalance_ratio: i32,
    
    _padding: [u8; 12],
}

impl FootprintLevel {
    pub fn new(price: f64) -> Self {
        Self {
            price: (price * 1e8) as i64,
            buy_volume: 0,
            sell_volume: 0,
            trade_count: 0,
            imbalance_ratio: 0,
            _padding: [0; 12],
        }
    }

    pub fn add_trade(&mut self, volume: f64, is_buyer_maker: bool) {
        let qty_fixed = (volume * 1e8) as i64;
        
        if is_buyer_maker {
            self.sell_volume += qty_fixed;
        } else {
            self.buy_volume += qty_fixed;
        }
        
        self.trade_count += 1;
        
        // Calculate imbalance ratio
        let total = self.buy_volume + self.sell_volume;
        if total > 0 && self.sell_volume > 0 {
            self.imbalance_ratio = ((self.buy_volume as f64 / self.sell_volume as f64) * 1000.0) as i32;
        }
    }

    pub fn price_f64(&self) -> f64 {
        self.price as f64 / 1e8
    }

    pub fn buy_volume_f64(&self) -> f64 {
        self.buy_volume as f64 / 1e8
    }

    pub fn sell_volume_f64(&self) -> f64 {
        self.sell_volume as f64 / 1e8
    }
}

/// Maximum footprint levels per bar
pub const MAX_FOOTPRINT_LEVELS: usize = 100;

/// Complete footprint bar data
#[repr(C, align(64))]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct FootprintBar {
    /// Symbol identifier hash
    pub symbol_hash: u64,
    
    /// Bar timestamp
    pub timestamp_ns: TimestampNs,
    
    /// Bar duration in nanoseconds
    pub duration_ns: i64,
    
    /// Number of valid footprint levels
    pub level_count: u32,
    
    /// POC (Point of Control) price index
    pub poc_index: u32,
    
    /// Value area high price index
    pub va_high_index: u32,
    
    /// Value area low price index
    pub va_low_index: u32,
    
    /// Footprint levels
    pub levels: [FootprintLevel; MAX_FOOTPRINT_LEVELS],
    
    _padding: [u8; 16],
}

impl FootprintBar {
    pub fn empty(symbol_hash: u64, timestamp_ns: i64, duration_ns: i64) -> Self {
        Self {
            symbol_hash,
            timestamp_ns,
            duration_ns,
            level_count: 0,
            poc_index: 0,
            va_high_index: 0,
            va_low_index: 0,
            levels: [FootprintLevel {
                price: 0,
                buy_volume: 0,
                sell_volume: 0,
                trade_count: 0,
                imbalance_ratio: 0,
                _padding: [0; 12],
            }; MAX_FOOTPRINT_LEVELS],
            _padding: [0; 16],
        }
    }

    pub fn add_level(&mut self, price: f64) -> Option<usize> {
        if self.level_count < MAX_FOOTPRINT_LEVELS as u32 {
            let idx = self.level_count as usize;
            self.levels[idx] = FootprintLevel::new(price);
            self.level_count += 1;
            Some(idx)
        } else {
            None
        }
    }

    pub fn get_or_find_level(&mut self, price: f64) -> Option<usize> {
        let price_fixed = (price * 1e8) as i64;
        
        // Search existing levels
        for i in 0..self.level_count as usize {
            if self.levels[i].price == price_fixed {
                return Some(i);
            }
        }
        
        // Add new level
        self.add_level(price)
    }
}

/// Liquidity sweep detection result
#[repr(C, align(32))]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct LiquiditySweep {
    /// Symbol identifier hash
    pub symbol_hash: u64,
    
    /// Sweep detection timestamp
    pub timestamp_ns: TimestampNs,
    
    /// Swept price (fixed-point)
    pub swept_price: FixedPrice,
    
    /// Price before sweep (fixed-point)
    pub pre_sweep_price: FixedPrice,
    
    /// Price after sweep (fixed-point)
    pub post_sweep_price: FixedPrice,
    
    /// Volume involved in sweep (fixed-point)
    pub sweep_volume: FixedQuantity,
    
    /// Sweep direction: 1 = upside, -1 = downside
    pub direction: i32,
    
    /// Sweep magnitude in ticks
    pub magnitude_ticks: u32,
    
    /// Was sweep followed by reversal?
    pub reversed: bool,
    
    _padding: [u8; 19],
}

impl LiquiditySweep {
    pub fn new(
        symbol_hash: u64,
        timestamp_ns: i64,
        swept_price: f64,
        pre_price: f64,
        post_price: f64,
        volume: f64,
        direction: i32,
    ) -> Self {
        Self {
            symbol_hash,
            timestamp_ns,
            swept_price: (swept_price * 1e8) as i64,
            pre_sweep_price: (pre_price * 1e8) as i64,
            post_sweep_price: (post_price * 1e8) as i64,
            sweep_volume: (volume * 1e8) as i64,
            direction,
            magnitude_ticks: 0,
            reversed: false,
            _padding: [0; 19],
        }
    }

    pub fn swept_price_f64(&self) -> f64 {
        self.swept_price as f64 / 1e8
    }

    pub fn is_upside_sweep(&self) -> bool {
        self.direction > 0
    }

    pub fn is_downside_sweep(&self) -> bool {
        self.direction < 0
    }
}

/// Smart Money Concepts (SMC) structure markers
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SMCType {
    OrderBlock = 0,
    FairValueGap = 1,
    BreakOfStructure = 2,
    ChangeOfCharacter = 3,
    LiquidityPool = 4,
    EqualHighLow = 5,
}

/// Detected SMC structure
#[repr(C, align(32))]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct SMCStructure {
    /// Structure type
    pub structure_type: u8,
    
    /// Bullish (1) or Bearish (0)
    pub is_bullish: u8,
    
    /// Reserved padding
    pub reserved: u16,
    
    /// Symbol identifier hash
    pub symbol_hash: u64,
    
    /// Structure formation timestamp
    pub timestamp_ns: TimestampNs,
    
    /// Structure high price (fixed-point)
    pub high: FixedPrice,
    
    /// Structure low price (fixed-point)
    pub low: FixedPrice,
    
    /// Mitigation price (where structure is invalidated, fixed-point)
    pub mitigation_price: FixedPrice,
    
    /// Strength score (0-100)
    pub strength_score: u32,
    
    _padding: [u8; 12],
}

impl SMCStructure {
    pub fn new(
        structure_type: SMCType,
        is_bullish: bool,
        symbol_hash: u64,
        timestamp_ns: i64,
        high: f64,
        low: f64,
    ) -> Self {
        Self {
            structure_type: structure_type as u8,
            is_bullish: if is_bullish { 1 } else { 0 },
            reserved: 0,
            symbol_hash,
            timestamp_ns,
            high: (high * 1e8) as i64,
            low: (low * 1e8) as i64,
            mitigation_price: 0,
            strength_score: 50,
            _padding: [0; 12],
        }
    }

    pub fn is_bullish(&self) -> bool {
        self.is_bullish != 0
    }

    pub fn high_f64(&self) -> f64 {
        self.high as f64 / 1e8
    }

    pub fn low_f64(&self) -> f64 {
        self.low as f64 / 1e8
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_order_flow_metrics() {
        let mut metrics = OrderFlowMetrics::new(0x123456, 1000000000);
        
        // Simulate trades
        metrics.update(50000.0, 1.0, false); // Aggressive buy
        metrics.update(50010.0, 2.0, false); // Aggressive buy
        metrics.update(50005.0, 1.5, true);  // Aggressive sell
        
        assert!(metrics.delta_f64() > 0.0); // More buying
        assert_eq!(metrics.trade_count, 3);
        assert_eq!(metrics.large_trade_count, 3); // All > 1.0
    }

    #[test]
    fn test_footprint_bar() {
        let mut footprint = FootprintBar::empty(0xABCDEF, 1000000000, 60_000_000_000);
        
        // Add levels
        footprint.add_level(50000.0);
        footprint.add_level(50001.0);
        
        assert_eq!(footprint.level_count, 2);
    }

    #[test]
    fn test_liquidity_sweep() {
        let sweep = LiquiditySweep::new(
            0x789,
            1000000000,
            50100.0,
            50000.0,
            50050.0,
            10.0,
            1, // Upside
        );
        
        assert!(sweep.is_upside_sweep());
        assert!(!sweep.is_downside_sweep());
        assert_eq!(sweep.swept_price_f64(), 50100.0);
    }

    #[test]
    fn test_smc_structure() {
        let structure = SMCStructure::new(
            SMCType::OrderBlock,
            true,
            0x456,
            1000000000,
            50100.0,
            50000.0,
        );
        
        assert!(structure.is_bullish());
        assert_eq!(structure.structure_type, SMCType::OrderBlock as u8);
    }

    #[test]
    fn test_alignment() {
        assert!(mem::align_of::<OrderFlowMetrics>() >= 64);
        assert!(mem::align_of::<FootprintBar>() >= 64);
        assert!(mem::align_of::<LiquiditySweep>() >= 32);
    }
}
