"""
High-Performance Market Data Models in Rust.

Defines Tick data, Bars, and Order Book snapshots using bytemuck for
zero-cost serialization/deserialization and aligned memory layouts
for optimal CPU cache efficiency on AMD Ryzen AI 5.
"""

use bytemuck::{Pod, Zeroable};
use std::mem;

/// Cache line size for AMD Zen architecture (typically 64 bytes)
const CACHE_LINE_SIZE: usize = 64;

/// Nanosecond-precision timestamp
pub type TimestampNs = i64;

/// Price represented as fixed-point for precision (price * 1e8)
pub type FixedPrice = i64;

/// Quantity represented as fixed-point (quantity * 1e8)
pub type FixedQuantity = i64;

/// Trade tick with zero-copy serialization support
#[repr(C, align(64))]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct TradeTick {
    /// Symbol identifier (first 12 bytes of symbol name hash)
    pub symbol_hash: u64,
    
    /// Trade price (fixed-point: actual_price * 1e8)
    pub price: FixedPrice,
    
    /// Trade quantity (fixed-point: actual_qty * 1e8)
    pub quantity: FixedQuantity,
    
    /// Timestamp in nanoseconds since Unix epoch
    pub timestamp_ns: TimestampNs,
    
    /// Trade ID from exchange
    pub trade_id: i64,
    
    /// Flags: bit 0 = is_buyer_maker
    pub flags: u32,
    
    /// Padding to maintain cache line alignment
    _padding: [u8; 12],
}

impl TradeTick {
    /// Create a new trade tick
    pub fn new(
        symbol_hash: u64,
        price: f64,
        quantity: f64,
        timestamp_ns: i64,
        trade_id: i64,
        is_buyer_maker: bool,
    ) -> Self {
        Self {
            symbol_hash,
            price: (price * 1e8) as i64,
            quantity: (quantity * 1e8) as i64,
            timestamp_ns,
            trade_id,
            flags: if is_buyer_maker { 1 } else { 0 },
            _padding: [0; 12],
        }
    }

    /// Get actual price as f64
    #[inline]
    pub fn price_f64(&self) -> f64 {
        self.price as f64 / 1e8
    }

    /// Get actual quantity as f64
    #[inline]
    pub fn quantity_f64(&self) -> f64 {
        self.quantity as f64 / 1e8
    }

    /// Check if buyer was maker
    #[inline]
    pub fn is_buyer_maker(&self) -> bool {
        self.flags & 1 != 0
    }

    /// Size in bytes
    #[inline]
    pub const fn size_bytes() -> usize {
        mem::size_of::<Self>()
    }

    /// Verify cache line alignment
    #[inline]
    pub const fn is_cache_aligned() -> bool {
        mem::align_of::<Self>() % CACHE_LINE_SIZE == 0
    }
}

/// Order book level (price + quantity)
#[repr(C, align(16))]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct BookLevel {
    pub price: FixedPrice,
    pub quantity: FixedQuantity,
}

impl BookLevel {
    pub fn new(price: f64, quantity: f64) -> Self {
        Self {
            price: (price * 1e8) as i64,
            quantity: (quantity * 1e8) as i64,
        }
    }

    pub fn price_f64(&self) -> f64 {
        self.price as f64 / 1e8
    }

    pub fn quantity_f64(&self) -> f64 {
        self.quantity as f64 / 1e8
    }
}

/// Maximum order book depth supported
pub const MAX_BOOK_DEPTH: usize = 50;

/// Order book snapshot with fixed-depth levels
#[repr(C, align(64))]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct OrderBookSnapshot {
    /// Symbol identifier hash
    pub symbol_hash: u64,
    
    /// Last update ID from exchange
    pub last_update_id: u64,
    
    /// Timestamp in nanoseconds
    pub timestamp_ns: TimestampNs,
    
    /// Number of valid bid levels
    pub bid_count: u32,
    
    /// Number of valid ask levels
    pub ask_count: u32,
    
    /// Bid levels (price, quantity)
    pub bids: [BookLevel; MAX_BOOK_DEPTH],
    
    /// Ask levels (price, quantity)
    pub asks: [BookLevel; MAX_BOOK_DEPTH],
    
    /// Checksum for validation
    pub checksum: u32,
    
    /// Padding for alignment
    _padding: [u8; 20],
}

impl OrderBookSnapshot {
    /// Create empty order book snapshot
    pub fn empty(symbol_hash: u64) -> Self {
        Self {
            symbol_hash,
            last_update_id: 0,
            timestamp_ns: 0,
            bid_count: 0,
            ask_count: 0,
            bids: [BookLevel { price: 0, quantity: 0 }; MAX_BOOK_DEPTH],
            asks: [BookLevel { price: 0, quantity: 0 }; MAX_BOOK_DEPTH],
            checksum: 0,
            _padding: [0; 20],
        }
    }

    /// Set bid level
    #[inline]
    pub fn set_bid(&mut self, index: usize, price: f64, quantity: f64) {
        if index < MAX_BOOK_DEPTH {
            self.bids[index] = BookLevel::new(price, quantity);
            if index >= self.bid_count as usize && quantity > 0.0 {
                self.bid_count = (index + 1) as u32;
            }
        }
    }

    /// Set ask level
    #[inline]
    pub fn set_ask(&mut self, index: usize, price: f64, quantity: f64) {
        if index < MAX_BOOK_DEPTH {
            self.asks[index] = BookLevel::new(price, quantity);
            if index >= self.ask_count as usize && quantity > 0.0 {
                self.ask_count = (index + 1) as u32;
            }
        }
    }

    /// Get best bid price
    #[inline]
    pub fn best_bid(&self) -> Option<f64> {
        if self.bid_count > 0 && self.bids[0].quantity > 0 {
            Some(self.bids[0].price_f64())
        } else {
            None
        }
    }

    /// Get best ask price
    #[inline]
    pub fn best_ask(&self) -> Option<f64> {
        if self.ask_count > 0 && self.asks[0].quantity > 0 {
            Some(self.asks[0].price_f64())
        } else {
            None
        }
    }

    /// Get mid price
    #[inline]
    pub fn mid_price(&self) -> Option<f64> {
        match (self.best_bid(), self.best_ask()) {
            (Some(bid), Some(ask)) => Some((bid + ask) / 2.0),
            _ => None,
        }
    }

    /// Calculate XOR checksum for validation
    pub fn calculate_checksum(&self, depth: usize) -> u32 {
        let mut checksum = 0u32;
        let actual_depth = depth.min(self.bid_count as usize).min(self.ask_count as usize);

        for i in 0..actual_depth {
            checksum ^= (self.bids[i].price as u32) ^ (self.bids[i].quantity as u32);
            checksum ^= (self.asks[i].price as u32) ^ (self.asks[i].quantity as u32);
        }

        checksum
    }

    /// Size in bytes
    #[inline]
    pub const fn size_bytes() -> usize {
        mem::size_of::<Self>()
    }
}

/// OHLCV bar data (1 second to 1 day resolution)
#[repr(C, align(32))]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct Bar {
    /// Symbol identifier hash
    pub symbol_hash: u64,
    
    /// Bar start timestamp (nanoseconds)
    pub timestamp_ns: TimestampNs,
    
    /// Open price (fixed-point)
    pub open: FixedPrice,
    
    /// High price (fixed-point)
    pub high: FixedPrice,
    
    /// Low price (fixed-point)
    pub low: FixedPrice,
    
    /// Close price (fixed-point)
    pub close: FixedPrice,
    
    /// Volume (fixed-point)
    pub volume: FixedQuantity,
    
    /// Number of trades in bar
    pub trade_count: u32,
    
    /// Bar duration in nanoseconds
    pub duration_ns: i64,
    
    /// Flags: bit 0 = is_complete
    pub flags: u32,
    
    _padding: [u8; 12],
}

impl Bar {
    pub fn new(symbol_hash: u64, timestamp_ns: i64, duration_ns: i64) -> Self {
        Self {
            symbol_hash,
            timestamp_ns,
            open: 0,
            high: 0,
            low: i64::MAX,
            close: 0,
            volume: 0,
            trade_count: 0,
            duration_ns,
            flags: 0,
            _padding: [0; 12],
        }
    }

    /// Update bar with trade
    #[inline]
    pub fn update(&mut self, price: f64, quantity: f64) {
        let price_fixed = (price * 1e8) as i64;
        let qty_fixed = (quantity * 1e8) as i64;

        if self.trade_count == 0 {
            self.open = price_fixed;
        }

        if price_fixed > self.high {
            self.high = price_fixed;
        }
        if price_fixed < self.low {
            self.low = price_fixed;
        }

        self.close = price_fixed;
        self.volume += qty_fixed;
        self.trade_count += 1;
    }

    /// Mark bar as complete
    #[inline]
    pub fn complete(&mut self) {
        self.flags |= 1;
    }

    /// Check if bar is complete
    #[inline]
    pub fn is_complete(&self) -> bool {
        self.flags & 1 != 0
    }

    /// Get open as f64
    #[inline]
    pub fn open_f64(&self) -> f64 {
        self.open as f64 / 1e8
    }

    /// Get high as f64
    #[inline]
    pub fn high_f64(&self) -> f64 {
        self.high as f64 / 1e8
    }

    /// Get low as f64
    #[inline]
    pub fn low_f64(&self) -> f64 {
        self.low as f64 / 1e8
    }

    /// Get close as f64
    #[inline]
    pub fn close_f64(&self) -> f64 {
        self.close as f64 / 1e8
    }

    /// Get volume as f64
    #[inline]
    pub fn volume_f64(&self) -> f64 {
        self.volume as f64 / 1e8
    }
}

/// Common event types that can be sent through the event bus
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MarketDataType {
    TradeTick = 0,
    OrderBookSnapshot = 1,
    OrderBookDelta = 2,
    Bar = 3,
}

/// Unified market data event header
#[repr(C, align(16))]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct MarketDataHeader {
    pub event_type: u8,
    pub version: u8,
    pub reserved: u16,
    pub payload_size: u32,
    pub timestamp_ns: TimestampNs,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trade_tick_alignment() {
        assert!(TradeTick::is_cache_aligned());
        println!("TradeTick size: {} bytes", TradeTick::size_bytes());
    }

    #[test]
    fn test_orderbook_snapshot_size() {
        let size = OrderBookSnapshot::size_bytes();
        println!("OrderBookSnapshot size: {} bytes", size);
        // Should be reasonable for 50 levels per side
        assert!(size < 10000);
    }

    #[test]
    fn test_trade_tick_roundtrip() {
        let tick = TradeTick::new(
            0x1234567890ABCDEF,
            50000.50,
            1.25,
            1234567890000000000,
            999999,
            true,
        );

        assert_eq!(tick.price_f64(), 50000.50);
        assert_eq!(tick.quantity_f64(), 1.25);
        assert!(tick.is_buyer_maker());
    }

    #[test]
    fn test_bar_update() {
        let mut bar = Bar::new(0xABCDEF, 1000000000, 60_000_000_000);

        bar.update(50000.0, 1.0);
        bar.update(50100.0, 2.0);
        bar.update(49900.0, 1.5);
        bar.update(50050.0, 0.5);

        assert_eq!(bar.open_f64(), 50000.0);
        assert_eq!(bar.high_f64(), 50100.0);
        assert_eq!(bar.low_f64(), 49900.0);
        assert_eq!(bar.close_f64(), 50050.0);
        assert_eq!(bar.volume_f64(), 5.0);
        assert_eq!(bar.trade_count, 4);
    }

    #[test]
    fn test_bytemuck_pod() {
        // Verify Pod trait is implemented (zero-copy safe)
        let tick = TradeTick::new(0x123, 100.0, 1.0, 0, 1, false);
        let bytes = bytemuck::bytes_of(&tick);
        assert_eq!(bytes.len(), TradeTick::size_bytes());
    }
}
