//! Lock-free, N-dimensional matrix tracking real-time mid-prices, bid/ask spreads, and depth
//! across Binance, Bybit, and OKX simultaneously. Updated in microseconds via atomic pointers.
//! Optimized for AMD Ryzen AI 5 with cache-line aligned structures.

use std::sync::atomic::{AtomicU64, AtomicI64, Ordering};
use std::ptr::NonNull;

/// Maximum number of venues supported
pub const MAX_VENUES: usize = 8;
/// Maximum number of symbols tracked
pub const MAX_SYMBOLS: usize = 100;

/// Venue identifier
#[repr(u8)]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum VenueId {
    Binance = 0,
    Bybit = 1,
    OKX = 2,
    Coinbase = 3,
    Kraken = 4,
    Huobi = 5,
    Kucoin = 6,
    Gateio = 7,
}

impl VenueId {
    pub fn from_u8(val: u8) -> Option<Self> {
        match val {
            0 => Some(VenueId::Binance),
            1 => Some(VenueId::Bybit),
            2 => Some(VenueId::OKX),
            3 => Some(VenueId::Coinbase),
            4 => Some(VenueId::Kraken),
            5 => Some(VenueId::Huobi),
            6 => Some(VenueId::Kucoin),
            7 => Some(VenueId::Gateio),
            _ => None,
        }
    }
}

/// Single venue data point - 64-byte aligned for cache efficiency
#[repr(C, align(64))]
pub struct VenueData {
    /// Mid price (fixed point: * 1e8)
    pub mid_price: AtomicU64,
    /// Best bid price
    pub best_bid: AtomicU64,
    /// Best ask price
    pub best_ask: AtomicU64,
    /// Bid depth (in base units)
    pub bid_depth: AtomicU64,
    /// Ask depth
    pub ask_depth: AtomicU64,
    /// Spread in basis points
    pub spread_bps: AtomicU64,
    /// Last update timestamp (ns)
    pub last_update_ns: AtomicU64,
    /// Sequence number for consistency checks
    pub sequence: AtomicU64,
    _padding: [u8; 16],
}

impl VenueData {
    pub fn new() -> Self {
        Self {
            mid_price: AtomicU64::new(0),
            best_bid: AtomicU64::new(0),
            best_ask: AtomicU64::new(0),
            bid_depth: AtomicU64::new(0),
            ask_depth: AtomicU64::new(0),
            spread_bps: AtomicU64::new(0),
            last_update_ns: AtomicU64::new(0),
            sequence: AtomicU64::new(0),
            _padding: [0u8; 16],
        }
    }
    
    #[inline]
    pub fn update(&self, mid: u64, bid: u64, ask: u64, bid_d: u64, ask_d: u64, seq: u64) {
        let spread = if mid > 0 {
            ((ask - bid) * 10000) / mid
        } else {
            0
        };
        
        self.mid_price.store(mid, Ordering::Relaxed);
        self.best_bid.store(bid, Ordering::Relaxed);
        self.best_ask.store(ask, Ordering::Relaxed);
        self.bid_depth.store(bid_d, Ordering::Relaxed);
        self.ask_depth.store(ask_d, Ordering::Relaxed);
        self.spread_bps.store(spread, Ordering::Relaxed);
        self.sequence.store(seq, Ordering::Relaxed);
    }
}

/// N-dimensional venue matrix [venues][symbols]
#[repr(C, align(64))]
pub struct VenueMatrix {
    /// Flat array: venue_idx * MAX_SYMBOLS + symbol_idx
    data: [VenueData; MAX_VENUES * MAX_SYMBOLS],
    /// Number of active venues
    active_venues: AtomicU64,
    /// Number of active symbols
    active_symbols: AtomicU64,
}

unsafe impl Send for VenueMatrix {}
unsafe impl Sync for VenueMatrix {}

impl VenueMatrix {
    pub fn new() -> Self {
        // Initialize array with default VenueData
        let mut data = Vec::with_capacity(MAX_VENUES * MAX_SYMBOLS);
        for _ in 0..(MAX_VENUES * MAX_SYMBOLS) {
            data.push(VenueData::new());
        }
        
        Self {
            data: data.try_into().unwrap_or_else(|_| panic!("Invalid array size")),
            active_venues: AtomicU64::new(0),
            active_symbols: AtomicU64::new(0),
        }
    }
    
    /// Get index into flat array
    #[inline]
    fn get_index(venue: usize, symbol: usize) -> usize {
        venue * MAX_SYMBOLS + symbol
    }
    
    /// Update venue data - O(1) microsecond operation
    #[inline]
    pub fn update(&self, venue: VenueId, symbol_idx: usize, mid: u64, bid: u64, ask: u64, 
                  bid_depth: u64, ask_depth: u64, sequence: u64) {
        if symbol_idx >= MAX_SYMBOLS {
            return;
        }
        
        let idx = Self::get_index(venue as usize, symbol_idx);
        self.data[idx].update(mid, bid, ask, bid_depth, ask_depth, sequence);
    }
    
    /// Get venue data reference
    #[inline]
    pub fn get(&self, venue: VenueId, symbol_idx: usize) -> Option<&VenueData> {
        if symbol_idx >= MAX_SYMBOLS {
            return None;
        }
        
        let idx = Self::get_index(venue as usize, symbol_idx);
        Some(&self.data[idx])
    }
    
    /// Find best bid across all venues for a symbol
    #[inline]
    pub fn get_best_bid(&self, symbol_idx: usize) -> Option<(VenueId, u64)> {
        if symbol_idx >= MAX_SYMBOLS {
            return None;
        }
        
        let mut best_bid = 0u64;
        let mut best_venue = VenueId::Binance;
        
        for venue_id in 0..MAX_VENUES {
            let idx = Self::get_index(venue_id, symbol_idx);
            let bid = self.data[idx].best_bid.load(Ordering::Relaxed);
            if bid > best_bid {
                best_bid = bid;
                best_venue = VenueId::from_u8(venue_id as u8).unwrap_or(VenueId::Binance);
            }
        }
        
        if best_bid > 0 {
            Some((best_venue, best_bid))
        } else {
            None
        }
    }
    
    /// Find best ask across all venues for a symbol
    #[inline]
    pub fn get_best_ask(&self, symbol_idx: usize) -> Option<(VenueId, u64)> {
        if symbol_idx >= MAX_SYMBOLS {
            return None;
        }
        
        let mut best_ask = u64::MAX;
        let mut best_venue = VenueId::Binance;
        
        for venue_id in 0..MAX_VENUES {
            let idx = Self::get_index(venue_id, symbol_idx);
            let ask = self.data[idx].best_ask.load(Ordering::Relaxed);
            if ask > 0 && ask < best_ask {
                best_ask = ask;
                best_venue = VenueId::from_u8(venue_id as u8).unwrap_or(VenueId::Binance);
            }
        }
        
        if best_ask < u64::MAX {
            Some((best_venue, best_ask))
        } else {
            None
        }
    }
    
    /// Calculate cross-venue spread for arbitrage detection
    #[inline]
    pub fn get_cross_venue_spread(&self, symbol_idx: usize) -> Option<u64> {
        let best_bid = self.get_best_bid(symbol_idx)?;
        let best_ask = self.get_best_ask(symbol_idx)?;
        
        if best_bid.1 > best_ask.1 {
            // Arbitrage opportunity!
            Some(best_bid.1 - best_ask.1)
        } else {
            None
        }
    }
    
    /// Activate a venue
    #[inline]
    pub fn activate_venue(&self, venue: VenueId) {
        let current = self.active_venues.load(Ordering::Relaxed);
        if current & (1 << venue as u8) == 0 {
            self.active_venues.fetch_or(1 << venue as u8, Ordering::Relaxed);
        }
    }
    
    /// Check if venue is active
    #[inline]
    pub fn is_venue_active(&self, venue: VenueId) -> bool {
        let mask = 1 << venue as u8;
        self.active_venues.load(Ordering::Relaxed) & mask != 0
    }
}

impl Default for VenueMatrix {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_venue_matrix_basic() {
        let matrix = VenueMatrix::new();
        
        // Update Binance data for symbol 0
        matrix.update(VenueId::Binance, 0, 10000, 9999, 10001, 1000, 1000, 1);
        
        let data = matrix.get(VenueId::Binance, 0).unwrap();
        assert_eq!(data.mid_price.load(Ordering::Relaxed), 10000);
    }
    
    #[test]
    fn test_cross_venue_arb() {
        let matrix = VenueMatrix::new();
        
        // Binance: bid 10000, ask 10002
        matrix.update(VenueId::Binance, 0, 10001, 10000, 10002, 1000, 1000, 1);
        
        // Bybit: bid 10003, ask 10005 (arb opportunity!)
        matrix.update(VenueId::Bybit, 0, 10004, 10003, 10005, 1000, 1000, 2);
        
        let spread = matrix.get_cross_venue_spread(0);
        assert!(spread.is_some());
        assert_eq!(spread.unwrap(), 1); // 10003 - 10002
    }
}
