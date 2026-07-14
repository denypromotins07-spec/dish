//! Lock-free state machine for Smart Money Concepts (SMC)
//! Tracks BOS, CHoCH, Swing Highs/Lows, and Premium/Discount zones

use crossbeam::atomic::AtomicCell;
use std::sync::Arc;

/// Market structure state
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MarketStructure {
    Bullish,
    Bearish,
    Ranging,
}

/// Break of Structure event
#[derive(Debug, Clone, Copy)]
pub struct BOS {
    pub timestamp_ns: u64,
    pub price: f64,
    pub direction: MarketStructure,
    pub strength: f64, // 0.0 to 1.0
}

/// Change of Character event
#[derive(Debug, Clone, Copy)]
pub struct CHoCH {
    pub timestamp_ns: u64,
    pub price: f64,
    pub from: MarketStructure,
    pub to: MarketStructure,
}

/// Swing point
#[derive(Debug, Clone, Copy)]
pub struct SwingPoint {
    pub timestamp_ns: u64,
    pub price: f64,
    pub is_high: bool,
    pub confirmed: bool,
}

/// Premium/Discount zone
#[repr(C, align(64))]
pub struct PDZone {
    pub premium_threshold: f64,
    pub discount_threshold: f64,
    pub equilibrium: f64,
    pub range_high: f64,
    pub range_low: f64,
}

impl PDZone {
    #[inline]
    pub fn new(range_high: f64, range_low: f64) -> Self {
        let range = range_high - range_low;
        Self {
            premium_threshold: range_low + range * 0.5,
            discount_threshold: range_low + range * 0.5,
            equilibrium: (range_high + range_low) / 2.0,
            range_high,
            range_low,
        }
    }

    #[inline]
    pub fn is_premium(&self, price: f64) -> bool {
        price > self.premium_threshold
    }

    #[inline]
    pub fn is_discount(&self, price: f64) -> bool {
        price < self.discount_threshold
    }

    #[inline]
    pub fn update_range(&mut self, high: f64, low: f64) {
        if high > self.range_high {
            self.range_high = high;
        }
        if low < self.range_low {
            self.range_low = low;
        }
        let range = self.range_high - self.range_low;
        self.premium_threshold = self.range_low + range * 0.5;
        self.discount_threshold = self.range_low + range * 0.5;
        self.equilibrium = (self.range_high + self.range_low) / 2.0;
    }
}

/// Main market structure tracker - lock-free state machine
#[repr(C, align(64))]
pub struct MarketStructureTracker {
    // Current state
    structure: AtomicCell<MarketStructure>,
    
    // Recent swing points (circular buffer)
    swing_highs: Vec<AtomicCell<Option<SwingPoint>>>,
    swing_lows: Vec<AtomicCell<Option<SwingPoint>>>,
    high_head: AtomicCell<usize>,
    low_head: AtomicCell<usize>,
    
    // Last confirmed levels
    last_swing_high: AtomicCell<f64>,
    last_swing_low: AtomicCell<f64>,
    
    // BOS tracking
    bos_count: AtomicCell<u32>,
    last_bos_price: AtomicCell<f64>,
    last_bos_time: AtomicCell<u64>,
    
    // CHoCH tracking
    choch_count: AtomicCell<u32>,
    last_choch: AtomicCell<Option<CHoCH>>,
    
    // PD Zone
    pd_zone: AtomicCell<PDZone>,
    
    // Pending swings (for confirmation)
    pending_high: AtomicCell<f64>,
    pending_low: AtomicCell<f64>,
    pending_high_time: AtomicCell<u64>,
    pending_low_time: AtomicCell<u64>,
    
    // Lookback period for swing detection
    swing_lookback: usize,
}

impl MarketStructureTracker {
    pub fn new(swing_lookback: usize, initial_high: f64, initial_low: f64) -> Self {
        let pd_zone = PDZone::new(initial_high, initial_low);
        
        Self {
            structure: AtomicCell::new(MarketStructure::Ranging),
            swing_highs: (0..swing_lookback).map(|_| AtomicCell::new(None)).collect(),
            swing_lows: (0..swing_lookback).map(|_| AtomicCell::new(None)).collect(),
            high_head: AtomicCell::new(0),
            low_head: AtomicCell::new(0),
            last_swing_high: AtomicCell::new(initial_high),
            last_swing_low: AtomicCell::new(initial_low),
            bos_count: AtomicCell::new(0),
            last_bos_price: AtomicCell::new(0.0),
            last_bos_time: AtomicCell::new(0),
            choch_count: AtomicCell::new(0),
            last_choch: AtomicCell::new(None),
            pd_zone: AtomicCell::new(pd_zone),
            pending_high: AtomicCell::new(0.0),
            pending_low: AtomicCell::new(0.0),
            pending_high_time: AtomicCell::new(0),
            pending_low_time: AtomicCell::new(0),
            swing_lookback,
        }
    }

    /// Update with new price data
    #[inline]
    pub fn update(&self, high: f64, low: f64, close: f64, timestamp_ns: u64) -> Option<BOS> {
        // Update pending swings
        self.update_pending_swings(high, low, timestamp_ns);
        
        // Check for confirmed swings
        if let Some(swing) = self.check_swing_high(high, timestamp_ns) {
            self.record_swing_high(swing);
        }
        if let Some(swing) = self.check_swing_low(low, timestamp_ns) {
            self.record_swing_low(swing);
        }
        
        // Check for BOS
        if let Some(bos) = self.check_bos(close, timestamp_ns) {
            return Some(bos);
        }
        
        None
    }

    #[inline]
    fn update_pending_swings(&self, high: f64, low: f64, timestamp_ns: u64) {
        let mut pending_high = self.pending_high.load();
        let mut pending_low = self.pending_low.load();
        
        if high > pending_high {
            self.pending_high.store(high);
            self.pending_high_time.store(timestamp_ns);
        }
        if low < pending_low || pending_low == 0.0 {
            self.pending_low.store(low);
            self.pending_low_time.store(timestamp_ns);
        }
    }

    #[inline]
    fn check_swing_high(&self, high: f64, timestamp_ns: u64) -> Option<SwingPoint> {
        let pending_high = self.pending_high.load();
        let pending_time = self.pending_high_time.load();
        
        // Simple swing detection: price moves away from pending high
        let pullback = (pending_high - high) / pending_high;
        if pullback > 0.001 && timestamp_ns > pending_time + 1_000_000_000 {
            // 0.1% pullback and 1 second elapsed
            let swing = SwingPoint {
                timestamp_ns: pending_time,
                price: pending_high,
                is_high: true,
                confirmed: true,
            };
            self.pending_high.store(high);
            self.pending_high_time.store(timestamp_ns);
            return Some(swing);
        }
        None
    }

    #[inline]
    fn check_swing_low(&self, low: f64, timestamp_ns: u64) -> Option<SwingPoint> {
        let pending_low = self.pending_low.load();
        let pending_time = self.pending_low_time.load();
        
        if pending_low == 0.0 {
            return None;
        }
        
        // Simple swing detection: price moves away from pending low
        let bounce = (low - pending_low) / pending_low;
        if bounce > 0.001 && timestamp_ns > pending_time + 1_000_000_000 {
            let swing = SwingPoint {
                timestamp_ns: pending_time,
                price: pending_low,
                is_high: false,
                confirmed: true,
            };
            self.pending_low.store(low);
            self.pending_low_time.store(timestamp_ns);
            return Some(swing);
        }
        None
    }

    #[inline]
    fn record_swing_high(&self, swing: SwingPoint) {
        let idx = self.high_head.fetch_add(1) % self.swing_lookback;
        self.swing_highs[idx].store(Some(swing));
        self.last_swing_high.store(swing.price);
        
        // Update PD zone
        let mut pd = self.pd_zone.load();
        pd.update_range(swing.price.max(self.last_swing_low.load()), self.last_swing_low.load());
        self.pd_zone.store(pd);
    }

    #[inline]
    fn record_swing_low(&self, swing: SwingPoint) {
        let idx = self.low_head.fetch_add(1) % self.swing_lookback;
        self.swing_lows[idx].store(Some(swing));
        self.last_swing_low.store(swing.price);
        
        // Update PD zone
        let mut pd = self.pd_zone.load();
        pd.update_range(self.last_swing_high.load(), swing.price.min(self.last_swing_high.load()));
        self.pd_zone.store(pd);
    }

    #[inline]
    fn check_bos(&self, price: f64, timestamp_ns: u64) -> Option<BOS> {
        let current_structure = self.structure.load();
        let last_high = self.last_swing_high.load();
        let last_low = self.last_swing_low.load();
        
        match current_structure {
            MarketStructure::Bullish => {
                // Bullish BOS: break above last swing high
                if price > last_high && last_high > 0.0 {
                    let strength = ((price - last_high) / last_high).min(1.0);
                    let bos = BOS {
                        timestamp_ns,
                        price,
                        direction: MarketStructure::Bullish,
                        strength,
                    };
                    self.last_bos_price.store(price);
                    self.last_bos_time.store(timestamp_ns);
                    self.bos_count.fetch_add(1);
                    return Some(bos);
                }
            }
            MarketStructure::Bearish => {
                // Bearish BOS: break below last swing low
                if price < last_low && last_low > 0.0 {
                    let strength = ((last_low - price) / last_low).min(1.0);
                    let bos = BOS {
                        timestamp_ns,
                        price,
                        direction: MarketStructure::Bearish,
                        strength,
                    };
                    self.last_bos_price.store(price);
                    self.last_bos_time.store(timestamp_ns);
                    self.bos_count.fetch_add(1);
                    return Some(bos);
                }
            }
            MarketStructure::Ranging => {
                // Determine initial structure
                if price > last_high && last_high > 0.0 {
                    self.structure.store(MarketStructure::Bullish);
                } else if price < last_low && last_low > 0.0 {
                    self.structure.store(MarketStructure::Bearish);
                }
            }
        }
        None
    }

    /// Check for Change of Character
    #[inline]
    pub fn check_choch(&self, price: f64, timestamp_ns: u64) -> Option<CHoCH> {
        let current = self.structure.load();
        let last_high = self.last_swing_high.load();
        let last_low = self.last_swing_low.load();
        
        match current {
            MarketStructure::Bullish => {
                // Potential CHoCH to bearish: break below last swing low
                if price < last_low && last_low > 0.0 {
                    let choch = CHoCH {
                        timestamp_ns,
                        price,
                        from: MarketStructure::Bullish,
                        to: MarketStructure::Bearish,
                    };
                    self.structure.store(MarketStructure::Bearish);
                    self.last_choch.store(Some(choch));
                    self.choc_count.fetch_add(1);
                    return Some(choch);
                }
            }
            MarketStructure::Bearish => {
                // Potential CHoCH to bullish: break above last swing high
                if price > last_high && last_high > 0.0 {
                    let choch = CHoCH {
                        timestamp_ns,
                        price,
                        from: MarketStructure::Bearish,
                        to: MarketStructure::Bullish,
                    };
                    self.structure.store(MarketStructure::Bullish);
                    self.last_choch.store(Some(choch));
                    self.choc_count.fetch_add(1);
                    return Some(choch);
                }
            }
            _ => {}
        }
        None
    }

    #[inline]
    pub fn current_structure(&self) -> MarketStructure {
        self.structure.load()
    }

    #[inline]
    pub fn last_swing_high(&self) -> f64 {
        self.last_swing_high.load()
    }

    #[inline]
    pub fn last_swing_low(&self) -> f64 {
        self.last_swing_low.load()
    }

    #[inline]
    pub fn pd_zone(&self) -> PDZone {
        self.pd_zone.load()
    }

    #[inline]
    pub fn bos_count(&self) -> u32 {
        self.bos_count.load()
    }

    #[inline]
    pub fn choch_count(&self) -> u32 {
        self.choc_count.load()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_market_structure() {
        let tracker = MarketStructureTracker::new(5, 100.0, 90.0);
        
        // Simulate bullish movement
        tracker.update(101.0, 99.0, 100.5, 1_000_000_000);
        tracker.update(102.0, 100.0, 101.5, 2_000_000_000);
        tracker.update(103.0, 101.0, 102.5, 3_000_000_000);
        
        assert_eq!(tracker.current_structure(), MarketStructure::Ranging);
        assert!(tracker.last_swing_high() >= 100.0);
    }

    #[test]
    fn test_pd_zone() {
        let mut pd = PDZone::new(110.0, 90.0);
        assert!(pd.is_premium(106.0));
        assert!(pd.is_discount(94.0));
        assert!(!pd.is_premium(95.0));
    }
}
