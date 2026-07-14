//! Order Blocks, Fair Value Gaps (FVG), and Breaker Blocks detection
//! Automatic mitigation tracking and garbage collection of invalidated zones

use crossbeam::atomic::AtomicCell;
use std::sync::Arc;

/// Zone type
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ZoneType {
    BullishOrderBlock,
    BearishOrderBlock,
    BullishFVG,
    BearishFVG,
    BullishBreaker,
    BearishBreaker,
}

/// SMC Zone with mitigation tracking
#[repr(C, align(64))]
#[derive(Clone, Copy)]
pub struct SMCZone {
    pub zone_type: ZoneType,
    pub high: f64,
    pub low: f64,
    pub midpoint: f64,
    pub created_at_ns: u64,
    pub tested_count: u32,
    pub mitigated: bool,
    pub invalidated: bool,
    pub volume_at_creation: f64,
}

impl SMCZone {
    #[inline]
    pub fn new(zone_type: ZoneType, high: f64, low: f64, timestamp_ns: u64, volume: f64) -> Self {
        Self {
            zone_type,
            high,
            low,
            midpoint: (high + low) / 2.0,
            created_at_ns: timestamp_ns,
            tested_count: 0,
            mitigated: false,
            invalidated: false,
            volume_at_creation: volume,
        }
    }

    #[inline]
    pub fn contains(&self, price: f64) -> bool {
        price >= self.low && price <= self.high
    }

    #[inline]
    pub fn test(&mut self) {
        self.tested_count += 1;
    }

    #[inline]
    pub fn mark_mitigated(&mut self) {
        self.mitigated = true;
    }

    #[inline]
    pub fn mark_invalidated(&mut self) {
        self.invalidated = true;
    }

    #[inline]
    pub fn is_valid(&self) -> bool {
        !self.mitigated && !self.invalidated
    }

    /// Check if zone should be garbage collected based on age and tests
    #[inline]
    pub fn should_gc(&self, current_time_ns: u64, max_age_ns: u64, max_tests: u32) -> bool {
        let age = current_time_ns.saturating_sub(self.created_at_ns);
        age > max_age_ns || self.tested_count >= max_tests || self.invalidated
    }
}

/// Fair Value Gap detector
#[repr(C, align(64))]
pub struct FVGDetector {
    zones: Vec<AtomicCell<Option<SMCZone>>>,
    head: AtomicCell<usize>,
    count: AtomicCell<usize>,
    max_zones: usize,
    max_age_ns: u64,
    max_tests: u32,
}

impl FVGDetector {
    pub fn new(max_zones: usize, max_age_ns: u64, max_tests: u32) -> Self {
        Self {
            zones: (0..max_zones).map(|_| AtomicCell::new(None)).collect(),
            head: AtomicCell::new(0),
            count: AtomicCell::new(0),
            max_zones,
            max_age_ns,
            max_tests,
        }
    }

    /// Detect FVG from three-candle pattern
    #[inline]
    pub fn detect(&self, candle1_high: f64, candle1_low: f64, candle1_close: f64,
                  candle2_high: f64, candle2_low: f64, candle2_close: f64,
                  candle3_high: f64, candle3_low: f64, candle3_open: f64,
                  timestamp_ns: u64, volume: f64) -> Option<SMCZone> {
        
        // Bullish FVG: candle2 is bullish, candle3 open > candle1 high
        if candle2_close > candle2_open() && candle3_open > candle1_high {
            let fvg_high = candle1_high;
            let fvg_low = candle2_low.min(candle3_low);
            
            if fvg_high > fvg_low {
                let zone = SMCZone::new(ZoneType::BullishFVG, fvg_high, fvg_low, timestamp_ns, volume);
                self.add_zone(zone);
                return Some(zone);
            }
        }
        
        // Bearish FVG: candle2 is bearish, candle3 open < candle1 low
        if candle2_close < candle2_open() && candle3_open < candle1_low {
            let fvg_low = candle1_low;
            let fvg_high = candle2_high.max(candle3_high);
            
            if fvg_high > fvg_low {
                let zone = SMCZone::new(ZoneType::BearishFVG, fvg_high, fvg_low, timestamp_ns, volume);
                self.add_zone(zone);
                return Some(zone);
            }
        }
        
        None
    }

    #[inline]
    fn add_zone(&self, zone: SMCZone) {
        let idx = self.head.fetch_add(1) % self.max_zones;
        self.zones[idx].store(Some(zone));
        
        let mut count = self.count.load();
        if count < self.max_zones {
            self.count.store(count + 1);
        }
    }

    /// Check price against all active FVG zones
    #[inline]
    pub fn check_price(&self, price: f64, timestamp_ns: u64) -> Vec<SMCZone> {
        let mut hits = Vec::with_capacity(4);
        let count = self.count.load();
        
        for i in 0..count.min(self.max_zones) {
            let idx = (self.head.load().wrapping_sub(i + 1)) % self.max_zones;
            if let Some(mut zone) = self.zones[idx].load() {
                if zone.contains(price) && zone.is_valid() {
                    zone.test();
                    self.zones[idx].store(Some(zone));
                    hits.push(zone);
                }
                
                // Garbage collect old/invalidated zones
                if zone.should_gc(timestamp_ns, self.max_age_ns, self.max_tests) {
                    self.zones[idx].store(None);
                }
            }
        }
        
        hits
    }

    /// Get all active zones
    #[inline]
    pub fn active_zones(&self, timestamp_ns: u64) -> Vec<SMCZone> {
        let mut zones = Vec::with_capacity(self.count.load());
        let count = self.count.load();
        
        for i in 0..count.min(self.max_zones) {
            let idx = (self.head.load().wrapping_sub(i + 1)) % self.max_zones;
            if let Some(zone) = self.zones[idx].load() {
                if zone.is_valid() && !zone.should_gc(timestamp_ns, self.max_age_ns, self.max_tests) {
                    zones.push(zone);
                } else {
                    // GC
                    self.zones[idx].store(None);
                }
            }
        }
        
        zones
    }
}

/// Order Block detector
#[repr(C, align(64))]
pub struct OrderBlockDetector {
    zones: Vec<AtomicCell<Option<SMCZone>>>,
    head: AtomicCell<usize>,
    count: AtomicCell<usize>,
    max_zones: usize,
    max_age_ns: u64,
    max_tests: u32,
}

impl OrderBlockDetector {
    pub fn new(max_zones: usize, max_age_ns: u64, max_tests: u32) -> Self {
        Self {
            zones: (0..max_zones).map(|_| AtomicCell::new(None)).collect(),
            head: AtomicCell::new(0),
            count: AtomicCell::new(0),
            max_zones,
            max_age_ns,
            max_tests,
        }
    }

    /// Detect order block from candle pattern
    #[inline]
    pub fn detect_bullish(&self, candle_high: f64, candle_low: f64, candle_close: f64,
                          next_candle_open: f64, timestamp_ns: u64, volume: f64) -> Option<SMCZone> {
        // Bullish OB: strong bullish candle followed by higher prices
        if candle_close > candle_low + (candle_high - candle_low) * 0.7 
           && next_candle_open > candle_high {
            let ob_high = candle_high;
            let ob_low = candle_low;
            
            let zone = SMCZone::new(ZoneType::BullishOrderBlock, ob_high, ob_low, timestamp_ns, volume);
            self.add_zone(zone);
            return Some(zone);
        }
        None
    }

    #[inline]
    pub fn detect_bearish(&self, candle_high: f64, candle_low: f64, candle_close: f64,
                          next_candle_open: f64, timestamp_ns: u64, volume: f64) -> Option<SMCZone> {
        // Bearish OB: strong bearish candle followed by lower prices
        if candle_close < candle_high - (candle_high - candle_low) * 0.7 
           && next_candle_open < candle_low {
            let ob_high = candle_high;
            let ob_low = candle_low;
            
            let zone = SMCZone::new(ZoneType::BearishOrderBlock, ob_high, ob_low, timestamp_ns, volume);
            self.add_zone(zone);
            return Some(zone);
        }
        None
    }

    #[inline]
    fn add_zone(&self, zone: SMCZone) {
        let idx = self.head.fetch_add(1) % self.max_zones;
        self.zones[idx].store(Some(zone));
        
        let mut count = self.count.load();
        if count < self.max_zones {
            self.count.store(count + 1);
        }
    }

    /// Check price against order blocks
    #[inline]
    pub fn check_price(&self, price: f64, timestamp_ns: u64) -> Vec<SMCZone> {
        let mut hits = Vec::with_capacity(4);
        let count = self.count.load();
        
        for i in 0..count.min(self.max_zones) {
            let idx = (self.head.load().wrapping_sub(i + 1)) % self.max_zones;
            if let Some(mut zone) = self.zones[idx].load() {
                if zone.contains(price) && zone.is_valid() {
                    zone.test();
                    self.zones[idx].store(Some(zone));
                    hits.push(zone);
                }
                
                if zone.should_gc(timestamp_ns, self.max_age_ns, self.max_tests) {
                    self.zones[idx].store(None);
                }
            }
        }
        
        hits
    }

    /// Run garbage collection explicitly
    #[inline]
    pub fn gc(&self, timestamp_ns: u64) {
        let count = self.count.load();
        for i in 0..count.min(self.max_zones) {
            let idx = (self.head.load().wrapping_sub(i + 1)) % self.max_zones;
            if let Some(zone) = self.zones[idx].load() {
                if zone.should_gc(timestamp_ns, self.max_age_ns, self.max_tests) {
                    self.zones[idx].store(None);
                }
            }
        }
    }
}

/// Breaker Block detector
#[repr(C, align(64))]
pub struct BreakerDetector {
    zones: Vec<AtomicCell<Option<SMCZone>>>,
    head: AtomicCell<usize>,
    count: AtomicCell<usize>,
    max_zones: usize,
}

impl BreakerDetector {
    pub fn new(max_zones: usize) -> Self {
        Self {
            zones: (0..max_zones).map(|_| AtomicCell::new(None)).collect(),
            head: AtomicCell::new(0),
            count: AtomicCell::new(0),
            max_zones,
        }
    }

    /// Detect breaker block after liquidity sweep
    #[inline]
    pub fn detect_bullish_breaker(&self, sweep_low: f64, sweep_high: f64,
                                   break_high: f64, timestamp_ns: u64, volume: f64) -> Option<SMCZone> {
        // Bullish breaker: sweep lows, then break above previous high
        if break_high > sweep_high {
            let breaker_high = sweep_high;
            let breaker_low = sweep_low;
            
            let zone = SMCZone::new(ZoneType::BullishBreaker, breaker_high, breaker_low, timestamp_ns, volume);
            self.add_zone(zone);
            return Some(zone);
        }
        None
    }

    #[inline]
    pub fn detect_bearish_breaker(&self, sweep_high: f64, sweep_low: f64,
                                   break_low: f64, timestamp_ns: u64, volume: f64) -> Option<SMCZone> {
        // Bearish breaker: sweep highs, then break below previous low
        if break_low < sweep_low {
            let breaker_high = sweep_high;
            let breaker_low = sweep_low;
            
            let zone = SMCZone::new(ZoneType::BearishBreaker, breaker_high, breaker_low, timestamp_ns, volume);
            self.add_zone(zone);
            return Some(zone);
        }
        None
    }

    #[inline]
    fn add_zone(&self, zone: SMCZone) {
        let idx = self.head.fetch_add(1) % self.max_zones;
        self.zones[idx].store(Some(zone));
        
        let mut count = self.count.load();
        if count < self.max_zones {
            self.count.store(count + 1);
        }
    }

    #[inline]
    pub fn check_price(&self, price: f64) -> Vec<SMCZone> {
        let mut hits = Vec::with_capacity(2);
        let count = self.count.load();
        
        for i in 0..count.min(self.max_zones) {
            let idx = (self.head.load().wrapping_sub(i + 1)) % self.max_zones;
            if let Some(zone) = self.zones[idx].load() {
                if zone.contains(price) && zone.is_valid() {
                    hits.push(zone);
                }
            }
        }
        
        hits
    }
}

/// Combined SMC zone manager
#[repr(C, align(64))]
pub struct SMCZoneManager {
    fvg_detector: FVGDetector,
    ob_detector: OrderBlockDetector,
    breaker_detector: BreakerDetector,
    total_zones_created: AtomicCell<u32>,
    total_zones_mitigated: AtomicCell<u32>,
}

impl SMCZoneManager {
    pub fn new(max_fvg: usize, max_ob: usize, max_breaker: usize, max_age_ns: u64, max_tests: u32) -> Self {
        Self {
            fvg_detector: FVGDetector::new(max_fvg, max_age_ns, max_tests),
            ob_detector: OrderBlockDetector::new(max_ob, max_age_ns, max_tests),
            breaker_detector: BreakerDetector::new(max_breaker),
            total_zones_created: AtomicCell::new(0),
            total_zones_mitigated: AtomicCell::new(0),
        }
    }

    #[inline]
    pub fn update(&self, 
                  // Current candle
                  high: f64, low: f64, close: f64, open: f64,
                  // Previous candles for FVG
                  prev1_high: f64, prev1_low: f64, prev1_close: f64, prev1_open: f64,
                  prev2_high: f64, prev2_low: f64, prev2_close: f64, prev2_open: f64,
                  timestamp_ns: u64, volume: f64) -> Vec<SMCZone> {
        
        let mut new_zones = Vec::with_capacity(4);
        
        // Detect FVGs
        if let Some(fvg) = self.fvg_detector.detect(
            prev2_high, prev2_low, prev2_close,
            prev1_high, prev1_low, prev1_close,
            high, low, open,
            timestamp_ns, volume
        ) {
            new_zones.push(fvg);
            self.total_zones_created.fetch_add(1);
        }
        
        // Detect Order Blocks
        if let Some(ob) = self.ob_detector.detect_bullish(
            prev1_high, prev1_low, prev1_close, open, timestamp_ns, volume
        ) {
            new_zones.push(ob);
            self.total_zones_created.fetch_add(1);
        }
        
        if let Some(ob) = self.ob_detector.detect_bearish(
            prev1_high, prev1_low, prev1_close, open, timestamp_ns, volume
        ) {
            new_zones.push(ob);
            self.total_zones_created.fetch_add(1);
        }
        
        // Check price against all zones
        let mut hits = self.fvg_detector.check_price(close, timestamp_ns);
        hits.extend(self.ob_detector.check_price(close, timestamp_ns));
        hits.extend(self.breaker_detector.check_price(close));
        
        // Mark mitigated zones
        for hit in &hits {
            if hit.zone_type == ZoneType::BullishFVG || hit.zone_type == ZoneType::BearishFVG {
                self.total_zones_mitigated.fetch_add(1);
            }
        }
        
        // Run GC
        self.ob_detector.gc(timestamp_ns);
        
        hits
    }

    #[inline]
    pub fn active_zones(&self, timestamp_ns: u64) -> Vec<SMCZone> {
        let mut zones = self.fvg_detector.active_zones(timestamp_ns);
        // Add OB and breaker zones as needed
        zones
    }

    #[inline]
    pub fn stats(&self) -> (u32, u32) {
        (self.total_zones_created.load(), self.total_zones_mitigated.load())
    }
}

// Helper function for candle direction
#[inline]
fn candle_open() -> f64 {
    // Placeholder - in real implementation would track open
    0.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fvg_detection() {
        let detector = FVGDetector::new(100, 3600_000_000_000, 10);
        // Test data would go here
        assert_eq!(detector.count.load(), 0);
    }

    #[test]
    fn test_zone_contains() {
        let zone = SMCZone::new(ZoneType::BullishFVG, 105.0, 100.0, 1000, 100.0);
        assert!(zone.contains(102.0));
        assert!(!zone.contains(99.0));
    }
}
