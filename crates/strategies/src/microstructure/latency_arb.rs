//! Cross-Venue Latency Arbitrage Detector and Execution Engine
//! Instantly spots and exploits microsecond price discrepancies between venues

use std::sync::atomic::{AtomicF64, AtomicU64, AtomicBool, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

/// Venue identifier
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Venue {
    BinanceSpot,
    BinanceFutures,
    Coinbase,
    Kraken,
    DEX,
    Custom(u8),
}

/// Price data from a single venue
pub struct VenuePrice {
    pub bid: AtomicF64,
    pub ask: AtomicF64,
    pub bid_size: AtomicF64,
    pub ask_size: AtomicF64,
    pub timestamp_ns: AtomicU64,
}

impl VenuePrice {
    pub fn new() -> Self {
        Self {
            bid: AtomicF64::new(0.0),
            ask: AtomicF64::new(0.0),
            bid_size: AtomicF64::new(0.0),
            ask_size: AtomicF64::new(0.0),
            timestamp_ns: AtomicU64::new(0),
        }
    }

    #[inline]
    pub fn update(&self, bid: f64, ask: f64, bid_size: f64, ask_size: f64) {
        self.bid.store(bid, Ordering::Relaxed);
        self.ask.store(ask, Ordering::Relaxed);
        self.bid_size.store(bid_size, Ordering::Relaxed);
        self.ask_size.store(ask_size, Ordering::Relaxed);
        self.timestamp_ns.store(
            SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos() as u64,
            Ordering::Relaxed,
        );
    }

    #[inline]
    pub fn mid(&self) -> f64 {
        (self.bid.load(Ordering::Relaxed) + self.ask.load(Ordering::Relaxed)) / 2.0
    }

    #[inline]
    pub fn spread(&self) -> f64 {
        self.ask.load(Ordering::Relaxed) - self.bid.load(Ordering::Relaxed)
    }
}

impl Default for VenuePrice {
    fn default() -> Self {
        Self::new()
    }
}

/// Latency arbitrage opportunity
#[derive(Clone, Copy, Debug)]
pub struct ArbitrageOpportunity {
    pub buy_venue: Venue,
    pub sell_venue: Venue,
    pub buy_price: f64,
    pub sell_price: f64,
    pub spread_bps: f64,
    pub max_size: f64,
    pub expected_profit: f64,
    pub confidence: f64,
    pub timestamp_ns: u64,
}

/// Cross-venue latency arbitrage detector
pub struct LatencyArbitrageDetector {
    /// Venue A prices
    pub venue_a: VenuePrice,
    /// Venue B prices
    pub venue_b: VenuePrice,
    /// Minimum spread threshold (bps) to trigger
    pub min_spread_bps: AtomicF64,
    /// Maximum age of quote (ns) to consider valid
    pub max_quote_age_ns: AtomicU64,
    /// Last detected opportunity
    pub last_opportunity: Option<ArbitrageOpportunity>,
    /// Opportunity count
    pub opportunity_count: AtomicU64,
    /// Enabled flag
    pub enabled: AtomicBool,
}

impl LatencyArbitrageDetector {
    pub fn new(min_spread_bps: f64, max_quote_age_ms: u64) -> Self {
        Self {
            venue_a: VenuePrice::new(),
            venue_b: VenuePrice::new(),
            min_spread_bps: AtomicF64::new(min_spread_bps),
            max_quote_age_ns: AtomicU64::new(max_quote_age_ms * 1_000_000),
            last_opportunity: None,
            opportunity_count: AtomicU64::new(0),
            enabled: AtomicBool::new(true),
        }
    }

    /// Update prices from venue A
    #[inline]
    pub fn update_venue_a(&self, bid: f64, ask: f64, bid_size: f64, ask_size: f64) {
        self.venue_a.update(bid, ask, bid_size, ask_size);
    }

    /// Update prices from venue B
    #[inline]
    pub fn update_venue_b(&self, bid: f64, ask: f64, bid_size: f64, ask_size: f64) {
        self.venue_b.update(bid, ask, bid_size, ask_size);
    }

    /// Check for arbitrage opportunity
    #[inline]
    pub fn check_arbitrage(&mut self) -> Option<ArbitrageOpportunity> {
        if !self.enabled.load(Ordering::Relaxed) {
            return None;
        }

        let now_ns = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64;

        // Check quote freshness
        let age_a = now_ns.saturating_sub(self.venue_a.timestamp_ns.load(Ordering::Relaxed));
        let age_b = now_ns.saturating_sub(self.venue_b.timestamp_ns.load(Ordering::Relaxed));
        let max_age = self.max_quote_age_ns.load(Ordering::Relaxed);

        if age_a > max_age || age_b > max_age {
            return None; // Quotes too stale
        }

        let bid_a = self.venue_a.bid.load(Ordering::Relaxed);
        let ask_a = self.venue_a.ask.load(Ordering::Relaxed);
        let bid_b = self.venue_b.bid.load(Ordering::Relaxed);
        let ask_b = self.venue_b.ask.load(Ordering::Relaxed);

        let size_a_bid = self.venue_a.bid_size.load(Ordering::Relaxed);
        let size_a_ask = self.venue_a.ask_size.load(Ordering::Relaxed);
        let size_b_bid = self.venue_b.bid_size.load(Ordering::Relaxed);
        let size_b_ask = self.venue_b.ask_size.load(Ordering::Relaxed);

        // Check both directions
        let mut opportunity: Option<ArbitrageOpportunity> = None;

        // Direction 1: Buy on A, Sell on B
        if bid_b > ask_a && bid_b > 0.0 && ask_a > 0.0 {
            let spread = (bid_b - ask_a) / ((bid_b + ask_a) / 2.0) * 10000.0;
            let min_spread = self.min_spread_bps.load(Ordering::Relaxed);
            
            if spread >= min_spread {
                let max_size = size_a_ask.min(size_b_bid);
                let expected_profit = (bid_b - ask_a) * max_size;
                
                // Confidence based on spread size and quote age
                let confidence = (spread / min_spread).min(2.0) / 2.0 
                    * (1.0 - (age_a.max(age_b) as f64 / max_age as f64));

                opportunity = Some(ArbitrageOpportunity {
                    buy_venue: Venue::BinanceSpot,
                    sell_venue: Venue::BinanceFutures,
                    buy_price: ask_a,
                    sell_price: bid_b,
                    spread_bps: spread,
                    max_size,
                    expected_profit,
                    confidence,
                    timestamp_ns: now_ns,
                });
            }
        }

        // Direction 2: Buy on B, Sell on A
        if bid_a > ask_b && bid_a > 0.0 && ask_b > 0.0 {
            let spread = (bid_a - ask_b) / ((bid_a + ask_b) / 2.0) * 10000.0;
            let min_spread = self.min_spread_bps.load(Ordering::Relaxed);
            
            if spread >= min_spread {
                let max_size = size_b_ask.min(size_a_bid);
                let expected_profit = (bid_a - ask_b) * max_size;
                
                let confidence = (spread / min_spread).min(2.0) / 2.0 
                    * (1.0 - (age_a.max(age_b) as f64 / max_age as f64));

                let opp = ArbitrageOpportunity {
                    buy_venue: Venue::BinanceFutures,
                    sell_venue: Venue::BinanceSpot,
                    buy_price: ask_b,
                    sell_price: bid_a,
                    spread_bps: spread,
                    max_size,
                    expected_profit,
                    confidence,
                    timestamp_ns: now_ns,
                };

                // Take the better opportunity
                if let Some(ref current) = opportunity {
                    if opp.spread_bps > current.spread_bps {
                        opportunity = Some(opp);
                    }
                } else {
                    opportunity = Some(opp);
                }
            }
        }

        if let Some(ref opp) = opportunity {
            self.last_opportunity = Some(*opp);
            self.opportunity_count.fetch_add(1, Ordering::Relaxed);
        }

        opportunity
    }

    /// Get cross-venue spread in bps
    #[inline]
    pub fn get_cross_spread_bps(&self) -> f64 {
        let mid_a = self.venue_a.mid();
        let mid_b = self.venue_b.mid();
        
        if mid_a <= 0.0 || mid_b <= 0.0 {
            return 0.0;
        }
        
        ((mid_b - mid_a).abs() / ((mid_a + mid_b) / 2.0)) * 10000.0
    }

    /// Get number of opportunities detected
    #[inline]
    pub fn get_opportunity_count(&self) -> u64 {
        self.opportunity_count.load(Ordering::Relaxed)
    }

    /// Enable/disable detection
    #[inline]
    pub fn set_enabled(&self, enabled: bool) {
        self.enabled.store(enabled, Ordering::Relaxed);
    }

    /// Update thresholds
    #[inline]
    pub fn update_thresholds(&self, min_spread_bps: f64, max_age_ms: u64) {
        self.min_spread_bps.store(min_spread_bps, Ordering::Relaxed);
        self.max_quote_age_ns.store(max_age_ms * 1_000_000, Ordering::Relaxed);
    }
}

/// Multi-venue arbitrage scanner
pub struct MultiVenueScanner {
    /// Reference venue (usually fastest/most liquid)
    pub reference_prices: VenuePrice,
    /// Target venues to scan
    pub target_venues: Vec<VenuePrice>,
    /// Min spread threshold
    pub min_spread_bps: AtomicF64,
}

impl MultiVenueScanner {
    pub fn new(num_targets: usize, min_spread_bps: f64) -> Self {
        let mut targets = Vec::with_capacity(num_targets);
        for _ in 0..num_targets {
            targets.push(VenuePrice::new());
        }
        
        Self {
            reference_prices: VenuePrice::new(),
            target_venues: targets,
            min_spread_bps: AtomicF64::new(min_spread_bps),
        }
    }

    /// Update reference venue prices
    #[inline]
    pub fn update_reference(&self, bid: f64, ask: f64, bid_size: f64, ask_size: f64) {
        self.reference_prices.update(bid, ask, bid_size, ask_size);
    }

    /// Update target venue prices
    #[inline]
    pub fn update_target(&self, index: usize, bid: f64, ask: f64, bid_size: f64, ask_size: f64) {
        if index < self.target_venues.len() {
            self.target_venues[index].update(bid, ask, bid_size, ask_size);
        }
    }

    /// Scan all venues for opportunities
    #[inline]
    pub fn scan(&self) -> Vec<ArbitrageOpportunity> {
        let mut opportunities = Vec::new();
        let ref_bid = self.reference_prices.bid.load(Ordering::Relaxed);
        let ref_ask = self.reference_prices.ask.load(Ordering::Relaxed);
        let ref_ts = self.reference_prices.timestamp_ns.load(Ordering::Relaxed);
        
        let now_ns = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64;
        let max_age = 5_000_000; // 5ms
        
        if now_ns.saturating_sub(ref_ts) > max_age {
            return opportunities; // Reference too stale
        }

        let min_spread = self.min_spread_bps.load(Ordering::Relaxed);

        for (i, venue) in self.target_venues.iter().enumerate() {
            let ts = venue.timestamp_ns.load(Ordering::Relaxed);
            if now_ns.saturating_sub(ts) > max_age {
                continue; // Venue too stale
            }

            let v_bid = venue.bid.load(Ordering::Relaxed);
            let v_ask = venue.ask.load(Ordering::Relaxed);
            let v_size_bid = venue.bid_size.load(Ordering::Relaxed);
            let v_size_ask = venue.ask_size.load(Ordering::Relaxed);

            // Check: Buy reference, sell venue
            if v_bid > ref_ask && v_bid > 0.0 {
                let spread = (v_bid - ref_ask) / ((v_bid + ref_ask) / 2.0) * 10000.0;
                if spread >= min_spread {
                    opportunities.push(ArbitrageOpportunity {
                        buy_venue: Venue::Custom(0),
                        sell_venue: Venue::Custom(i as u8),
                        buy_price: ref_ask,
                        sell_price: v_bid,
                        spread_bps: spread,
                        max_size: ref_ask.min(v_size_bid),
                        expected_profit: (v_bid - ref_ask) * ref_ask.min(v_size_bid),
                        confidence: 0.8,
                        timestamp_ns: now_ns,
                    });
                }
            }

            // Check: Buy venue, sell reference
            if ref_bid > v_ask && ref_bid > 0.0 {
                let spread = (ref_bid - v_ask) / ((ref_bid + v_ask) / 2.0) * 10000.0;
                if spread >= min_spread {
                    opportunities.push(ArbitrageOpportunity {
                        buy_venue: Venue::Custom(i as u8),
                        sell_venue: Venue::Custom(0),
                        buy_price: v_ask,
                        sell_price: ref_bid,
                        spread_bps: spread,
                        max_size: v_size_ask.min(ref_bid),
                        expected_profit: (ref_bid - v_ask) * v_size_ask.min(ref_bid),
                        confidence: 0.8,
                        timestamp_ns: now_ns,
                    });
                }
            }
        }

        opportunities
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_arbitrage_detection() {
        let mut detector = LatencyArbitrageDetector::new(5.0, 100); // 5 bps, 100ms
        
        // Set up clear arbitrage: A cheaper than B
        detector.update_venue_a(99.9, 100.0, 10.0, 10.0); // Ask at 100
        detector.update_venue_b(100.2, 100.3, 10.0, 10.0); // Bid at 100.2
        
        let opp = detector.check_arbitrage();
        assert!(opp.is_some());
        
        let o = opp.unwrap();
        assert!(o.spread_bps > 5.0);
        assert!((o.buy_price - 100.0).abs() < 0.01);
        assert!((o.sell_price - 100.2).abs() < 0.01);
    }

    #[test]
    fn test_no_arbitrage() {
        let mut detector = LatencyArbitrageDetector::new(5.0, 100);
        
        // Overlapping prices, no arb
        detector.update_venue_a(99.9, 100.1, 10.0, 10.0);
        detector.update_venue_b(99.95, 100.05, 10.0, 10.0);
        
        let opp = detector.check_arbitrage();
        assert!(opp.is_none());
    }

    #[test]
    fn test_stale_quotes() {
        let mut detector = LatencyArbitrageDetector::new(1.0, 1); // 1ms max age
        
        detector.update_venue_a(99.0, 99.5, 10.0, 10.0);
        detector.update_venue_b(100.0, 100.5, 10.0, 10.0);
        
        // Wait a bit
        std::thread::sleep(std::time::Duration::from_millis(10));
        
        let opp = detector.check_arbitrage();
        assert!(opp.is_none()); // Should be stale
    }

    #[test]
    fn test_multi_venue_scan() {
        let scanner = MultiVenueScanner::new(3, 5.0);
        
        scanner.update_reference(99.9, 100.0, 10.0, 10.0);
        scanner.update_target(0, 100.2, 100.3, 10.0, 10.0); // Arb opportunity
        scanner.update_target(1, 99.95, 100.05, 10.0, 10.0); // No arb
        scanner.update_target(2, 100.5, 100.6, 10.0, 10.0); // Better arb
        
        let opportunities = scanner.scan();
        assert!(!opportunities.is_empty());
    }
}
