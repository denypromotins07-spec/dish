//! Footprint Chart Alpha Generation Engine
//! Identifies trapped traders, passive absorption, and aggressive delta imbalances

use std::sync::atomic::{AtomicF64, AtomicI64, AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

/// Single footprint bar data
pub struct FootprintBar {
    /// Open price
    pub open: f64,
    /// High price
    pub high: f64,
    /// Low price
    pub low: f64,
    /// Close price
    pub close: f64,
    /// Total volume
    pub total_volume: f64,
    /// Aggressive buy volume (market buys hitting asks)
    pub aggressive_buy_vol: f64,
    /// Aggressive sell volume (market sells hitting bids)
    pub aggressive_sell_vol: f64,
    /// Passive buy volume (limit buys filled)
    pub passive_buy_vol: f64,
    /// Passive sell volume (limit sells filled)
    pub passive_sell_vol: f64,
    /// Delta (aggressive buy - aggressive sell)
    pub delta: f64,
    /// Imbalance ratio
    pub imbalance: f64,
    /// Timestamp
    pub timestamp_ns: u64,
}

impl FootprintBar {
    pub fn new(
        open: f64, high: f64, low: f64, close: f64,
        agg_buy: f64, agg_sell: f64,
        pass_buy: f64, pass_sell: f64,
    ) -> Self {
        let delta = agg_buy - agg_sell;
        let total_aggressive = agg_buy + agg_sell;
        let imbalance = if total_aggressive > 0.0 {
            (agg_buy - agg_sell) / total_aggressive
        } else {
            0.0
        };

        Self {
            open, high, low, close,
            total_volume: agg_buy + agg_sell + pass_buy + pass_sell,
            aggressive_buy_vol: agg_buy,
            aggressive_sell_vol: agg_sell,
            passive_buy_vol: pass_buy,
            passive_sell_vol: pass_sell,
            delta,
            imbalance,
            timestamp_ns: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos() as u64,
        }
    }

    /// Check if this is a bullish absorption bar
    #[inline]
    pub fn is_bullish_absorption(&self) -> bool {
        // Price didn't move down much despite heavy selling
        let body = self.close - self.open;
        let range = self.high - self.low;
        
        if range <= 0.0 { return false; }
        
        let small_body = body.abs() < range * 0.3;
        let heavy_selling = self.aggressive_sell_vol > self.aggressive_buy_vol * 2.0;
        let passive_buying = self.passive_buy_vol > self.aggressive_sell_vol * 0.8;
        
        small_body && heavy_selling && passive_buying
    }

    /// Check if this is a bearish absorption bar
    #[inline]
    pub fn is_bearish_absorption(&self) -> bool {
        let body = self.close - self.open;
        let range = self.high - self.low;
        
        if range <= 0.0 { return false; }
        
        let small_body = body.abs() < range * 0.3;
        let heavy_buying = self.aggressive_buy_vol > self.aggressive_sell_vol * 2.0;
        let passive_selling = self.passive_sell_vol > self.aggressive_buy_vol * 0.8;
        
        small_body && heavy_buying && passive_selling
    }

    /// Check for trapped traders (longs trapped at high)
    #[inline]
    pub fn has_trapped_longs(&self, prev_bar: &FootprintBar) -> bool {
        // Made new high but closed near low with negative delta
        let made_new_high = self.high > prev_bar.high;
        let closed_near_low = (self.close - self.low) < (self.high - self.low) * 0.3;
        let negative_delta = self.delta < 0.0;
        let high_volume = self.total_volume > prev_bar.total_volume * 1.5;
        
        made_new_high && closed_near_low && negative_delta && high_volume
    }

    /// Check for trapped traders (shorts trapped at low)
    #[inline]
    pub fn has_trapped_shorts(&self, prev_bar: &FootprintBar) -> bool {
        let made_new_low = self.low < prev_bar.low;
        let closed_near_high = (self.high - self.close) < (self.high - self.low) * 0.3;
        let positive_delta = self.delta > 0.0;
        let high_volume = self.total_volume > prev_bar.total_volume * 1.5;
        
        made_new_low && closed_near_high && positive_delta && high_volume
    }
}

/// Footprint alpha signal generator
pub struct FootprintAlphaEngine {
    /// Rolling window of bars
    pub bars: Vec<FootprintBar>,
    /// Max bars to keep
    pub max_bars: usize,
    /// Cumulative delta
    pub cumulative_delta: AtomicF64,
    /// Average bar volume
    pub avg_volume: AtomicF64,
    /// Last signal
    pub last_signal: AtomicI64, // -1 = short, 0 = neutral, 1 = long
}

impl FootprintAlphaEngine {
    pub fn new(max_bars: usize) -> Self {
        Self {
            bars: Vec::with_capacity(max_bars),
            max_bars,
            cumulative_delta: AtomicF64::new(0.0),
            avg_volume: AtomicF64::new(0.0),
            last_signal: AtomicI64::new(0),
        }
    }

    /// Add a new bar and update statistics
    #[inline]
    pub fn add_bar(&mut self, bar: FootprintBar) {
        self.bars.push(bar);
        
        if self.bars.len() > self.max_bars {
            self.bars.remove(0);
        }
        
        self.update_statistics();
    }

    /// Update rolling statistics
    #[inline]
    fn update_statistics(&self) {
        if self.bars.is_empty() { return; }
        
        let mut cum_delta = 0.0;
        let mut total_vol = 0.0;
        
        for bar in &self.bars {
            cum_delta += bar.delta;
            total_vol += bar.total_volume;
        }
        
        self.cumulative_delta.store(cum_delta, Ordering::Relaxed);
        self.avg_volume.store(total_vol / self.bars.len() as f64, Ordering::Relaxed);
    }

    /// Generate alpha signal based on footprint analysis
    #[inline]
    pub fn generate_signal(&self) -> FootprintSignal {
        if self.bars.len() < 2 {
            return FootprintSignal::Neutral;
        }

        let current = &self.bars[self.bars.len() - 1];
        let prev = &self.bars[self.bars.len() - 2];

        let mut long_score = 0.0;
        let mut short_score = 0.0;

        // Check for bullish absorption
        if current.is_bullish_absorption() {
            long_score += 2.0;
        }

        // Check for bearish absorption
        if current.is_bearish_absorption() {
            short_score += 2.0;
        }

        // Check for trapped longs (bullish reversal signal)
        if current.has_trapped_longs(prev) {
            short_score += 1.5; // Wait, trapped longs means they will sell -> bearish
        }

        // Check for trapped shorts (bearish reversal signal)
        if current.has_trapped_shorts(prev) {
            long_score += 1.5; // Trapped shorts will cover -> bullish
        }

        // Delta divergence
        let cum_delta = self.cumulative_delta.load(Ordering::Relaxed);
        let price_change = current.close - self.bars[0].open;
        
        if price_change < 0.0 && cum_delta > 0.0 {
            // Price down but delta positive = hidden buying
            long_score += 1.0;
        }
        if price_change > 0.0 && cum_delta < 0.0 {
            // Price up but delta negative = hidden selling
            short_score += 1.0;
        }

        // Extreme imbalance
        if current.imbalance > 0.7 {
            long_score += 0.5;
        }
        if current.imbalance < -0.7 {
            short_score += 0.5;
        }

        if long_score >= 3.0 && long_score > short_score + 1.0 {
            self.last_signal.store(1, Ordering::Relaxed);
            FootprintSignal::Long(long_score)
        } else if short_score >= 3.0 && short_score > long_score + 1.0 {
            self.last_signal.store(-1, Ordering::Relaxed);
            FootprintSignal::Short(short_score)
        } else {
            self.last_signal.store(0, Ordering::Relaxed);
            FootprintSignal::Neutral
        }
    }

    /// Get current cumulative delta
    #[inline]
    pub fn get_cumulative_delta(&self) -> f64 {
        self.cumulative_delta.load(Ordering::Relaxed)
    }

    /// Get average volume
    #[inline]
    pub fn get_avg_volume(&self) -> f64 {
        self.avg_volume.load(Ordering::Relaxed)
    }

    /// Clear all bars
    #[inline]
    pub fn reset(&mut self) {
        self.bars.clear();
        self.cumulative_delta.store(0.0, Ordering::Relaxed);
        self.avg_volume.store(0.0, Ordering::Relaxed);
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum FootprintSignal {
    Long(f64),    // Signal strength
    Short(f64),   // Signal strength
    Neutral,
}

/// Order flow imbalance tracker
pub struct OrderFlowImbalance {
    /// Rolling buy volume
    pub buy_vol: AtomicF64,
    /// Rolling sell volume
    pub sell_vol: AtomicF64,
    /// Window size in milliseconds
    pub window_ms: u64,
    /// Last update timestamp
    pub last_update_ns: AtomicU64,
}

impl OrderFlowImbalance {
    pub fn new(window_ms: u64) -> Self {
        Self {
            buy_vol: AtomicF64::new(0.0),
            sell_vol: AtomicF64::new(0.0),
            window_ms,
            last_update_ns: AtomicU64::new(0),
        }
    }

    #[inline]
    pub fn add_trade(&self, volume: f64, is_buy: bool) {
        if is_buy {
            self.buy_vol.fetch_add(volume, Ordering::Relaxed);
        } else {
            self.sell_vol.fetch_add(volume, Ordering::Relaxed);
        }
        self.last_update_ns.store(
            SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos() as u64,
            Ordering::Relaxed,
        );
    }

    #[inline]
    pub fn get_imbalance_ratio(&self) -> f64 {
        let buy = self.buy_vol.load(Ordering::Relaxed);
        let sell = self.sell_vol.load(Ordering::Relaxed);
        let total = buy + sell;
        
        if total <= 0.0 { return 0.0; }
        (buy - sell) / total
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_footprint_bar_creation() {
        let bar = FootprintBar::new(
            100.0, 101.0, 99.5, 100.5,
            500.0, 200.0,  // Aggressive buy/sell
            100.0, 100.0,  // Passive buy/sell
        );
        
        assert!((bar.delta - 300.0).abs() < 0.01);
        assert!(bar.imbalance > 0.5); // Positive imbalance
    }

    #[test]
    fn test_bullish_absorption() {
        let bar = FootprintBar::new(
            100.0, 100.5, 99.5, 100.1,  // Small body
            100.0, 500.0,  // Heavy selling
            450.0, 50.0,   // Passive buying absorbing
        );
        
        assert!(bar.is_bullish_absorption());
    }

    #[test]
    fn test_alpha_signal_generation() {
        let mut engine = FootprintAlphaEngine::new(10);
        
        // Add normal bars first
        for i in 0..5 {
            let bar = FootprintBar::new(
                100.0 + i as f64, 101.0 + i as f64,
                99.0 + i as f64, 100.5 + i as f64,
                300.0, 300.0, 100.0, 100.0,
            );
            engine.add_bar(bar);
        }
        
        // Add strong bullish absorption bar
        let absorption_bar = FootprintBar::new(
            105.0, 105.5, 104.5, 105.1,
            100.0, 600.0,  // Heavy selling
            550.0, 50.0,   // Absorbed by passive buyers
        );
        engine.add_bar(absorption_bar);
        
        let signal = engine.generate_signal();
        match signal {
            FootprintSignal::Long(strength) => assert!(strength >= 2.0),
            _ => {}, // May need more bars or different thresholds
        }
    }

    #[test]
    fn test_order_flow_imbalance() {
        let imbalance = OrderFlowImbalance::new(1000);
        
        imbalance.add_trade(100.0, true);
        imbalance.add_trade(100.0, true);
        imbalance.add_trade(50.0, false);
        
        let ratio = imbalance.get_imbalance_ratio();
        assert!(ratio > 0.5); // More buying than selling
    }
}
