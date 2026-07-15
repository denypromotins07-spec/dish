//! "Smart Money" Liquidity Sniper Strategy
//! Detects liquidity sweeps (stop hunts) and enters in sweep direction

use std::sync::atomic::{AtomicF64, AtomicU64, AtomicBool, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

/// Liquidity pool level (key support/resistance)
#[derive(Clone, Copy, Debug)]
pub struct LiquidityLevel {
    pub price: f64,
    pub liquidity_amount: f64,
    pub is_above: bool, // true = above current price (resistance), false = below (support)
    pub touched_count: u32,
    pub last_touched_ns: u64,
}

/// Sweep detection result
#[derive(Clone, Copy, Debug)]
pub struct SweepEvent {
    pub direction: SweepDirection,
    pub sweep_price: f64,
    pub liquidity_taken: f64,
    pub speed_ms: f64,
    pub confidence: f64,
    pub timestamp_ns: u64,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum SweepDirection {
    Upward,   // Swept liquidity above (bullish continuation or reversal)
    Downward, // Swept liquidity below (bearish continuation or reversal)
}

/// Liquidity sniper engine
pub struct LiquiditySniper {
    /// Known liquidity levels
    pub liquidity_levels: Vec<LiquidityLevel>,
    /// Recent price history for sweep detection
    pub price_history: Vec<(f64, u64)>,
    /// Max price history size
    pub max_history: usize,
    /// Current market price
    pub current_price: AtomicF64,
    /// Last detected sweep
    pub last_sweep: Option<SweepEvent>,
    /// Active position flag
    pub in_position: AtomicBool,
    /// Enabled flag
    pub enabled: AtomicBool,
    /// Sweep detection threshold (price movement %)
    pub sweep_threshold_pct: AtomicF64,
    /// Min liquidity to consider significant
    pub min_liquidity: AtomicF64,
}

impl LiquiditySniper {
    pub fn new(max_history: usize) -> Self {
        Self {
            liquidity_levels: Vec::with_capacity(20),
            price_history: Vec::with_capacity(max_history),
            max_history,
            current_price: AtomicF64::new(0.0),
            last_sweep: None,
            in_position: AtomicBool::new(false),
            enabled: AtomicBool::new(true),
            sweep_threshold_pct: AtomicF64::new(0.5),
            min_liquidity: AtomicF64::new(1000.0),
        }
    }

    /// Add a known liquidity level
    #[inline]
    pub fn add_liquidity_level(&mut self, level: LiquidityLevel) {
        self.liquidity_levels.push(level);
        // Keep sorted by price
        self.liquidity_levels.sort_by(|a, b| a.price.partial_cmp(&b.price).unwrap());
    }

    /// Update current price and check for sweeps
    #[inline]
    pub fn update_price(&mut self, price: f64, volume: f64) -> Option<SweepEvent> {
        if !self.enabled.load(Ordering::Relaxed) {
            return None;
        }

        let now_ns = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64;

        self.current_price.store(price, Ordering::Relaxed);
        self.price_history.push((price, now_ns));

        if self.price_history.len() > self.max_history {
            self.price_history.remove(0);
        }

        // Check for sweep
        if let Some(sweep) = self.detect_sweep(price, volume, now_ns) {
            self.last_sweep = Some(sweep);
            return Some(sweep);
        }

        None
    }

    /// Detect liquidity sweep
    #[inline]
    fn detect_sweep(&self, price: f64, volume: f64, now_ns: u64) -> Option<SweepEvent> {
        if self.price_history.len() < 5 {
            return None;
        }

        let threshold = self.sweep_threshold_pct.load(Ordering::Relaxed);
        let min_liq = self.min_liquidity.load(Ordering::Relaxed);

        // Find recent high and low
        let mut recent_high = f64::MIN;
        let mut recent_low = f64::MAX;
        let lookback = self.price_history.len().min(20);

        for i in (self.price_history.len() - lookback)..self.price_history.len() {
            let (p, _) = self.price_history[i];
            if p > recent_high { recent_high = p; }
            if p < recent_low { recent_low = p; }
        }

        // Check each liquidity level
        for level in &self.liquidity_levels {
            if level.liquidity_amount < min_liq {
                continue;
            }

            // Check if price just swept this level
            if level.is_above && price >= level.price && recent_high < level.price {
                // Upward sweep - price moved from below to above resistance
                let sweep_pct = (price - level.price) / level.price * 100.0;
                if sweep_pct.abs() < threshold && volume >= min_liq {
                    // Calculate sweep speed
                    let speed = self.calculate_sweep_speed(level.price, price, now_ns);
                    
                    return Some(SweepEvent {
                        direction: SweepDirection::Upward,
                        sweep_price: level.price,
                        liquidity_taken: level.liquidity_amount,
                        speed_ms: speed,
                        confidence: self.calculate_confidence(volume, level, sweep_pct),
                        timestamp_ns: now_ns,
                    });
                }
            }

            if !level.is_above && price <= level.price && recent_low > level.price {
                // Downward sweep - price moved from above to below support
                let sweep_pct = (level.price - price) / level.price * 100.0;
                if sweep_pct.abs() < threshold && volume >= min_liq {
                    let speed = self.calculate_sweep_speed(level.price, price, now_ns);
                    
                    return Some(SweepEvent {
                        direction: SweepDirection::Downward,
                        sweep_price: level.price,
                        liquidity_taken: level.liquidity_amount,
                        speed_ms: speed,
                        confidence: self.calculate_confidence(volume, level, sweep_pct),
                        timestamp_ns: now_ns,
                    });
                }
            }
        }

        None
    }

    /// Calculate sweep speed in ms
    #[inline]
    fn calculate_sweep_speed(&self, level_price: f64, current_price: f64, now_ns: u64) -> f64 {
        // Find when price was near level
        for (price, ts) in self.price_history.iter().rev() {
            if (price - level_price).abs() / level_price < 0.001 {
                let elapsed_ms = (now_ns - ts) as f64 / 1_000_000.0;
                return elapsed_ms.max(1.0);
            }
        }
        100.0 // Default if not found
    }

    /// Calculate confidence score (0-1)
    #[inline]
    fn calculate_confidence(&self, volume: f64, level: &LiquidityLevel, sweep_pct: f64) -> f64 {
        let mut confidence = 0.5;

        // Higher volume = higher confidence
        let vol_factor = (volume / level.liquidity_amount).min(2.0) / 2.0;
        confidence += vol_factor * 0.3;

        // Faster sweep = higher confidence (aggressive move)
        // But not too fast (might be wick)
        confidence += 0.1;

        // More touches historically = more significant level
        let touch_factor = (level.touched_count as f64).min(5.0) / 5.0;
        confidence += touch_factor * 0.1;

        // Smaller overshoot = cleaner sweep
        if sweep_pct.abs() < 0.2 {
            confidence += 0.1;
        }

        confidence.min(1.0)
    }

    /// Get entry signal based on sweep
    #[inline]
    pub fn get_entry_signal(&self, sweep: &SweepEvent) -> EntrySignal {
        match sweep.direction {
            SweepDirection::Upward => {
                // Swept highs - could be continuation or reversal
                // Wait for confirmation (price holding above level)
                let current = self.current_price.load(Ordering::Relaxed);
                if current > sweep.sweep_price * 1.001 {
                    EntrySignal::LongContinuation
                } else if current < sweep.sweep_price {
                    EntrySignal::LongReversal // Failed sweep, reversal down
                } else {
                    EntrySignal::Wait
                }
            }
            SweepDirection::Downward => {
                // Swept lows
                let current = self.current_price.load(Ordering::Relaxed);
                if current < sweep.sweep_price * 0.999 {
                    EntrySignal::ShortContinuation
                } else if current > sweep.sweep_price {
                    EntrySignal::ShortReversal // Failed sweep, reversal up
                } else {
                    EntrySignal::Wait
                }
            }
        }
    }

    /// Execute entry
    #[inline]
    pub fn enter_position(&self, direction: SweepDirection, size: f64) -> bool {
        if self.in_position.load(Ordering::Relaxed) {
            return false;
        }
        self.in_position.store(true, Ordering::Relaxed);
        true
    }

    /// Exit position
    #[inline]
    pub fn exit_position(&self) {
        self.in_position.store(false, Ordering::Relaxed);
    }

    /// Check if currently in position
    #[inline]
    pub fn is_in_position(&self) -> bool {
        self.in_position.load(Ordering::Relaxed)
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum EntrySignal {
    LongContinuation,
    LongReversal,
    ShortContinuation,
    ShortReversal,
    Wait,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_liquidity_sweep_detection() {
        let mut sniper = LiquiditySniper::new(100);
        
        // Add liquidity level above current price
        sniper.add_liquidity_level(LiquidityLevel {
            price: 100.0,
            liquidity_amount: 5000.0,
            is_above: true,
            touched_count: 3,
            last_touched_ns: 0,
        });
        
        // Simulate price approaching and sweeping the level
        for i in 0..10 {
            let price = 99.0 + (i as f64 * 0.2);
            sniper.update_price(price, 100.0);
        }
        
        // Final sweep with volume
        let sweep = sniper.update_price(100.05, 6000.0);
        
        assert!(sweep.is_some());
        let s = sweep.unwrap();
        assert_eq!(s.direction, SweepDirection::Upward);
        assert!(s.confidence > 0.5);
    }

    #[test]
    fn test_entry_signal_generation() {
        let mut sniper = LiquiditySniper::new(100);
        
        sniper.add_liquidity_level(LiquidityLevel {
            price: 100.0,
            liquidity_amount: 5000.0,
            is_above: true,
            touched_count: 2,
            last_touched_ns: 0,
        });
        
        // Create sweep event manually for testing
        let sweep = SweepEvent {
            direction: SweepDirection::Upward,
            sweep_price: 100.0,
            liquidity_taken: 5000.0,
            speed_ms: 50.0,
            confidence: 0.8,
            timestamp_ns: 0,
        };
        
        sniper.current_price.store(100.2, Ordering::Relaxed);
        let signal = sniper.get_entry_signal(&sweep);
        assert_eq!(signal, EntrySignal::LongContinuation);
    }
}
