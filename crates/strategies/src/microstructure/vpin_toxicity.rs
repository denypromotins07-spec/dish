//! Volume-Synchronized Probability of Informed Trading (VPIN) Calculator
//! Detects toxic order flow and automatically widens spreads or halts quoting

use std::sync::atomic::{AtomicF64, AtomicU64, AtomicBool, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

/// VPIN calculation state using volume buckets
pub struct VpinState {
    /// Current VPIN value
    pub vpin: AtomicF64,
    /// Buy volume in current bucket
    pub buy_volume: AtomicF64,
    /// Sell volume in current bucket
    pub sell_volume: AtomicF64,
    /// Target bucket size (volume)
    pub bucket_size: AtomicF64,
    /// Number of buckets for rolling average
    pub num_buckets: usize,
    /// Bucket index
    pub bucket_index: AtomicU64,
    /// Rolling sum of absolute imbalances
    pub rolling_imbalance_sum: AtomicF64,
    /// Count of filled buckets
    pub filled_buckets: AtomicU64,
}

impl VpinState {
    pub fn new(bucket_size: f64, num_buckets: usize) -> Self {
        Self {
            vpin: AtomicF64::new(0.0),
            buy_volume: AtomicF64::new(0.0),
            sell_volume: AtomicF64::new(0.0),
            bucket_size: AtomicF64::new(bucket_size),
            num_buckets,
            bucket_index: AtomicU64::new(0),
            rolling_imbalance_sum: AtomicF64::new(0.0),
            filled_buckets: AtomicU64::new(0),
        }
    }

    /// Process a trade and update VPIN
    #[inline]
    pub fn process_trade(&self, volume: f64, is_buy: bool) -> f64 {
        if is_buy {
            self.buy_volume.fetch_add(volume, Ordering::Relaxed);
        } else {
            self.sell_volume.fetch_add(volume, Ordering::Relaxed);
        }

        let buy_vol = self.buy_volume.load(Ordering::Relaxed);
        let sell_vol = self.sell_volume.load(Ordering::Relaxed);
        let total_vol = buy_vol + sell_vol;
        let bucket_size = self.bucket_size.load(Ordering::Relaxed);

        // Check if bucket is full
        if total_vol >= bucket_size {
            return self.complete_bucket(buy_vol, sell_vol);
        }

        self.vpin.load(Ordering::Relaxed)
    }

    /// Complete a bucket and calculate VPIN
    #[inline]
    fn complete_bucket(&self, buy_vol: f64, sell_vol: f64) -> f64 {
        // Calculate imbalance for this bucket
        let imbalance = (buy_vol - sell_vol).abs();
        let total = buy_vol + sell_vol;
        
        let bucket_vpin = if total > 0.0 { imbalance / total } else { 0.0 };

        // Update rolling sum
        let num_buckets = self.num_buckets as f64;
        let old_sum = self.rolling_imbalance_sum.load(Ordering::Relaxed);
        let filled = self.filled_buckets.load(Ordering::Relaxed);

        let new_sum = if filled < self.num_buckets as u64 {
            // Still filling initial buckets
            old_sum + bucket_vpin
        } else {
            // Rolling window: remove oldest, add newest
            let old_avg = old_sum / num_buckets;
            old_sum - old_avg + bucket_vpin
        };

        self.rolling_imbalance_sum.store(new_sum, Ordering::Relaxed);

        // Update filled bucket count
        let current_filled = self.filled_buckets.load(Ordering::Relaxed);
        if current_filled < self.num_buckets as u64 {
            self.filled_buckets.fetch_add(1, Ordering::Relaxed);
        }

        // Increment bucket index
        self.bucket_index.fetch_add(1, Ordering::Relaxed);

        // Reset current bucket volumes
        self.buy_volume.store(0.0, Ordering::Relaxed);
        self.sell_volume.store(0.0, Ordering::Relaxed);

        // Calculate new VPIN
        let actual_buckets = self.filled_buckets.load(Ordering::Relaxed) as f64;
        let new_vpin = new_sum / actual_buckets;
        self.vpin.store(new_vpin, Ordering::Relaxed);

        new_vpin
    }

    /// Get current VPIN value
    #[inline]
    pub fn get_vpin(&self) -> f64 {
        self.vpin.load(Ordering::Relaxed)
    }

    /// Reset state
    #[inline]
    pub fn reset(&self) {
        self.vpin.store(0.0, Ordering::Relaxed);
        self.buy_volume.store(0.0, Ordering::Relaxed);
        self.sell_volume.store(0.0, Ordering::Relaxed);
        self.bucket_index.store(0, Ordering::Relaxed);
        self.rolling_imbalance_sum.store(0.0, Ordering::Relaxed);
        self.filled_buckets.store(0, Ordering::Relaxed);
    }
}

/// Toxicity detection and response engine
pub struct ToxicityDetector {
    /// VPIN state
    pub vpin_state: VpinState,
    /// Toxicity threshold (VPIN above this = toxic)
    pub toxicity_threshold: AtomicF64,
    /// Critical toxicity threshold (halt trading)
    pub critical_threshold: AtomicF64,
    /// Spread widening factor when toxic
    pub spread_widening_factor: AtomicF64,
    /// Currently in toxic state
    pub is_toxic: AtomicBool,
    /// Trading halted flag
    pub trading_halted: AtomicBool,
    /// Last toxicity event timestamp
    pub last_toxic_ns: AtomicU64,
    /// Cooldown period after toxicity (ms)
    pub cooldown_ms: AtomicU64,
}

impl ToxicityDetector {
    pub fn new(
        bucket_size: f64,
        num_buckets: usize,
        toxicity_threshold: f64,
        critical_threshold: f64,
    ) -> Self {
        let now_ns = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64;
        
        Self {
            vpin_state: VpinState::new(bucket_size, num_buckets),
            toxicity_threshold: AtomicF64::new(toxicity_threshold),
            critical_threshold: AtomicF64::new(critical_threshold),
            spread_widening_factor: AtomicF64::new(1.0),
            is_toxic: AtomicBool::new(false),
            trading_halted: AtomicBool::new(false),
            last_toxic_ns: AtomicU64::new(0),
            cooldown_ms: AtomicU64::new(5000), // 5 second default cooldown
        }
    }

    /// Process a trade and check for toxicity
    #[inline]
    pub fn process_trade(&self, volume: f64, is_buy: bool) -> ToxicityState {
        let vpin = self.vpin_state.process_trade(volume, is_buy);
        
        let toxic_thresh = self.toxicity_threshold.load(Ordering::Relaxed);
        let critical_thresh = self.critical_threshold.load(Ordering::Relaxed);
        
        let now_ns = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64;

        // Check for critical toxicity
        if vpin >= critical_thresh {
            self.is_toxic.store(true, Ordering::Relaxed);
            self.trading_halted.store(true, Ordering::Relaxed);
            self.last_toxic_ns.store(now_ns, Ordering::Relaxed);
            self.spread_widening_factor.store(5.0, Ordering::Relaxed); // Max widening
            
            return ToxicityState {
                vpin,
                state: ToxicityLevel::Critical,
                action: ToxicityAction::Halt,
                spread_multiplier: 5.0,
            };
        }

        // Check for elevated toxicity
        if vpin >= toxic_thresh {
            self.is_toxic.store(true, Ordering::Relaxed);
            self.last_toxic_ns.store(now_ns, Ordering::Relaxed);
            
            // Linear scaling of spread widening
            let factor = 1.0 + (vpin - toxic_thresh) * 10.0;
            let factor = factor.min(3.0); // Cap at 3x
            self.spread_widening_factor.store(factor, Ordering::Relaxed);
            
            return ToxicityState {
                vpin,
                state: ToxicityLevel::Elevated,
                action: ToxicityAction::WidenSpread,
                spread_multiplier: factor,
            };
        }

        // Check cooldown expiration
        let cooldown_expired = self.check_cooldown();
        
        if cooldown_expired && self.is_toxic.load(Ordering::Relaxed) {
            self.is_toxic.store(false, Ordering::Relaxed);
            self.trading_halted.store(false, Ordering::Relaxed);
            self.spread_widening_factor.store(1.0, Ordering::Relaxed);
        }

        ToxicityState {
            vpin,
            state: ToxicityLevel::Normal,
            action: ToxicityAction::None,
            spread_multiplier: 1.0,
        }
    }

    /// Check if cooldown period has expired
    #[inline]
    fn check_cooldown(&self) -> bool {
        let last_ns = self.last_toxic_ns.load(Ordering::Relaxed);
        if last_ns == 0 { return true; }
        
        let now_ns = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64;
        let cooldown_ns = self.cooldown_ms.load(Ordering::Relaxed) * 1_000_000;
        
        (now_ns - last_ns) > cooldown_ns
    }

    /// Get current spread multiplier
    #[inline]
    pub fn get_spread_multiplier(&self) -> f64 {
        if self.check_cooldown() {
            return 1.0;
        }
        self.spread_widening_factor.load(Ordering::Relaxed)
    }

    /// Check if trading should proceed
    #[inline]
    pub fn can_trade(&self) -> bool {
        !self.trading_halted.load(Ordering::Relaxed) && self.check_cooldown()
    }

    /// Manually halt trading
    #[inline]
    pub fn halt_trading(&self) {
        self.trading_halted.store(true, Ordering::Relaxed);
        self.last_toxic_ns.store(
            SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos() as u64,
            Ordering::Relaxed,
        );
    }

    /// Manually resume trading
    #[inline]
    pub fn resume_trading(&self) {
        self.trading_halted.store(false, Ordering::Relaxed);
        self.is_toxic.store(false, Ordering::Relaxed);
        self.spread_widening_factor.store(1.0, Ordering::Relaxed);
    }

    /// Update thresholds dynamically
    #[inline]
    pub fn update_thresholds(&self, toxic: f64, critical: f64) {
        self.toxicity_threshold.store(toxic, Ordering::Relaxed);
        self.critical_threshold.store(critical, Ordering::Relaxed);
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ToxicityLevel {
    Normal,
    Elevated,
    Critical,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ToxicityAction {
    None,
    WidenSpread,
    Halt,
}

#[derive(Clone, Copy, Debug)]
pub struct ToxicityState {
    pub vpin: f64,
    pub state: ToxicityLevel,
    pub action: ToxicityAction,
    pub spread_multiplier: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vpin_calculation() {
        let detector = ToxicityDetector::new(100.0, 10, 0.5, 0.8);
        
        // Process balanced trades
        for _ in 0..5 {
            detector.process_trade(50.0, true);  // Buy
            detector.process_trade(50.0, false); // Sell
        }
        
        let vpin = detector.vpin_state.get_vpin();
        assert!(vpin < 0.3); // Should be low with balanced flow
    }

    #[test]
    fn test_toxic_flow_detection() {
        let detector = ToxicityDetector::new(100.0, 5, 0.5, 0.8);
        
        // Process heavily one-sided trades (toxic)
        for _ in 0..10 {
            detector.process_trade(100.0, true); // All buys
        }
        
        let vpin = detector.vpin_state.get_vpin();
        assert!(vpin > 0.7); // Should detect high VPIN
        
        let state = detector.process_trade(1.0, true);
        assert!(state.state == ToxicityLevel::Elevated || state.state == ToxicityLevel::Critical);
    }

    #[test]
    fn test_trading_halt() {
        let detector = ToxicityDetector::new(100.0, 3, 0.5, 0.7);
        
        // Create extremely toxic flow
        for _ in 0..20 {
            detector.process_trade(100.0, true);
        }
        
        assert!(!detector.can_trade()); // Should be halted
    }

    #[test]
    fn test_spread_widening() {
        let detector = ToxicityDetector::new(100.0, 5, 0.4, 0.8);
        
        // Normal conditions
        assert_eq!(detector.get_spread_multiplier(), 1.0);
        
        // Elevated toxicity
        for _ in 0..8 {
            detector.process_trade(100.0, true);
        }
        
        let multiplier = detector.get_spread_multiplier();
        assert!(multiplier > 1.0); // Should widen spread
    }
}
