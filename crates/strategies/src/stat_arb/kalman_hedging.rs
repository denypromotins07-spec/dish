//! Real-time Kalman Filter for Dynamic Hedge Ratio Adjustment
//! Ensures dollar-neutral and beta-neutral portfolio despite shifting correlations

use std::sync::atomic::{AtomicF64, AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

/// Kalman Filter state for hedge ratio estimation
pub struct KalmanHedgeFilter {
    /// Current hedge ratio estimate (state)
    pub hedge_ratio: AtomicF64,
    /// Error covariance
    pub p: AtomicF64,
    /// Process noise variance
    pub q: AtomicF64,
    /// Measurement noise variance
    pub r: AtomicF64,
    /// Previous price A (for returns calculation)
    pub prev_price_a: AtomicF64,
    /// Previous price B (for returns calculation)
    pub prev_price_b: AtomicF64,
    /// Last update timestamp
    pub last_update_ns: AtomicU64,
}

impl KalmanHedgeFilter {
    pub fn new(
        initial_hedge_ratio: f64,
        initial_covariance: f64,
        process_noise: f64,
        measurement_noise: f64,
    ) -> Self {
        let now_ns = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64;
        
        Self {
            hedge_ratio: AtomicF64::new(initial_hedge_ratio),
            p: AtomicF64::new(initial_covariance),
            q: AtomicF64::new(process_noise),
            r: AtomicF64::new(measurement_noise),
            prev_price_a: AtomicF64::new(0.0),
            prev_price_b: AtomicF64::new(0.0),
            last_update_ns: AtomicU64::new(now_ns),
        }
    }

    /// Update filter with new prices using returns-based observation
    #[inline]
    pub fn update(&self, price_a: f64, price_b: f64) -> f64 {
        let prev_a = self.prev_price_a.load(Ordering::Relaxed);
        let prev_b = self.prev_price_b.load(Ordering::Relaxed);
        
        // Skip if no previous prices
        if prev_a <= 0.0 || prev_b <= 0.0 {
            self.prev_price_a.store(price_a, Ordering::Relaxed);
            self.prev_price_b.store(price_b, Ordering::Relaxed);
            return self.hedge_ratio.load(Ordering::Relaxed);
        }

        // Calculate returns
        let ret_a = (price_a - prev_a) / prev_a;
        let ret_b = (price_b - prev_b) / prev_b;

        // Predict step
        let p_pred = self.p.load(Ordering::Relaxed) + self.q.load(Ordering::Relaxed);
        
        // Update step
        // Observation model: ret_a = hedge_ratio * ret_b + noise
        // Kalman gain: K = P_pred * ret_b / (ret_b^2 * P_pred + R)
        let k = (p_pred * ret_b) / (ret_b * ret_b * p_pred + self.r.load(Ordering::Relaxed));
        
        // Innovation: y = ret_a - hedge_ratio * ret_b
        let current_hedge = self.hedge_ratio.load(Ordering::Relaxed);
        let innovation = ret_a - current_hedge * ret_b;
        
        // Updated state estimate
        let new_hedge = current_hedge + k * innovation;
        self.hedge_ratio.store(new_hedge, Ordering::Relaxed);
        
        // Updated covariance
        let new_p = (1.0 - k * ret_b) * p_pred;
        self.p.store(new_p.max(1e-10), Ordering::Relaxed);
        
        // Store prices for next iteration
        self.prev_price_a.store(price_a, Ordering::Relaxed);
        self.prev_price_b.store(price_b, Ordering::Relaxed);
        
        self.last_update_ns.store(
            SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos() as u64,
            Ordering::Relaxed,
        );
        
        new_hedge
    }

    /// Get current hedge ratio
    #[inline]
    pub fn get_hedge_ratio(&self) -> f64 {
        self.hedge_ratio.load(Ordering::Relaxed)
    }

    /// Get current uncertainty (covariance)
    #[inline]
    pub fn get_uncertainty(&self) -> f64 {
        self.p.load(Ordering::Relaxed)
    }

    /// Reset filter with new parameters
    #[inline]
    pub fn reset(&self, new_hedge: f64, new_p: f64) {
        self.hedge_ratio.store(new_hedge, Ordering::Relaxed);
        self.p.store(new_p, Ordering::Relaxed);
        self.prev_price_a.store(0.0, Ordering::Relaxed);
        self.prev_price_b.store(0.0, Ordering::Relaxed);
    }

    /// Adjust process noise dynamically based on market regime
    #[inline]
    pub fn adjust_process_noise(&self, volatility_regime: f64) {
        // Higher volatility = higher process noise (faster adaptation)
        let base_q = self.q.load(Ordering::Relaxed);
        let adjusted = base_q * (1.0 + volatility_regime);
        self.q.store(adjusted, Ordering::Relaxed);
    }
}

/// Beta-neutral hedging calculator
pub struct BetaNeutralHedger {
    /// Kalman filter for dynamic beta estimation
    pub kalman: KalmanHedgeFilter,
    /// Portfolio value in asset A
    pub value_a: AtomicF64,
    /// Portfolio value in asset B
    pub value_b: AtomicF64,
    /// Target beta (usually 0 for neutral)
    pub target_beta: AtomicF64,
}

impl BetaNeutralHedger {
    pub fn new(
        initial_hedge: f64,
        initial_p: f64,
        process_noise: f64,
        measurement_noise: f64,
    ) -> Self {
        Self {
            kalman: KalmanHedgeFilter::new(initial_hedge, initial_p, process_noise, measurement_noise),
            value_a: AtomicF64::new(0.0),
            value_b: AtomicF64::new(0.0),
            target_beta: AtomicF64::new(0.0),
        }
    }

    /// Update with new prices and get recommended hedge adjustment
    #[inline]
    pub fn update_and_get_hedge(&self, price_a: f64, price_b: f64) -> HedgeRecommendation {
        let new_hedge = self.kalman.update(price_a, price_b);
        let uncertainty = self.kalman.get_uncertainty();
        
        let value_a = self.value_a.load(Ordering::Relaxed);
        let value_b = self.value_b.load(Ordering::Relaxed);
        
        // Calculate current dollar exposure
        let exposure_a = value_a;
        let exposure_b = value_b * new_hedge;
        
        // Net exposure
        let net_exposure = exposure_a - exposure_b;
        
        // Recommended adjustment to achieve neutrality
        let target_b_value = exposure_a / new_hedge;
        let adjustment_needed = target_b_value - value_b;
        
        HedgeRecommendation {
            hedge_ratio: new_hedge,
            uncertainty,
            current_exposure_a: exposure_a,
            current_exposure_b: exposure_b,
            net_exposure,
            target_b_value,
            adjustment_needed,
            timestamp_ns: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos() as u64,
        }
    }

    /// Update portfolio values
    #[inline]
    pub fn update_portfolio(&self, value_a: f64, value_b: f64) {
        self.value_a.store(value_a, Ordering::Relaxed);
        self.value_b.store(value_b, Ordering::Relaxed);
    }

    /// Set target beta (for beta-weighted portfolios)
    #[inline]
    pub fn set_target_beta(&self, beta: f64) {
        self.target_beta.store(beta, Ordering::Relaxed);
    }
}

/// Hedge recommendation result
#[derive(Clone, Copy, Debug)]
pub struct HedgeRecommendation {
    pub hedge_ratio: f64,
    pub uncertainty: f64,
    pub current_exposure_a: f64,
    pub current_exposure_b: f64,
    pub net_exposure: f64,
    pub target_b_value: f64,
    pub adjustment_needed: f64,
    pub timestamp_ns: u64,
}

impl HedgeRecommendation {
    /// Check if hedge adjustment is significant enough to act on
    #[inline]
    pub fn should_adjust(&self, threshold_pct: f64) -> bool {
        if self.current_exposure_b == 0.0 {
            return self.adjustment_needed.abs() > 0.01; // Small absolute threshold
        }
        let pct_change = (self.adjustment_needed / self.current_exposure_b).abs();
        pct_change > threshold_pct / 100.0
    }

    /// Get signal direction
    #[inline]
    pub fn get_signal(&self) -> HedgeSignal {
        if self.net_exposure > 0.01 {
            HedgeSignal::ReduceA  // Too long A, reduce or short more B
        } else if self.net_exposure < -0.01 {
            HedgeSignal::ReduceB  // Too short A or too long B
        } else {
            HedgeSignal::Neutral
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum HedgeSignal {
    ReduceA,   // Reduce exposure to asset A
    ReduceB,   // Reduce exposure to asset B
    Neutral,   // Properly hedged
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kalman_convergence() {
        let filter = KalmanHedgeFilter::new(1.0, 0.1, 0.001, 0.01);
        
        // Simulate correlated prices with known hedge ratio of 1.5
        let mut price_a = 100.0;
        let mut price_b = 100.0;
        
        for i in 0..100 {
            let shock = (i as f64 * 0.01).sin() * 0.5;
            price_a += shock * 1.5 + 0.1;
            price_b += shock + 0.05;
            
            filter.update(price_a, price_b);
        }
        
        let hedge = filter.get_hedge_ratio();
        // Should converge near 1.5
        assert!((hedge - 1.5).abs() < 0.3);
    }

    #[test]
    fn test_beta_neutral_hedger() {
        let hedger = BetaNeutralHedger::new(1.0, 0.1, 0.001, 0.01);
        
        hedger.update_portfolio(10000.0, 10000.0);
        
        let mut price_a = 100.0;
        let mut price_b = 100.0;
        
        for _ in 0..50 {
            price_a += 0.5;
            price_b += 0.3;
            let rec = hedger.update_and_get_hedge(price_a, price_b);
            
            // After warmup, should have reasonable recommendations
            if _ > 10 {
                assert!(rec.hedge_ratio > 0.0);
                assert!(rec.uncertainty > 0.0);
            }
        }
    }

    #[test]
    fn test_hedge_signal() {
        let rec = HedgeRecommendation {
            hedge_ratio: 1.0,
            uncertainty: 0.01,
            current_exposure_a: 10000.0,
            current_exposure_b: 8000.0,
            net_exposure: 2000.0,
            target_b_value: 10000.0,
            adjustment_needed: 2000.0,
            timestamp_ns: 0,
        };
        
        assert_eq!(rec.get_signal(), HedgeSignal::ReduceA);
        assert!(rec.should_adjust(10.0)); // 20% adjustment needed
    }
}
