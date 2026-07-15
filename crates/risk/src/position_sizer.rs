//! Microsecond position sizing engine implementing Kelly Criterion, Volatility Targeting, and Fixed Fractional sizing.
//! Uses lock-free atomic state to read current equity and calculate exact order quantities without blocking.

use std::sync::atomic::{AtomicF64, AtomicU64, Ordering};
use std::time::Instant;

/// Lock-free position sizing calculator optimized for microsecond execution
pub struct PositionSizer {
    /// Current account equity (lock-free atomic)
    equity: AtomicF64,
    /// Risk-free rate for Kelly calculations
    risk_free_rate: AtomicF64,
    /// Maximum position size as fraction of equity
    max_position_fraction: AtomicF64,
    /// Volatility target (annualized)
    vol_target: AtomicF64,
    /// Kelly multiplier (0.0 to 1.0)
    kelly_fraction: AtomicF64,
    /// Last calculation timestamp for latency tracking
    last_calc_ts: AtomicU64,
}

/// Position sizing result with metadata
#[derive(Debug, Clone)]
pub struct PositionSizeResult {
    pub quantity: f64,
    pub notional_value: f64,
    pub risk_amount: f64,
    pub kelly_fraction_used: f64,
    pub volatility_adjusted: bool,
    pub calculation_latency_ns: u64,
}

impl PositionSizer {
    /// Create a new position sizer with default parameters
    pub fn new(initial_equity: f64) -> Self {
        Self {
            equity: AtomicF64::new(initial_equity),
            risk_free_rate: AtomicF64::new(0.02), // 2% annual
            max_position_fraction: AtomicF64::new(0.25), // Max 25% per position
            vol_target: AtomicF64::new(0.15), // 15% annual volatility target
            kelly_fraction: AtomicF64::new(0.5), // Half-Kelly for safety
            last_calc_ts: AtomicU64::new(0),
        }
    }

    /// Update current equity (lock-free, called on every PnL update)
    #[inline(always)]
    pub fn update_equity(&self, new_equity: f64) {
        self.equity.store(new_equity, Ordering::Relaxed);
    }

    /// Get current equity reading
    #[inline(always)]
    pub fn get_equity(&self) -> f64 {
        self.equity.load(Ordering::Relaxed)
    }

    /// Calculate position size using Kelly Criterion
    /// Kelly % = W - [(1-W)/R] where W=win probability, R=win/loss ratio
    #[inline(always)]
    pub fn kelly_criterion(&self, win_probability: f64, win_loss_ratio: f64) -> f64 {
        if win_loss_ratio <= 0.0 || win_probability <= 0.0 || win_probability >= 1.0 {
            return 0.0;
        }
        
        let kelly = win_probability - ((1.0 - win_probability) / win_loss_ratio);
        let kelly_frac = self.kelly_fraction.load(Ordering::Relaxed);
        
        // Apply fractional Kelly and cap at max position
        (kelly * kelly_frac).max(0.0).min(self.max_position_fraction.load(Ordering::Relaxed))
    }

    /// Calculate position size using Volatility Targeting
    /// Adjusts position based on current volatility vs target volatility
    #[inline(always)]
    pub fn volatility_targeting(&self, current_volatility: f64, price: f64) -> f64 {
        let equity = self.equity.load(Ordering::Relaxed);
        let vol_target = self.vol_target.load(Ordering::Relaxed);
        let max_fraction = self.max_position_fraction.load(Ordering::Relaxed);
        
        if current_volatility <= 0.0 || price <= 0.0 {
            return 0.0;
        }
        
        // Volatility scaling: reduce position when vol is high
        let vol_scalar = (vol_target / current_volatility).min(2.0).max(0.1);
        let target_notional = equity * max_fraction * vol_scalar;
        
        target_notional / price
    }

    /// Fixed Fractional position sizing (simple risk-per-trade)
    #[inline(always)]
    pub fn fixed_fractional(&self, risk_per_trade: f64, stop_loss_distance: f64, price: f64) -> f64 {
        let equity = self.equity.load(Ordering::Relaxed);
        
        if stop_loss_distance <= 0.0 || price <= 0.0 {
            return 0.0;
        }
        
        let risk_amount = equity * risk_per_trade;
        let shares = risk_amount / stop_loss_distance;
        
        shares.min((equity * self.max_position_fraction.load(Ordering::Relaxed)) / price)
    }

    /// Combined position sizing with all methods and constraints
    pub fn calculate_optimal_size(
        &self,
        win_prob: f64,
        win_loss_ratio: f64,
        current_vol: f64,
        price: f64,
        stop_loss_pct: f64,
        method: SizingMethod,
    ) -> PositionSizeResult {
        let start = Instant::now();
        let equity = self.equity.load(Ordering::Relaxed);
        let max_fraction = self.max_position_fraction.load(Ordering::Relaxed);
        
        let (quantity, risk_amount, kelly_frac_used, vol_adjusted) = match method {
            SizingMethod::Kelly => {
                let kelly_frac = self.kelly_criterion(win_prob, win_loss_ratio);
                let notional = equity * kelly_frac;
                let qty = notional / price;
                (qty, notional * stop_loss_pct, kelly_frac, false)
            }
            SizingMethod::Volatility => {
                let qty = self.volatility_targeting(current_vol, price);
                let notional = qty * price;
                (qty, notional * stop_loss_pct, max_fraction, true)
            }
            SizingMethod::FixedFractional => {
                let qty = self.fixed_fractional(0.01, price * stop_loss_pct, price); // 1% risk
                let notional = qty * price;
                (qty, notional * stop_loss_pct, max_fraction, false)
            }
            SizingMethod::Conservative => {
                // Use minimum of all three methods
                let kelly_qty = self.kelly_criterion(win_prob, win_loss_ratio) * equity / price;
                let vol_qty = self.volatility_targeting(current_vol, price);
                let fixed_qty = self.fixed_fractional(0.005, price * stop_loss_pct, price); // 0.5% risk
                
                let qty = kelly_qty.min(vol_qty).min(fixed_qty);
                let notional = qty * price;
                (qty, notional * stop_loss_pct, max_fraction, true)
            }
        };
        
        let latency_ns = start.elapsed().as_nanos() as u64;
        self.last_calc_ts.store(latency_ns, Ordering::Relaxed);
        
        PositionSizeResult {
            quantity,
            notional_value: quantity * price,
            risk_amount,
            kelly_fraction_used: kelly_frac_used,
            volatility_adjusted: vol_adjusted,
            calculation_latency_ns: latency_ns,
        }
    }

    /// Set Kelly fraction multiplier (thread-safe)
    #[inline(always)]
    pub fn set_kelly_fraction(&self, fraction: f64) {
        self.kelly_fraction.store(fraction.clamp(0.0, 1.0), Ordering::Relaxed);
    }

    /// Set volatility target (thread-safe)
    #[inline(always)]
    pub fn set_vol_target(&self, vol: f64) {
        self.vol_target.store(vol.clamp(0.01, 1.0), Ordering::Relaxed);
    }

    /// Set maximum position fraction (thread-safe)
    #[inline(always)]
    pub fn set_max_position_fraction(&self, fraction: f64) {
        self.max_position_fraction.store(fraction.clamp(0.01, 1.0), Ordering::Relaxed);
    }
}

/// Available position sizing methods
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SizingMethod {
    Kelly,
    Volatility,
    FixedFractional,
    Conservative,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kelly_criterion() {
        let sizer = PositionSizer::new(100_000.0);
        // 60% win rate, 2:1 reward/risk
        let kelly = sizer.kelly_criterion(0.6, 2.0);
        assert!(kelly > 0.0 && kelly < 1.0);
    }

    #[test]
    fn test_volatility_targeting() {
        let sizer = PositionSizer::new(100_000.0);
        let qty = sizer.volatility_targeting(0.25, 50_000.0); // High vol, BTC price
        assert!(qty > 0.0);
    }

    #[test]
    fn test_combined_sizing() {
        let sizer = PositionSizer::new(100_000.0);
        let result = sizer.calculate_optimal_size(
            0.55, 0.25, 1.8, 50_000.0, 0.02, SizingMethod::Kelly
        );
        assert!(result.quantity > 0.0);
        assert!(result.calculation_latency_ns < 10_000); // Sub-10 microsecond
    }
}
