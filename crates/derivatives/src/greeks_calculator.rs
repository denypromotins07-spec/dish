//! Microsecond calculation engine for first and second-order Greeks
//! Uses lock-free atomic state for portfolio risk metrics updates

use std::sync::atomic::{AtomicF64, Ordering};
use crate::black_scholes::{BlackScholes, norm_cdf, norm_pdf};

/// First and second-order Greeks for a single option position
#[derive(Debug, Clone, Copy)]
pub struct Greeks {
    pub delta: f64,
    pub gamma: f64,
    pub theta: f64,
    pub vega: f64,
    pub rho: f64,
    pub vanna: f64,
    pub volga: f64,
}

impl Greeks {
    pub fn zero() -> Self {
        Self {
            delta: 0.0,
            gamma: 0.0,
            theta: 0.0,
            vega: 0.0,
            rho: 0.0,
            vanna: 0.0,
            volga: 0.0,
        }
    }
}

/// Lock-free Greeks calculator using atomic operations
pub struct GreeksCalculator {
    portfolio_delta: AtomicF64,
    portfolio_gamma: AtomicF64,
    portfolio_theta: AtomicF64,
    portfolio_vega: AtomicF64,
}

impl GreeksCalculator {
    pub fn new() -> Self {
        Self {
            portfolio_delta: AtomicF64::new(0.0),
            portfolio_gamma: AtomicF64::new(0.0),
            portfolio_theta: AtomicF64::new(0.0),
            portfolio_vega: AtomicF64::new(0.0),
        }
    }

    /// Calculate all Greeks for a European option in microseconds
    #[inline(always)]
    pub fn calculate_greeks(&self, bs: &BlackScholes, is_call: bool) -> Greeks {
        if bs.time_to_expiry <= 0.0 || bs.volatility <= 0.0 {
            return Greeks::zero();
        }

        let sqrt_t = bs.time_to_expiry.sqrt();
        let vol_sqrt_t = bs.volatility * sqrt_t;
        let drift = (bs.risk_free_rate - bs.dividend_yield + 0.5 * bs.volatility.powi(2))
            * bs.time_to_expiry;

        let d1 = ((bs.spot / bs.strike).ln() + drift) / vol_sqrt_t;
        let d2 = d1 - vol_sqrt_t;

        let nd1 = norm_cdf(d1);
        let nd2 = norm_cdf(d2);
        let npd1 = norm_pdf(d1);

        let discount_factor = (-bs.risk_free_rate * bs.time_to_expiry).exp();
        let spot_discount = (-bs.dividend_yield * bs.time_to_expiry).exp();

        // First-order Greeks
        let delta = if is_call {
            spot_discount * nd1
        } else {
            spot_discount * (nd1 - 1.0)
        };

        let gamma = (spot_discount * npd1) / (bs.spot * bs.volatility * sqrt_t);

        let theta_term1 = -(bs.spot * spot_discount * npd1 * bs.volatility) / (2.0 * sqrt_t);
        let theta = if is_call {
            theta_term1 
                - bs.risk_free_rate * bs.strike * discount_factor * nd2
                + bs.dividend_yield * bs.spot * spot_discount * nd1
        } else {
            theta_term1 
                + bs.risk_free_rate * bs.strike * discount_factor * (1.0 - nd2)
                - bs.dividend_yield * bs.spot * spot_discount * (1.0 - nd1)
        };

        let vega = bs.spot * spot_discount * npd1 * sqrt_t / 100.0; // Per 1% vol change

        let rho = if is_call {
            bs.strike * bs.time_to_expiry * discount_factor * nd2 / 100.0
        } else {
            -bs.strike * bs.time_to_expiry * discount_factor * (1.0 - nd2) / 100.0
        };

        // Second-order Greeks
        let vanna = (npd1 * (1.0 - d1 / (bs.volatility * sqrt_t))) / (bs.volatility * 100.0);
        
        let volga = (bs.spot * spot_discount * npd1 * sqrt_t * d1 * d2) / (bs.volatility * 100.0);

        Greeks {
            delta,
            gamma,
            theta,
            vega,
            rho,
            vanna,
            volga,
        }
    }

    /// Atomically update portfolio-level risk metrics
    #[inline]
    pub fn update_portfolio_risk(&self, greeks: &Greeks, quantity: f64, is_long: bool) {
        let multiplier = if is_long { quantity } else { -quantity };
        
        self.portfolio_delta.fetch_add(greeks.delta * multiplier, Ordering::Relaxed);
        self.portfolio_gamma.fetch_add(greeks.gamma * multiplier, Ordering::Relaxed);
        self.portfolio_theta.fetch_add(greeks.theta * multiplier, Ordering::Relaxed);
        self.portfolio_vega.fetch_add(greeks.vega * multiplier, Ordering::Relaxed);
    }

    /// Get current portfolio delta (lock-free read)
    #[inline]
    pub fn get_portfolio_delta(&self) -> f64 {
        self.portfolio_delta.load(Ordering::Relaxed)
    }

    /// Get current portfolio gamma
    #[inline]
    pub fn get_portfolio_gamma(&self) -> f64 {
        self.portfolio_gamma.load(Ordering::Relaxed)
    }

    /// Get current portfolio theta
    #[inline]
    pub fn get_portfolio_theta(&self) -> f64 {
        self.portfolio_theta.load(Ordering::Relaxed)
    }

    /// Get current portfolio vega
    #[inline]
    pub fn get_portfolio_vega(&self) -> f64 {
        self.portfolio_vega.load(Ordering::Relaxed)
    }

    /// Reset all portfolio metrics atomically
    pub fn reset_portfolio(&self) {
        self.portfolio_delta.store(0.0, Ordering::Relaxed);
        self.portfolio_gamma.store(0.0, Ordering::Relaxed);
        self.portfolio_theta.store(0.0, Ordering::Relaxed);
        self.portfolio_vega.store(0.0, Ordering::Relaxed);
    }
}

/// Batch Greeks calculator for entire portfolios
pub struct BatchGreeksCalculator {
    calculator: GreeksCalculator,
}

impl BatchGreeksCalculator {
    pub fn new() -> Self {
        Self {
            calculator: GreeksCalculator::new(),
        }
    }

    /// Calculate Greeks for an entire options chain efficiently
    pub fn calculate_chain(&self, positions: &[(&BlackScholes, bool, f64, bool)]) -> Vec<Greeks> {
        let mut results = Vec::with_capacity(positions.len());

        for (bs, is_call, quantity, is_long) in positions {
            let greeks = self.calculator.calculate_greeks(bs, *is_call);
            self.calculator.update_portfolio_risk(&greeks, *quantity, *is_long);
            results.push(greeks);
        }

        results
    }

    pub fn get_portfolio_summary(&self) -> Greeks {
        Greeks {
            delta: self.calculator.get_portfolio_delta(),
            gamma: self.calculator.get_portfolio_gamma(),
            theta: self.calculator.get_portfolio_theta(),
            vega: self.calculator.get_portfolio_vega(),
            rho: 0.0,
            vanna: 0.0,
            volga: 0.0,
        }
    }
}

impl Default for GreeksCalculator {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for BatchGreeksCalculator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_greeks_calculation() {
        let calc = GreeksCalculator::new();
        let bs = BlackScholes::new(100.0, 100.0, 1.0, 0.05, 0.2);
        let greeks = calc.calculate_greeks(&bs, true);
        
        assert!(greeks.delta > 0.0 && greeks.delta < 1.0);
        assert!(greeks.gamma > 0.0);
        assert!(greeks.theta < 0.0); // Long calls have negative theta
        assert!(greeks.vega > 0.0);
    }

    #[test]
    fn test_portfolio_aggregation() {
        let calc = GreeksCalculator::new();
        let bs = BlackScholes::new(100.0, 100.0, 1.0, 0.05, 0.2);
        let greeks = calc.calculate_greeks(&bs, true);
        
        calc.update_portfolio_risk(&greeks, 10.0, true);
        assert!(calc.get_portfolio_delta() > 0.0);
    }
}
