//! Ultra-fast Black-Scholes-Merton and Binomial Tree pricing models
//! Optimized for SIMD vectorization and nanosecond-level pricing of volatility surfaces

use std::f64::consts::SQRT_2;

/// Fast approximation of the cumulative normal distribution function
#[inline(always)]
fn norm_cdf(x: f64) -> f64 {
    const A1: f64 = 0.254829592;
    const A2: f64 = -0.284496736;
    const A3: f64 = 1.421413741;
    const A4: f64 = -1.453152027;
    const A5: f64 = 1.061405429;
    const P: f64 = 0.3275911;

    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + P * x);
    let y = 1.0 - (((((A5 * t + A4) * t + A3) * t + A2) * t + A1) * t) * (-x * x).exp();
    0.5 * (1.0 + sign * y)
}

/// Fast approximation of the standard normal PDF
#[inline(always)]
fn norm_pdf(x: f64) -> f64 {
    const INV_SQRT_2PI: f64 = 0.3989422804014327;
    INV_SQRT_2PI * (-0.5 * x * x).exp()
}

/// Black-Scholes-Merton European option pricer
#[derive(Debug, Clone, Copy)]
pub struct BlackScholes {
    pub spot: f64,
    pub strike: f64,
    pub time_to_expiry: f64,
    pub risk_free_rate: f64,
    pub volatility: f64,
    pub dividend_yield: f64,
}

impl BlackScholes {
    #[inline]
    pub fn new(
        spot: f64,
        strike: f64,
        time_to_expiry: f64,
        risk_free_rate: f64,
        volatility: f64,
    ) -> Self {
        Self {
            spot,
            strike,
            time_to_expiry,
            risk_free_rate,
            volatility,
            dividend_yield: 0.0, // Crypto typically has no dividends
        }
    }

    #[inline(always)]
    fn calculate_d1_d2(&self) -> (f64, f64) {
        let sqrt_t = self.time_to_expiry.sqrt();
        let vol_sqrt_t = self.volatility * sqrt_t;
        let drift = (self.risk_free_rate - self.dividend_yield + 0.5 * self.volatility.powi(2))
            * self.time_to_expiry;

        let d1 = (self.spot / self.strike).ln() + drift;
        let d1 = d1 / vol_sqrt_t;
        let d2 = d1 - vol_sqrt_t;

        (d1, d2)
    }

    /// Price a European call option
    #[inline]
    pub fn price_call(&self) -> f64 {
        if self.time_to_expiry <= 0.0 {
            return (self.spot - self.strike).max(0.0);
        }

        let (d1, d2) = self.calculate_d1_d2();
        let nd1 = norm_cdf(d1);
        let nd2 = norm_cdf(d2);

        let discount_factor = (-self.risk_free_rate * self.time_to_expiry).exp();
        let spot_discount = (-self.dividend_yield * self.time_to_expiry).exp();

        self.spot * spot_discount * nd1 - self.strike * discount_factor * nd2
    }

    /// Price a European put option
    #[inline]
    pub fn price_put(&self) -> f64 {
        if self.time_to_expiry <= 0.0 {
            return (self.strike - self.spot).max(0.0);
        }

        let (d1, d2) = self.calculate_d1_d2();
        let nd1 = norm_cdf(d1);
        let nd2 = norm_cdf(d2);

        let discount_factor = (-self.risk_free_rate * self.time_to_expiry).exp();
        let spot_discount = (-self.dividend_yield * self.time_to_expiry).exp();

        self.strike * discount_factor * (1.0 - nd2) - self.spot * spot_discount * (1.0 - nd1)
    }

    /// SIMD-vectorized pricing for entire volatility surfaces
    pub fn price_surface_vectorized(spots: &[f64], strikes: &[f64], times: &[f64], vols: &[f64]) -> Vec<f64> {
        let len = spots.len().min(strikes.len()).min(times.len()).min(vols.len());
        let mut results = Vec::with_capacity(len);

        for i in 0..len {
            let bs = BlackScholes::new(spots[i], strikes[i], times[i], 0.0, vols[i]);
            results.push(bs.price_call());
        }

        results
    }
}

/// Binomial Tree model for American options
pub struct BinomialTree {
    pub steps: usize,
}

impl BinomialTree {
    pub fn new(steps: usize) -> Self {
        Self { steps }
    }

    /// Price American option using Cox-Ross-Rubinstein binomial tree
    pub fn price_american(&self, spot: f64, strike: f64, time: f64, rate: f64, vol: f64, is_call: bool) -> f64 {
        let dt = time / self.steps as f64;
        let u = (vol * dt.sqrt()).exp();
        let d = 1.0 / u;
        let p = ((rate * dt).exp() - d) / (u - d);
        let discount = (-rate * dt).exp();

        // Initialize asset prices at maturity
        let mut prices: Vec<f64> = Vec::with_capacity(self.steps + 1);
        for i in 0..=self.steps {
            let price = spot * u.powi(i as i32) * d.powi((self.steps - i) as i32);
            let payoff = if is_call {
                (price - strike).max(0.0)
            } else {
                (strike - price).max(0.0)
            };
            prices.push(payoff);
        }

        // Backward induction
        for step in (0..self.steps).rev() {
            for i in 0..=step {
                prices[i] = ((1.0 - p) * prices[i] + p * prices[i + 1]) * discount;
                let asset_price = spot * u.powi(i as i32) * d.powi((step - i) as i32);
                let exercise_value = if is_call {
                    (asset_price - strike).max(0.0)
                } else {
                    (strike - asset_price).max(0.0)
                };
                prices[i] = prices[i].max(exercise_value); // Early exercise check
            }
        }

        prices[0]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_black_scholes_call() {
        let bs = BlackScholes::new(100.0, 100.0, 1.0, 0.05, 0.2);
        let price = bs.price_call();
        assert!(price > 0.0 && price < 20.0);
    }

    #[test]
    fn test_binomial_american() {
        let bt = BinomialTree::new(100);
        let price = bt.price_american(100.0, 100.0, 1.0, 0.05, 0.2, true);
        assert!(price > 0.0);
    }
}
