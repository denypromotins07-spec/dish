//! Statistical Arbitrage: Cointegration, Correlation, and Z-score mean reversion
//! Engle-Granger two-step method for pairs trading

use crossbeam::atomic::AtomicCell;
use std::sync::Arc;

/// Rolling correlation calculator
#[repr(C, align(64))]
pub struct RollingCorrelation<const N: usize> {
    x_buffer: [f64; N],
    y_buffer: [f64; N],
    head: AtomicCell<usize>,
    sum_x: AtomicCell<f64>,
    sum_y: AtomicCell<f64>,
    sum_xy: AtomicCell<f64>,
    sum_x2: AtomicCell<f64>,
    sum_y2: AtomicCell<f64>,
    count: AtomicCell<usize>,
}

impl<const N: usize> RollingCorrelation<N> {
    #[inline]
    pub const fn new() -> Self {
        Self {
            x_buffer: [0.0; N],
            y_buffer: [0.0; N],
            head: AtomicCell::new(0),
            sum_x: AtomicCell::new(0.0),
            sum_y: AtomicCell::new(0.0),
            sum_xy: AtomicCell::new(0.0),
            sum_x2: AtomicCell::new(0.0),
            sum_y2: AtomicCell::new(0.0),
            count: AtomicCell::new(0),
        }
    }

    #[inline]
    pub fn update(&self, x: f64, y: f64) {
        let idx = self.head.load();
        
        let old_x = unsafe { *self.x_buffer.get_unchecked(idx) };
        let old_y = unsafe { *self.y_buffer.get_unchecked(idx) };
        
        unsafe {
            *self.x_buffer.get_unchecked_mut(idx) = x;
            *self.y_buffer.get_unchecked_mut(idx) = y;
        }
        
        self.head.store((idx + 1) % N);

        // Update sums atomically (simplified - in production use more sophisticated locking)
        let mut sum_x = self.sum_x.load();
        let mut sum_y = self.sum_y.load();
        let mut sum_xy = self.sum_xy.load();
        let mut sum_x2 = self.sum_x2.load();
        let mut sum_y2 = self.sum_y2.load();

        sum_x = sum_x - old_x + x;
        sum_y = sum_y - old_y + y;
        sum_xy = sum_xy - old_x * old_y + x * y;
        sum_x2 = sum_x2 - old_x * old_x + x * x;
        sum_y2 = sum_y2 - old_y * old_y + y * y;

        self.sum_x.store(sum_x);
        self.sum_y.store(sum_y);
        self.sum_xy.store(sum_xy);
        self.sum_x2.store(sum_x2);
        self.sum_y2.store(sum_y2);

        let cnt = self.count.load();
        if cnt < N {
            self.count.store(cnt + 1);
        }
    }

    #[inline]
    pub fn correlation(&self) -> f64 {
        let n = self.count.load().min(N) as f64;
        if n < 2.0 { return 0.0; }

        let sum_x = self.sum_x.load();
        let sum_y = self.sum_y.load();
        let sum_xy = self.sum_xy.load();
        let sum_x2 = self.sum_x2.load();
        let sum_y2 = self.sum_y2.load();

        let numerator = n * sum_xy - sum_x * sum_y;
        let denom_x = n * sum_x2 - sum_x * sum_x;
        let denom_y = n * sum_y2 - sum_y * sum_y;

        if denom_x <= 0.0 || denom_y <= 0.0 {
            return 0.0;
        }

        let denominator = (denom_x * denom_y).sqrt();
        if denominator == 0.0 {
            return 0.0;
        }

        numerator / denominator
    }

    #[inline]
    pub fn count(&self) -> usize {
        self.count.load()
    }
}

/// OLS regression results
#[derive(Clone, Copy)]
pub struct RegressionResult {
    pub alpha: f64,
    pub beta: f64,
    pub r_squared: f64,
    pub residual_std: f64,
}

/// Rolling OLS regression for cointegration analysis
#[repr(C, align(64))]
pub struct RollingRegression<const N: usize> {
    x_buffer: [f64; N],
    y_buffer: [f64; N],
    head: AtomicCell<usize>,
    sum_x: AtomicCell<f64>,
    sum_y: AtomicCell<f64>,
    sum_xy: AtomicCell<f64>,
    sum_x2: AtomicCell<f64>,
    sum_y2: AtomicCell<f64>,
    count: AtomicCell<usize>,
}

impl<const N: usize> RollingRegression<N> {
    #[inline]
    pub const fn new() -> Self {
        Self {
            x_buffer: [0.0; N],
            y_buffer: [0.0; N],
            head: AtomicCell::new(0),
            sum_x: AtomicCell::new(0.0),
            sum_y: AtomicCell::new(0.0),
            sum_xy: AtomicCell::new(0.0),
            sum_x2: AtomicCell::new(0.0),
            sum_y2: AtomicCell::new(0.0),
            count: AtomicCell::new(0),
        }
    }

    #[inline]
    pub fn update(&self, x: f64, y: f64) -> RegressionResult {
        let idx = self.head.load();
        
        let old_x = unsafe { *self.x_buffer.get_unchecked(idx) };
        let old_y = unsafe { *self.y_buffer.get_unchecked(idx) };
        
        unsafe {
            *self.x_buffer.get_unchecked_mut(idx) = x;
            *self.y_buffer.get_unchecked_mut(idx) = y;
        }
        
        self.head.store((idx + 1) % N);

        let mut sum_x = self.sum_x.load();
        let mut sum_y = self.sum_y.load();
        let mut sum_xy = self.sum_xy.load();
        let mut sum_x2 = self.sum_x2.load();
        let mut sum_y2 = self.sum_y2.load();

        sum_x = sum_x - old_x + x;
        sum_y = sum_y - old_y + y;
        sum_xy = sum_xy - old_x * old_y + x * y;
        sum_x2 = sum_x2 - old_x * old_x + x * x;
        sum_y2 = sum_y2 - old_y * old_y + y * y;

        self.sum_x.store(sum_x);
        self.sum_y.store(sum_y);
        self.sum_xy.store(sum_xy);
        self.sum_x2.store(sum_x2);
        self.sum_y2.store(sum_y2);

        let cnt = self.count.load();
        if cnt < N {
            self.count.store(cnt + 1);
        }

        self.compute_regression()
    }

    #[inline]
    fn compute_regression(&self) -> RegressionResult {
        let n = self.count.load().min(N) as f64;
        if n < 2.0 {
            return RegressionResult {
                alpha: 0.0,
                beta: 0.0,
                r_squared: 0.0,
                residual_std: 0.0,
            };
        }

        let sum_x = self.sum_x.load();
        let sum_y = self.sum_y.load();
        let sum_xy = self.sum_xy.load();
        let sum_x2 = self.sum_x2.load();
        let sum_y2 = self.sum_y2.load();

        let mean_x = sum_x / n;
        let mean_y = sum_y / n;

        let ss_xx = sum_x2 - n * mean_x * mean_x;
        let ss_xy = sum_xy - n * mean_x * mean_y;
        let ss_yy = sum_y2 - n * mean_y * mean_y;

        let beta = if ss_xx != 0.0 { ss_xy / ss_xx } else { 0.0 };
        let alpha = mean_y - beta * mean_x;

        let r_squared = if ss_xx > 0.0 && ss_yy > 0.0 {
            (ss_xy * ss_xy) / (ss_xx * ss_yy)
        } else {
            0.0
        };

        // Residual standard error
        let sse = ss_yy - beta * ss_xy;
        let residual_std = if n > 2.0 { (sse / (n - 2.0)).sqrt() } else { 0.0 };

        RegressionResult {
            alpha,
            beta,
            r_squared,
            residual_std,
        }
    }
}

/// Engle-Granger cointegration test
#[repr(C, align(64))]
pub struct CointegrationTest<const N: usize> {
    regression: RollingRegression<N>,
    residuals: [f64; N],
    res_head: AtomicCell<usize>,
    hedge_ratio: AtomicCell<f64>,
    is_cointegrated: AtomicCell<bool>,
    adf_statistic: AtomicCell<f64>,
}

impl<const N: usize> CointegrationTest<N> {
    #[inline]
    pub const fn new() -> Self {
        Self {
            regression: RollingRegression::new(),
            residuals: [0.0; N],
            res_head: AtomicCell::new(0),
            hedge_ratio: AtomicCell::new(0.0),
            is_cointegrated: AtomicCell::new(false),
            adf_statistic: AtomicCell::new(0.0),
        }
    }

    /// Update with new price pair and compute cointegration metrics
    #[inline]
    pub fn update(&self, price_x: f64, price_y: f64) -> (f64, bool) {
        // Run regression to get hedge ratio
        let result = self.regression.update(price_x, price_y);
        self.hedge_ratio.store(result.beta);

        // Compute residual (spread)
        let spread = price_y - result.alpha - result.beta * price_x;

        // Store residual
        let idx = self.res_head.load();
        unsafe { *self.residuals.get_unchecked_mut(idx) = spread };
        self.res_head.store((idx + 1) % N);

        // Simplified ADF-like test (in production, use full ADF)
        let is_coint = self.test_stationarity(spread, result.r_squared);
        self.is_cointegrated.store(is_coint);

        (spread, is_coint)
    }

    #[inline]
    fn test_stationarity(&self, current_residual: f64, r_squared: f64) -> bool {
        // Simplified stationarity test based on:
        // 1. High R-squared indicates strong linear relationship
        // 2. Check if residuals are mean-reverting
        
        if r_squared < 0.7 {
            return false;
        }

        // Check mean reversion of residuals
        let mut sum_res = 0.0;
        let count = self.res_head.load().min(N);
        
        for i in 0..count {
            sum_res += unsafe { *self.residuals.get_unchecked(i) };
        }
        
        let mean_res = sum_res / count as f64;
        
        // Simple test: current residual within 2 std of mean
        let mut sum_sq = 0.0;
        for i in 0..count {
            let diff = unsafe { *self.residuals.get_unchecked(i) } - mean_res;
            sum_sq += diff * diff;
        }
        
        let std_res = (sum_sq / count as f64).sqrt();
        
        if std_res == 0.0 {
            return true;
        }

        let z_score = (current_residual - mean_res).abs() / std_res;
        z_score < 2.5 // Within ~99% confidence
    }

    #[inline]
    pub fn hedge_ratio(&self) -> f64 {
        self.hedge_ratio.load()
    }

    #[inline]
    pub fn is_cointegrated(&self) -> bool {
        self.is_cointegrated.load()
    }

    #[inline]
    pub fn spread_zscore(&self) -> f64 {
        let count = self.res_head.load().min(N);
        if count == 0 { return 0.0; }

        let mut sum = 0.0;
        for i in 0..count {
            sum += unsafe { *self.residuals.get_unchecked(i) };
        }
        let mean = sum / count as f64;

        let mut sum_sq = 0.0;
        for i in 0..count {
            let diff = unsafe { *self.residuals.get_unchecked(i) } - mean;
            sum_sq += diff * diff;
        }
        let std = (sum_sq / count as f64).sqrt();

        if std == 0.0 { return 0.0; }

        let latest_idx = if count == 0 { 0 } else { (self.res_head.load().wrapping_sub(1)) % N };
        let latest = unsafe { *self.residuals.get_unchecked(latest_idx) };

        (latest - mean) / std
    }
}

/// Pairs trading signal generator
#[repr(C, align(64))]
pub struct PairsTrader<const N: usize> {
    coint_test: CointegrationTest<N>,
    entry_threshold: f64,
    exit_threshold: f64,
    position: AtomicCell<i8>, // -1: short spread, 0: flat, 1: long spread
    entry_price: AtomicCell<f64>,
}

impl<const N: usize> PairsTrader<N> {
    #[inline]
    pub fn new(entry_threshold: f64, exit_threshold: f64) -> Self {
        Self {
            coint_test: CointegrationTest::new(),
            entry_threshold,
            exit_threshold,
            position: AtomicCell::new(0),
            entry_price: AtomicCell::new(0.0),
        }
    }

    /// Update prices and get trading signal
    #[inline]
    pub fn update(&self, price_x: f64, price_y: f64) -> i8 {
        let (spread, is_coint) = self.coint_test.update(price_x, price_y);
        let z_score = self.coint_test.spread_zscore();

        if !is_coint {
            self.position.store(0);
            return 0;
        }

        let current_pos = self.position.load();

        // Entry signals
        if current_pos == 0 {
            if z_score > self.entry_threshold {
                // Short the spread (sell Y, buy X)
                self.position.store(-1);
                self.entry_price.store(spread);
                return -1;
            } else if z_score < -self.entry_threshold {
                // Long the spread (buy Y, sell X)
                self.position.store(1);
                self.entry_price.store(spread);
                return 1;
            }
        }

        // Exit signals
        if current_pos != 0 {
            if z_score.abs() < self.exit_threshold {
                self.position.store(0);
                return 0;
            }
            // Stop loss at extreme levels
            if z_score.abs() > 4.0 {
                self.position.store(0);
                return 0;
            }
        }

        current_pos
    }

    #[inline]
    pub fn position(&self) -> i8 {
        self.position.load()
    }

    #[inline]
    pub fn hedge_ratio(&self) -> f64 {
        self.coint_test.hedge_ratio()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_correlation() {
        let corr: RollingCorrelation<20> = RollingCorrelation::new();
        
        // Perfect positive correlation
        for i in 0..20 {
            corr.update(i as f64, i as f64);
        }
        assert!(corr.correlation() > 0.99);
    }

    #[test]
    fn test_regression() {
        let reg: RollingRegression<20> = RollingRegression::new();
        
        for i in 0..20 {
            let x = i as f64;
            let y = 2.0 * x + 1.0; // y = 2x + 1
            reg.update(x, y);
        }
        
        let result = reg.compute_regression();
        assert!((result.beta - 2.0).abs() < 0.01);
        assert!(result.r_squared > 0.99);
    }
}
