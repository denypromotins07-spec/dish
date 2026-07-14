//! Quantitative Finance: Kalman Filters and GARCH models
//! Fast math implementations with aligned memory structs

use crossbeam::atomic::AtomicCell;

/// Kalman Filter for dynamic state estimation
#[repr(C, align(64))]
pub struct KalmanFilter {
    // State
    x: AtomicCell<f64>,      // State estimate
    p: AtomicCell<f64>,      // Error covariance
    k: AtomicCell<f64>,      // Kalman gain
    
    // Process parameters
    q: f64,                  // Process noise
    r: f64,                  // Measurement noise
    
    // For multidimensional support (simplified 1D here)
    initialized: AtomicCell<bool>,
}

impl KalmanFilter {
    #[inline]
    pub fn new(initial_state: f64, process_noise: f64, measurement_noise: f64) -> Self {
        Self {
            x: AtomicCell::new(initial_state),
            p: AtomicCell::new(1.0),
            k: AtomicCell::new(0.0),
            q: process_noise,
            r: measurement_noise,
            initialized: AtomicCell::new(false),
        }
    }

    /// Update filter with new measurement
    #[inline]
    pub fn update(&self, measurement: f64) -> f64 {
        let mut x = self.x.load();
        let mut p = self.p.load();
        
        if !self.initialized.load() {
            self.initialized.store(true);
            self.x.store(measurement);
            return measurement;
        }

        // Prediction step (for random walk model, prediction = previous state)
        let x_pred = x;
        let p_pred = p + self.q;

        // Update step
        let k = p_pred / (p_pred + self.r);
        let x_new = x_pred + k * (measurement - x_pred);
        let p_new = (1.0 - k) * p_pred;

        self.x.store(x_new);
        self.p.store(p_new);
        self.k.store(k);

        x_new
    }

    #[inline]
    pub fn state(&self) -> f64 {
        self.x.load()
    }

    #[inline]
    pub fn covariance(&self) -> f64 {
        self.p.load()
    }

    #[inline]
    pub fn gain(&self) -> f64 {
        self.k.load()
    }

    /// Reset filter
    #[inline]
    pub fn reset(&self, initial_state: f64) {
        self.x.store(initial_state);
        self.p.store(1.0);
        self.k.store(0.0);
        self.initialized.store(false);
    }
}

/// Extended Kalman Filter for non-linear systems
#[repr(C, align(64))]
pub struct ExtendedKalmanFilter {
    base: KalmanFilter,
    linearization_point: AtomicCell<f64>,
}

impl ExtendedKalmanFilter {
    #[inline]
    pub fn new(initial_state: f64, q: f64, r: f64) -> Self {
        Self {
            base: KalmanFilter::new(initial_state, q, r),
            linearization_point: AtomicCell::new(initial_state),
        }
    }

    /// Update with non-linear measurement function
    #[inline]
    pub fn update_nonlinear<F>(&self, measurement: f64, h: F, dh: F) -> f64
    where
        F: Fn(f64) -> f64,
    {
        let x = self.base.state();
        self.linearization_point.store(x);

        // Linearize around current state
        let h_x = h(x);
        let h_jacobian = dh(x);

        // Modified innovation
        let innovation = measurement - h_x;

        // Standard Kalman update with linearized model
        let mut p = self.base.covariance();
        let q = self.base.q;
        let r = self.base.r;

        let p_pred = p + q;
        let s = h_jacobian * h_jacobian * p_pred + r;
        let k = p_pred * h_jacobian / s;

        let x_new = x + k * innovation;
        let p_new = (1.0 - k * h_jacobian) * p_pred;

        self.base.x.store(x_new);
        self.base.p.store(p_new);

        x_new
    }

    #[inline]
    pub fn state(&self) -> f64 {
        self.base.state()
    }
}

/// GARCH(1,1) model for volatility forecasting
#[repr(C, align(64))]
pub struct GARCH11 {
    // Parameters
    omega: f64,   // Constant term
    alpha: f64,   // ARCH term coefficient
    beta: f64,    // GARCH term coefficient
    
    // State
    sigma_sq: AtomicCell<f64>,  // Conditional variance
    epsilon_sq: AtomicCell<f64>, // Last squared residual
    
    // Long-run variance
    long_run_variance: AtomicCell<f64>,
    initialized: AtomicCell<bool>,
}

impl GARCH11 {
    /// Create GARCH(1,1) model with constraints: alpha + beta < 1
    #[inline]
    pub fn new(omega: f64, alpha: f64, beta: f64) -> Option<Self> {
        if alpha < 0.0 || beta < 0.0 || omega <= 0.0 {
            return None;
        }
        if alpha + beta >= 1.0 {
            return None; // Non-stationary
        }

        let long_run_var = omega / (1.0 - alpha - beta);

        Some(Self {
            omega,
            alpha,
            beta,
            sigma_sq: AtomicCell::new(long_run_var),
            epsilon_sq: AtomicCell::new(0.0),
            long_run_variance: AtomicCell::new(long_run_var),
            initialized: AtomicCell::new(false),
        })
    }

    /// Update with new return/residual
    #[inline]
    pub fn update(&self, epsilon: f64) -> f64 {
        let epsilon_sq = epsilon * epsilon;
        let prev_sigma_sq = self.sigma_sq.load();

        if !self.initialized.load() {
            self.initialized.store(true);
            self.epsilon_sq.store(epsilon_sq);
            return prev_sigma_sq.sqrt();
        }

        // GARCH(1,1): sigma²_t = ω + α*ε²_{t-1} + β*σ²_{t-1}
        let new_sigma_sq = self.omega 
            + self.alpha * self.epsilon_sq.load() 
            + self.beta * prev_sigma_sq;

        self.sigma_sq.store(new_sigma_sq);
        self.epsilon_sq.store(epsilon_sq);

        new_sigma_sq.sqrt()
    }

    /// Forecast volatility h steps ahead
    #[inline]
    pub fn forecast(&self, h: usize) -> f64 {
        let current_sigma_sq = self.sigma_sq.load();
        let lr_var = self.long_run_variance.load();
        let ab = self.alpha + self.beta;

        // E[sigma²_{t+h}] = lr_var + ab^h * (sigma²_t - lr_var)
        let forecast_var = lr_var + ab.powi(h as i32) * (current_sigma_sq - lr_var);
        forecast_var.sqrt()
    }

    #[inline]
    pub fn current_volatility(&self) -> f64 {
        self.sigma_sq.load().sqrt()
    }

    #[inline]
    pub fn long_run_volatility(&self) -> f64 {
        self.long_run_variance.load().sqrt()
    }

    /// Get persistence (alpha + beta)
    #[inline]
    pub fn persistence(&self) -> f64 {
        self.alpha + self.beta
    }
}

/// EGARCH model for asymmetric volatility (leverage effect)
#[repr(C, align(64))]
pub struct EGARCH {
    omega: f64,
    alpha: f64,   // ARCH term
    gamma: f64,   // Leverage term
    beta: f64,    // GARCH term
    
    log_sigma: AtomicCell<f64>,
    last_z: AtomicCell<f64>,
    initialized: AtomicCell<bool>,
}

impl EGARCH {
    #[inline]
    pub fn new(omega: f64, alpha: f64, gamma: f64, beta: f64) -> Option<Self> {
        if beta >= 1.0 || beta < 0.0 {
            return None;
        }

        Some(Self {
            omega,
            alpha,
            gamma,
            beta,
            log_sigma: AtomicCell::new(omega / (1.0 - beta)),
            last_z: AtomicCell::new(0.0),
            initialized: AtomicCell::new(false),
        })
    }

    /// Update with standardized residual
    #[inline]
    pub fn update(&self, z: f64) -> f64 {
        let prev_log_sigma = self.log_sigma.load();

        if !self.initialized.load() {
            self.initialized.store(true);
            self.last_z.store(z);
            return prev_log_sigma.exp();
        }

        // EGARCH: log(σ²_t) = ω + α*(|z_{t-1}| - E[|z|]) + γ*z_{t-1} + β*log(σ²_{t-1})
        // Assuming z ~ N(0,1), E[|z|] ≈ 0.798
        let e_abs_z = 0.7978845608028654; // sqrt(2/pi)
        
        let log_sigma_sq_new = self.omega 
            + self.alpha * ((z.abs() - e_abs_z))
            + self.gamma * z 
            + self.beta * prev_log_sigma;

        self.log_sigma.store(log_sigma_sq_new);
        self.last_z.store(z);

        log_sigma_sq_new.exp()
    }

    #[inline]
    pub fn volatility(&self) -> f64 {
        self.log_sigma.load().exp()
    }
}

/// Rolling statistics for quantitative analysis
#[repr(C, align(64))]
pub struct RollingStats<const N: usize> {
    buffer: [f64; N],
    head: AtomicCell<usize>,
    sum: AtomicCell<f64>,
    sum_sq: AtomicCell<f64>,
    count: AtomicCell<usize>,
}

impl<const N: usize> RollingStats<N> {
    #[inline]
    pub const fn new() -> Self {
        Self {
            buffer: [0.0; N],
            head: AtomicCell::new(0),
            sum: AtomicCell::new(0.0),
            sum_sq: AtomicCell::new(0.0),
            count: AtomicCell::new(0),
        }
    }

    #[inline]
    pub fn update(&self, value: f64) {
        let idx = self.head.load();
        let old = unsafe { *self.buffer.get_unchecked(idx) };
        unsafe { *self.buffer.get_unchecked_mut(idx) = value };
        
        self.head.store((idx + 1) % N);

        // Update sums atomically
        let mut sum = self.sum.load();
        let mut sum_sq = self.sum_sq.load();
        
        loop {
            let new_sum = sum - old + value;
            let new_sum_sq = sum_sq - old * old + value * value;
            
            match (self.sum.compare_exchange(sum, new_sum), self.sum_sq.compare_exchange(sum_sq, new_sum_sq)) {
                (Ok(_), Ok(_)) => break,
                (Err(s), Err(sq)) | (Ok(_), Err(sq)) | (Err(s), Ok(_)) => {
                    sum = self.sum.load();
                    sum_sq = self.sum_sq.load();
                }
            }
        }

        let cnt = self.count.load();
        if cnt < N {
            self.count.store(cnt + 1);
        }
    }

    #[inline]
    pub fn mean(&self) -> f64 {
        let cnt = self.count.load().min(N);
        if cnt == 0 { return 0.0; }
        self.sum.load() / cnt as f64
    }

    #[inline]
    pub fn variance(&self) -> f64 {
        let cnt = self.count.load().min(N);
        if cnt < 2 { return 0.0; }
        
        let mean = self.mean();
        let sum_sq = self.sum_sq.load();
        let n = cnt as f64;
        
        // Sample variance with Bessel's correction
        (sum_sq - n * mean * mean) / (n - 1.0)
    }

    #[inline]
    pub fn std(&self) -> f64 {
        self.variance().sqrt()
    }

    #[inline]
    pub fn skewness(&self) -> f64 {
        let cnt = self.count.load().min(N);
        if cnt < 3 { return 0.0; }

        let mean = self.mean();
        let std = self.std();
        if std == 0.0 { return 0.0; }

        let n = cnt as f64;
        let mut sum_cubed = 0.0;
        
        for i in 0..cnt {
            let val = unsafe { *self.buffer.get_unchecked(i) };
            let diff = (val - mean) / std;
            sum_cubed += diff * diff * diff;
        }

        // Adjusted Fisher-Pearson coefficient
        (n / ((n - 1.0) * (n - 2.0))) * sum_cubed
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kalman_filter() {
        let kf = KalmanFilter::new(0.0, 0.01, 0.1);
        
        // Simulate noisy measurements of constant value 10
        for i in 0..20 {
            let measurement = 10.0 + (i as f64 * 0.1 - 1.0).sin();
            kf.update(measurement);
        }
        
        let state = kf.state();
        assert!((state - 10.0).abs() < 1.0);
    }

    #[test]
    fn test_garch() {
        let garch = GARCH11::new(0.00001, 0.1, 0.85).unwrap();
        
        // Simulate returns
        for i in 0..100 {
            let ret = (i as f64 * 0.1).sin() * 0.02;
            garch.update(ret);
        }
        
        let vol = garch.current_volatility();
        assert!(vol > 0.0);
    }

    #[test]
    fn test_rolling_stats() {
        let stats: RollingStats<10> = RollingStats::new();
        
        for i in 1..=10 {
            stats.update(i as f64);
        }
        
        assert!((stats.mean() - 5.5).abs() < 0.01);
        assert!(stats.std() > 0.0);
    }
}
