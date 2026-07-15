//! High-performance Historical and Parametric Value at Risk (VaR) and Expected Shortfall (ES) calculator.
//! Utilizes fast math and rolling ring buffers to compute portfolio risk metrics on every tick.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicF64, AtomicUsize, Ordering};
use std::time::Instant;

/// Rolling ring buffer for returns data - lock-free design
struct RingBuffer {
    data: Vec<AtomicF64>,
    capacity: usize,
    index: AtomicUsize,
    count: AtomicUsize,
}

impl RingBuffer {
    fn new(capacity: usize) -> Self {
        let mut data = Vec::with_capacity(capacity);
        for _ in 0..capacity {
            data.push(AtomicF64::new(0.0));
        }
        Self {
            data,
            capacity,
            index: AtomicUsize::new(0),
            count: AtomicUsize::new(0),
        }
    }

    #[inline(always)]
    fn push(&self, value: f64) {
        let idx = self.index.fetch_add(1, Ordering::Relaxed) % self.capacity;
        self.data[idx].store(value, Ordering::Relaxed);
        
        if self.count.load(Ordering::Relaxed) < self.capacity {
            self.count.fetch_add(1, Ordering::Relaxed);
        }
    }

    #[inline(always)]
    fn get_all(&self) -> Vec<f64> {
        let count = self.count.load(Ordering::Relaxed);
        let mut result = Vec::with_capacity(count);
        let current_idx = self.index.load(Ordering::Relaxed);
        
        for i in 0..count {
            let idx = (current_idx + i) % self.capacity;
            result.push(self.data[idx].load(Ordering::Relaxed));
        }
        result
    }

    #[inline(always)]
    fn len(&self) -> usize {
        self.count.load(Ordering::Relaxed)
    }
}

/// VaR calculation method
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum VarMethod {
    Historical,
    Parametric,
    MonteCarlo,
}

/// VaR/ES calculation result
#[derive(Debug, Clone)]
pub struct VarResult {
    pub var_95: f64,
    pub var_99: f64,
    pub expected_shortfall_95: f64,
    pub expected_shortfall_99: f64,
    pub calculation_method: VarMethod,
    pub sample_count: usize,
    pub calculation_latency_ns: u64,
    pub mean_return: f64,
    pub volatility: f64,
}

/// High-performance VaR calculator with multiple methods
pub struct VarCalculator {
    /// Returns ring buffer (lock-free)
    returns_buffer: RingBuffer,
    /// Confidence levels
    confidence_95: f64,
    confidence_99: f64,
    /// Decay factor for EWMA volatility
    lambda: f64,
    /// Current EWMA volatility estimate
    ewma_vol: AtomicF64,
    /// Mean return estimate
    mean_return: AtomicF64,
}

impl VarCalculator {
    /// Create new VaR calculator with specified lookback period
    pub fn new(lookback_days: usize, lambda: f64) -> Self {
        Self {
            returns_buffer: RingBuffer::new(lookback_days.max(252)), // At least 1 year of daily data
            confidence_95: 0.95,
            confidence_99: 0.99,
            lambda,
            ewma_vol: AtomicF64::new(0.015), // Default ~1.5% daily vol
            mean_return: AtomicF64::new(0.0),
        }
    }

    /// Add new return observation (call on every tick or bar close)
    #[inline(always)]
    pub fn add_return(&self, return_pct: f64) {
        self.returns_buffer.push(return_pct);
        
        // Update EWMA volatility
        let current_vol = self.ewma_vol.load(Ordering::Relaxed);
        let new_vol = ((1.0 - self.lambda) * return_pct.powi(2) + self.lambda * current_vol.powi(2)).sqrt();
        self.ewma_vol.store(new_vol, Ordering::Relaxed);
        
        // Update mean return (simple moving average approximation)
        let count = self.returns_buffer.len() as f64;
        let current_mean = self.mean_return.load(Ordering::Relaxed);
        let new_mean = ((current_mean * (count - 1.0)) + return_pct) / count;
        self.mean_return.store(new_mean, Ordering::Relaxed);
    }

    /// Calculate Historical VaR using sorted returns
    fn historical_var(&self, confidence: f64) -> f64 {
        let mut returns = self.returns_buffer.get_all();
        if returns.is_empty() {
            return 0.0;
        }
        
        // Sort for percentile calculation (optimized for small arrays)
        returns.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        
        let index = ((1.0 - confidence) * returns.len() as f64).floor() as usize;
        -returns[index.min(returns.len() - 1)] // Negative because we want loss
    }

    /// Calculate Parametric VaR assuming normal distribution
    fn parametric_var(&self, confidence: f64) -> f64 {
        let vol = self.ewma_vol.load(Ordering::Relaxed);
        let mean = self.mean_return.load(Ordering::Relaxed);
        
        // Z-scores for confidence levels
        let z_score = match confidence {
            c if c >= 0.99 => 2.326,
            c if c >= 0.975 => 1.96,
            c if c >= 0.95 => 1.645,
            _ => 1.282,
        };
        
        -(mean - z_score * vol)
    }

    /// Calculate Expected Shortfall (CVaR) - average loss beyond VaR
    fn expected_shortfall(&self, confidence: f64, method: VarMethod) -> f64 {
        let returns = self.returns_buffer.get_all();
        if returns.is_empty() {
            return 0.0;
        }
        
        let var = match method {
            VarMethod::Historical => self.historical_var(confidence),
            _ => self.parametric_var(confidence),
        };
        
        // Average all returns worse than VaR
        let tail_returns: Vec<f64> = returns.iter()
            .filter(|&&r| -r > var)
            .map(|&r| -r)
            .collect();
        
        if tail_returns.is_empty() {
            return var;
        }
        
        tail_returns.iter().sum::<f64>() / tail_returns.len() as f64
    }

    /// Main VaR calculation entry point
    pub fn calculate_var(&self, method: VarMethod) -> VarResult {
        let start = Instant::now();
        
        let (var_95, var_99, es_95, es_99) = match method {
            VarMethod::Historical => (
                self.historical_var(0.95),
                self.historical_var(0.99),
                self.expected_shortfall(0.95, method),
                self.expected_shortfall(0.99, method),
            ),
            VarMethod::Parametric => {
                let v95 = self.parametric_var(0.95);
                let v99 = self.parametric_var(0.99);
                (v95, v99, self.expected_shortfall(0.95, method), self.expected_shortfall(0.99, method))
            }
            VarMethod::MonteCarlo => {
                // Simplified Monte Carlo using parametric with fat tails
                let base_95 = self.parametric_var(0.95);
                let base_99 = self.parametric_var(0.99);
                // Adjust for fat tails (crypto typically has kurtosis ~5-10)
                let fat_tail_adjustment = 1.15;
                (base_95 * fat_tail_adjustment, base_99 * fat_tail_adjustment,
                 base_95 * fat_tail_adjustment * 1.2, base_99 * fat_tail_adjustment * 1.2)
            }
        };
        
        let latency_ns = start.elapsed().as_nanos() as u64;
        
        VarResult {
            var_95,
            var_99,
            expected_shortfall_95: es_95,
            expected_shortfall_99: es_99,
            calculation_method: method,
            sample_count: self.returns_buffer.len(),
            calculation_latency_ns: latency_ns,
            mean_return: self.mean_return.load(Ordering::Relaxed),
            volatility: self.ewma_vol.load(Ordering::Relaxed),
        }
    }

    /// Get current EWMA volatility estimate
    #[inline(always)]
    pub fn get_current_volatility(&self) -> f64 {
        self.ewma_vol.load(Ordering::Relaxed)
    }

    /// Get annualized volatility (assuming daily data)
    #[inline(always)]
    pub fn get_annualized_volatility(&self) -> f64 {
        self.ewma_vol.load(Ordering::Relaxed) * (252.0_f64).sqrt()
    }

    /// Calculate portfolio VaR given position values and correlation matrix
    pub fn portfolio_var(&self, positions: &[f64], correlations: &[f64]) -> f64 {
        let vol = self.ewma_vol.load(Ordering::Relaxed);
        let total_exposure: f64 = positions.iter().map(|p| p.abs()).sum();
        
        // Simplified portfolio VaR (assumes equal correlation)
        let avg_correlation = if correlations.is_empty() {
            0.3 // Default crypto correlation
        } else {
            correlations.iter().sum::<f64>() / correlations.len() as f64
        };
        
        let diversification_ratio = (1.0 + (positions.len() as f64 - 1.0) * avg_correlation).sqrt();
        let portfolio_vol = vol * diversification_ratio / (positions.len() as f64).sqrt();
        
        total_exposure * portfolio_vol * 1.645 // 95% confidence
    }

    /// Set decay factor for EWMA
    #[inline(always)]
    pub fn set_lambda(&mut self, lambda: f64) {
        self.lambda = lambda.clamp(0.8, 0.99);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_var_calculation() {
        let calc = VarCalculator::new(252, 0.94);
        
        // Add some sample returns
        for i in 0..100 {
            let ret = (i as f64 - 50.0) * 0.001; // Returns from -5% to +5%
            calc.add_return(ret);
        }
        
        let result = calc.calculate_var(VarMethod::Historical);
        assert!(result.var_95 > 0.0);
        assert!(result.var_99 >= result.var_95);
        assert!(result.calculation_latency_ns < 100_000); // Sub-100 microseconds
    }

    #[test]
    fn test_ewma_volatility() {
        let calc = VarCalculator::new(252, 0.94);
        calc.add_return(0.02);
        calc.add_return(-0.015);
        calc.add_return(0.03);
        
        let vol = calc.get_current_volatility();
        assert!(vol > 0.0);
    }
}
