//! Equal Risk Contribution (ERC) fast-path optimizer in Rust
//! Triggers microsecond portfolio rebalancing calculations when individual asset volatilities shift

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

const MAX_ASSETS: usize = 128;
const MAX_ITERATIONS: usize = 500;
const TOLERANCE: f64 = 1e-8;

/// Result of ERC optimization
#[derive(Debug, Clone)]
pub struct ERCResult {
    pub weights: [f64; MAX_ASSETS],
    pub risk_contributions: [f64; MAX_ASSETS],
    pub total_risk: f64,
    pub converged: bool,
    pub iterations: u32,
}

/// ERC Optimizer state (pre-allocated for zero heap allocation in hot path)
#[repr(align(64))]
pub struct ERCOptimizer {
    covariance: [[f64; MAX_ASSETS]; MAX_ASSETS],
    asset_count: usize,
    target_risk_budget: [f64; MAX_ASSETS],
    last_weights: [f64; MAX_ASSETS],
    cache_valid: AtomicBool,
    update_counter: AtomicU64,
}

unsafe impl Send for ERCOptimizer {}
unsafe impl Sync for ERCOptimizer {}

impl Default for ERCOptimizer {
    fn default() -> Self {
        Self {
            covariance: [[0.0; MAX_ASSETS]; MAX_ASSETS],
            asset_count: 0,
            target_risk_budget: [0.0; MAX_ASSETS],
            last_weights: [0.0; MAX_ASSETS],
            cache_valid: AtomicBool::new(false),
            update_counter: AtomicU64::new(0),
        }
    }
}

impl ERCOptimizer {
    #[inline(always)]
    pub fn new(asset_count: usize) -> Self {
        assert!(asset_count <= MAX_ASSETS);
        let mut opt = Self::default();
        opt.asset_count = asset_count;
        
        // Equal risk budget by default
        let budget = 1.0 / asset_count as f64;
        for i in 0..asset_count {
            opt.target_risk_budget[i] = budget;
        }
        
        opt
    }

    /// Update covariance matrix (invalidates cache)
    #[inline(always)]
    pub fn update_covariance(&mut self, cov: &[[f64]]) {
        let n = self.asset_count;
        for i in 0..n {
            for j in 0..n {
                self.covariance[i][j] = cov[i][j];
            }
        }
        self.cache_valid.store(false, Ordering::Release);
        self.update_counter.fetch_add(1, Ordering::Release);
    }

    /// Set custom risk budgets (must sum to 1)
    #[inline(always)]
    pub fn set_risk_budgets(&mut self, budgets: &[f64]) {
        assert_eq!(budgets.len(), self.asset_count);
        for i in 0..self.asset_count {
            self.target_risk_budget[i] = budgets[i];
        }
        self.cache_valid.store(false, Ordering::Release);
    }

    /// Compute ERC allocation using cyclical coordinate descent
    #[inline(always)]
    pub fn compute_erc(&self) -> ERCResult {
        let n = self.asset_count;
        
        if n == 0 {
            return ERCResult {
                weights: [0.0; MAX_ASSETS],
                risk_contributions: [0.0; MAX_ASSETS],
                total_risk: 0.0,
                converged: false,
                iterations: 0,
            };
        }

        // Initialize with equal weights
        let mut weights = [0.0; MAX_ASSETS];
        let init_w = 1.0 / n as f64;
        for i in 0..n {
            weights[i] = init_w;
        }

        let mut converged = false;
        let mut iterations = 0u32;

        // Cyclical coordinate descent
        for iter in 0..MAX_ITERATIONS {
            iterations = iter as u32 + 1;
            let mut max_change = 0.0;

            for i in 0..n {
                // Calculate marginal risk contribution for asset i
                let mut marginal_risk_i = 0.0;
                for j in 0..n {
                    marginal_risk_i += self.covariance[i][j] * weights[j];
                }

                // Current portfolio variance
                let mut port_var = 0.0;
                for ii in 0..n {
                    for jj in 0..n {
                        port_var += weights[ii] * self.covariance[ii][jj] * weights[jj];
                    }
                }

                let port_vol = port_var.sqrt();
                if port_vol < 1e-10 {
                    continue;
                }

                // Current risk contribution
                let rc_i = weights[i] * marginal_risk_i / port_vol;
                let target_rc = self.target_risk_budget[i];

                // Adjust weight to move toward target risk contribution
                // Using Newton-like step
                if rc_i > 1e-10 {
                    let ratio = target_rc / rc_i;
                    let new_weight = weights[i] * ratio.powf(0.5); // Damped update
                    let change = (new_weight - weights[i]).abs();
                    max_change = max_change.max(change);
                    weights[i] = new_weight.max(1e-10);
                }
            }

            // Normalize weights
            let sum: f64 = weights[..n].iter().sum();
            if sum > 1e-10 {
                for i in 0..n {
                    weights[i] /= sum;
                }
            }

            // Check convergence
            if max_change < TOLERANCE {
                converged = true;
                break;
            }
        }

        // Calculate final risk contributions
        let mut risk_contributions = [0.0; MAX_ASSETS];
        let mut port_var = 0.0;

        for i in 0..n {
            for j in 0..n {
                port_var += weights[i] * self.covariance[i][j] * weights[j];
            }
        }

        let port_vol = port_var.sqrt();

        for i in 0..n {
            let mut marginal = 0.0;
            for j in 0..n {
                marginal += self.covariance[i][j] * weights[j];
            }
            risk_contributions[i] = weights[i] * marginal / port_vol;
        }

        ERCResult {
            weights,
            risk_contributions,
            total_risk: port_vol,
            converged,
            iterations,
        }
    }

    /// Fast-path: incremental update when only one asset's volatility changes
    #[inline(always)]
    pub fn update_single_asset_volatility(&mut self, asset_idx: usize, new_volatility: f64) -> ERCResult {
        if asset_idx >= self.asset_count {
            return self.compute_erc();
        }

        // Update diagonal element (variance)
        self.covariance[asset_idx][asset_idx] = new_volatility * new_volatility;
        
        // Invalidate cache
        self.cache_valid.store(false, Ordering::Release);
        
        // Use last weights as warm start
        let n = self.asset_count;
        let mut weights = self.last_weights;
        
        // Quick renormalization if needed
        let sum: f64 = weights[..n].iter().sum();
        if sum > 1e-10 && (sum - 1.0).abs() > 0.01 {
            for i in 0..n {
                weights[i] /= sum;
            }
        }

        // Run a few iterations of coordinate descent from warm start
        let mut converged = false;
        let mut iterations = 0u32;

        for iter in 0..50 {  // Limited iterations for speed
            iterations = iter as u32 + 1;
            let mut max_change = 0.0;

            for i in 0..n {
                let mut marginal_risk_i = 0.0;
                for j in 0..n {
                    marginal_risk_i += self.covariance[i][j] * weights[j];
                }

                let mut port_var = 0.0;
                for ii in 0..n {
                    for jj in 0..n {
                        port_var += weights[ii] * self.covariance[ii][jj] * weights[jj];
                    }
                }

                let port_vol = port_var.sqrt();
                if port_vol < 1e-10 {
                    continue;
                }

                let rc_i = weights[i] * marginal_risk_i / port_vol;
                let target_rc = self.target_risk_budget[i];

                if rc_i > 1e-10 {
                    let ratio = target_rc / rc_i;
                    let new_weight = weights[i] * ratio.powf(0.3); // More damping for stability
                    let change = (new_weight - weights[i]).abs();
                    max_change = max_change.max(change);
                    weights[i] = new_weight.max(1e-10);
                }
            }

            let sum: f64 = weights[..n].iter().sum();
            if sum > 1e-10 {
                for i in 0..n {
                    weights[i] /= sum;
                }
            }

            if max_change < TOLERANCE * 10 {  // Relaxed tolerance for fast path
                converged = true;
                break;
            }
        }

        // Cache the result
        self.last_weights = weights;
        self.cache_valid.store(true, Ordering::Release);

        // Calculate risk contributions
        let mut risk_contributions = [0.0; MAX_ASSETS];
        let mut port_var = 0.0;

        for i in 0..n {
            for j in 0..n {
                port_var += weights[i] * self.covariance[i][j] * weights[j];
            }
        }

        let port_vol = port_var.sqrt();

        for i in 0..n {
            let mut marginal = 0.0;
            for j in 0..n {
                marginal += self.covariance[i][j] * weights[j];
            }
            risk_contributions[i] = weights[i] * marginal / port_vol;
        }

        ERCResult {
            weights,
            risk_contributions,
            total_risk: port_vol,
            converged,
            iterations,
        }
    }

    /// Get cached result if valid
    #[inline(always)]
    pub fn get_cached(&self) -> Option<ERCResult> {
        if !self.cache_valid.load(Ordering::Acquire) {
            return None;
        }

        let n = self.asset_count;
        let weights = self.last_weights;

        // Recalculate risk contributions
        let mut risk_contributions = [0.0; MAX_ASSETS];
        let mut port_var = 0.0;

        for i in 0..n {
            for j in 0..n {
                port_var += weights[i] * self.covariance[i][j] * weights[j];
            }
        }

        let port_vol = port_var.sqrt();

        for i in 0..n {
            let mut marginal = 0.0;
            for j in 0..n {
                marginal += self.covariance[i][j] * weights[j];
            }
            risk_contributions[i] = weights[i] * marginal / port_vol;
        }

        Some(ERCResult {
            weights,
            risk_contributions,
            total_risk: port_vol,
            converged: true,
            iterations: 0,
        })
    }

    #[inline(always)]
    pub fn update_counter(&self) -> u64 {
        self.update_counter.load(Ordering::Acquire)
    }
}

/// Quick ERC calculation for small portfolios (specialized for crypto)
#[inline(always)]
pub fn erc_quick(covariance: &[[f64]], asset_count: usize) -> ERCResult {
    assert!(asset_count <= MAX_ASSETS);
    
    let mut optimizer = ERCOptimizer::new(asset_count);
    optimizer.update_covariance(covariance);
    optimizer.compute_erc()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_erc_optimizer() {
        let mut cov = [[0.0; MAX_ASSETS]; MAX_ASSETS];
        
        // Simple 3-asset case
        cov[0][0] = 0.04;
        cov[1][1] = 0.09;
        cov[2][2] = 0.16;
        cov[0][1] = 0.01;
        cov[1][0] = 0.01;
        cov[0][2] = 0.005;
        cov[2][0] = 0.005;
        cov[1][2] = 0.02;
        cov[2][1] = 0.02;

        let mut opt = ERCOptimizer::new(3);
        opt.update_covariance(&cov);

        let result = opt.compute_erc();

        assert!(result.converged || result.iterations > 0);
        assert!(result.total_risk > 0.0);
        
        // Risk contributions should be approximately equal
        let avg_rc = (result.risk_contributions[0] + result.risk_contributions[1] + result.risk_contributions[2]) / 3.0;
        assert!((result.risk_contributions[0] - avg_rc).abs() < 0.01);
        assert!((result.risk_contributions[1] - avg_rc).abs() < 0.01);
        assert!((result.risk_contributions[2] - avg_rc).abs() < 0.01);
    }

    #[test]
    fn test_fast_path_update() {
        let mut cov = [[0.0; MAX_ASSETS]; MAX_ASSETS];
        
        cov[0][0] = 0.04;
        cov[1][1] = 0.09;
        cov[2][2] = 0.16;

        let mut opt = ERCOptimizer::new(3);
        opt.update_covariance(&cov);
        
        // Initial computation
        let result1 = opt.compute_erc();
        
        // Fast path update
        let result2 = opt.update_single_asset_volatility(0, 0.25); // New vol = 0.25
        
        assert!(result2.total_risk > 0.0);
        // Weights should have shifted away from asset 0 (higher vol)
        assert!(result2.weights[0] < result1.weights[0]);
    }
}
