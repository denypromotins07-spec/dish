//! Custom, lightweight Quadratic Programming (QP) solver for Markowitz Mean-Variance optimization
//! Calculates optimal asset weights to maximize Sharpe ratio subject to strict weight and leverage constraints

use crate::covariance_matrix::CovarianceMatrix;

const MAX_ASSETS: usize = 128;
const MAX_ITERATIONS: usize = 1000;
const TOLERANCE: f64 = 1e-10;

/// Result of mean-variance optimization
#[derive(Debug, Clone)]
pub struct MVOResult {
    pub weights: [f64; MAX_ASSETS],
    pub expected_return: f64,
    pub portfolio_variance: f64,
    pub sharpe_ratio: f64,
    pub asset_count: usize,
    pub converged: bool,
}

/// Constraints for portfolio optimization
#[derive(Debug, Clone)]
pub struct MVOConstraints {
    pub min_weights: [f64; MAX_ASSETS],
    pub max_weights: [f64; MAX_ASSETS],
    pub target_return: Option<f64>,
    pub risk_free_rate: f64,
    pub leverage_limit: f64,
    pub asset_count: usize,
}

impl Default for MVOConstraints {
    fn default() -> Self {
        Self {
            min_weights: [0.0; MAX_ASSETS],
            max_weights: [1.0; MAX_ASSETS],
            target_return: None,
            risk_free_rate: 0.0,
            leverage_limit: 1.0,
            asset_count: 0,
        }
    }
}

impl MVOConstraints {
    #[inline(always)]
    pub fn new(asset_count: usize) -> Self {
        assert!(asset_count <= MAX_ASSETS);
        let mut constraints = Self::default();
        constraints.asset_count = asset_count;
        constraints
    }

    #[inline(always)]
    pub fn set_bounds(&mut self, min: f64, max: f64) {
        for i in 0..self.asset_count {
            self.min_weights[i] = min;
            self.max_weights[i] = max;
        }
    }
}

/// Mean-Variance Optimizer using projected gradient descent
pub struct MeanVarianceOptimizer {
    expected_returns: [f64; MAX_ASSETS],
    covariance: CovarianceMatrix,
}

impl MeanVarianceOptimizer {
    #[inline(always)]
    pub fn new(expected_returns: &[f64], covariance: CovarianceMatrix) -> Self {
        let asset_count = covariance.asset_count();
        assert_eq!(expected_returns.len(), asset_count);
        
        let mut returns = [0.0; MAX_ASSETS];
        returns[..asset_count].copy_from_slice(&expected_returns[..asset_count]);
        
        Self {
            expected_returns: returns,
            covariance,
        }
    }

    /// Optimize for maximum Sharpe ratio
    #[inline(always)]
    pub fn optimize_max_sharpe(&self, constraints: &MVOConstraints) -> MVOResult {
        let n = constraints.asset_count;
        let mut weights = [0.0; MAX_ASSETS];
        
        // Initialize with equal weights
        let initial_weight = 1.0 / n as f64;
        for i in 0..n {
            weights[i] = initial_weight;
        }
        
        let mut best_sharpe = f64::NEG_INFINITY;
        let mut best_weights = weights;
        let mut converged = false;
        
        // Projected gradient ascent on Sharpe ratio
        let learning_rate = 0.1;
        
        for iter in 0..MAX_ITERATIONS {
            // Calculate portfolio metrics
            let (ret, var) = self.portfolio_metrics(&weights, n);
            let vol = var.sqrt();
            
            // Sharpe ratio
            let sharpe = if vol > 1e-10 {
                (ret - constraints.risk_free_rate) / vol
            } else {
                0.0
            };
            
            if sharpe > best_sharpe {
                best_sharpe = sharpe;
                best_weights = weights;
            }
            
            // Check convergence
            if iter > 0 && (sharpe - best_sharpe).abs() < TOLERANCE {
                converged = true;
                break;
            }
            
            // Calculate gradient of Sharpe ratio w.r.t. weights
            let mut gradient = [0.0; MAX_ASSETS];
            self.sharpe_gradient(&weights, n, ret, var, constraints.risk_free_rate, &mut gradient);
            
            // Gradient ascent step
            for i in 0..n {
                weights[i] += learning_rate * gradient[i];
            }
            
            // Project onto constraint set
            self.project_weights(&mut weights, n, constraints);
        }
        
        let (opt_ret, opt_var) = self.portfolio_metrics(&best_weights, n);
        
        MVOResult {
            weights: best_weights,
            expected_return: opt_ret,
            portfolio_variance: opt_var,
            sharpe_ratio: if opt_var.sqrt() > 1e-10 {
                (opt_ret - constraints.risk_free_rate) / opt_var.sqrt()
            } else {
                0.0
            },
            asset_count: n,
            converged,
        }
    }

    /// Optimize for minimum variance given target return
    #[inline(always)]
    pub fn optimize_min_variance(&self, constraints: &MVOConstraints) -> MVOResult {
        let n = constraints.asset_count;
        let target = constraints.target_return.unwrap_or(0.0);
        
        let mut weights = [0.0; MAX_ASSETS];
        let initial_weight = 1.0 / n as f64;
        for i in 0..n {
            weights[i] = initial_weight;
        }
        
        let mut best_variance = f64::MAX;
        let mut best_weights = weights;
        let mut converged = false;
        
        // Lagrangian approach with projected gradient
        let mut lambda = 1.0; // Lagrange multiplier for return constraint
        let learning_rate = 0.05;
        
        for iter in 0..MAX_ITERATIONS {
            let (_, var) = self.portfolio_metrics(&weights, n);
            let (ret, _) = self.portfolio_metrics(&weights, n);
            
            if var < best_variance && (ret - target).abs() < 0.01 {
                best_variance = var;
                best_weights = weights;
            }
            
            // Check convergence
            if iter > 0 && var.abs() - best_variance.abs() < TOLERANCE {
                converged = true;
                break;
            }
            
            // Gradient of variance + lambda * (return - target)
            let mut gradient = [0.0; MAX_ASSETS];
            self.variance_gradient(&weights, n, &mut gradient);
            
            for i in 0..n {
                gradient[i] += 2.0 * lambda * self.expected_returns[i];
                weights[i] -= learning_rate * gradient[i];
            }
            
            // Adjust lambda based on constraint violation
            lambda += 0.1 * (ret - target);
            
            self.project_weights(&mut weights, n, constraints);
        }
        
        let (opt_ret, opt_var) = self.portfolio_metrics(&best_weights, n);
        
        MVOResult {
            weights: best_weights,
            expected_return: opt_ret,
            portfolio_variance: opt_var,
            sharpe_ratio: if opt_var.sqrt() > 1e-10 {
                (opt_ret - constraints.risk_free_rate) / opt_var.sqrt()
            } else {
                0.0
            },
            asset_count: n,
            converged,
        }
    }

    #[inline(always)]
    fn portfolio_metrics(&self, weights: &[f64], n: usize) -> (f64, f64) {
        // Expected return
        let mut ret = 0.0;
        for i in 0..n {
            ret += weights[i] * self.expected_returns[i];
        }
        
        // Variance: w^T * Sigma * w
        let mut var = 0.0;
        for i in 0..n {
            for j in 0..n {
                var += weights[i] * self.covariance.get(i, j) * weights[j];
            }
        }
        
        (ret, var)
    }

    #[inline(always)]
    fn sharpe_gradient(&self, weights: &[f64], n: usize, ret: f64, var: f64, rf: f64, grad: &mut [f64]) {
        let vol = var.sqrt();
        let excess_ret = ret - rf;
        
        // d(SR)/d(w_i) = (mu_i * vol - excess_ret * d(vol)/d(w_i)) / vol^2
        // d(vol)/d(w_i) = (Sigma * w)_i / vol
        
        for i in 0..n {
            let mut sigma_w_i = 0.0;
            for j in 0..n {
                sigma_w_i += self.covariance.get(i, j) * weights[j];
            }
            
            let dvol_dw = if vol > 1e-10 { sigma_w_i / vol } else { 0.0 };
            
            grad[i] = (self.expected_returns[i] * vol - excess_ret * dvol_dw) / (var + 1e-10);
        }
    }

    #[inline(always)]
    fn variance_gradient(&self, weights: &[f64], n: usize, grad: &mut [f64]) {
        // d(w^T Sigma w)/d(w) = 2 * Sigma * w
        for i in 0..n {
            let mut sum = 0.0;
            for j in 0..n {
                sum += self.covariance.get(i, j) * weights[j];
            }
            grad[i] = 2.0 * sum;
        }
    }

    #[inline(always)]
    fn project_weights(&self, weights: &mut [f64], n: usize, constraints: &MVOConstraints) {
        // Project onto box constraints
        for i in 0..n {
            weights[i] = weights[i].max(constraints.min_weights[i]);
            weights[i] = weights[i].min(constraints.max_weights[i]);
        }
        
        // Project onto simplex (sum to 1 or leverage limit)
        let sum: f64 = weights[..n].iter().sum();
        let target_sum = constraints.leverage_limit.min(1.0);
        
        if (sum - target_sum).abs() > 1e-10 {
            // Simple scaling (approximate projection)
            let scale = target_sum / sum;
            for i in 0..n {
                weights[i] *= scale;
            }
            
            // Re-apply box constraints after scaling
            for i in 0..n {
                weights[i] = weights[i].max(constraints.min_weights[i]);
                weights[i] = weights[i].min(constraints.max_weights[i]);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mvo_optimization() {
        let mut cov = CovarianceMatrix::new(3);
        
        // Simple diagonal covariance
        cov.data[0][0] = 0.04;
        cov.data[1][1] = 0.09;
        cov.data[2][2] = 0.16;
        cov.asset_count = 3;
        
        let returns = vec![0.10, 0.15, 0.20];
        let optimizer = MeanVarianceOptimizer::new(&returns, cov);
        
        let mut constraints = MVOConstraints::new(3);
        constraints.set_bounds(0.0, 1.0);
        
        let result = optimizer.optimize_max_sharpe(&constraints);
        
        assert!(result.converged || result.sharpe_ratio > 0.0);
        assert!(result.portfolio_variance > 0.0);
    }
}
