//! Rust implementation of the Black-Litterman model
//! Combines market equilibrium (CAPM implied returns) with strategy "views" to generate posterior expected returns

use crate::covariance_matrix::CovarianceMatrix;

const MAX_ASSETS: usize = 128;
const MAX_VIEWS: usize = 32;

/// Result of Black-Litterman optimization
#[derive(Debug, Clone)]
pub struct BlackLittermanResult {
    pub posterior_returns: [f64; MAX_ASSETS],
    pub posterior_covariance: CovarianceMatrix,
    pub asset_count: usize,
}

/// A single view in the Black-Litterman model
#[derive(Debug, Clone)]
pub struct View {
    /// Pick matrix row: which assets this view affects (P vector)
    pub pick_vector: [f64; MAX_ASSETS],
    /// Expected return of the view
    pub expected_return: f64,
    /// Confidence in the view (0.0 to 1.0)
    pub confidence: f64,
    /// Number of assets in this view
    pub asset_count: usize,
}

impl View {
    #[inline(always)]
    pub fn new(asset_count: usize) -> Self {
        assert!(asset_count <= MAX_ASSETS);
        Self {
            pick_vector: [0.0; MAX_ASSETS],
            expected_return: 0.0,
            confidence: 0.5,
            asset_count,
        }
    }

    /// Create an absolute view (e.g., "BTC will return 10%")
    #[inline(always)]
    pub fn absolute(asset_index: usize, expected_return: f64, confidence: f64, total_assets: usize) -> Self {
        let mut view = Self::new(total_assets);
        view.pick_vector[asset_index] = 1.0;
        view.expected_return = expected_return;
        view.confidence = confidence.clamp(0.0, 1.0);
        view
    }

    /// Create a relative view (e.g., "BTC will outperform ETH by 5%")
    #[inline(always)]
    pub fn relative(
        long_asset: usize,
        short_asset: usize,
        outperformance: f64,
        confidence: f64,
        total_assets: usize,
    ) -> Self {
        let mut view = Self::new(total_assets);
        view.pick_vector[long_asset] = 1.0;
        view.pick_vector[short_asset] = -1.0;
        view.expected_return = outperformance;
        view.confidence = confidence.clamp(0.0, 1.0);
        view
    }
}

/// Black-Litterman Model
pub struct BlackLittermanModel {
    /// Market capitalization weights
    market_weights: [f64; MAX_ASSETS],
    /// Risk aversion coefficient (lambda)
    risk_aversion: f64,
    /// Prior covariance matrix
    covariance: CovarianceMatrix,
    /// Asset count
    asset_count: usize,
}

impl BlackLittermanModel {
    #[inline(always)]
    pub fn new(market_weights: &[f64], covariance: CovarianceMatrix, risk_aversion: f64) -> Self {
        let asset_count = covariance.asset_count();
        assert_eq!(market_weights.len(), asset_count);
        
        let mut weights = [0.0; MAX_ASSETS];
        weights[..asset_count].copy_from_slice(&market_weights[..asset_count]);
        
        Self {
            market_weights: weights,
            risk_aversion,
            covariance,
            asset_count,
        }
    }

    /// Calculate implied equilibrium returns (CAPM-style)
    #[inline(always)]
    pub fn implied_returns(&self) -> [f64; MAX_ASSETS] {
        let mut returns = [0.0; MAX_ASSETS];
        
        // Pi = lambda * Sigma * w_mkt
        for i in 0..self.asset_count {
            let mut sum = 0.0;
            for j in 0..self.asset_count {
                sum += self.covariance.get(i, j) * self.market_weights[j];
            }
            returns[i] = self.risk_aversion * sum;
        }
        
        returns
    }

    /// Apply Black-Litterman formula with views
    /// Posterior = [(tau * Sigma)^-1 + P' * Omega^-1 * P]^-1 * [(tau * Sigma)^-1 * Pi + P' * Omega^-1 * Q]
    #[inline(always)]
    pub fn compute_posterior(&self, views: &[View], tau: f64) -> BlackLittermanResult {
        let num_views = views.len();
        
        // Get prior returns
        let pi = self.implied_returns();
        
        // Build Omega matrix (uncertainty of views)
        // Omega is diagonal with elements proportional to view variance
        let mut omega_diag = [0.0; MAX_VIEWS];
        for (k, view) in views.iter().enumerate() {
            // Omega_kk = (P_k * Sigma * P_k') / confidence
            // Simplified: use scalar proportional to view variance
            let mut view_variance = 0.0;
            for i in 0..self.asset_count {
                for j in 0..self.asset_count {
                    view_variance += view.pick_vector[i] * self.covariance.get(i, j) * view.pick_vector[j];
                }
            }
            
            // Scale by confidence (lower confidence = higher uncertainty)
            let confidence_factor = if view.confidence > 1e-6 { 1.0 / view.confidence } else { 1e6 };
            omega_diag[k] = tau * view_variance * confidence_factor;
        }
        
        // Compute posterior using simplified formula for diagonal Omega
        // This avoids full matrix inversion
        
        let mut posterior_returns = [0.0; MAX_ASSETS];
        let mut posterior_cov = CovarianceMatrix::new(self.asset_count);
        
        // For each asset, compute posterior return
        // Using the formula: E[R] = [(tau*Sigma)^-1 + P'*Omega^-1*P]^-1 * [(tau*Sigma)^-1*Pi + P'*Omega^-1*Q]
        
        // Simplified approach: iterative update for each view
        for i in 0..self.asset_count {
            posterior_returns[i] = pi[i]; // Start with prior
            
            // Adjust for each view
            for (k, view) in views.iter().enumerate() {
                if view.pick_vector[i].abs() > 1e-10 {
                    let p_i = view.pick_vector[i];
                    let omega_inv = 1.0 / omega_diag[k].max(1e-10);
                    
                    // View adjustment
                    let view_adjustment = p_i * omega_inv * view.expected_return;
                    let prior_adjustment = p_i * omega_inv * pi[i];
                    
                    // Weight by confidence
                    let weight = view.confidence * tau;
                    posterior_returns[i] += weight * (view_adjustment - prior_adjustment);
                }
            }
        }
        
        // Compute posterior covariance (simplified)
        // Sigma_post = Sigma - Sigma * P' * (Omega + P * Sigma * P')^-1 * P * Sigma
        for i in 0..self.asset_count {
            for j in 0..self.asset_count {
                let mut cov_adj = 0.0;
                
                for (k, view) in views.iter().enumerate() {
                    let omega_k = omega_diag[k];
                    
                    // P * Sigma for this view
                    let mut p_sigma_i = 0.0;
                    let mut p_sigma_j = 0.0;
                    for l in 0..self.asset_count {
                        p_sigma_i += view.pick_vector[l] * self.covariance.get(i, l);
                        p_sigma_j += view.pick_vector[l] * self.covariance.get(j, l);
                    }
                    
                    // Adjustment term
                    let denom = omega_k + p_sigma_i * view.pick_vector[i];
                    if denom > 1e-10 {
                        cov_adj += (p_sigma_i * p_sigma_j) / denom;
                    }
                }
                
                let post_cov = self.covariance.get(i, j) - cov_adj * tau;
                posterior_cov.data[i][j] = post_cov.max(0.0);
            }
        }
        posterior_cov.asset_count = self.asset_count;
        
        BlackLittermanResult {
            posterior_returns,
            posterior_covariance: posterior_cov,
            asset_count: self.asset_count,
        }
    }

    /// Apply views with automatic Omega scaling based on view type
    #[inline(always)]
    pub fn compute_with_auto_omega(&self, views: &[View]) -> BlackLittermanResult {
        // Use standard tau = 0.05 (common in practice)
        self.compute_posterior(views, 0.05)
    }

    /// Merge BL returns with MVO optimizer
    #[inline(always)]
    pub fn get_optimization_inputs(&self, views: &[View]) -> ([f64; MAX_ASSETS], CovarianceMatrix) {
        let result = self.compute_with_auto_omega(views);
        (result.posterior_returns, result.posterior_covariance)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_black_litterman() {
        let mut cov = CovarianceMatrix::new(3);
        cov.data[0][0] = 0.04;
        cov.data[1][1] = 0.09;
        cov.data[2][2] = 0.16;
        cov.asset_count = 3;
        
        let market_weights = vec![0.4, 0.35, 0.25];
        let bl = BlackLittermanModel::new(&market_weights, cov, 2.5);
        
        // Get implied returns
        let implied = bl.implied_returns();
        assert!(implied[0] > 0.0);
        
        // Add a view: BTC will outperform ETH by 5% with 60% confidence
        let views = vec![
            View::relative(0, 1, 0.05, 0.6, 3),
        ];
        
        let result = bl.compute_with_auto_omega(&views);
        assert_eq!(result.asset_count, 3);
        
        // View should increase return for asset 0 relative to asset 1
        assert!(result.posterior_returns[0] > implied[0] || result.posterior_returns[1] < implied[1]);
    }
}
