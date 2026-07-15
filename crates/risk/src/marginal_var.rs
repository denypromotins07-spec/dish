// crates/risk/src/marginal_var.rs
// High-performance Component VaR and Marginal VaR calculator
// Determines which asset/strategy contributes most to tail risk

use std::sync::atomic::{AtomicU64, Ordering};

/// Portfolio position with risk metrics
#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct Position {
    pub asset_id: u32,
    pub quantity: f64,
    pub current_price: f64,
    pub weight: f64,
}

/// VaR calculation result for a single asset
#[derive(Clone, Copy, Debug, Default)]
#[repr(C)]
pub struct AssetVaR {
    pub asset_id: u32,
    pub marginal_var: f64,      // Change in portfolio VaR per unit change in position
    pub component_var: f64,     // Asset's contribution to total portfolio VaR
    pub incremental_var: f64,   // Approximate change in VaR if position is liquidated
    pub standalone_var: f64,    // VaR of the asset in isolation
    pub correlation_contrib: f64, // Risk contribution from correlations
}

/// Portfolio-level VaR summary
#[derive(Clone, Copy, Debug, Default)]
#[repr(C)]
pub struct PortfolioVaRSummary {
    pub total_var_95: f64,
    pub total_var_99: f64,
    pub total_cvar_95: f64,     // Conditional VaR (Expected Shortfall)
    pub total_cvar_99: f64,
    pub diversification_ratio: f64,
    pub concentration_index: f64, // Herfindahl index of risk contributions
}

/// Covariance matrix cache (flattened, row-major)
/// Pre-allocated for up to 50 assets to avoid runtime allocation
pub struct CovarianceCache<const N: usize> {
    data: [f64; N * N],
    size: usize,
    is_valid: bool,
}

impl<const N: usize> CovarianceCache<N> {
    pub const fn new() -> Self {
        Self {
            data: [0.0; N * N],
            size: 0,
            is_valid: false,
        }
    }

    #[inline]
    pub fn set(&mut self, i: usize, j: usize, value: f64) {
        if i < self.size && j < self.size {
            self.data[i * self.size + j] = value;
        }
    }

    #[inline]
    pub fn get(&self, i: usize, j: usize) -> f64 {
        if i < self.size && j < self.size {
            self.data[i * self.size + j]
        } else {
            0.0
        }
    }

    #[inline]
    pub fn set_size(&mut self, size: usize) {
        self.size = size.min(N);
        self.is_valid = true;
    }

    #[inline]
    pub const fn size(&self) -> usize {
        self.size
    }
}

impl<const N: usize> Default for CovarianceCache<N> {
    fn default() -> Self {
        Self::new()
    }
}

/// Historical returns buffer (circular, fixed-size)
pub struct ReturnsBuffer<const ASSETS: usize, const SAMPLES: usize> {
    data: [[f64; SAMPLES]; ASSETS],
    head: [usize; ASSETS],
    count: [usize; ASSETS],
    num_assets: usize,
}

impl<const ASSETS: usize, const SAMPLES: usize> ReturnsBuffer<ASSETS, SAMPLES> {
    pub const fn new() -> Self {
        Self {
            data: [[0.0; SAMPLES]; ASSETS],
            head: [0; ASSETS],
            count: [0; ASSETS],
            num_assets: 0,
        }
    }

    #[inline]
    pub fn push_return(&mut self, asset_idx: usize, return_pct: f64) {
        if asset_idx >= ASSETS { return; }
        
        let idx = self.head[asset_idx];
        self.data[asset_idx][idx] = return_pct;
        self.head[asset_idx] = (idx + 1) % SAMPLES;
        
        if self.count[asset_idx] < SAMPLES {
            self.count[asset_idx] += 1;
        }
        
        if asset_idx >= self.num_assets && self.count[asset_idx] > 0 {
            self.num_assets = asset_idx + 1;
        }
    }

    #[inline]
    pub fn get_returns(&self, asset_idx: usize, out: &mut [f64]) -> usize {
        if asset_idx >= self.num_assets { return 0; }
        
        let count = self.count[asset_idx].min(out.len());
        let start = if self.head[asset_idx] >= count {
            self.head[asset_idx] - count
        } else {
            SAMPLES - (count - self.head[asset_idx])
        };

        for i in 0..count {
            let idx = (start + i) % SAMPLES;
            out[i] = self.data[asset_idx][idx];
        }
        count
    }

    #[inline]
    pub const fn get_num_assets(&self) -> usize {
        self.num_assets
    }

    #[inline]
    pub fn compute_variance(&self, asset_idx: usize) -> f64 {
        let mut returns = [0.0f64; SAMPLES];
        let n = self.get_returns(asset_idx, &mut returns);
        if n == 0 { return 0.0; }

        let mean = returns.iter().take(n).sum::<f64>() / n as f64;
        let variance = returns.iter().take(n)
            .map(|r| (r - mean).powi(2))
            .sum::<f64>() / (n - 1) as f64;
        variance
    }
}

impl<const ASSETS: usize, const SAMPLES: usize> Default for ReturnsBuffer<ASSETS, SAMPLES> {
    fn default() -> Self {
        Self::new()
    }
}

/// Main Marginal VaR calculator - parametric approach with delta-normal method
pub struct MarginalVaRCalculator<const MAX_ASSETS: usize, const SAMPLES: usize> {
    returns_buffer: ReturnsBuffer<MAX_ASSETS, SAMPLES>,
    cov_cache: CovarianceCache<MAX_ASSETS>,
    positions: [Position; MAX_ASSETS],
    num_positions: usize,
    confidence_level: f64,
    time_horizon_days: u32,
    portfolio_value: f64,
}

impl<const MAX_ASSETS: usize, const SAMPLES: usize> MarginalVaRCalculator<MAX_ASSETS, SAMPLES> {
    pub const fn new(confidence_level: f64, time_horizon_days: u32) -> Self {
        Self {
            returns_buffer: ReturnsBuffer::new(),
            cov_cache: CovarianceCache::new(),
            positions: [Position {
                asset_id: 0, quantity: 0.0, current_price: 0.0, weight: 0.0
            }; MAX_ASSETS],
            num_positions: 0,
            confidence_level,
            time_horizon_days,
            portfolio_value: 0.0,
        }
    }

    #[inline]
    pub fn set_position(&mut self, pos: Position) {
        for i in 0..self.num_positions {
            if self.positions[i].asset_id == pos.asset_id {
                self.positions[i] = pos;
                return;
            }
        }
        if self.num_positions < MAX_ASSETS {
            self.positions[self.num_positions] = pos;
            self.num_positions += 1;
        }
    }

    #[inline]
    pub fn add_return(&mut self, asset_idx: usize, return_pct: f64) {
        self.returns_buffer.push_return(asset_idx, return_pct);
    }

    /// Compute covariance matrix from historical returns
    #[inline]
    pub fn compute_covariance_matrix(&mut self) {
        let num_assets = self.returns_buffer.get_num_assets();
        self.cov_cache.set_size(num_assets);

        // First compute variances and standard deviations
        let mut std_devs = [0.0f64; MAX_ASSETS];
        for i in 0..num_assets {
            let var = self.returns_buffer.compute_variance(i);
            std_devs[i] = var.sqrt();
            self.cov_cache.set(i, i, var);
        }

        // Compute correlations and covariances
        let mut returns_i = [0.0f64; SAMPLES];
        let mut returns_j = [0.0f64; SAMPLES];
        
        for i in 0..num_assets {
            let n_i = self.returns_buffer.get_returns(i, &mut returns_i);
            for j in (i + 1)..num_assets {
                let n_j = self.returns_buffer.get_returns(j, &mut returns_j);
                let n = n_i.min(n_j);
                
                if n < 2 { continue; }

                // Compute correlation
                let mean_i = returns_i.iter().take(n).sum::<f64>() / n as f64;
                let mean_j = returns_j.iter().take(n).sum::<f64>() / n as f64;
                
                let mut cov_sum = 0.0;
                for k in 0..n {
                    cov_sum += (returns_i[k] - mean_i) * (returns_j[k] - mean_j);
                }
                let corr = cov_sum / ((n - 1) as f64 * std_devs[i] * std_devs[j]);
                let cov = corr * std_devs[i] * std_devs[j];

                self.cov_cache.set(i, j, cov);
                self.cov_cache.set(j, i, cov);
            }
        }
    }

    /// Calculate Marginal VaR using delta-normal method
    /// Marginal VaR_i = (Σ * w)_i / σ_p * z_α
    /// where Σ is covariance matrix, w is weight vector, σ_p is portfolio vol
    #[inline]
    pub fn calculate_marginal_var(&self, asset_idx: usize) -> f64 {
        if asset_idx >= self.num_positions || self.portfolio_value <= 0.0 {
            return 0.0;
        }

        let weight = self.positions[asset_idx].weight;
        if weight <= 0.0 { return 0.0; }

        // Get portfolio volatility (sqrt of portfolio variance)
        let port_var = self.calculate_portfolio_variance();
        let port_vol = port_var.sqrt();
        
        if port_vol <= 0.0 { return 0.0; }

        // Calculate (Σ * w)_i - the i-th element of covariance times weights
        let mut cov_weight_sum = 0.0;
        for j in 0..self.num_positions {
            let cov = self.cov_cache.get(asset_idx, j);
            let w_j = self.positions[j].weight;
            cov_weight_sum += cov * w_j;
        }

        // Marginal VaR = (Σ * w)_i / σ_p * z_α
        let z_alpha = self.get_z_score(self.confidence_level);
        let marginal_var = (cov_weight_sum / port_vol) * z_alpha;

        marginal_var
    }

    /// Calculate Component VaR = weight_i * Marginal VaR_i
    #[inline]
    pub fn calculate_component_var(&self, asset_idx: usize) -> f64 {
        let weight = self.positions[asset_idx].weight;
        let marginal_var = self.calculate_marginal_var(asset_idx);
        weight * marginal_var
    }

    /// Calculate total portfolio VaR
    #[inline]
    pub fn calculate_portfolio_var(&self) -> f64 {
        let port_var = self.calculate_portfolio_variance();
        let port_vol = port_var.sqrt();
        let z_alpha = self.get_z_score(self.confidence_level);
        
        // Scale by time horizon (square root of time rule)
        let time_scale = (self.time_horizon_days as f64 / 252.0).sqrt();
        
        port_vol * z_alpha * time_scale * self.portfolio_value
    }

    /// Calculate portfolio variance = w' * Σ * w
    #[inline]
    fn calculate_portfolio_variance(&self) -> f64 {
        let mut variance = 0.0;
        for i in 0..self.num_positions {
            for j in 0..self.num_positions {
                let cov = self.cov_cache.get(i, j);
                let w_i = self.positions[i].weight;
                let w_j = self.positions[j].weight;
                variance += w_i * w_j * cov;
            }
        }
        variance.max(0.0)
    }

    /// Get all asset VaR breakdowns
    #[inline]
    pub fn get_asset_var_breakdown(&self, out: &mut [AssetVaR]) -> usize {
        let count = self.num_positions.min(out.len());
        let total_var = self.calculate_portfolio_var();

        for i in 0..count {
            let asset = self.positions[i];
            out[i].asset_id = asset.asset_id;
            out[i].marginal_var = self.calculate_marginal_var(i);
            out[i].component_var = self.calculate_component_var(i);
            
            // Incremental VaR approximation
            out[i].incremental_var = out[i].marginal_var * asset.weight * self.portfolio_value;
            
            // Standalone VaR (asset in isolation)
            let asset_var = self.cov_cache.get(i, i);
            let z_alpha = self.get_z_score(self.confidence_level);
            let time_scale = (self.time_horizon_days as f64 / 252.0).sqrt();
            out[i].standalone_var = asset_var.sqrt() * z_alpha * time_scale 
                * asset.quantity * asset.current_price;
            
            // Correlation contribution
            out[i].correlation_contrib = out[i].component_var - 
                (out[i].standalone_var * asset.weight);
        }
        count
    }

    #[inline]
    pub fn get_portfolio_summary(&self) -> PortfolioVaRSummary {
        let mut summary = PortfolioVaRSummary::default();
        
        let port_var_95 = self.calculate_portfolio_variance();
        let port_vol = port_var_95.sqrt();
        
        summary.total_var_95 = port_vol * self.get_z_score(0.95) * self.portfolio_value;
        summary.total_var_99 = port_vol * self.get_z_score(0.99) * self.portfolio_value;
        
        // CVaR (Expected Shortfall) for normal distribution
        // ES_α = φ(Φ^{-1}(α)) / (1-α) * σ
        summary.total_cvar_95 = port_vol * self.get_es_factor(0.95) * self.portfolio_value;
        summary.total_cvar_99 = port_vol * self.get_es_factor(0.99) * self.portfolio_value;
        
        // Diversification ratio = sum of standalone VaRs / portfolio VaR
        let mut standalone_sum = 0.0;
        for i in 0..self.num_positions {
            let asset_var = self.cov_cache.get(i, i);
            standalone_sum += asset_var.sqrt() * self.positions[i].weight;
        }
        if port_vol > 0.0 {
            summary.diversification_ratio = standalone_sum / port_vol;
        }
        
        // Concentration index (Herfindahl of component VaRs)
        let total_var = summary.total_var_95.abs();
        if total_var > 0.0 {
            let mut hhi = 0.0;
            for i in 0..self.num_positions {
                let comp_var = self.calculate_component_var(i);
                let share = comp_var / total_var;
                hhi += share * share;
            }
            summary.concentration_index = hhi;
        }
        
        summary
    }

    #[inline]
    pub fn set_portfolio_value(&mut self, value: f64) {
        self.portfolio_value = value;
    }

    /// Standard normal quantile (Z-score) for confidence level
    #[inline]
    fn get_z_score(&self, confidence: f64) -> f64 {
        // Approximation using Abramowitz and Stegun
        let p = if confidence > 0.5 { 1.0 - confidence } else { confidence };
        let t = (-2.0 * p.ln()).sqrt();
        
        let c0 = 2.515517;
        let c1 = 0.802853;
        let c2 = 0.010328;
        let d1 = 1.432788;
        let d2 = 0.189269;
        let d3 = 0.001308;
        
        let z = t - (c0 + c1*t + c2*t*t) / (1.0 + d1*t + d2*t*t + d3*t*t*t);
        
        if confidence > 0.5 { z } else { -z }
    }

    /// Expected Shortfall factor for normal distribution
    #[inline]
    fn get_es_factor(&self, confidence: f64) -> f64 {
        let z = self.get_z_score(confidence);
        // φ(z) = (1/√(2π)) * e^(-z²/2)
        let phi_z = (-0.5 * z * z).exp() / (2.0 * std::f64::consts::PI).sqrt();
        phi_z / (1.0 - confidence)
    }
}

impl<const MAX_ASSETS: usize, const SAMPLES: usize> Default 
    for MarginalVaRCalculator<MAX_ASSETS, SAMPLES> 
{
    fn default() -> Self {
        Self::new(0.99, 1)
    }
}

/// Lock-free aggregate risk statistics
pub struct RiskAggregate {
    max_component_var_asset: AtomicU64,
    max_component_var_value: AtomicU64,
    total_var_samples: AtomicU64,
    running_avg_var: AtomicU64,
}

impl RiskAggregate {
    pub const fn new() -> Self {
        Self {
            max_component_var_asset: AtomicU64::new(0),
            max_component_var_value: AtomicU64::new(0),
            total_var_samples: AtomicU64::new(0),
            running_avg_var: AtomicU64::new(0),
        }
    }

    #[inline]
    pub fn record_max_component(&self, asset_id: u32, component_var: f64) {
        let scaled = (component_var.abs() * 1_000_000.0) as u64;
        let current_max = self.max_component_var_value.load(Ordering::Relaxed);
        
        if scaled > current_max {
            if self.max_component_var_value.compare_exchange_weak(
                current_max, scaled, Ordering::Relaxed, Ordering::Relaxed).is_ok() {
                self.max_component_var_asset.store(asset_id as u64, Ordering::Relaxed);
            }
        }
    }

    #[inline]
    pub fn get_max_risk_contributor(&self) -> (u32, f64) {
        let asset = self.max_component_var_asset.load(Ordering::Relaxed);
        let value = self.max_component_var_value.load(Ordering::Relaxed);
        (asset as u32, value as f64 / 1_000_000.0)
    }
}

impl Default for RiskAggregate {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_marginal_var_calculation() {
        let mut calc: MarginalVaRCalculator<10, 252> = 
            MarginalVaRCalculator::new(0.99, 1);
        
        calc.set_portfolio_value(1_000_000.0);
        
        // Add two positions
        calc.set_position(Position {
            asset_id: 1, quantity: 10.0, current_price: 100.0, weight: 0.5
        });
        calc.set_position(Position {
            asset_id: 2, quantity: 20.0, current_price: 25.0, weight: 0.5
        });

        // Add some historical returns
        for i in 0..100 {
            calc.add_return(0, 0.01 * (i as f64 % 10 - 5.0) / 100.0);
            calc.add_return(1, 0.015 * (i as f64 % 10 - 5.0) / 100.0);
        }

        calc.compute_covariance_matrix();
        
        let marginal_0 = calc.calculate_marginal_var(0);
        assert!(marginal_0.is_finite());
    }

    #[test]
    fn test_zero_allocation() {
        let mut calc: MarginalVaRCalculator<50, 500> = 
            MarginalVaRCalculator::new(0.95, 1);
        
        for _ in 0..1000 {
            calc.add_return(0, 0.001);
        }
        calc.compute_covariance_matrix();
        let _ = calc.calculate_portfolio_var();
    }
}
