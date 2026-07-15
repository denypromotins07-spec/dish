//! Efficient frontier generator that rapidly calculates the optimal risk-return tradeoff curve
//! Feeds target returns to the execution engine

use crate::covariance_matrix::CovarianceMatrix;
use crate::mean_variance_optimizer::{MeanVarianceOptimizer, MVOConstraints, MVOResult};

const MAX_POINTS: usize = 100;
const MAX_ASSETS: usize = 128;

/// A single point on the efficient frontier
#[derive(Debug, Clone)]
pub struct FrontierPoint {
    pub expected_return: f64,
    pub portfolio_variance: f64,
    pub sharpe_ratio: f64,
    pub weights: [f64; MAX_ASSETS],
    pub asset_count: usize,
}

/// Efficient frontier result
#[derive(Debug, Clone)]
pub struct EfficientFrontier {
    pub points: Vec<FrontierPoint>,
    pub min_return: f64,
    pub max_return: f64,
    pub min_variance: f64,
    pub max_sharpe_point: Option<usize>,
}

impl EfficientFrontier {
    #[inline(always)]
    pub fn new() -> Self {
        Self {
            points: Vec::with_capacity(MAX_POINTS),
            min_return: 0.0,
            max_return: 0.0,
            min_variance: f64::MAX,
            max_sharpe_point: None,
        }
    }
}

impl Default for EfficientFrontier {
    fn default() -> Self {
        Self::new()
    }
}

/// Efficient Frontier Generator
pub struct FrontierGenerator {
    expected_returns: [f64; MAX_ASSETS],
    covariance: CovarianceMatrix,
    asset_count: usize,
}

impl FrontierGenerator {
    #[inline(always)]
    pub fn new(expected_returns: &[f64], covariance: CovarianceMatrix) -> Self {
        let asset_count = covariance.asset_count();
        assert_eq!(expected_returns.len(), asset_count);
        
        let mut returns = [0.0; MAX_ASSETS];
        returns[..asset_count].copy_from_slice(&expected_returns[..asset_count]);
        
        Self {
            expected_returns: returns,
            covariance,
            asset_count,
        }
    }

    /// Generate the full efficient frontier
    #[inline(always)]
    pub fn generate(&self, num_points: usize) -> EfficientFrontier {
        let mut frontier = EfficientFrontier::new();
        let num_points = num_points.min(MAX_POINTS);
        
        // Find min and max feasible returns
        let min_return = *self.expected_returns[..self.asset_count].iter().min_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
        let max_return = *self.expected_returns[..self.asset_count].iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
        
        frontier.min_return = min_return;
        frontier.max_return = max_return;
        
        let mut max_sharpe = f64::NEG_INFINITY;
        let mut max_sharpe_idx = None;
        
        // Generate points along the return spectrum
        for i in 0..num_points {
            let target_return = min_return + (max_return - min_return) * (i as f64 / (num_points - 1) as f64);
            
            let optimizer = MeanVarianceOptimizer::new(&self.expected_returns[..self.asset_count], self.covariance.clone());
            
            let mut constraints = MVOConstraints::new(self.asset_count);
            constraints.set_bounds(0.0, 1.0);
            constraints.target_return = Some(target_return);
            
            let result = optimizer.optimize_min_variance(&constraints);
            
            if result.portfolio_variance > 0.0 && result.portfolio_variance.is_finite() {
                let point = FrontierPoint {
                    expected_return: result.expected_return,
                    portfolio_variance: result.portfolio_variance,
                    sharpe_ratio: result.sharpe_ratio,
                    weights: result.weights,
                    asset_count: result.asset_count,
                };
                
                if result.portfolio_variance < frontier.min_variance {
                    frontier.min_variance = result.portfolio_variance;
                }
                
                if result.sharpe_ratio > max_sharpe {
                    max_sharpe = result.sharpe_ratio;
                    max_sharpe_idx = Some(frontier.points.len());
                }
                
                frontier.points.push(point);
            }
        }
        
        frontier.max_sharpe_point = max_sharpe_idx;
        frontier
    }

    /// Generate frontier with custom weight constraints
    #[inline(always)]
    pub fn generate_with_constraints(
        &self,
        num_points: usize,
        min_weights: &[f64],
        max_weights: &[f64],
        leverage_limit: f64,
    ) -> EfficientFrontier {
        let mut frontier = EfficientFrontier::new();
        let num_points = num_points.min(MAX_POINTS);
        
        let min_return = *self.expected_returns[..self.asset_count].iter().min_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
        let max_return = *self.expected_returns[..self.asset_count].iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
        
        frontier.min_return = min_return;
        frontier.max_return = max_return;
        
        let mut max_sharpe = f64::NEG_INFINITY;
        let mut max_sharpe_idx = None;
        
        for i in 0..num_points {
            let target_return = min_return + (max_return - min_return) * (i as f64 / (num_points - 1) as f64);
            
            let optimizer = MeanVarianceOptimizer::new(&self.expected_returns[..self.asset_count], self.covariance.clone());
            
            let mut constraints = MVOConstraints::new(self.asset_count);
            for j in 0..self.asset_count {
                constraints.min_weights[j] = min_weights.get(j).copied().unwrap_or(0.0);
                constraints.max_weights[j] = max_weights.get(j).copied().unwrap_or(1.0);
            }
            constraints.target_return = Some(target_return);
            constraints.leverage_limit = leverage_limit;
            
            let result = optimizer.optimize_min_variance(&constraints);
            
            if result.portfolio_variance > 0.0 && result.portfolio_variance.is_finite() {
                let point = FrontierPoint {
                    expected_return: result.expected_return,
                    portfolio_variance: result.portfolio_variance,
                    sharpe_ratio: result.sharpe_ratio,
                    weights: result.weights,
                    asset_count: result.asset_count,
                };
                
                if result.portfolio_variance < frontier.min_variance {
                    frontier.min_variance = result.portfolio_variance;
                }
                
                if result.sharpe_ratio > max_sharpe {
                    max_sharpe = result.sharpe_ratio;
                    max_sharpe_idx = Some(frontier.points.len());
                }
                
                frontier.points.push(point);
            }
        }
        
        frontier.max_sharpe_point = max_sharpe_idx;
        frontier
    }

    /// Get the tangency portfolio (maximum Sharpe ratio point)
    #[inline(always)]
    pub fn tangency_portfolio(&self, risk_free_rate: f64) -> FrontierPoint {
        let optimizer = MeanVarianceOptimizer::new(&self.expected_returns[..self.asset_count], self.covariance.clone());
        
        let mut constraints = MVOConstraints::new(self.asset_count);
        constraints.set_bounds(0.0, 1.0);
        constraints.risk_free_rate = risk_free_rate;
        
        let result = optimizer.optimize_max_sharpe(&constraints);
        
        FrontierPoint {
            expected_return: result.expected_return,
            portfolio_variance: result.portfolio_variance,
            sharpe_ratio: result.sharpe_ratio,
            weights: result.weights,
            asset_count: result.asset_count,
        }
    }

    /// Interpolate weights for a target return on the frontier
    #[inline(always)]
    pub fn interpolate_weights(&self, frontier: &EfficientFrontier, target_return: f64) -> Option<[f64; MAX_ASSETS]> {
        if frontier.points.is_empty() {
            return None;
        }
        
        // Find bracketing points
        let mut lower_idx = None;
        let mut upper_idx = None;
        
        for (i, point) in frontier.points.iter().enumerate() {
            if point.expected_return <= target_return {
                lower_idx = Some(i);
            }
            if point.expected_return >= target_return && upper_idx.is_none() {
                upper_idx = Some(i);
                break;
            }
        }
        
        match (lower_idx, upper_idx) {
            (Some(lower), Some(upper)) if lower == upper => {
                Some(frontier.points[lower].weights)
            }
            (Some(lower), Some(upper)) => {
                let lower_ret = frontier.points[lower].expected_return;
                let upper_ret = frontier.points[upper].expected_return;
                
                if (upper_ret - lower_ret).abs() < 1e-10 {
                    return Some(frontier.points[lower].weights);
                }
                
                let alpha = (target_return - lower_ret) / (upper_ret - lower_ret);
                
                let mut weights = [0.0; MAX_ASSETS];
                for i in 0..self.asset_count {
                    weights[i] = frontier.points[lower].weights[i] * (1.0 - alpha) 
                               + frontier.points[upper].weights[i] * alpha;
                }
                
                Some(weights)
            }
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_frontier_generation() {
        let mut cov = CovarianceMatrix::new(3);
        cov.data[0][0] = 0.04;
        cov.data[1][1] = 0.09;
        cov.data[2][2] = 0.16;
        cov.asset_count = 3;
        
        let returns = vec![0.10, 0.15, 0.20];
        let generator = FrontierGenerator::new(&returns, cov);
        
        let frontier = generator.generate(20);
        
        assert!(!frontier.points.is_empty());
        assert!(frontier.min_return > 0.0);
        assert!(frontier.max_return <= 0.20);
        assert!(frontier.max_sharpe_point.is_some());
    }
}
