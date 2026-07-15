//! Delta-neutral hedging logic for complex portfolios.
//! Automatically calculates hedge ratios and executes offsetting trades for market neutrality.

use std::sync::atomic::{AtomicF64, AtomicBool, Ordering};
use std::collections::HashMap;

/// Portfolio exposure summary
#[derive(Debug, Clone)]
pub struct PortfolioExposure {
    pub total_delta: f64,
    pub total_gamma: f64,
    pub total_vega: f64,
    pub net_notional: f64,
    pub gross_notional: f64,
    pub long_exposure: f64,
    pub short_exposure: f64,
}

/// Hedge recommendation
#[derive(Debug, Clone)]
pub struct HedgeRecommendation {
    pub symbol: String,
    pub side: HedgeSide,
    pub quantity: f64,
    pub notional_value: f64,
    pub expected_delta_reduction: f64,
    pub hedge_ratio: f64,
    pub urgency: f64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum HedgeSide {
    Buy,
    Sell,
}

/// Delta hedging engine
pub struct DeltaHedger {
    /// Target delta (usually 0 for neutral)
    target_delta: AtomicF64,
    /// Maximum allowed delta deviation
    max_delta_deviation: AtomicF64,
    /// Current portfolio delta
    current_delta: AtomicF64,
    /// Hedge threshold (rebalance when delta exceeds this)
    hedge_threshold: AtomicF64,
    /// Enable auto-hedging
    auto_hedge_enabled: AtomicBool,
    /// Correlation matrix for cross-asset hedging
    correlations: HashMap<(String, String), f64>,
    /// Beta coefficients for each asset vs hedge instrument
    betas: HashMap<String, f64>,
}

impl DeltaHedger {
    /// Create new delta hedger
    pub fn new(target_delta: f64, max_deviation: f64) -> Self {
        Self {
            target_delta: AtomicF64::new(target_delta),
            max_delta_deviation: AtomicF64::new(max_deviation),
            current_delta: AtomicF64::new(0.0),
            hedge_threshold: AtomicF64::new(max_deviation * 0.5),
            auto_hedge_enabled: AtomicBool::new(true),
            correlations: HashMap::new(),
            betas: HashMap::new(),
        }
    }

    /// Update current portfolio delta
    #[inline(always)]
    pub fn update_portfolio_delta(&self, delta: f64) {
        self.current_delta.store(delta, Ordering::Relaxed);
    }

    /// Get current delta
    #[inline(always)]
    pub fn get_current_delta(&self) -> f64 {
        self.current_delta.load(Ordering::Relaxed)
    }

    /// Check if rebalancing is needed
    pub fn needs_rebalancing(&self) -> bool {
        if !self.auto_hedge_enabled.load(Ordering::Relaxed) {
            return false;
        }
        
        let current = self.current_delta.load(Ordering::Relaxed);
        let target = self.target_delta.load(Ordering::Relaxed);
        let threshold = self.hedge_threshold.load(Ordering::Relaxed);
        
        (current - target).abs() > threshold
    }

    /// Calculate hedge quantity needed to reach target delta
    pub fn calculate_hedge_quantity(
        &self,
        hedge_instrument_price: f64,
        hedge_instrument_delta: f64,
    ) -> Option<HedgeRecommendation> {
        let current_delta = self.current_delta.load(Ordering::Relaxed);
        let target_delta = self.target_delta.load(Ordering::Relaxed);
        
        let delta_gap = target_delta - current_delta;
        if delta_gap.abs() < self.hedge_threshold.load(Ordering::Relaxed) {
            return None;
        }
        
        if hedge_instrument_price <= 0.0 || hedge_instrument_delta == 0.0 {
            return None;
        }
        
        // Quantity needed to offset delta
        let hedge_quantity = delta_gap.abs() / (hedge_instrument_price * hedge_instrument_delta);
        
        let side = if delta_gap > 0.0 {
            HedgeSide::Buy  // Need positive delta
        } else {
            HedgeSide::Sell // Need negative delta
        };
        
        let notional = hedge_quantity * hedge_instrument_price;
        let expected_reduction = hedge_quantity * hedge_instrument_price * hedge_instrument_delta;
        
        Some(HedgeRecommendation {
            symbol: "HEDGE".to_string(),
            side,
            quantity: hedge_quantity,
            notional_value: notional,
            expected_delta_reduction: expected_reduction,
            hedge_ratio: expected_reduction / delta_gap.abs(),
            urgency: (delta_gap.abs() / self.max_delta_deviation.load(Ordering::Relaxed)).min(1.0),
        })
    }

    /// Calculate optimal hedge using futures (basis-aware)
    pub fn calculate_futures_hedge(
        &self,
        spot_positions: &HashMap<String, f64>, // symbol -> notional
        futures_price: f64,
        spot_price: f64,
        basis_bps: f64,
    ) -> Vec<HedgeRecommendation> {
        let mut recommendations = Vec::new();
        
        let total_spot_exposure: f64 = spot_positions.values().sum();
        if total_spot_exposure.abs() < 0.001 {
            return recommendations;
        }
        
        // Adjust for basis (futures might be at premium/discount)
        let basis_adjustment = 1.0 + (basis_bps / 10000.0);
        let fair_futures_price = spot_price * basis_adjustment;
        
        // Calculate hedge ratio considering basis
        let hedge_ratio = if fair_futures_price > 0.0 {
            spot_price / fair_futures_price
        } else {
            1.0
        };
        
        let futures_qty = (total_spot_exposure.abs() / futures_price) * hedge_ratio;
        
        let side = if total_spot_exposure > 0.0 {
            HedgeSide::Sell // Long spot, short futures
        } else {
            HedgeSide::Buy // Short spot, long futures
        };
        
        recommendations.push(HedgeRecommendation {
            symbol: "FUTURES_HEDGE".to_string(),
            side,
            quantity: futures_qty,
            notional_value: futures_qty * futures_price,
            expected_delta_reduction: total_spot_exposure.abs() * 0.95, // ~95% effective
            hedge_ratio,
            urgency: 0.5,
        });
        
        recommendations
    }

    /// Cross-asset hedging using correlated instruments
    pub fn calculate_cross_asset_hedge(
        &self,
        primary_symbol: &str,
        primary_exposure: f64,
        hedge_symbol: &str,
        hedge_price: f64,
    ) -> Option<HedgeRecommendation> {
        // Get correlation between assets
        let correlation = self.correlations.get(&(primary_symbol.to_string(), hedge_symbol.to_string()))
            .copied()
            .unwrap_or(0.5); // Default moderate correlation
        
        // Get beta (sensitivity of primary to hedge)
        let beta = self.betas.get(primary_symbol)
            .copied()
            .unwrap_or(1.0);
        
        if hedge_price <= 0.0 || primary_exposure.abs() < 0.001 {
            return None;
        }
        
        // Optimal hedge ratio with correlation adjustment
        let adjusted_hedge_ratio = beta * correlation;
        
        let hedge_qty = (primary_exposure.abs() / hedge_price) * adjusted_hedge_ratio;
        
        let side = if primary_exposure > 0.0 {
            HedgeSide::Sell
        } else {
            HedgeSide::Buy
        };
        
        Some(HedgeRecommendation {
            symbol: hedge_symbol.to_string(),
            side,
            quantity: hedge_qty,
            notional_value: hedge_qty * hedge_price,
            expected_delta_reduction: primary_exposure.abs() * correlation,
            hedge_ratio: adjusted_hedge_ratio,
            urgency: correlation * 0.7, // Higher correlation = higher urgency
        })
    }

    /// Set correlation between two assets
    #[inline(always)]
    pub fn set_correlation(&mut self, asset_a: &str, asset_b: &str, correlation: f64) {
        self.correlations.insert(
            (asset_a.to_string(), asset_b.to_string()),
            correlation.clamp(-1.0, 1.0),
        );
    }

    /// Set beta coefficient for an asset
    #[inline(always)]
    pub fn set_beta(&mut self, asset: &str, beta: f64) {
        self.betas.insert(asset.to_string(), beta);
    }

    /// Enable/disable auto-hedging
    #[inline(always)]
    pub fn set_auto_hedge_enabled(&self, enabled: bool) {
        self.auto_hedge_enabled.store(enabled, Ordering::Relaxed);
    }

    /// Set hedge threshold
    #[inline(always)]
    pub fn set_hedge_threshold(&self, threshold: f64) {
        self.hedge_threshold.store(threshold.max(0.0), Ordering::Relaxed);
    }

    /// Calculate portfolio Greeks summary
    pub fn calculate_portfolio_greeks(
        &self,
        positions: &[PositionGreeks],
    ) -> PortfolioExposure {
        let mut total_delta = 0.0;
        let mut total_gamma = 0.0;
        let mut total_vega = 0.0;
        let mut net_notional = 0.0;
        let mut gross_notional = 0.0;
        let mut long_exposure = 0.0;
        let mut short_exposure = 0.0;
        
        for pos in positions {
            total_delta += pos.delta;
            total_gamma += pos.gamma;
            total_vega += pos.vega;
            
            let notional = pos.quantity * pos.price;
            net_notional += notional * pos.side_multiplier;
            gross_notional += notional.abs();
            
            if pos.side_multiplier > 0.0 {
                long_exposure += notional;
            } else {
                short_exposure += notional.abs();
            }
        }
        
        // Update internal delta state
        self.update_portfolio_delta(total_delta);
        
        PortfolioExposure {
            total_delta,
            total_gamma,
            total_vega,
            net_notional,
            gross_notional,
            long_exposure,
            short_exposure,
        }
    }

    /// Execute hedge (returns order details, actual execution handled elsewhere)
    pub fn execute_hedge(&self, recommendation: &HedgeRecommendation) -> HedgeExecution {
        HedgeExecution {
            symbol: recommendation.symbol.clone(),
            side: recommendation.side,
            quantity: recommendation.quantity,
            estimated_cost: recommendation.notional_value,
            expected_delta_after: self.current_delta.load(Ordering::Relaxed) 
                + recommendation.expected_delta_reduction * if matches!(recommendation.side, HedgeSide::Buy) { 1.0 } else { -1.0 },
            execution_priority: recommendation.urgency > 0.7,
        }
    }
}

/// Position Greeks for portfolio calculation
#[derive(Debug, Clone)]
pub struct PositionGreeks {
    pub symbol: String,
    pub quantity: f64,
    pub price: f64,
    pub delta: f64,
    pub gamma: f64,
    pub vega: f64,
    pub side_multiplier: f64, // 1.0 for long, -1.0 for short
}

/// Hedge execution result
#[derive(Debug, Clone)]
pub struct HedgeExecution {
    pub symbol: String,
    pub side: HedgeSide,
    pub quantity: f64,
    pub estimated_cost: f64,
    pub expected_delta_after: f64,
    pub execution_priority: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_delta_hedger() {
        let hedger = DeltaHedger::new(0.0, 10_000.0); // Target 0 delta, max 10k deviation
        
        hedger.update_portfolio_delta(5_000.0); // Long 5k delta
        
        assert!(hedger.needs_rebalancing());
        
        let hedge = hedger.calculate_hedge_quantity(50_000.0, 1.0);
        assert!(hedge.is_some());
        
        let hedge = hedge.unwrap();
        assert_eq!(hedge.side, HedgeSide::Sell);
        assert!((hedge.quantity - 0.1).abs() < 0.001); // 5000 / 50000 = 0.1
    }

    #[test]
    fn test_futures_hedge() {
        let hedger = DeltaHedger::new(0.0, 10_000.0);
        
        let mut spots = HashMap::new();
        spots.insert("BTC".to_string(), 100_000.0); // Long 100k spot
        
        let hedges = hedger.calculate_futures_hedge(&spots, 50_000.0, 49_500.0, 100.0);
        
        assert!(!hedges.is_empty());
        assert_eq!(hedges[0].side, HedgeSide::Sell);
    }

    #[test]
    fn test_cross_asset_hedge() {
        let mut hedger = DeltaHedger::new(0.0, 10_000.0);
        hedger.set_correlation("ETH", "BTC", 0.85);
        hedger.set_beta("ETH", 1.2);
        
        let hedge = hedger.calculate_cross_asset_hedge("ETH", 50_000.0, "BTC", 50_000.0);
        
        assert!(hedge.is_some());
        let hedge = hedge.unwrap();
        // Hedge ratio = 1.2 * 0.85 = 1.02
        assert!((hedge.hedge_ratio - 1.02).abs() < 0.01);
    }

    #[test]
    fn test_portfolio_greeks() {
        let hedger = DeltaHedger::new(0.0, 10_000.0);
        
        let positions = vec![
            PositionGreeks {
                symbol: "BTC".to_string(),
                quantity: 1.0,
                price: 50_000.0,
                delta: 50_000.0,
                gamma: 0.0,
                vega: 0.0,
                side_multiplier: 1.0,
            },
            PositionGreeks {
                symbol: "ETH".to_string(),
                quantity: -5.0,
                price: 3_000.0,
                delta: -15_000.0,
                gamma: 0.0,
                vega: 0.0,
                side_multiplier: -1.0,
            },
        ];
        
        let exposure = hedger.calculate_portfolio_greeks(&positions);
        
        assert!((exposure.total_delta - 35_000.0).abs() < 0.01);
        assert!((exposure.net_notional - 35_000.0).abs() < 0.01);
    }
}
