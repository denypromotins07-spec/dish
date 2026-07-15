//! Advanced cross-margin calculator that identifies offsetting exposures.
//! Minimizes global maintenance margin requirements across Spot, Perps, and Options.
//! Frees maximum capital for the alpha engine.

use std::collections::HashMap;

/// Fixed-point precision multiplier (1e8)
const FP_MULTIPLIER: u128 = 1_000_000_000;

/// Represents a position in any instrument type
#[derive(Debug, Clone, Copy)]
pub struct Position {
    pub instrument_id: u64,
    pub symbol: [u8; 16], // Fixed-size string for zero allocation
    pub quantity: i64,    // Signed quantity
    pub notional_value: u128, // In quote currency * 1e8
    pub position_type: PositionType,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PositionType {
    Spot,
    Perp,
    Future,
    CallOption,
    PutOption,
}

/// Margin requirements for an instrument
#[derive(Debug, Clone, Copy)]
pub struct MarginRequirement {
    pub initial_margin: u128, // In quote currency * 1e8
    pub maintenance_margin: u128, // In quote currency * 1e8
    pub margin_rate: u32, // Basis points (e.g., 500 = 5%)
}

/// Net exposure after cross-margin offsetting
#[derive(Debug, Clone)]
pub struct NetExposure {
    pub instrument_id: u64,
    pub gross_long: u128,
    pub gross_short: u128,
    pub net_notional: i128, // Can be negative for net short
    pub offset_amount: u128, // Amount offset by opposite positions
    pub margin_savings: u128, // Margin saved due to offsetting
}

/// Cross-margin optimization result
#[derive(Debug, Clone)]
pub struct CrossMarginResult {
    pub total_gross_exposure: u128,
    pub total_net_exposure: u128,
    pub total_initial_margin_required: u128,
    pub total_maintenance_margin_required: u128,
    pub margin_savings_from_offsetting: u128,
    pub capital_efficiency_ratio: f64, // net / gross (higher is better)
    pub net_exposures: Vec<NetExposure>,
}

/// Advanced cross-margin optimizer
pub struct CrossMarginOptimizer {
    /// All positions by instrument
    positions: HashMap<u64, Vec<Position>>,
    /// Margin rates by instrument (basis points)
    margin_rates: HashMap<u64, u32>,
    /// Correlation matrix for portfolio margin (simplified)
    correlation_groups: HashMap<u64, u64>, // instrument -> group_id
}

impl CrossMarginOptimizer {
    pub fn new() -> Self {
        Self {
            positions: HashMap::with_capacity(256),
            margin_rates: HashMap::with_capacity(256),
            correlation_groups: HashMap::new(),
        }
    }

    /// Add or update a position
    pub fn add_position(&mut self, position: Position) {
        self.positions
            .entry(position.instrument_id)
            .or_insert_with(|| Vec::with_capacity(8))
            .push(position);
    }

    /// Set margin rate for an instrument (in basis points)
    pub fn set_margin_rate(&mut self, instrument_id: u64, rate_bps: u32) {
        self.margin_rates.insert(instrument_id, rate_bps);
    }

    /// Set correlation group for portfolio margin calculation
    pub fn set_correlation_group(&mut self, instrument_id: u64, group_id: u64) {
        self.correlation_groups.insert(instrument_id, group_id);
    }

    /// Calculate net exposure for a single instrument
    fn calculate_net_exposure(&self, instrument_id: u64) -> NetExposure {
        let positions = self.positions.get(&instrument_id);
        
        if positions.is_none() || positions.unwrap().is_empty() {
            return NetExposure {
                instrument_id,
                gross_long: 0,
                gross_short: 0,
                net_notional: 0,
                offset_amount: 0,
                margin_savings: 0,
            };
        }

        let mut gross_long = 0u128;
        let mut gross_short = 0u128;

        for pos in positions.unwrap() {
            if pos.quantity > 0 {
                gross_long += pos.notional_value;
            } else {
                gross_short += pos.notional_value;
            }
        }

        let net_notional = gross_long as i128 - gross_short as i128;
        let offset_amount = gross_long.min(gross_short);
        
        // Calculate margin savings from offsetting
        let margin_rate = self.margin_rates.get(&instrument_id).copied().unwrap_or(500); // Default 5%
        let margin_savings = (offset_amount * margin_rate as u128) / 10000;

        NetExposure {
            instrument_id,
            gross_long,
            gross_short,
            net_notional,
            offset_amount,
            margin_savings,
        }
    }

    /// Calculate portfolio-wide cross-margin optimization
    pub fn optimize(&self) -> CrossMarginResult {
        let mut total_gross = 0u128;
        let mut total_net_abs = 0u128;
        let mut total_initial_margin = 0u128;
        let mut total_maintenance_margin = 0u128;
        let mut total_margin_savings = 0u128;
        let mut net_exposures = Vec::with_capacity(self.positions.len());

        // Calculate net exposure per instrument
        for &instrument_id in self.positions.keys() {
            let net_exp = self.calculate_net_exposure(instrument_id);
            
            total_gross += net_exp.gross_long + net_exp.gross_short;
            total_net_abs += net_exp.net_notional.abs() as u128;
            total_margin_savings += net_exp.margin_savings;
            net_exposures.push(net_exp);

            // Calculate margin requirements for net exposure
            let margin_rate = self.margin_rates.get(&instrument_id).copied().unwrap_or(500);
            let net_notional_abs = net_exp.net_notional.abs() as u128;
            
            // Initial margin (typically higher)
            let initial_rate = margin_rate; 
            let maintenance_rate = (margin_rate as f64 * 0.8) as u32; // 80% of initial

            total_initial_margin += (net_notional_abs * initial_rate as u128) / 10000;
            total_maintenance_margin += (net_notional_abs * maintenance_rate as u128) / 10000;
        }

        // Apply portfolio margin benefits for correlated instruments
        let portfolio_adjustment = self.calculate_portfolio_margin_adjustment();
        total_initial_margin = (total_initial_margin as f64 * portfolio_adjustment) as u128;
        total_maintenance_margin = (total_maintenance_margin as f64 * portfolio_adjustment) as u128;

        let capital_efficiency = if total_gross > 0 {
            total_net_abs as f64 / total_gross as f64
        } else {
            1.0
        };

        CrossMarginResult {
            total_gross_exposure: total_gross,
            total_net_exposure: total_net_abs,
            total_initial_margin_required: total_initial_margin,
            total_maintenance_margin_required: total_maintenance_margin,
            margin_savings_from_offsetting: total_margin_savings,
            capital_efficiency_ratio: capital_efficiency,
            net_exposures,
        }
    }

    /// Calculate portfolio margin adjustment based on correlations
    /// Returns a multiplier < 1.0 for diversification benefit
    fn calculate_portfolio_margin_adjustment(&self) -> f64 {
        if self.correlation_groups.is_empty() {
            return 1.0; // No portfolio margin benefit
        }

        // Group positions by correlation group
        let mut group_exposures: HashMap<u64, i128> = HashMap::new();
        
        for (&instrument_id, positions) in &self.positions {
            let group_id = self.correlation_groups.get(&instrument_id).copied().unwrap_or(0);
            let net: i128 = positions.iter().map(|p| {
                if p.quantity > 0 { p.notional_value as i128 } else { -(p.notional_value as i128) }
            }).sum();
            
            *group_exposures.entry(group_id).or_insert(0) += net;
        }

        // Calculate diversification benefit
        let sum_abs = group_exposures.values().map(|v| v.abs()).sum::<i128>() as f64;
        let abs_sum = group_exposures.values().sum::<i128>().abs() as f64;
        
        if sum_abs > 0.0 {
            // Diversification ratio (lower means more diversification)
            let ratio = abs_sum / sum_abs;
            // Map to adjustment factor (0.7 to 1.0 range)
            0.7 + (ratio * 0.3)
        } else {
            1.0
        }
    }

    /// Get freed capital available for alpha engine
    pub fn get_freed_capital(&self) -> u128 {
        let result = self.optimize();
        result.margin_savings_from_offsetting
    }

    /// Clear all positions
    pub fn clear(&mut self) {
        self.positions.clear();
        self.margin_rates.clear();
        self.correlation_groups.clear();
    }
}

impl Default for CrossMarginOptimizer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cross_margin_offsetting() {
        let mut optimizer = CrossMarginOptimizer::new();
        
        // Add long spot position
        optimizer.add_position(Position {
            instrument_id: 1,
            symbol: *b"BTC             ",
            quantity: 100,
            notional_value: 5_000_000_000_000, // $5M
            position_type: PositionType::Spot,
        });
        
        // Add short perp position (hedging)
        optimizer.add_position(Position {
            instrument_id: 1,
            symbol: *b"BTC-PERP        ",
            quantity: -100,
            notional_value: 5_000_000_000_000, // $5M
            position_type: PositionType::Perp,
        });
        
        optimizer.set_margin_rate(1, 500); // 5% margin
        
        let result = optimizer.optimize();
        
        // Net exposure should be near zero
        assert_eq!(result.total_net_exposure, 0);
        
        // Should have significant margin savings
        assert!(result.margin_savings_from_offsetting > 0);
        
        // Capital efficiency should be low (well-hedged)
        assert!(result.capital_efficiency_ratio < 0.1);
    }
}
