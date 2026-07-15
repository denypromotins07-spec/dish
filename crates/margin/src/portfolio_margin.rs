//! Portfolio Margin / Multi-asset offset calculator
//! Identifies hedged positions to calculate true, reduced margin requirement

use std::collections::HashMap;

/// Single position in portfolio
#[derive(Debug, Clone, Copy)]
pub struct PortfolioPosition {
    pub symbol: [u8; 16],
    pub asset_type: AssetType,
    pub side: PositionSide,
    pub size: f64,
    pub price: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AssetType {
    Spot,
    Perpetual,
    Futures,
    OptionCall,
    OptionPut,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PositionSide {
    Long,
    Short,
}

/// Correlation group for offsetting positions
#[derive(Debug, Clone)]
pub struct CorrelationGroup {
    pub assets: Vec<[u8; 16]>,
    /// Correlation coefficient between assets (0.0 to 1.0)
    pub correlation: f64,
    /// Offset percentage allowed
    pub offset_pct: f64,
}

/// Portfolio margin calculation result
#[derive(Debug, Clone, Copy)]
pub struct PortfolioMarginResult {
    /// Total gross notional value
    pub gross_notional: f64,
    /// Net notional after offsets
    pub net_notional: f64,
    /// Gross margin requirement (no offsets)
    pub gross_margin: f64,
    /// Net margin requirement (with offsets)
    pub net_margin: f64,
    /// Margin saved through offsets
    pub margin_savings: f64,
    /// Savings percentage
    pub savings_pct: f64,
}

/// Portfolio Margin Calculator with multi-asset offsets
pub struct PortfolioMarginCalculator {
    /// Margin rates by asset type
    margin_rates: HashMap<AssetType, f64>,
    /// Correlation groups for offsetting
    correlation_groups: Vec<CorrelationGroup>,
    /// Symbol to correlation group mapping
    symbol_group_map: HashMap<[u8; 16], usize>,
    /// Cross-margin discount factors
    cross_margin_discount: f64,
}

impl PortfolioMarginCalculator {
    pub fn new() -> Self {
        let mut calc = Self {
            margin_rates: HashMap::new(),
            correlation_groups: Vec::new(),
            symbol_group_map: HashMap::new(),
            cross_margin_discount: 0.5, // 50% offset for perfectly correlated hedges
        };

        calc.set_defaults();
        calc
    }

    fn set_defaults(&mut self) {
        // Set margin rates by asset type
        self.margin_rates.insert(AssetType::Spot, 1.0); // 100% for spot
        self.margin_rates.insert(AssetType::Perpetual, 0.01); // 1% initial
        self.margin_rates.insert(AssetType::Futures, 0.01);
        self.margin_rates.insert(AssetType::OptionCall, 0.15); // 15% for options
        self.margin_rates.insert(AssetType::OptionPut, 0.15);

        // Create BTC correlation group
        let btc_assets = vec![
            *b"BTC           ",
            *b"BTC-PERP      ",
            *b"BTC-FUTURES   ",
        ];
        
        self.correlation_groups.push(CorrelationGroup {
            assets: btc_assets.clone(),
            correlation: 0.99,
            offset_pct: 0.95, // 95% offset for BTC hedge
        });

        // Map symbols to group
        for (idx, &symbol) in btc_assets.iter().enumerate() {
            self.symbol_group_map.insert(symbol, idx);
        }

        // Create ETH correlation group
        let eth_assets = vec![
            *b"ETH           ",
            *b"ETH-PERP      ",
            *b"ETH-FUTURES   ",
        ];

        let eth_idx = self.correlation_groups.len();
        self.correlation_groups.push(CorrelationGroup {
            assets: eth_assets.clone(),
            correlation: 0.99,
            offset_pct: 0.95,
        });

        for symbol in eth_assets {
            self.symbol_group_map.insert(symbol, eth_idx);
        }
    }

    /// Calculate portfolio margin with offsets
    pub fn calculate_portfolio_margin(&self, positions: &[PortfolioPosition]) -> PortfolioMarginResult {
        if positions.is_empty() {
            return PortfolioMarginResult {
                gross_notional: 0.0,
                net_notional: 0.0,
                gross_margin: 0.0,
                net_margin: 0.0,
                margin_savings: 0.0,
                savings_pct: 0.0,
            };
        }

        // Calculate gross notional and margin
        let mut gross_notional = 0.0;
        let mut gross_margin = 0.0;

        for pos in positions {
            let notional = pos.size * pos.price;
            gross_notional += notional.abs();

            let rate = self.get_margin_rate(pos.asset_type);
            gross_margin += notional.abs() * rate;
        }

        // Calculate net margin with offsets
        let net_margin = self.calculate_net_margin_with_offsets(positions);

        let margin_savings = gross_margin - net_margin;
        let savings_pct = if gross_margin > 0.0 {
            margin_savings / gross_margin * 100.0
        } else {
            0.0
        };

        // Net notional is the sum of absolute net exposures per group
        let net_notional = self.calculate_net_notional(positions);

        PortfolioMarginResult {
            gross_notional,
            net_notional,
            gross_margin,
            net_margin,
            margin_savings,
            savings_pct,
        }
    }

    /// Get margin rate for asset type
    #[inline]
    fn get_margin_rate(&self, asset_type: AssetType) -> f64 {
        *self.margin_rates.get(&asset_type).unwrap_or(&0.01)
    }

    /// Calculate net margin with correlation-based offsets
    fn calculate_net_margin_with_offsets(&self, positions: &[PortfolioPosition]) -> f64 {
        // Group positions by correlation group
        let mut group_exposures: HashMap<usize, HashMap<(AssetType, PositionSide), f64>> = HashMap::new();
        let mut ungrouped_margin = 0.0;

        for pos in positions {
            if let Some(&group_idx) = self.symbol_group_map.get(&pos.symbol) {
                let group = group_exposures.entry(group_idx).or_insert_with(HashMap::new);
                let key = (pos.asset_type, pos.side);
                let notional = pos.size * pos.price;
                
                *group.entry(key).or_insert(0.0) += notional;
            } else {
                // Ungrouped positions get full margin
                let rate = self.get_margin_rate(pos.asset_type);
                ungrouped_margin += (pos.size * pos.price).abs() * rate;
            }
        }

        // Calculate offset margin for each group
        let mut offset_margin = 0.0;

        for (group_idx, exposures) in group_exposures {
            let group = &self.correlation_groups[group_idx];
            
            // Sum long and short exposures separately
            let mut long_exposure = 0.0;
            let mut short_exposure = 0.0;

            for ((_, side), &notional) in &exposures {
                match side {
                    PositionSide::Long => long_exposure += notional,
                    PositionSide::Short => short_exposure += notional.abs(),
                }
            }

            // Calculate offset
            let matched = long_exposure.min(short_exposure);
            let unmatched_long = (long_exposure - matched).max(0.0);
            let unmatched_short = (short_exposure - matched).max(0.0);

            // Apply offset percentage to matched portion
            let matched_margin = matched * self.get_margin_rate(AssetType::Perpetual) * (1.0 - group.offset_pct);
            let unmatched_margin = (unmatched_long + unmatched_short) * self.get_margin_rate(AssetType::Perpetual);

            offset_margin += matched_margin + unmatched_margin;
        }

        offset_margin + ungrouped_margin
    }

    /// Calculate net notional exposure
    fn calculate_net_notional(&self, positions: &[PortfolioPosition]) -> f64 {
        let mut net_by_group: HashMap<usize, f64> = HashMap::new();
        let mut ungrouped_notional = 0.0;

        for pos in positions {
            let notional = pos.size * pos.price * match pos.side {
                PositionSide::Long => 1.0,
                PositionSide::Short => -1.0,
            };

            if let Some(&group_idx) = self.symbol_group_map.get(&pos.symbol) {
                *net_by_group.entry(group_idx).or_insert(0.0) += notional;
            } else {
                ungrouped_notional += notional.abs();
            }
        }

        let grouped_net: f64 = net_by_group.values().map(|v| v.abs()).sum();
        grouped_net + ungrouped_notional
    }

    /// Identify hedged positions in portfolio
    pub fn identify_hedges(&self, positions: &[PortfolioPosition]) -> Vec<HedgeInfo> {
        let mut hedges = Vec::new();

        // Group by correlation group
        let mut group_positions: HashMap<usize, Vec<&PortfolioPosition>> = HashMap::new();

        for pos in positions {
            if let Some(&group_idx) = self.symbol_group_map.get(&pos.symbol) {
                group_positions.entry(group_idx).or_insert_with(Vec::new).push(pos);
            }
        }

        // Find offsetting positions within each group
        for (group_idx, group_pos) in group_positions {
            let group = &self.correlation_groups[group_idx];
            
            let mut longs = Vec::new();
            let mut shorts = Vec::new();

            for pos in group_pos {
                let notional = pos.size * pos.price;
                match pos.side {
                    PositionSide::Long => longs.push((*pos, notional)),
                    PositionSide::Short => shorts.push((*pos, notional)),
                }
            }

            // Check for hedge opportunities
            let total_long: f64 = longs.iter().map(|(_, n)| n).sum();
            let total_short: f64 = shorts.iter().map(|(_, n)| n).sum();

            if !longs.is_empty() && !shorts.is_empty() {
                let hedge_ratio = total_long.min(total_short) / total_long.max(total_short).max(1e-10);
                
                hedges.push(HedgeInfo {
                    group_index: group_idx,
                    assets: group.assets.clone(),
                    long_notional: total_long,
                    short_notional: total_short,
                    hedge_ratio,
                    offset_benefit: hedge_ratio * group.offset_pct * 100.0,
                });
            }
        }

        hedges
    }

    /// Add custom correlation group
    pub fn add_correlation_group(&mut self, assets: Vec<[u8; 16]>, correlation: f64, offset_pct: f64) {
        let group_idx = self.correlation_groups.len();
        
        self.correlation_groups.push(CorrelationGroup {
            assets,
            correlation: correlation.clamp(0.0, 1.0),
            offset_pct: offset_pct.clamp(0.0, 1.0),
        });

        for &symbol in &self.correlation_groups[group_idx].assets {
            self.symbol_group_map.insert(symbol, group_idx);
        }
    }

    /// Set cross-margin discount factor
    pub fn set_cross_margin_discount(&mut self, discount: f64) {
        self.cross_margin_discount = discount.clamp(0.0, 1.0);
    }
}

/// Hedge identification info
#[derive(Debug, Clone)]
pub struct HedgeInfo {
    pub group_index: usize,
    pub assets: Vec<[u8; 16]>,
    pub long_notional: f64,
    pub short_notional: f64,
    pub hedge_ratio: f64,
    pub offset_benefit: f64,
}

impl Default for PortfolioMarginCalculator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_portfolio_margin_calculation() {
        let calc = PortfolioMarginCalculator::new();

        let positions = vec![
            PortfolioPosition {
                symbol: *b"BTC           ",
                asset_type: AssetType::Spot,
                side: PositionSide::Long,
                size: 1.0,
                price: 50000.0,
            },
            PortfolioPosition {
                symbol: *b"BTC-PERP      ",
                asset_type: AssetType::Perpetual,
                side: PositionSide::Short,
                size: 1.0,
                price: 50000.0,
            },
        ];

        let result = calc.calculate_portfolio_margin(&positions);
        
        assert!(result.gross_margin > result.net_margin);
        assert!(result.savings_pct > 0.0);
    }

    #[test]
    fn test_hedge_identification() {
        let calc = PortfolioMarginCalculator::new();

        let positions = vec![
            PortfolioPosition {
                symbol: *b"ETH           ",
                asset_type: AssetType::Spot,
                side: PositionSide::Long,
                size: 10.0,
                price: 3000.0,
            },
            PortfolioPosition {
                symbol: *b"ETH-PERP      ",
                asset_type: AssetType::Perpetual,
                side: PositionSide::Short,
                size: 10.0,
                price: 3000.0,
            },
        ];

        let hedges = calc.identify_hedges(&positions);
        assert!(!hedges.is_empty());
        assert!(hedges[0].hedge_ratio > 0.9);
    }
}
