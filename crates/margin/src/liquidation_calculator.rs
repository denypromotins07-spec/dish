//! Exact liquidation price calculator with exchange-specific ADL mechanics
//! Factors in bankruptcy prices, insurance fund mechanics, and auto-deleveraging

use std::collections::HashMap;

/// Liquidation calculation parameters
#[derive(Debug, Clone, Copy)]
pub struct LiquidationParams {
    pub entry_price: f64,
    pub position_size: f64,
    pub leverage: f64,
    pub margin_balance: f64,
    pub maintenance_margin_rate: f64,
    pub taker_fee_rate: f64,
    pub funding_rate: f64,
}

/// Liquidation calculation result
#[derive(Debug, Clone, Copy)]
pub struct LiquidationResult {
    /// Price at which liquidation occurs
    pub liquidation_price: f64,
    /// Bankruptcy price (equity = 0)
    pub bankruptcy_price: f64,
    /// Margin required at liquidation
    pub liquidation_margin: f64,
    /// Estimated loss given default
    pub expected_loss_pct: f64,
    /// ADL tier that would be triggered
    pub adl_tier: u32,
}

impl LiquidationResult {
    pub fn is_liquidated(&self, current_price: f64, side: PositionSide) -> bool {
        match side {
            PositionSide::Long => current_price <= self.liquidation_price,
            PositionSide::Short => current_price >= self.liquidation_price,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PositionSide {
    Long,
    Short,
}

/// Auto-Deleveraging configuration
#[derive(Debug, Clone)]
pub struct ADLConfig {
    /// Number of ADL tiers
    pub tiers: u32,
    /// Leverage thresholds per tier
    pub leverage_thresholds: Vec<f64>,
    /// Priority scores based on profitability and leverage
    pub use_profitability_priority: bool,
}

impl Default for ADLConfig {
    fn default() -> Self {
        Self {
            tiers: 5,
            leverage_thresholds: vec![10.0, 25.0, 50.0, 75.0, 100.0],
            use_profitability_priority: true,
        }
    }
}

/// Insurance fund state
#[derive(Debug, Clone)]
pub struct InsuranceFund {
    pub balance: f64,
    pub currency: [u8; 8],
    /// Minimum balance before ADL triggers
    pub min_balance: f64,
}

impl InsuranceFund {
    pub fn new(currency: &str, initial_balance: f64) -> Self {
        let mut curr = [0u8; 8];
        curr[..currency.len().min(8)].copy_from_slice(&currency.as_bytes()[..currency.len().min(8)]);
        
        Self {
            balance: initial_balance,
            currency: curr,
            min_balance: initial_balance * 0.1, // 10% minimum
        }
    }

    /// Add realized PnL from liquidations to insurance fund
    pub fn add_liquidation_surplus(&mut self, surplus: f64) {
        if surplus > 0.0 {
            self.balance += surplus;
        }
    }

    /// Withdraw from insurance fund to cover liquidation deficit
    pub fn cover_deficit(&mut self, deficit: f64) -> f64 {
        let available = (self.balance - self.min_balance).max(0.0);
        let covered = available.min(deficit);
        self.balance -= covered;
        covered
    }

    /// Check if insurance fund is depleted
    #[inline]
    pub fn is_depleted(&self) -> bool {
        self.balance <= self.min_balance
    }
}

/// Liquidation Calculator with exchange-specific mechanics
pub struct LiquidationCalculator {
    /// Maintenance margin rates by symbol
    maintenance_rates: HashMap<[u8; 16], f64>,
    /// Taker fee rates
    taker_fees: HashMap<[u8; 16], f64>,
    /// ADL configurations
    adl_configs: HashMap<[u8; 16], ADLConfig>,
    /// Insurance funds by currency
    insurance_funds: HashMap<[u8; 8], InsuranceFund>,
}

impl LiquidationCalculator {
    pub fn new() -> Self {
        let mut calc = Self {
            maintenance_rates: HashMap::new(),
            taker_fees: HashMap::new(),
            adl_configs: HashMap::new(),
            insurance_funds: HashMap::new(),
        };

        calc.set_defaults();
        calc
    }

    fn set_defaults(&mut self) {
        // BTC perpetual
        let btc = *b"BTC-PERP      ";
        self.maintenance_rates.insert(btc, 0.005);
        self.taker_fees.insert(btc, 0.0004); // 4 bps
        self.adl_configs.insert(btc, ADLConfig::default());

        // ETH perpetual
        let eth = *b"ETH-PERP      ";
        self.maintenance_rates.insert(eth, 0.005);
        self.taker_fees.insert(eth, 0.0004);
        self.adl_configs.insert(eth, ADLConfig::default());

        // USD insurance fund
        let usd = *b"USD     ";
        self.insurance_funds.insert(usd, InsuranceFund::new("USD", 100_000_000.0));
    }

    /// Calculate exact liquidation price for a position
    #[inline]
    pub fn calculate_liquidation_price(&self, params: &LiquidationParams, side: PositionSide) -> LiquidationResult {
        let mm_rate = params.maintenance_margin_rate;
        let taker_fee = params.taker_fee_rate;
        
        // Initial margin
        let initial_margin = params.margin_balance;
        
        // Position value
        let position_value = params.position_size * params.entry_price;
        
        // Effective leverage
        let effective_leverage = position_value / initial_margin;

        match side {
            PositionSide::Long => {
                // For long positions, liquidation occurs when:
                // Equity = Maintenance Margin + Fees
                // (Position Value - Entry Value) + Initial Margin = MM + Fees
                
                let numerator = position_value * (mm_rate + taker_fee) - initial_margin;
                let denominator = params.position_size * (mm_rate + taker_fee - 1.0);
                
                let liquidation_price = if denominator.abs() > 1e-10 {
                    (numerator / denominator).max(0.0)
                } else {
                    params.entry_price
                };

                // Bankruptcy price is where equity = 0
                let bankruptcy_price = params.entry_price - initial_margin / params.position_size;

                // Expected loss as percentage of position
                let expected_loss = if liquidation_price > 0.0 {
                    (params.entry_price - liquidation_price) / params.entry_price
                } else {
                    1.0
                };

                // ADL tier based on leverage
                let adl_tier = self.get_adl_tier(effective_leverage, &*b"BTC-PERP      ");

                LiquidationResult {
                    liquidation_price,
                    bankruptcy_price,
                    liquidation_margin: position_value * mm_rate,
                    expected_loss_pct: expected_loss,
                    adl_tier,
                }
            }
            PositionSide::Short => {
                // For short positions
                let numerator = initial_margin + position_value * (1.0 - mm_rate - taker_fee);
                let denominator = params.position_size * (1.0 - mm_rate - taker_fee);
                
                let liquidation_price = if denominator.abs() > 1e-10 {
                    numerator / denominator
                } else {
                    params.entry_price
                };

                // Bankruptcy price
                let bankruptcy_price = params.entry_price + initial_margin / params.position_size;

                let expected_loss = if liquidation_price > 0.0 {
                    (liquidation_price - params.entry_price) / params.entry_price
                } else {
                    1.0
                };

                let adl_tier = self.get_adl_tier(effective_leverage, &*b"BTC-PERP      ");

                LiquidationResult {
                    liquidation_price,
                    bankruptcy_price,
                    liquidation_margin: position_value * mm_rate,
                    expected_loss_pct: expected_loss,
                    adl_tier,
                }
            }
        }
    }

    /// Get ADL tier for a position
    fn get_adl_tier(&self, leverage: f64, symbol: &[u8; 16]) -> u32 {
        if let Some(config) = self.adl_configs.get(symbol) {
            for (tier, &threshold) in config.leverage_thresholds.iter().enumerate() {
                if leverage >= threshold {
                    return (tier + 1) as u32;
                }
            }
        }
        1
    }

    /// Calculate liquidation PnL and insurance fund impact
    pub fn calculate_liquidation_pnl(
        &self,
        params: &LiquidationParams,
        side: PositionSide,
        liquidation_price: f64,
    ) -> LiquidationPnL {
        let entry_value = params.position_size * params.entry_price;
        let exit_value = params.position_size * liquidation_price;
        let taker_fee = exit_value * params.taker_fee_rate;

        let gross_pnl = match side {
            PositionSide::Long => exit_value - entry_value,
            PositionSide::Short => entry_value - exit_value,
        };

        let net_pnl = gross_pnl - taker_fee;
        
        // Surplus goes to insurance fund, deficit is covered by fund
        let insurance_impact = if net_pnl < -params.margin_balance {
            // Deficit exceeds margin - insurance fund covers
            net_pnl + params.margin_balance
        } else if net_pnl > 0.0 {
            // Surplus after covering losses
            net_pnl.min(params.margin_balance)
        } else {
            0.0
        };

        LiquidationPnL {
            gross_pnl,
            net_pnl,
            taker_fee,
            insurance_impact,
            trader_loss: params.margin_balance.min(-net_pnl.max(0.0)),
        }
    }

    /// Simulate ADL queue priority score
    pub fn calculate_adl_score(&self, position_value: f64, leverage: f64, unrealized_pnl: f64) -> f64 {
        // Higher leverage and higher profitability = higher priority for ADL
        let leverage_score = leverage / 100.0; // Normalize
        let pnl_score = if unrealized_pnl > 0.0 {
            unrealized_pnl / position_value
        } else {
            0.0
        };

        leverage_score * (1.0 + pnl_score)
    }

    /// Set custom maintenance margin rate
    pub fn set_maintenance_rate(&mut self, symbol: [u8; 16], rate: f64) {
        self.maintenance_rates.insert(symbol, rate.clamp(0.001, 0.5));
    }

    /// Get insurance fund for currency
    pub fn get_insurance_fund(&self, currency: &[u8; 8]) -> Option<&InsuranceFund> {
        self.insurance_funds.get(currency)
    }

    /// Update insurance fund
    pub fn update_insurance_fund(&mut self, currency: [u8; 8], balance: f64) {
        if let Some(fund) = self.insurance_funds.get_mut(&currency) {
            fund.balance = balance;
        }
    }
}

/// Liquidation PnL breakdown
#[derive(Debug, Clone, Copy)]
pub struct LiquidationPnL {
    pub gross_pnl: f64,
    pub net_pnl: f64,
    pub taker_fee: f64,
    pub insurance_impact: f64,
    pub trader_loss: f64,
}

/// ADL event simulation
#[derive(Debug, Clone)]
pub struct ADLEvent {
    pub symbol: [u8; 16],
    pub affected_positions: u32,
    pub total_volume_reduced: f64,
    pub average_price_impact_bps: f64,
    pub insurance_used: f64,
}

impl Default for LiquidationCalculator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_long_liquidation() {
        let calc = LiquidationCalculator::new();
        
        let params = LiquidationParams {
            entry_price: 50000.0,
            position_size: 1.0,
            leverage: 10.0,
            margin_balance: 5000.0,
            maintenance_margin_rate: 0.005,
            taker_fee_rate: 0.0004,
            funding_rate: 0.0,
        };

        let result = calc.calculate_liquidation_price(&params, PositionSide::Long);
        
        assert!(result.liquidation_price < params.entry_price);
        assert!(result.bankruptcy_price < result.liquidation_price);
    }

    #[test]
    fn test_short_liquidation() {
        let calc = LiquidationCalculator::new();
        
        let params = LiquidationParams {
            entry_price: 50000.0,
            position_size: 1.0,
            leverage: 10.0,
            margin_balance: 5000.0,
            maintenance_margin_rate: 0.005,
            taker_fee_rate: 0.0004,
            funding_rate: 0.0,
        };

        let result = calc.calculate_liquidation_price(&params, PositionSide::Short);
        
        assert!(result.liquidation_price > params.entry_price);
        assert!(result.bankruptcy_price > result.liquidation_price);
    }

    #[test]
    fn test_insurance_fund() {
        let mut fund = InsuranceFund::new("USD", 1000000.0);
        
        fund.add_liquidation_surplus(50000.0);
        assert!(fund.balance > 1000000.0);
        
        let covered = fund.cover_deficit(100000.0);
        assert!(covered > 0.0);
    }
}
