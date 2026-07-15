//! Cross vs. Isolated margin simulator
//! Tracks exact maintenance margin requirements, initial margin, and available balance

use std::collections::HashMap;

/// Position representation
#[derive(Debug, Clone, Copy)]
pub struct Position {
    pub symbol: [u8; 16],
    pub side: PositionSide,
    pub size: f64,
    pub entry_price: f64,
    pub mark_price: f64,
    pub leverage: f64,
    pub margin_mode: MarginMode,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PositionSide {
    Long,
    Short,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MarginMode {
    Cross,
    Isolated,
}

impl Position {
    /// Calculate unrealized PnL
    #[inline]
    pub fn unrealized_pnl(&self) -> f64 {
        match self.side {
            PositionSide::Long => (self.mark_price - self.entry_price) * self.size,
            PositionSide::Short => (self.entry_price - self.mark_price) * self.size,
        }
    }

    /// Calculate position value in USD
    #[inline]
    pub fn position_value(&self) -> f64 {
        self.size * self.mark_price
    }

    /// Calculate liquidation price (simplified)
    #[inline]
    pub fn liquidation_price(&self, maintenance_margin_rate: f64) -> f64 {
        let position_value = self.position_value();
        let margin = position_value / self.leverage;
        
        match self.side {
            PositionSide::Long => {
                let mm = position_value * maintenance_margin_rate;
                if margin <= mm {
                    return self.mark_price;
                }
                self.entry_price - (margin - mm) / self.size
            }
            PositionSide::Short => {
                let mm = position_value * maintenance_margin_rate;
                if margin <= mm {
                    return self.mark_price;
                }
                self.entry_price + (margin - mm) / self.size
            }
        }
    }
}

/// Margin account state
#[derive(Debug, Clone)]
pub struct MarginAccount {
    /// Total wallet balance
    pub wallet_balance: f64,
    /// Available balance (not used as margin)
    pub available_balance: f64,
    /// Total margin in use
    pub total_margin_used: f64,
    /// Unrealized PnL across all positions
    pub unrealized_pnl: f64,
    /// Realized PnL (session)
    pub realized_pnl: f64,
    /// Positions in cross margin mode
    cross_positions: HashMap<[u8; 16], Position>,
    /// Isolated margin allocations per position
    isolated_allocations: HashMap<[u8; 16], f64>,
}

impl MarginAccount {
    pub fn new(initial_balance: f64) -> Self {
        Self {
            wallet_balance: initial_balance,
            available_balance: initial_balance,
            total_margin_used: 0.0,
            unrealized_pnl: 0.0,
            realized_pnl: 0.0,
            cross_positions: HashMap::new(),
            isolated_allocations: HashMap::new(),
        }
    }

    /// Update mark prices and recalculate margin metrics
    pub fn update_mark_prices(&mut self, prices: &HashMap<[u8; 16], f64>) {
        self.unrealized_pnl = 0.0;

        // Update cross positions
        for (_, position) in self.cross_positions.iter_mut() {
            if let Some(&price) = prices.get(&position.symbol) {
                position.mark_price = price;
                self.unrealized_pnl += position.unrealized_pnl();
            }
        }

        // Recalculate available balance
        self.recalculate_available_balance();
    }

    /// Add or update a cross margin position
    pub fn update_cross_position(&mut self, position: Position) -> MarginUpdateResult {
        let required_margin = position.position_value() / position.leverage;
        
        if required_margin > self.available_balance {
            return MarginUpdateResult::InsufficientMargin;
        }

        self.cross_positions.insert(position.symbol, position);
        self.total_margin_used += required_margin;
        self.available_balance -= required_margin;

        MarginUpdateResult::Success
    }

    /// Allocate isolated margin to a position
    pub fn allocate_isolated_margin(&mut self, symbol: [u8; 16], amount: f64) -> bool {
        if amount > self.available_balance {
            return false;
        }

        self.isolated_allocations.insert(symbol, amount);
        self.available_balance -= amount;
        self.total_margin_used += amount;
        true
    }

    /// Remove isolated margin allocation
    pub fn remove_isolated_margin(&mut self, symbol: &[u8; 16]) -> Option<f64> {
        if let Some(amount) = self.isolated_allocations.remove(symbol) {
            self.available_balance += amount;
            self.total_margin_used -= amount;
            Some(amount)
        } else {
            None
        }
    }

    /// Recalculate available balance after PnL changes
    fn recalculate_available_balance(&mut self) {
        let equity = self.wallet_balance + self.unrealized_pnl;
        self.available_balance = equity - self.total_margin_used;
    }

    /// Get account equity
    #[inline]
    pub fn equity(&self) -> f64 {
        self.wallet_balance + self.unrealized_pnl
    }

    /// Get margin ratio (used / equity)
    #[inline]
    pub fn margin_ratio(&self) -> f64 {
        let equity = self.equity();
        if equity <= 0.0 {
            return f64::MAX;
        }
        self.total_margin_used / equity
    }

    /// Check if account is at risk of liquidation
    #[inline]
    pub fn is_at_liquidation_risk(&self, maintenance_margin_rate: f64) -> bool {
        let equity = self.equity();
        let required_mm = self.total_margin_used * maintenance_margin_rate;
        equity < required_mm
    }

    /// Get total position value
    pub fn total_position_value(&self) -> f64 {
        self.cross_positions.values().map(|p| p.position_value()).sum()
    }

    /// Get leveraged exposure
    pub fn total_leveraged_exposure(&self) -> f64 {
        self.cross_positions.values().map(|p| p.position_value()).sum()
    }
}

/// Result of margin update operation
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MarginUpdateResult {
    Success,
    InsufficientMargin,
    InvalidPosition,
    RiskLimitExceeded,
}

/// Margin simulation engine with stress testing
pub struct MarginSimulator {
    /// Default maintenance margin rates per symbol
    maintenance_rates: HashMap<[u8; 16], f64>,
    /// Default initial margin rates
    initial_rates: HashMap<[u8; 16], f64>,
    /// Maximum leverage per symbol
    max_leverage: HashMap<[u8; 16], f64>,
}

impl MarginSimulator {
    pub fn new() -> Self {
        let mut sim = Self {
            maintenance_rates: HashMap::new(),
            initial_rates: HashMap::new(),
            max_leverage: HashMap::new(),
        };

        // Set defaults for common crypto instruments
        sim.set_defaults();
        sim
    }

    fn set_defaults(&mut self) {
        // BTC/ETH perpetuals
        let btc = *b"BTC-PERP      ";
        let eth = *b"ETH-PERP      ";

        self.maintenance_rates.insert(btc, 0.005); // 0.5%
        self.maintenance_rates.insert(eth, 0.005);
        self.initial_rates.insert(btc, 0.01); // 1%
        self.initial_rates.insert(eth, 0.01);
        self.max_leverage.insert(btc, 100.0);
        self.max_leverage.insert(eth, 100.0);
    }

    /// Set maintenance margin rate for symbol
    pub fn set_maintenance_rate(&mut self, symbol: [u8; 16], rate: f64) {
        self.maintenance_rates.insert(symbol, rate.clamp(0.001, 0.5));
    }

    /// Get maintenance margin requirement for position
    #[inline]
    pub fn get_maintenance_requirement(&self, position: &Position) -> f64 {
        let rate = self.maintenance_rates.get(&position.symbol)
            .copied()
            .unwrap_or(0.005);
        position.position_value() * rate
    }

    /// Get initial margin requirement for new position
    #[inline]
    pub fn get_initial_requirement(&self, position_value: f64, leverage: f64, symbol: &[u8; 16]) -> f64 {
        let base_rate = self.initial_rates.get(symbol).copied().unwrap_or(0.01);
        let leverage_rate = 1.0 / leverage;
        position_value * base_rate.max(leverage_rate)
    }

    /// Validate if new position is allowed
    pub fn validate_position(&self, account: &MarginAccount, position: &Position) -> ValidationResult {
        // Check leverage limit
        if let Some(&max_leverage) = self.max_leverage.get(&position.symbol) {
            if position.leverage > max_leverage {
                return ValidationResult::LeverageExceeded { max: max_leverage };
            }
        }

        // Check margin requirement
        let required = self.get_initial_requirement(position.position_value(), position.leverage, &position.symbol);
        if required > account.available_balance {
            return ValidationResult::InsufficientMargin { required };
        }

        // Check risk limits
        let total_exposure = account.total_leveraged_exposure() + position.position_value();
        if total_exposure > account.equity() * 10.0 { // Example: 10x total exposure limit
            return ValidationResult::RiskLimitExceeded;
        }

        ValidationResult::Valid
    }

    /// Run stress test scenario on account
    pub fn stress_test(&self, account: &MarginAccount, price_shock_pct: f64) -> StressTestResult {
        let mut stressed_pnl = 0.0;
        let mut liquidated_positions = Vec::new();

        for (&symbol, position) in &account.cross_positions {
            let shocked_price = position.mark_price * (1.0 + price_shock_pct / 100.0);
            let mm_rate = self.maintenance_rates.get(&symbol).copied().unwrap_or(0.005);
            
            let shocked_position = Position {
                mark_price: shocked_price,
                ..*position
            };

            if shocked_position.unrealized_pnl() < -(position.position_value() * mm_rate) {
                liquidated_positions.push(symbol);
            }

            stressed_pnl += shocked_position.unrealized_pnl();
        }

        StressTestResult {
            original_equity: account.equity(),
            stressed_equity: account.wallet_balance + stressed_pnl,
            equity_change_pct: (stressed_pnl - account.unrealized_pnl) / account.equity() * 100.0,
            liquidated_positions,
            price_shock_pct,
        }
    }

    /// Calculate maximum safe position size given current account state
    pub fn max_safe_position_size(&self, account: &MarginAccount, symbol: &[u8; 16], leverage: f64, mark_price: f64) -> f64 {
        let available = account.available_balance;
        let im_rate = self.initial_rates.get(symbol).copied().unwrap_or(0.01);
        let leverage_rate = 1.0 / leverage;
        let effective_rate = im_rate.max(leverage_rate);

        // Account for existing exposure
        let max_notional = available / effective_rate;
        max_notional / mark_price
    }
}

/// Validation result for position checks
#[derive(Debug, Clone, Copy)]
pub enum ValidationResult {
    Valid,
    LeverageExceeded { max: f64 },
    InsufficientMargin { required: f64 },
    RiskLimitExceeded,
}

/// Stress test result
#[derive(Debug, Clone)]
pub struct StressTestResult {
    pub original_equity: f64,
    pub stressed_equity: f64,
    pub equity_change_pct: f64,
    pub liquidated_positions: Vec<[u8; 16]>,
    pub price_shock_pct: f64,
}

impl Default for MarginSimulator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_position_pnl() {
        let position = Position {
            symbol: *b"BTC-PERP      ",
            side: PositionSide::Long,
            size: 1.0,
            entry_price: 50000.0,
            mark_price: 51000.0,
            leverage: 10.0,
            margin_mode: MarginMode::Cross,
        };

        let pnl = position.unrealized_pnl();
        assert_eq!(pnl, 1000.0);
    }

    #[test]
    fn test_margin_account() {
        let mut account = MarginAccount::new(100000.0);
        
        let position = Position {
            symbol: *b"BTC-PERP      ",
            side: PositionSide::Long,
            size: 1.0,
            entry_price: 50000.0,
            mark_price: 50000.0,
            leverage: 10.0,
            margin_mode: MarginMode::Cross,
        };

        let result = account.update_cross_position(position);
        assert_eq!(result, MarginUpdateResult::Success);
        assert!(account.available_balance < 100000.0);
    }

    #[test]
    fn test_stress_test() {
        let simulator = MarginSimulator::new();
        let account = MarginAccount::new(100000.0);
        
        let result = simulator.stress_test(&account, -10.0);
        assert!(result.stressed_equity <= result.original_equity);
    }
}
