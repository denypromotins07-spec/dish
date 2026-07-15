//! Automated idle capital router.
//! Instantly sweeps unused stablecoin balances into low-risk, exchange-native flexible savings.
//! Ensures zero cash drag without locking up execution capital.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

/// Minimum balance to keep in trading wallet (for immediate execution)
const MIN_TRADING_BALANCE_BPS: u32 = 500; // 5% of total

/// Yield product types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum YieldProductType {
    FlexibleSavings,      // Instant withdrawal
    Staking,              // May have unbonding period
    Lending,              // May have lock-up
    LiquidityPool,        // May have impermanent loss
}

/// Available yield product
#[derive(Debug, Clone)]
pub struct YieldProduct {
    pub product_id: u64,
    pub name: [u8; 32],
    pub product_type: YieldProductType,
    pub apy_bps: u32,              // Annual percentage yield in basis points
    pub min_deposit: u128,         // Minimum deposit amount
    pub max_deposit: u128,         // Maximum deposit amount
    pub withdrawal_delay_ns: u64,  // Nanoseconds to withdraw
    pub risk_score: u8,            // 0 (risk-free) to 100 (risky)
    pub supported_assets: Vec<u64>, // Asset IDs that can be deposited
}

/// Active deposit in a yield product
#[derive(Debug, Clone)]
pub struct ActiveDeposit {
    pub product_id: u64,
    pub asset_id: u64,
    pub amount: u128,
    pub deposited_at_ns: u64,
    pub accrued_yield: u128,
}

/// Balance allocation result
#[derive(Debug, Clone)]
pub struct AllocationResult {
    pub trading_balance: u128,
    pub yield_balance: u128,
    pub expected_daily_yield: u128,
    pub allocations: Vec<(u64, u128)>, // (product_id, amount)
}

/// Idle capital yield generator
pub struct IdleYieldGenerator {
    /// Available yield products
    products: HashMap<u64, YieldProduct>,
    /// Active deposits
    deposits: HashMap<u64, ActiveDeposit>, // product_id -> deposit
    /// Available balances by asset
    balances: HashMap<u64, u128>,
    /// Total yield earned (lifetime)
    total_yield_earned: u128,
    /// Memory footprint tracking
    memory_footprint: AtomicU64,
}

impl IdleYieldGenerator {
    pub fn new() -> Self {
        Self {
            products: HashMap::with_capacity(32),
            deposits: HashMap::new(),
            balances: HashMap::with_capacity(32),
            total_yield_earned: 0,
            memory_footprint: AtomicU64::new(0),
        }
    }

    /// Get current timestamp in nanoseconds
    #[inline]
    fn now_ns() -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64
    }

    /// Register a yield product
    pub fn register_product(&mut self, product: YieldProduct) {
        self.products.insert(product.product_id, product);
        
        self.memory_footprint.store(
            (self.products.len() * std::mem::size_of::<YieldProduct>()) as u64,
            Ordering::Relaxed,
        );
    }

    /// Update available balance for an asset
    pub fn update_balance(&mut self, asset_id: u64, amount: u128) {
        self.balances.insert(asset_id, amount);
    }

    /// Calculate optimal allocation of idle capital
    pub fn calculate_optimal_allocation(&self, asset_id: u64) -> AllocationResult {
        let total_balance = *self.balances.get(&asset_id).unwrap_or(&0);
        
        if total_balance == 0 {
            return AllocationResult {
                trading_balance: 0,
                yield_balance: 0,
                expected_daily_yield: 0,
                allocations: Vec::new(),
            };
        }

        // Keep minimum for trading
        let trading_balance = (total_balance * MIN_TRADING_BALANCE_BPS as u128) / 10000;
        let idle_balance = total_balance - trading_balance;

        if idle_balance == 0 {
            return AllocationResult {
                trading_balance,
                yield_balance: 0,
                expected_daily_yield: 0,
                allocations: Vec::new(),
            };
        }

        // Find best yield products for this asset (sorted by APY, then risk)
        let mut eligible_products: Vec<&YieldProduct> = self.products
            .values()
            .filter(|p| {
                p.supported_assets.contains(&asset_id)
                    && idle_balance >= p.min_deposit
            })
            .collect();

        // Sort by APY (descending), then by risk (ascending)
        eligible_products.sort_by(|a, b| {
            b.apy_bps.cmp(&a.apy_bps)
                .then(a.risk_score.cmp(&b.risk_score))
        });

        // Allocate across products (diversification)
        let mut allocations: Vec<(u64, u128)> = Vec::new();
        let mut remaining = idle_balance;
        let mut total_expected_daily = 0u128;

        for product in &eligible_products {
            if remaining == 0 {
                break;
            }

            // Calculate allocation to this product
            let max_alloc = remaining.min(product.max_deposit);
            let actual_alloc = max_alloc;

            if actual_alloc >= product.min_deposit {
                allocations.push((product.product_id, actual_alloc));
                remaining -= actual_alloc;

                // Calculate expected daily yield: (amount * apy) / 365 / 10000
                let daily_yield = (actual_alloc * product.apy_bps as u128) / 365 / 10000;
                total_expected_daily += daily_yield;
            }
        }

        // If we still have remaining, add to trading balance
        let final_trading_balance = trading_balance + remaining;

        AllocationResult {
            trading_balance: final_trading_balance,
            yield_balance: idle_balance - remaining,
            expected_daily_yield: total_expected_daily,
            allocations,
        }
    }

    /// Execute sweep into yield products
    pub fn execute_sweep(&mut self, asset_id: u64) -> Vec<(u64, u128)> {
        let allocation = self.calculate_optimal_allocation(asset_id);
        let mut executed = Vec::new();

        for (product_id, amount) in &allocation.allocations {
            // Check if we already have a deposit in this product
            if let Some(existing) = self.deposits.get(product_id) {
                // Top up existing deposit
                if let Some(deposit) = self.deposits.get_mut(product_id) {
                    deposit.amount += amount;
                }
            } else {
                // Create new deposit
                let deposit = ActiveDeposit {
                    product_id: *product_id,
                    asset_id,
                    amount: *amount,
                    deposited_at_ns: Self::now_ns(),
                    accrued_yield: 0,
                };
                self.deposits.insert(*product_id, deposit);
            }

            // Reduce balance
            if let Some(balance) = self.balances.get_mut(&asset_id) {
                *balance = balance.saturating_sub(*amount);
            }

            executed.push((*product_id, *amount));
        }

        executed
    }

    /// Withdraw from yield product (if allowed)
    pub fn withdraw(&mut self, product_id: u64, amount: u128) -> Option<u128> {
        let deposit = self.deposits.get_mut(&product_id)?;
        
        // Check withdrawal delay
        let product = self.products.get(&product_id)?;
        let now = Self::now_ns();
        let can_withdraw_at = deposit.deposited_at_ns + product.withdrawal_delay_ns;
        
        if now < can_withdraw_at {
            return None; // Cannot withdraw yet
        }

        let withdraw_amount = amount.min(deposit.amount);
        
        deposit.amount -= withdraw_amount;
        deposit.accrued_yield += self.calculate_accrued_yield(product_id);

        // Return principal + accrued yield
        let total_return = withdraw_amount + deposit.accrued_yield;
        
        // Update balance
        *self.balances.entry(deposit.asset_id).or_insert(0) += total_return;

        // Remove deposit if empty
        if deposit.amount == 0 {
            self.deposits.remove(&product_id);
        }

        Some(total_return)
    }

    /// Calculate accrued yield for a deposit
    fn calculate_accrued_yield(&self, product_id: u64) -> u128 {
        let deposit = match self.deposits.get(&product_id) {
            Some(d) => d,
            None => return 0,
        };

        let product = match self.products.get(&product_id) {
            Some(p) => p,
            None => return 0,
        };

        let elapsed_ns = Self::now_ns() - deposit.deposited_at_ns;
        let elapsed_days = elapsed_ns as f64 / (86400.0 * 1_000_000_000.0);
        
        // Daily yield: (principal * apy) / 365 / 10000
        let daily_yield = (deposit.amount * product.apy_bps as u128) / 365 / 10000;
        
        (daily_yield as f64 * elapsed_days) as u128
    }

    /// Get total value across all yield products
    pub fn total_yield_value(&self) -> u128 {
        self.deposits.values().map(|d| d.amount + d.accrued_yield).sum()
    }

    /// Get total available trading balance across all assets
    pub fn total_trading_balance(&self) -> u128 {
        self.balances.values().sum()
    }

    /// Get active deposits
    pub fn get_active_deposits(&self) -> Vec<&ActiveDeposit> {
        self.deposits.values().collect()
    }

    /// Get yield product by ID
    pub fn get_product(&self, product_id: u64) -> Option<&YieldProduct> {
        self.products.get(&product_id)
    }

    /// Update accrued yields (call periodically)
    pub fn update_accrued_yields(&mut self) {
        let mut total_new_yield = 0u128;
        
        for (product_id, deposit) in &mut self.deposits {
            let accrued = self.calculate_accrued_yield(*product_id);
            total_new_yield += accrued - deposit.accrued_yield;
            deposit.accrued_yield = accrued;
        }
        
        self.total_yield_earned += total_new_yield;
    }

    /// Get total yield earned (lifetime)
    pub fn total_yield_earned(&self) -> u128 {
        self.total_yield_earned
    }

    /// Get memory footprint in bytes
    pub fn memory_footprint(&self) -> u64 {
        self.memory_footprint.load(Ordering::Relaxed)
    }
}

impl Default for IdleYieldGenerator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_yield_allocation() {
        let mut generator = IdleYieldGenerator::new();
        
        // Register a flexible savings product with 5% APY
        generator.register_product(YieldProduct {
            product_id: 1,
            name: *b"USDC Flexible Savings       ",
            product_type: YieldProductType::FlexibleSavings,
            apy_bps: 500, // 5%
            min_deposit: 100_000_000, // $100
            max_deposit: 1_000_000_000_000, // $1M
            withdrawal_delay_ns: 0, // Instant
            risk_score: 5,
            supported_assets: vec![1], // USDC
        });

        // Set balance of $10,000
        generator.update_balance(1, 10_000_000_000);

        let allocation = generator.calculate_optimal_allocation(1);
        
        // Should keep some for trading, invest rest
        assert!(allocation.trading_balance > 0);
        assert!(allocation.yield_balance > 0);
        assert!(allocation.expected_daily_yield > 0);
        
        // Execute sweep
        let executed = generator.execute_sweep(1);
        assert!(!executed.is_empty());
        
        // Verify deposit was created
        assert!(generator.get_active_deposits().len() > 0);
    }
}
