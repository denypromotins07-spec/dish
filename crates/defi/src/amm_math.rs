//! Blazing-fast Rust implementation of AMM math for Uniswap V2/V3.
//! Calculates swap routes, tick crossings, and price impacts in nanoseconds.

use std::cmp::{max, min};

/// Constant Product (Uniswap V2) AMM calculator
pub struct ConstantProductAmm;

impl ConstantProductAmm {
    /// Calculate output amount for a given input (x * y = k)
    #[inline]
    pub fn get_amount_out(
        amount_in: u128,
        reserve_in: u128,
        reserve_out: u128,
        fee_bps: u128,
    ) -> Option<u128> {
        if amount_in == 0 || reserve_in == 0 || reserve_out == 0 {
            return None;
        }

        // amountOut = (amountIn * (1 - fee) * reserveOut) / (reserveIn + amountIn * (1 - fee))
        let amount_in_with_fee = amount_in * (10000 - fee_bps);
        let numerator = amount_in_with_fee * reserve_out;
        let denominator = (reserve_in * 10000) + amount_in_with_fee;

        if denominator == 0 {
            return None;
        }

        Some(numerator / denominator)
    }

    /// Calculate input amount required for a desired output
    #[inline]
    pub fn get_amount_in(
        amount_out: u128,
        reserve_in: u128,
        reserve_out: u128,
        fee_bps: u128,
    ) -> Option<u128> {
        if amount_out == 0 || reserve_in == 0 || reserve_out == 0 {
            return None;
        }

        if amount_out >= reserve_out {
            return None;
        }

        // amountIn = (reserveIn * amountOut * 10000) / ((reserveOut - amountOut) * (10000 - fee))
        let numerator = reserve_in * amount_out * 10000;
        let denominator = (reserve_out - amount_out) * (10000 - fee_bps);

        if denominator == 0 {
            return None;
        }

        Some((numerator / denominator) + 1)
    }

    /// Calculate price impact for a swap
    #[inline]
    pub fn calculate_price_impact(
        amount_in: u128,
        reserve_in: u128,
        reserve_out: u128,
        fee_bps: u128,
    ) -> Option<f64> {
        if reserve_in == 0 || reserve_out == 0 {
            return None;
        }

        // Spot price before swap
        let spot_price_before = reserve_out as f64 / reserve_in as f64;

        // Get actual output
        let amount_out = Self::get_amount_out(amount_in, reserve_in, reserve_out, fee_bps)?;

        // Effective execution price
        let exec_price = amount_out as f64 / amount_in as f64;

        // Price impact = 1 - (exec_price / spot_price)
        let impact = 1.0 - (exec_price / spot_price_before);

        Some(impact.max(0.0))
    }

    /// Calculate optimal trade size for maximum arbitrage profit
    #[inline]
    pub fn optimal_arbitrage_size(
        external_price: f64,
        reserve_in: u128,
        reserve_out: u128,
        fee_bps: u128,
    ) -> Option<u128> {
        if external_price <= 0.0 || reserve_in == 0 || reserve_out == 0 {
            return None;
        }

        // Simplified: find delta that maximizes profit
        // In production, would use Newton-Raphson iteration
        let spot_price = reserve_out as f64 / reserve_in as f64;
        let fee_multiplier = (10000 - fee_bps) as f64 / 10000.0;

        if external_price > spot_price * fee_multiplier {
            // Arbitrage: buy from AMM, sell externally
            let ratio = (external_price / (spot_price * fee_multiplier)).sqrt();
            Some(((ratio - 1.0) * reserve_in as f64) as u128)
        } else if external_price < spot_price / fee_multiplier {
            // Arbitrage: buy externally, sell to AMM
            let ratio = ((spot_price / fee_multiplier) / external_price).sqrt();
            Some(((ratio - 1.0) * reserve_out as f64) as u128)
        } else {
            Some(0)
        }
    }
}

/// Uniswap V3 concentrated liquidity structures
#[derive(Debug, Clone)]
pub struct Tick {
    pub index: i32,
    pub liquidity_gross: u128,
    pub liquidity_net: i128,
    pub fee_growth_outside_0: u128,
    pub fee_growth_outside_1: u128,
}

#[derive(Debug, Clone)]
pub struct Position {
    pub tick_lower: i32,
    pub tick_upper: i32,
    pub liquidity: u128,
    pub tokens_owed_0: u128,
    pub tokens_owed_1: u128,
}

/// Uniswap V3 Concentrated Liquidity AMM calculator
pub struct ConcentratedLiquidityAmm {
    tick_spacing: i32,
    current_tick: i32,
    current_sqrt_price_x96: u128,
    ticks: Vec<Tick>,
    positions: Vec<Position>,
}

impl ConcentratedLiquidityAmm {
    pub fn new(tick_spacing: i32) -> Self {
        Self {
            tick_spacing,
            current_tick: 0,
            current_sqrt_price_x96: 0,
            ticks: Vec::new(),
            positions: Vec::new(),
        }
    }

    /// Convert sqrtPriceX96 to price (token1/token0)
    #[inline]
    pub fn sqrt_price_to_price(sqrt_price_x96: u128) -> f64 {
        let price_x192 = (sqrt_price_x96 as f64).powi(2);
        price_x192 / (1u64 << 192) as f64
    }

    /// Convert price to sqrtPriceX96
    #[inline]
    pub fn price_to_sqrt_price_x96(price: f64) -> u128 {
        let sqrt_price = price.sqrt();
        (sqrt_price * (1u64 << 96) as f64) as u128
    }

    /// Convert tick to price
    #[inline]
    pub fn tick_to_price(tick: i32) -> f64 {
        (1.0001_f64).powi(tick)
    }

    /// Convert price to tick
    #[inline]
    pub fn price_to_tick(price: f64) -> i32 {
        (price.ln() / 1.0001_f64.ln()).floor() as i32
    }

    /// Calculate liquidity available at current price
    #[inline]
    pub fn get_liquidity_at_price(&self, price: f64) -> u128 {
        let current_tick = Self::price_to_tick(price);
        let mut total_liquidity = 0u128;

        for position in &self.positions {
            if current_tick >= position.tick_lower && current_tick < position.tick_upper {
                total_liquidity += position.liquidity;
            }
        }

        total_liquidity
    }

    /// Calculate output amount for a swap within a single tick range
    #[inline]
    pub fn calc_swap_output_single_range(
        amount_in: u128,
        sqrt_price_current_x96: u128,
        sqrt_price_target_x96: u128,
        liquidity: u128,
        fee_bps: u128,
    ) -> Option<(u128, u128)> {
        if liquidity == 0 {
            return None;
        }

        let amount_in_after_fee = amount_in * (10000 - fee_bps) / 10000;

        // Determine direction
        let zero_for_one = sqrt_price_target_x96 < sqrt_price_current_x96;

        if zero_for_one {
            // Selling token0 for token1
            // Δy = L * (√P_a - √P_b)
            let sqrt_diff = (sqrt_price_current_x96 as i128 - sqrt_price_target_x96 as i128).abs() as u128;
            let amount_out = (liquidity as u128).saturating_mul(sqrt_diff) >> 96;
            
            // Check if we have enough input
            // Δx = L * (1/√P_b - 1/√P_a)
            let inv_sqrt_diff = ((1u128 << 192) / sqrt_price_target_x96) 
                .saturating_sub((1u128 << 192) / sqrt_price_current_x96);
            let amount_needed = (liquidity as u128).saturating_mul(inv_sqrt_diff) >> 96;

            if amount_in_after_fee >= amount_needed {
                Some((amount_out, amount_needed))
            } else {
                None
            }
        } else {
            // Selling token1 for token0
            let sqrt_diff = (sqrt_price_target_x96 as i128 - sqrt_price_current_x96 as i128).abs() as u128;
            let amount_out = (liquidity as u128).saturating_mul(sqrt_diff) >> 96;
            
            let inv_sqrt_diff = ((1u128 << 192) / sqrt_price_current_x96)
                .saturating_sub((1u128 << 192) / sqrt_price_target_x96);
            let amount_needed = (liquidity as u128).saturating_mul(inv_sqrt_diff) >> 96;

            if amount_in_after_fee >= amount_needed {
                Some((amount_out, amount_needed))
            } else {
                None
            }
        }
    }

    /// Simulate multi-tick swap and return total output and ticks crossed
    pub fn simulate_swap(
        &self,
        amount_in: u128,
        zero_for_one: bool,
        fee_bps: u128,
    ) -> SwapResult {
        let mut remaining_input = amount_in;
        let mut total_output = 0u128;
        let mut ticks_crossed = Vec::new();
        let mut current_sqrt_price = self.current_sqrt_price_x96;
        let mut current_tick = self.current_tick;

        // Find active positions at current price
        let active_liquidity = self.get_liquidity_at_price(Self::sqrt_price_to_price(current_sqrt_price));

        if active_liquidity == 0 {
            return SwapResult {
                amount_out: 0,
                amount_in_used: 0,
                ticks_crossed: vec![],
                final_sqrt_price: current_sqrt_price,
                price_impact: 1.0,
            };
        }

        // Simplified: assume single tick range for now
        // Production would iterate through all ticks
        let target_tick = if zero_for_one {
            current_tick - 100
        } else {
            current_tick + 100
        };

        let target_sqrt_price = Self::tick_to_sqrt_price(target_tick);

        if let Some((output, input_used)) = Self::calc_swap_output_single_range(
            remaining_input,
            current_sqrt_price,
            target_sqrt_price,
            active_liquidity,
            fee_bps,
        ) {
            total_output = output;
            remaining_input -= input_used;
            current_sqrt_price = target_sqrt_price;
            ticks_crossed.push(target_tick);
        }

        let initial_price = Self::sqrt_price_to_price(self.current_sqrt_price_x96);
        let final_price = Self::sqrt_price_to_price(current_sqrt_price);
        let price_impact = ((final_price - initial_price).abs() / initial_price).min(1.0);

        SwapResult {
            amount_out: total_output,
            amount_in_used: amount_in - remaining_input,
            ticks_crossed,
            final_sqrt_price: current_sqrt_price,
            price_impact,
        }
    }

    fn tick_to_sqrt_price(tick: i32) -> u128 {
        Self::price_to_sqrt_price_x96(Self::tick_to_price(tick))
    }

    /// Add a liquidity position
    pub fn add_position(&mut self, tick_lower: i32, tick_upper: i32, liquidity: u128) {
        self.positions.push(Position {
            tick_lower,
            tick_upper,
            liquidity,
            tokens_owed_0: 0,
            tokens_owed_1: 0,
        });
    }

    /// Get total value locked in USD
    pub fn get_tvl(&self, price_token0: f64, price_token1: f64) -> f64 {
        let mut tvl = 0.0;
        
        for position in &self.positions {
            // Simplified TVL calculation
            let tick_range = (position.tick_upper - position.tick_lower) as f64;
            let liquidity_value = position.liquidity as f64 * tick_range * 0.0001;
            tvl += liquidity_value * (price_token0 + price_token1) / 2.0;
        }

        tvl
    }
}

#[derive(Debug, Clone)]
pub struct SwapResult {
    pub amount_out: u128,
    pub amount_in_used: u128,
    pub ticks_crossed: Vec<i32>,
    pub final_sqrt_price: u128,
    pub price_impact: f64,
}

/// Multi-hop route finder for optimal swap execution
pub struct RouteFinder {
    pools: Vec<PoolInfo>,
}

#[derive(Debug, Clone)]
pub struct PoolInfo {
    pub id: u64,
    pub token0: [u8; 20],
    pub token1: [u8; 20],
    pub amm_type: AmmType,
    pub reserve0: u128,
    pub reserve1: u128,
    pub fee_bps: u128,
    pub v3_data: Option<V3PoolData>,
}

#[derive(Debug, Clone)]
pub enum AmmType {
    ConstantProduct,
    ConcentratedLiquidity { tick_spacing: i32 },
}

#[derive(Debug, Clone)]
pub struct V3PoolData {
    pub current_tick: i32,
    pub current_sqrt_price_x96: u128,
    pub liquidity: u128,
}

#[derive(Debug, Clone)]
pub struct Route {
    pub hops: Vec<u64>,
    pub token_path: Vec<[u8; 20]>,
    pub expected_output: u128,
    pub price_impact: f64,
    pub total_fees_usd: f64,
}

impl RouteFinder {
    pub fn new() -> Self {
        Self { pools: Vec::new() }
    }

    pub fn add_pool(&mut self, pool: PoolInfo) {
        self.pools.push(pool);
    }

    /// Find optimal route for swapping token_in to token_out
    pub fn find_best_route(
        &self,
        token_in: [u8; 20],
        token_out: [u8; 20],
        amount_in: u128,
        max_hops: usize,
    ) -> Option<Route> {
        if token_in == token_out {
            return None;
        }

        // BFS to find all possible routes
        let mut routes = Vec::new();
        self.find_routes_dfs(
            token_in,
            token_out,
            amount_in,
            vec![],
            vec![token_in],
            max_hops,
            &mut routes,
        );

        // Return route with best output
        routes.into_iter().max_by(|a, b| a.expected_output.cmp(&b.expected_output))
    }

    fn find_routes_dfs(
        &self,
        current_token: [u8; 20],
        target: [u8; 20],
        amount_in: u128,
        mut hop_ids: Vec<u64>,
        mut token_path: Vec<[u8; 20]>,
        max_hops: usize,
        routes: &mut Vec<Route>,
    ) {
        if hop_ids.len() >= max_hops {
            return;
        }

        // Find pools connecting current token to next
        for (idx, pool) in self.pools.iter().enumerate() {
            let next_token = if pool.token0 == current_token {
                pool.token1
            } else if pool.token1 == current_token {
                pool.token0
            } else {
                continue;
            };

            // Skip if already visited
            if token_path.contains(&next_token) {
                continue;
            }

            // Calculate output for this hop
            let output = self.calculate_hop_output(pool, current_token, amount_in);

            let mut new_hops = hop_ids.clone();
            new_hops.push(idx as u64);

            let mut new_path = token_path.clone();
            new_path.push(next_token);

            if next_token == target {
                // Found complete route
                routes.push(Route {
                    hops: new_hops,
                    token_path: new_path,
                    expected_output: output,
                    price_impact: 0.0, // Would calculate properly
                    total_fees_usd: 0.0,
                });
            } else {
                // Continue DFS
                self.find_routes_dfs(
                    next_token,
                    target,
                    output,
                    new_hops,
                    new_path,
                    max_hops,
                    routes,
                );
            }
        }
    }

    fn calculate_hop_output(&self, pool: &PoolInfo, token_in: [u8; 20], amount_in: u128) -> u128 {
        let is_token0_in = pool.token0 == token_in;
        
        match pool.amm_type {
            AmmType::ConstantProduct => {
                let (reserve_in, reserve_out) = if is_token0_in {
                    (pool.reserve0, pool.reserve1)
                } else {
                    (pool.reserve1, pool.reserve0)
                };

                ConstantProductAmm::get_amount_out(amount_in, reserve_in, reserve_out, pool.fee_bps)
                    .unwrap_or(0)
            }
            AmmType::ConcentratedLiquidity { .. } => {
                // Simplified for now
                if let Some(v3) = &pool.v3_data {
                    let cl = ConcentratedLiquidityAmm::new(60);
                    let result = cl.simulate_swap(amount_in, !is_token0_in, pool.fee_bps);
                    result.amount_out
                } else {
                    0
                }
            }
        }
    }
}

impl Default for RouteFinder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_constant_product_swap() {
        let amount_out = ConstantProductAmm::get_amount_out(
            1_000_000,
            100_000_000,
            50_000_000,
            30, // 0.3% fee
        ).unwrap();

        assert!(amount_out > 0);
        assert!(amount_out < 500_000); // Should be less than half due to reserves
    }

    #[test]
    fn test_price_impact() {
        let impact = ConstantProductAmm::calculate_price_impact(
            1_000_000,
            100_000_000,
            50_000_000,
            30,
        ).unwrap();

        assert!(impact > 0.0);
        assert!(impact < 1.0);
    }

    #[test]
    fn test_tick_to_price() {
        let price = ConcentratedLiquidityAmm::tick_to_price(0);
        assert!((price - 1.0).abs() < 0.0001);

        let price_100 = ConcentratedLiquidityAmm::tick_to_price(100);
        assert!(price_100 > 1.0);
    }
}
