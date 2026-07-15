//! Maker/Taker fee optimization logic for intelligent order routing.
//! Captures maker rebates or minimizes taker fees based on execution urgency.

use std::sync::atomic::{AtomicF64, AtomicBool, AtomicU64, Ordering};
use std::collections::HashMap;

/// Fee tier structure
#[derive(Debug, Clone)]
pub struct FeeTier {
    pub volume_threshold: f64,      // 30-day volume threshold (USD)
    pub maker_fee_bps: f64,
    pub taker_fee_bps: f64,
    pub maker_rebate_bps: f64,      // Negative = rebate
}

/// Current fee status
#[derive(Debug, Clone)]
pub struct FeeStatus {
    pub current_tier: usize,
    pub thirty_day_volume: f64,
    pub maker_fee_bps: f64,
    pub taker_fee_bps: f64,
    pub maker_rebate_bps: f64,
    pub next_tier_volume: f64,
    pub volume_to_next_tier: f64,
}

/// Fee optimization result
#[derive(Debug, Clone)]
pub struct FeeOptimizationResult {
    pub recommended_order_type: OrderType,
    pub expected_fee_bps: f64,
    pub potential_rebate_bps: f64,
    pub urgency_adjustment: f64,
    pub cost_savings_bps: f64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum OrderType {
    MakerLimit,
    TakerMarket,
    TakerLimit,
    PostOnly,
}

/// Fee optimizer main engine
pub struct FeeOptimizer {
    /// Fee tiers by exchange
    fee_tiers: HashMap<String, Vec<FeeTier>>,
    /// Current tier per exchange
    current_tiers: HashMap<String, usize>,
    /// 30-day rolling volume per exchange
    rolling_volumes: HashMap<String, AtomicF64>,
    /// Target fee tier for optimization
    target_tier: AtomicU64,
    /// Prefer maker orders when possible
    prefer_maker: AtomicBool,
    /// Minimum spread for maker orders (bps)
    min_maker_spread_bps: f64,
}

impl FeeOptimizer {
    /// Create new fee optimizer
    pub fn new() -> Self {
        let mut fee_tiers = HashMap::new();
        
        // Default Binance fee tiers
        fee_tiers.insert(
            "BINANCE".to_string(),
            vec![
                FeeTier {
                    volume_threshold: 0.0,
                    maker_fee_bps: 10.0,
                    taker_fee_bps: 10.0,
                    maker_rebate_bps: 0.0,
                },
                FeeTier {
                    volume_threshold: 1_000_000.0,
                    maker_fee_bps: 8.0,
                    taker_fee_bps: 10.0,
                    maker_rebate_bps: 0.0,
                },
                FeeTier {
                    volume_threshold: 5_000_000.0,
                    maker_fee_bps: 6.0,
                    taker_fee_bps: 8.0,
                    maker_rebate_bps: 0.0,
                },
                FeeTier {
                    volume_threshold: 20_000_000.0,
                    maker_fee_bps: 4.0,
                    taker_fee_bps: 6.0,
                    maker_rebate_bps: 0.0,
                },
                FeeTier {
                    volume_threshold: 100_000_000.0,
                    maker_fee_bps: 2.0,
                    taker_fee_bps: 4.0,
                    maker_rebate_bps: 0.0,
                },
            ],
        );
        
        // Binance Futures (lower fees)
        fee_tiers.insert(
            "BINANCE_FUTURES".to_string(),
            vec![
                FeeTier {
                    volume_threshold: 0.0,
                    maker_fee_bps: 2.0,
                    taker_fee_bps: 4.0,
                    maker_rebate_bps: 0.0,
                },
                FeeTier {
                    volume_threshold: 1_000_000.0,
                    maker_fee_bps: 0.0,
                    taker_fee_bps: 2.0,
                    maker_rebate_bps: 0.0,
                },
                FeeTier {
                    volume_threshold: 5_000_000.0,
                    maker_fee_bps: -2.0, // Rebate!
                    taker_fee_bps: 2.0,
                    maker_rebate_bps: 2.0,
                },
            ],
        );
        
        Self {
            fee_tiers,
            current_tiers: HashMap::new(),
            rolling_volumes: HashMap::new(),
            target_tier: AtomicU64::new(0),
            prefer_maker: AtomicBool::new(true),
            min_maker_spread_bps: 2.0,
        }
    }

    /// Update 30-day volume for an exchange
    #[inline(always)]
    pub fn update_volume(&self, exchange: &str, volume_usd: f64) {
        if let Some(vol) = self.rolling_volumes.get(exchange) {
            vol.store(volume_usd, Ordering::Relaxed);
        } else {
            self.rolling_volumes.insert(exchange.to_string(), AtomicF64::new(volume_usd));
        }
        
        // Update current tier
        self.update_current_tier(exchange);
    }

    /// Get current fee status for an exchange
    pub fn get_fee_status(&self, exchange: &str) -> FeeStatus {
        let tiers = match self.fee_tiers.get(exchange) {
            Some(t) => t,
            None => return self.get_default_status(),
        };
        
        let volume = self.rolling_volumes.get(exchange)
            .map(|v| v.load(Ordering::Relaxed))
            .unwrap_or(0.0);
        
        let current_tier_idx = self.current_tiers.get(exchange).copied().unwrap_or(0);
        let current_tier = &tiers[current_tier_idx.min(tiers.len() - 1)];
        
        let next_tier_volume = if current_tier_idx < tiers.len() - 1 {
            tiers[current_tier_idx + 1].volume_threshold
        } else {
            f64::MAX
        };
        
        FeeStatus {
            current_tier: current_tier_idx,
            thirty_day_volume: volume,
            maker_fee_bps: current_tier.maker_fee_bps,
            taker_fee_bps: current_tier.taker_fee_bps,
            maker_rebate_bps: current_tier.maker_rebate_bps,
            next_tier_volume,
            volume_to_next_tier: (next_tier_volume - volume).max(0.0),
        }
    }

    /// Update current tier based on volume
    fn update_current_tier(&self, exchange: &str) {
        let tiers = match self.fee_tiers.get(exchange) {
            Some(t) => t,
            None => return,
        };
        
        let volume = self.rolling_volumes.get(exchange)
            .map(|v| v.load(Ordering::Relaxed))
            .unwrap_or(0.0);
        
        let mut tier_idx = 0;
        for (i, tier) in tiers.iter().enumerate() {
            if volume >= tier.volume_threshold {
                tier_idx = i;
            } else {
                break;
            }
        }
        
        self.current_tiers.insert(exchange.to_string(), tier_idx);
    }

    /// Get default status when exchange not found
    fn get_default_status(&self) -> FeeStatus {
        FeeStatus {
            current_tier: 0,
            thirty_day_volume: 0.0,
            maker_fee_bps: 10.0,
            taker_fee_bps: 10.0,
            maker_rebate_bps: 0.0,
            next_tier_volume: 1_000_000.0,
            volume_to_next_tier: 1_000_000.0,
        }
    }

    /// Optimize fee strategy based on urgency and market conditions
    pub fn optimize_fees(
        &self,
        exchange: &str,
        urgency: f64,           // 0.0 (no rush) to 1.0 (immediate)
        spread_bps: f64,
        order_size_usd: f64,
    ) -> FeeOptimizationResult {
        let status = self.get_fee_status(exchange);
        
        let prefer_maker = self.prefer_maker.load(Ordering::Relaxed);
        
        // Calculate effective costs
        let maker_cost = status.maker_fee_bps - status.maker_rebate_bps;
        let taker_cost = status.taker_fee_bps;
        
        // Fee savings from using maker
        let fee_savings = taker_cost - maker_cost;
        
        // Spread cost for maker order (might not fill immediately)
        let spread_opportunity_cost = if spread_bps > self.min_maker_spread_bps {
            spread_bps * 0.5 // Half spread risk
        } else {
            0.0
        };
        
        // Net benefit of maker order
        let maker_net_benefit = fee_savings - spread_opportunity_cost;
        
        // Determine optimal order type
        let (recommended_type, expected_fee, potential_rebate) = if urgency > 0.8 {
            // High urgency: use taker
            (OrderType::TakerMarket, taker_cost, 0.0)
        } else if urgency > 0.5 && maker_net_benefit > 0.0 {
            // Medium urgency: aggressive limit (cross spread slightly)
            (OrderType::TakerLimit, taker_cost * 0.8, 0.0)
        } else if prefer_maker && maker_net_benefit > 0.0 {
            // Low urgency: pure maker
            (OrderType::PostOnly, maker_cost, status.maker_rebate_bps)
        } else {
            // Default to maker if spread is tight
            if spread_bps < self.min_maker_spread_bps {
                (OrderType::MakerLimit, maker_cost, status.maker_rebate_bps)
            } else {
                (OrderType::TakerLimit, taker_cost * 0.9, 0.0)
            }
        };
        
        let cost_savings = taker_cost - expected_fee;
        
        FeeOptimizationResult {
            recommended_order_type: recommended_type,
            expected_fee_bps: expected_fee,
            potential_rebate_bps: potential_rebate,
            urgency_adjustment: urgency,
            cost_savings_bps: cost_savings,
        }
    }

    /// Calculate total fees for a set of executions
    pub fn calculate_total_fees(
        &self,
        exchange: &str,
        maker_notional: f64,
        taker_notional: f64,
    ) -> (f64, f64) {
        let status = self.get_fee_status(exchange);
        
        let maker_fee = maker_notional * (status.maker_fee_bps / 10000.0);
        let taker_fee = taker_notional * (status.taker_fee_bps / 10000.0);
        
        // Apply rebate
        let net_maker_fee = maker_fee - (maker_notional * (status.maker_rebate_bps / 10000.0));
        
        (net_maker_fee.max(0.0), taker_fee)
    }

    /// Set preference for maker orders
    #[inline(always)]
    pub fn set_prefer_maker(&self, prefer: bool) {
        self.prefer_maker.store(prefer, Ordering::Relaxed);
    }

    /// Set minimum spread for maker orders
    #[inline(always)]
    pub fn set_min_maker_spread(&mut self, spread_bps: f64) {
        self.min_maker_spread_bps = spread_bps;
    }

    /// Get projected fees at next tier
    pub fn get_next_tier_projection(&self, exchange: &str) -> Option<(f64, f64)> {
        let status = self.get_fee_status(exchange);
        let tiers = self.fee_tiers.get(exchange)?;
        
        if status.current_tier >= tiers.len() - 1 {
            return None; // Already at highest tier
        }
        
        let next_tier = &tiers[status.current_tier + 1];
        Some((next_tier.maker_fee_bps, next_tier.taker_fee_bps))
    }

    /// Add custom fee tiers for an exchange
    pub fn add_exchange_tiers(&mut self, exchange: &str, tiers: Vec<FeeTier>) {
        self.fee_tiers.insert(exchange.to_string(), tiers);
        self.current_tiers.insert(exchange.to_string(), 0);
    }
}

impl Default for FeeOptimizer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fee_status() {
        let optimizer = FeeOptimizer::new();
        
        optimizer.update_volume("BINANCE", 500_000.0);
        let status = optimizer.get_fee_status("BINANCE");
        
        assert_eq!(status.current_tier, 0);
        assert_eq!(status.maker_fee_bps, 10.0);
        assert_eq!(status.taker_fee_bps, 10.0);
        
        optimizer.update_volume("BINANCE", 2_000_000.0);
        let status = optimizer.get_fee_status("BINANCE");
        
        assert_eq!(status.current_tier, 1);
        assert_eq!(status.maker_fee_bps, 8.0);
    }

    #[test]
    fn test_fee_optimization() {
        let optimizer = FeeOptimizer::new();
        
        // Low urgency should recommend maker
        let result = optimizer.optimize_fees("BINANCE", 0.2, 5.0, 10_000.0);
        assert!(result.expected_fee_bps <= 10.0);
        
        // High urgency should recommend taker
        let result = optimizer.optimize_fees("BINANCE", 0.9, 5.0, 10_000.0);
        assert_eq!(result.recommended_order_type, OrderType::TakerMarket);
    }

    #[test]
    fn test_fee_calculation() {
        let optimizer = FeeOptimizer::new();
        optimizer.update_volume("BINANCE_FUTURES", 10_000_000.0);
        
        let (maker_fee, taker_fee) = optimizer.calculate_total_fees(
            "BINANCE_FUTURES",
            1_000_000.0,
            1_000_000.0,
        );
        
        // Should have rebate at high tier
        assert!(maker_fee < taker_fee);
    }
}
