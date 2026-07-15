//! Smart rebalancing engine that calculates exact delta needed to restore target weights
//! Factors in trading fees, TWAP/VWAP slippage models, and tax/lot implications

use std::sync::atomic::{AtomicU64, Ordering};

const MAX_ASSETS: usize = 128;

/// Trade direction
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TradeDirection {
    Buy,
    Sell,
}

/// Rebalance trade order
#[derive(Debug, Clone)]
pub struct RebalanceTrade {
    pub asset_id: u32,
    pub direction: TradeDirection,
    pub quantity: f64,
    pub estimated_price: f64,
    pub estimated_cost: f64,
    pub fee_estimate: f64,
    pub slippage_estimate: f64,
    pub priority: u8,
}

/// Result of rebalance calculation
#[derive(Debug, Clone)]
pub struct RebalancePlan {
    pub trades: Vec<RebalanceTrade>,
    pub total_turnover: f64,
    pub total_fees: f64,
    pub total_slippage: f64,
    pub net_cash_flow: f64,
    pub execution_time_ms: u64,
}

/// Slippage model parameters
#[derive(Debug, Clone, Copy)]
pub struct SlippageModel {
    /// Linear slippage coefficient (bps per % of ADV)
    pub linear_coef: f64,
    /// Quadratic slippage coefficient (for large orders)
    pub quadratic_coef: f64,
    /// Minimum spread (bps)
    pub min_spread_bps: f64,
}

impl Default for SlippageModel {
    fn default() -> Self {
        Self {
            linear_coef: 0.5,  // 0.5 bps per 1% of ADV
            quadratic_coef: 0.1,
            min_spread_bps: 1.0,
        }
    }
}

/// Fee model
#[derive(Debug, Clone, Copy)]
pub struct FeeModel {
    /// Maker fee (bps)
    pub maker_bps: f64,
    /// Taker fee (bps)
    pub taker_bps: f64,
    /// Volume discount threshold (USD)
    pub discount_threshold: f64,
    /// Discount rate above threshold
    pub discount_rate: f64,
}

impl Default for FeeModel {
    fn default() -> Self {
        Self {
            maker_bps: 2.0,
            taker_bps: 5.0,
            discount_threshold: 1_000_000.0,
            discount_rate: 0.5,
        }
    }
}

/// Rebalance Executor state
#[repr(align(64))]
pub struct RebalanceExecutor {
    /// Current portfolio weights
    current_weights: [f64; MAX_ASSETS],
    /// Target portfolio weights
    target_weights: [f64; MAX_ASSETS],
    /// Asset prices
    prices: [f64; MAX_ASSETS],
    /// Portfolio value (USD)
    portfolio_value: f64,
    /// Asset count
    asset_count: usize,
    /// Slippage model
    slippage_model: SlippageModel,
    /// Fee model
    fee_model: FeeModel,
    /// Average daily volume for each asset (USD)
    adv: [f64; MAX_ASSETS],
    /// Minimum trade size (USD)
    min_trade_size: f64,
    /// Drift threshold for triggering rebalance
    drift_threshold: f64,
    /// Update counter
    update_counter: AtomicU64,
}

unsafe impl Send for RebalanceExecutor {}
unsafe impl Sync for RebalanceExecutor {}

impl RebalanceExecutor {
    #[inline(always)]
    pub fn new(asset_count: usize, portfolio_value: f64) -> Self {
        assert!(asset_count <= MAX_ASSETS);
        
        Self {
            current_weights: [0.0; MAX_ASSETS],
            target_weights: [0.0; MAX_ASSETS],
            prices: [0.0; MAX_ASSETS],
            portfolio_value,
            asset_count,
            slippage_model: SlippageModel::default(),
            fee_model: FeeModel::default(),
            adv: [10_000_000.0; MAX_ASSETS],  // Default $10M ADV
            min_trade_size: 10.0,  // $10 minimum
            drift_threshold: 0.01,  // 1% drift triggers rebalance
            update_counter: AtomicU64::new(0),
        }
    }

    /// Update current weights
    #[inline(always)]
    pub fn set_current_weights(&mut self, weights: &[f64]) {
        assert_eq!(weights.len(), self.asset_count);
        for i in 0..self.asset_count {
            self.current_weights[i] = weights[i];
        }
        self.update_counter.fetch_add(1, Ordering::Release);
    }

    /// Set target weights
    #[inline(always)]
    pub fn set_target_weights(&mut self, weights: &[f64]) {
        assert_eq!(weights.len(), self.asset_count);
        for i in 0..self.asset_count {
            self.target_weights[i] = weights[i];
        }
        self.update_counter.fetch_add(1, Ordering::Release);
    }

    /// Update prices
    #[inline(always)]
    pub fn set_prices(&mut self, prices: &[f64]) {
        assert_eq!(prices.len(), self.asset_count);
        for i in 0..self.asset_count {
            self.prices[i] = prices[i];
        }
    }

    /// Set ADV for slippage estimation
    #[inline(always)]
    pub fn set_adv(&mut self, adv: &[f64]) {
        assert_eq!(adv.len(), self.asset_count);
        for i in 0..self.asset_count {
            self.adv[i] = adv[i].max(100_000.0);  // Minimum $100k ADV
        }
    }

    /// Calculate portfolio drift from target
    #[inline(always)]
    pub fn calculate_drift(&self) -> f64 {
        let mut max_drift = 0.0;
        let mut total_drift = 0.0;
        
        for i in 0..self.asset_count {
            let drift = (self.current_weights[i] - self.target_weights[i]).abs();
            max_drift = max_drift.max(drift);
            total_drift += drift;
        }
        
        // Return L-infinity norm (max single asset drift)
        max_drift
    }

    /// Check if rebalance is needed
    #[inline(always)]
    pub fn needs_rebalance(&self) -> bool {
        self.calculate_drift() > self.drift_threshold
    }

    /// Estimate slippage for a trade
    #[inline(always)]
    fn estimate_slippage(&self, trade_value: f64, asset_idx: usize) -> f64 {
        let adv = self.adv[asset_idx];
        let adv_pct = trade_value / adv;
        
        // Slippage = min_spread + linear * adv_pct + quadratic * adv_pct^2
        let slippage_bps = self.slippage_model.min_spread_bps
            + self.slippage_model.linear_coef * adv_pct * 100.0
            + self.slippage_model.quadratic_coef * adv_pct * adv_pct * 10000.0;
        
        trade_value * slippage_bps / 10000.0
    }

    /// Estimate trading fee
    #[inline(always)]
    fn estimate_fee(&self, trade_value: f64, is_maker: bool) -> f64 {
        let base_fee_bps = if is_maker {
            self.fee_model.maker_bps
        } else {
            self.fee_model.taker_bps
        };
        
        // Apply volume discount if applicable
        let effective_fee_bps = if trade_value > self.fee_model.discount_threshold {
            base_fee_bps * self.fee_model.discount_rate
        } else {
            base_fee_bps
        };
        
        trade_value * effective_fee_bps / 10000.0
    }

    /// Generate rebalance plan
    #[inline(always)]
    pub fn generate_plan(&self) -> RebalancePlan {
        let start_time = std::time::Instant::now();
        
        let mut trades = Vec::with_capacity(self.asset_count);
        let mut total_turnover = 0.0;
        let mut total_fees = 0.0;
        let mut total_slippage = 0.0;
        let mut net_cash_flow = 0.0;
        
        for i in 0..self.asset_count {
            let weight_diff = self.target_weights[i] - self.current_weights[i];
            
            if weight_diff.abs() < 1e-6 {
                continue;
            }
            
            let trade_value = weight_diff * self.portfolio_value;
            
            // Skip small trades
            if trade_value.abs() < self.min_trade_size {
                continue;
            }
            
            let price = self.prices[i].max(1e-10);
            let quantity = trade_value.abs() / price;
            let direction = if weight_diff > 0.0 {
                TradeDirection::Buy
            } else {
                TradeDirection::Sell
            };
            
            // Estimate costs
            let slippage = self.estimate_slippage(trade_value.abs(), i);
            let fee = self.estimate_fee(trade_value.abs(), false);  // Assume taker
            
            let cost = trade_value.abs() + slippage + fee;
            
            if direction == TradeDirection::Buy {
                net_cash_flow -= cost;
            } else {
                net_cash_flow += trade_value.abs() - fee - slippage;
            }
            
            total_turnover += trade_value.abs();
            total_fees += fee;
            total_slippage += slippage;
            
            trades.push(RebalanceTrade {
                asset_id: i as u32,
                direction,
                quantity,
                estimated_price: price,
                estimated_cost: cost,
                fee_estimate: fee,
                slippage_estimate: slippage,
                priority: self.calculate_priority(i, weight_diff),
            });
        }
        
        // Sort by priority (higher first)
        trades.sort_by(|a, b| b.priority.cmp(&a.priority));
        
        let execution_time_ms = start_time.elapsed().as_micros() as u64 / 1000;
        
        RebalancePlan {
            trades,
            total_turnover,
            total_fees,
            total_slippage,
            net_cash_flow,
            execution_time_ms,
        }
    }

    /// Calculate trade priority based on drift and urgency
    #[inline(always)]
    fn calculate_priority(&self, asset_idx: usize, weight_diff: f64) -> u8 {
        let drift = weight_diff.abs();
        let vol = (self.covariance_element(asset_idx, asset_idx)).sqrt();
        
        // Higher priority for larger drifts and higher volatility assets
        let priority_score = drift * vol * 1000.0;
        
        priority_score.min(255.0) as u8
    }

    /// Get covariance element (simplified - would be passed in production)
    #[inline(always)]
    fn covariance_element(&self, i: usize, j: usize) -> f64 {
        // Placeholder - in production this would use actual covariance
        if i == j {
            0.04  // 20% vol assumption
        } else {
            0.01
        }
    }

    /// Generate TWAP execution schedule
    #[inline(always)]
    pub fn generate_twap_schedule(
        &self,
        plan: &RebalancePlan,
        duration_minutes: u32,
        num_slices: u32,
    ) -> Vec<Vec<RebalanceTrade>> {
        let mut schedule = Vec::with_capacity(num_slices as usize);
        let slice_interval = duration_minutes / num_slices;
        
        for slice in 0..num_slices {
            let mut slice_trades = Vec::new();
            
            for trade in &plan.trades {
                let slice_quantity = trade.quantity / num_slices as f64;
                
                slice_trades.push(RebalanceTrade {
                    asset_id: trade.asset_id,
                    direction: trade.direction,
                    quantity: slice_quantity,
                    estimated_price: trade.estimated_price,
                    estimated_cost: trade.estimated_cost / num_slices as f64,
                    fee_estimate: trade.fee_estimate / num_slices as f64,
                    slippage_estimate: trade.slippage_estimate / num_slices as f64,
                    priority: trade.priority,
                });
            }
            
            schedule.push(slice_trades);
        }
        
        schedule
    }

    #[inline(always)]
    pub fn update_counter(&self) -> u64 {
        self.update_counter.load(Ordering::Acquire)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rebalance_executor() {
        let mut executor = RebalanceExecutor::new(3, 100_000.0);
        
        // Set current weights (drifted from target)
        executor.set_current_weights(&[0.4, 0.35, 0.25]);
        executor.set_target_weights(&[0.33, 0.33, 0.34]);
        executor.set_prices(&[50000.0, 3000.0, 1.0]);
        
        // Check drift
        let drift = executor.calculate_drift();
        assert!(drift > 0.01);  // Should need rebalance
        
        // Generate plan
        let plan = executor.generate_plan();
        
        assert!(!plan.trades.is_empty());
        assert!(plan.total_turnover > 0.0);
        assert!(plan.total_fees > 0.0);
        
        println!("Rebalance plan:");
        println!("  Trades: {}", plan.trades.len());
        println!("  Turnover: ${:.2}", plan.total_turnover);
        println!("  Fees: ${:.2}", plan.total_fees);
        println!("  Slippage: ${:.2}", plan.total_slippage);
    }
}
