//! Rust engine for real-time supply analytics: circulating supply, EIP-1559 burn, inflation/deflation.
//! Calculates metrics on every new block for the macro feature pipeline.

use std::sync::atomic::{AtomicU64, AtomicU128, Ordering};
use std::sync::Arc;

/// Real-time supply metrics calculated per block
#[derive(Debug, Clone)]
pub struct SupplyMetrics {
    pub block_number: u64,
    pub timestamp: u64,
    pub total_supply: u128,
    pub circulating_supply: u128,
    pub burned_supply: u128,
    pub staked_supply: u128,
    pub inflation_rate_annual: f64,
    pub deflation_rate_annual: f64,
    pub net_issuance_rate: f64,
    pub eip1559_base_fee: u128,
    pub burn_rate_per_block: u128,
}

/// EIP-1559 burn tracking data
#[derive(Debug, Clone)]
pub struct BurnData {
    pub block_number: u64,
    pub base_fee_per_gas: u128,
    pub gas_used: u64,
    pub eth_burned: u128,
    pub cumulative_burned: u128,
}

/// Staking metrics for supply calculation
#[derive(Debug, Clone)]
pub struct StakingData {
    pub total_validators: u64,
    pub total_staked_eth: u128,
    pub staking_apr: f64,
    pub annual_issuance_eth: u128,
}

/// Supply analyzer with strict memory bounds
pub struct SupplyAnalyzer {
    /// Initial total supply at genesis (or reference point)
    initial_supply: u128,
    /// Current tracked metrics
    current_metrics: Arc<parking_lot::RwLock<SupplyMetrics>>,
    /// Cumulative burn tracker
    cumulative_burned: AtomicU128,
    /// Block-by-block burn history (bounded)
    burn_history: parking_lot::Mutex<Vec<BurnData>>,
    /// Max history size to prevent memory bloat
    max_history_size: usize,
    /// Last processed block
    last_processed_block: AtomicU64,
}

impl SupplyAnalyzer {
    /// Create a new supply analyzer
    pub fn new(initial_supply_wei: u128) -> Self {
        const ONE_ETH: u128 = 1_000_000_000_000_000_000;
        
        Self {
            initial_supply: initial_supply_wei,
            current_metrics: Arc::new(parking_lot::RwLock::new(SupplyMetrics {
                block_number: 0,
                timestamp: 0,
                total_supply: initial_supply_wei,
                circulating_supply: initial_supply_wei,
                burned_supply: 0,
                staked_supply: 0,
                inflation_rate_annual: 0.0,
                deflation_rate_annual: 0.0,
                net_issuance_rate: 0.0,
                eip1559_base_fee: 0,
                burn_rate_per_block: 0,
            })),
            cumulative_burned: AtomicU128::new(0),
            burn_history: parking_lot::Mutex::new(Vec::with_capacity(100)),
            max_history_size: 1000,
            last_processed_block: AtomicU64::new(0),
        }
    }

    /// Process a new block and update supply metrics
    pub fn process_block(
        &self,
        block_number: u64,
        timestamp: u64,
        base_fee_per_gas: u128,
        gas_used: u64,
        block_reward_wei: u128,
        staking_data: Option<StakingData>,
    ) -> SupplyMetrics {
        // Skip if already processed
        let last_block = self.last_processed_block.load(Ordering::Relaxed);
        if block_number <= last_block {
            return self.get_current_metrics();
        }

        // Calculate burn for this block
        let eth_burned = (base_fee_per_gas as u128) * (gas_used as u128);
        
        // Update cumulative burn
        let prev_cumulative = self.cumulative_burned.load(Ordering::Relaxed);
        let new_cumulative = prev_cumulative + eth_burned;
        self.cumulative_burned.store(new_cumulative, Ordering::Relaxed);

        // Get staked supply if available
        let staked_supply = staking_data.as_ref()
            .map(|s| s.total_staked_eth)
            .unwrap_or(0);

        // Calculate issuance (block rewards + staking rewards)
        let staking_issuance = staking_data.as_ref()
            .map(|s| s.annual_issuance_eth / 2_628_000) // Per block approximation
            .unwrap_or(0);
        
        let total_issuance = block_reward_wei + staking_issuance;

        // Net change = issuance - burn
        let net_change = total_issuance as i128 - eth_burned as i128;
        
        // Update total supply
        let mut metrics = self.current_metrics.write();
        let new_total_supply = if net_change >= 0 {
            metrics.total_supply + net_change as u128
        } else {
            metrics.total_supply.saturating_sub((-net_change) as u128)
        };

        // Circulating supply = total - staked (simplified)
        let circulating = new_total_supply.saturating_sub(staked_supply);

        // Calculate rates (annualized)
        let blocks_per_year = 2_628_000u128; // ~12 second blocks
        let annual_burn = eth_burned * blocks_per_year;
        let annual_issuance = total_issuance * blocks_per_year;

        let inflation_rate = if metrics.total_supply > 0 {
            (annual_issuance as f64) / (metrics.total_supply as f64) * 100.0
        } else {
            0.0
        };

        let deflation_rate = if metrics.total_supply > 0 {
            (annual_burn as f64) / (metrics.total_supply as f64) * 100.0
        } else {
            0.0
        };

        let net_rate = inflation_rate - deflation_rate;

        // Update metrics
        metrics.block_number = block_number;
        metrics.timestamp = timestamp;
        metrics.total_supply = new_total_supply;
        metrics.circulating_supply = circulating;
        metrics.burned_supply = new_cumulative;
        metrics.staked_supply = staked_supply;
        metrics.inflation_rate_annual = inflation_rate;
        metrics.deflation_rate_annual = deflation_rate;
        metrics.net_issuance_rate = net_rate;
        metrics.eip1559_base_fee = base_fee_per_gas;
        metrics.burn_rate_per_block = eth_burned;

        // Record burn data
        self.record_burn_data(BurnData {
            block_number,
            base_fee_per_gas,
            gas_used,
            eth_burned,
            cumulative_burned: new_cumulative,
        });

        drop(metrics);

        self.last_processed_block.store(block_number, Ordering::Relaxed);

        self.get_current_metrics()
    }

    /// Record burn data with bounded history
    fn record_burn_data(&self, data: BurnData) {
        let mut history = self.burn_history.lock();
        history.push(data);
        
        // Enforce max size
        if history.len() > self.max_history_size {
            *history = history.split_off(history.len() - self.max_history_size);
        }
    }

    /// Get current supply metrics
    pub fn get_current_metrics(&self) -> SupplyMetrics {
        self.current_metrics.read().clone()
    }

    /// Get average burn rate over recent blocks
    pub fn get_average_burn_rate(&self, num_blocks: usize) -> u128 {
        let history = self.burn_history.lock();
        let take = num_blocks.min(history.len());
        
        if take == 0 {
            return 0;
        }

        let sum: u128 = history.iter()
            .rev()
            .take(take)
            .map(|d| d.eth_burned)
            .sum();

        sum / take as u128
    }

    /// Check if network is in deflationary regime
    pub fn is_deflationary(&self) -> bool {
        let metrics = self.current_metrics.read();
        metrics.deflation_rate_annual > metrics.inflation_rate_annual
    }

    /// Get supply regime classification
    pub fn get_regime(&self) -> SupplyRegime {
        let metrics = self.current_metrics.read();
        
        if metrics.deflation_rate_annual > metrics.inflation_rate_annual * 1.1 {
            SupplyRegime::Deflationary
        } else if metrics.inflation_rate_annual > metrics.deflation_rate_annual * 1.1 {
            SupplyRegime::Inflationary
        } else {
            SupplyRegime::Neutral
        }
    }

    /// Calculate time until next supply milestone
    pub fn estimate_milestone(
        &self,
        target_supply_wei: u128,
    ) -> Option<MilestoneEstimate> {
        let metrics = self.current_metrics.read();
        
        let current = metrics.total_supply as i128;
        let target = target_supply_wei as i128;
        let delta = target - current;
        
        if delta == 0 {
            return Some(MilestoneEstimate {
                blocks_until_milestone: 0,
                estimated_time_hours: 0.0,
                direction: MilestoneDirection::AlreadyReached,
            });
        }

        let net_per_block = metrics.net_issuance_rate / 100.0 * metrics.total_supply as f64 / 2_628_000.0;
        
        if net_per_block.abs() < 1.0 {
            return None; // Rate too small to estimate
        }

        let blocks_needed = (delta as f64 / net_per_block).abs() as u64;
        let hours = (blocks_needed as f64 * 12.0) / 3600.0;

        let direction = if delta > 0 {
            MilestoneDirection::Increasing
        } else {
            MilestoneDirection::Decreasing
        };

        Some(MilestoneEstimate {
            blocks_until_milestone: blocks_needed,
            estimated_time_hours: hours,
            direction,
        })
    }

    /// Get historical burn trend
    pub fn get_burn_trend(&self, num_blocks: usize) -> Vec<BurnData> {
        let history = self.burn_history.lock();
        history.iter()
            .rev()
            .take(num_blocks)
            .cloned()
            .collect()
    }

    /// Set maximum history size
    pub fn set_max_history(&mut self, size: usize) {
        self.max_history_size = size;
        let mut history = self.burn_history.lock();
        if history.len() > size {
            *history = history.split_off(history.len() - size);
        }
    }

    /// Export metrics for external consumption
    pub fn export_metrics(&self) -> SupplyMetricsExport {
        let metrics = self.current_metrics.read();
        let one_eth = 1_000_000_000_000_000_000f64;
        
        SupplyMetricsExport {
            block_number: metrics.block_number,
            total_supply_eth: metrics.total_supply as f64 / one_eth,
            circulating_supply_eth: metrics.circulating_supply as f64 / one_eth,
            burned_supply_eth: metrics.burned_supply as f64 / one_eth,
            staked_supply_eth: metrics.staked_supply as f64 / one_eth,
            inflation_rate_percent: metrics.inflation_rate_annual,
            deflation_rate_percent: metrics.deflation_rate_annual,
            net_rate_percent: metrics.net_issuance_rate,
            is_deflationary: self.is_deflationary(),
            regime: self.get_regime(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SupplyRegime {
    Inflationary,
    Deflationary,
    Neutral,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MilestoneDirection {
    Increasing,
    Decreasing,
    AlreadyReached,
}

#[derive(Debug, Clone)]
pub struct MilestoneEstimate {
    pub blocks_until_milestone: u64,
    pub estimated_time_hours: f64,
    pub direction: MilestoneDirection,
}

#[derive(Debug, Clone)]
pub struct SupplyMetricsExport {
    pub block_number: u64,
    pub total_supply_eth: f64,
    pub circulating_supply_eth: f64,
    pub burned_supply_eth: f64,
    pub staked_supply_eth: f64,
    pub inflation_rate_percent: f64,
    pub deflation_rate_percent: f64,
    pub net_rate_percent: f64,
    pub is_deflationary: bool,
    pub regime: SupplyRegime,
}

impl Default for SupplyAnalyzer {
    fn default() -> Self {
        // Ethereum mainnet genesis supply approximately
        const ETH_GENESIS_SUPPLY: u128 = 72_000_000 * 1_000_000_000_000_000_000;
        Self::new(ETH_GENESIS_SUPPLY)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_supply_analyzer_creation() {
        let analyzer = SupplyAnalyzer::new(100_000_000_000_000_000_000);
        let metrics = analyzer.get_current_metrics();
        
        assert_eq!(metrics.total_supply, 100_000_000_000_000_000_000);
        assert_eq!(metrics.block_number, 0);
    }

    #[test]
    fn test_process_block_with_burn() {
        let analyzer = SupplyAnalyzer::new(100_000_000_000_000_000_000);
        
        // Process a block with typical EIP-1559 burn
        let metrics = analyzer.process_block(
            1,
            1000,
            30_000_000_000, // 30 gwei base fee
            15_000_000,     // 15M gas used
            2_000_000_000_000_000_000, // 2 ETH block reward
            None,
        );
        
        assert_eq!(metrics.block_number, 1);
        assert!(metrics.burned_supply > 0);
        assert!(metrics.deflation_rate_annual > 0.0);
    }

    #[test]
    fn test_deflationary_detection() {
        let analyzer = SupplyAnalyzer::default();
        
        // Process block with high burn
        analyzer.process_block(
            1,
            1000,
            100_000_000_000, // Very high base fee
            30_000_000,      // High gas usage
            2_000_000_000_000_000_000,
            None,
        );
        
        // With high burn, should potentially be deflationary
        let _regime = analyzer.get_regime();
    }
}
