//! Real-time collateral haircut engine.
//! Evaluates volatility and liquidity of held assets.
//! Dynamically adjusts margin value to prevent liquidation cascades.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};

/// Fixed-point precision multiplier (1e8)
const FP_MULTIPLIER: u128 = 1_000_000_000;

/// Haircut configuration for an asset
#[derive(Debug, Clone, Copy)]
pub struct HaircutConfig {
    pub base_haircut_bps: u32,      // Base haircut in basis points
    pub vol_scaling_factor: f64,    // Multiplier for volatility adjustment
    pub liq_scaling_factor: f64,    // Multiplier for liquidity adjustment
    pub max_haircut_bps: u32,       // Maximum allowed haircut
    pub min_haircut_bps: u32,       // Minimum allowed haircut
}

impl Default for HaircutConfig {
    fn default() -> Self {
        Self {
            base_haircut_bps: 500,       // 5% base haircut
            vol_scaling_factor: 1.5,     // 1.5x scaling for vol
            liq_scaling_factor: 1.2,     // 1.2x scaling for liquidity
            max_haircut_bps: 5000,       // Max 50% haircut
            min_haircut_bps: 100,        // Min 1% haircut
        }
    }
}

/// Collateral asset with real-time metrics
#[derive(Debug, Clone, Copy)]
pub struct CollateralAsset {
    pub asset_id: u64,
    pub quantity: u128,           // Asset quantity * 1e8
    pub price_usd: u128,          // Price in USD * 1e8
    pub notional_value: u128,     // Quantity * Price / 1e8
    pub volatility_30d: f64,      // 30-day annualized volatility
    pub liquidity_score: f64,     // 0.0 (illiquid) to 1.0 (highly liquid)
    pub current_haircut_bps: u32, // Current applied haircut
    pub adjusted_value: u128,     // Value after haircut
}

/// Haircut calculation result
#[derive(Debug, Clone, Copy)]
pub struct HaircutResult {
    pub asset_id: u64,
    pub original_value: u128,
    pub haircut_bps: u32,
    pub haircut_amount: u128,
    pub adjusted_value: u128,
    pub haircuts: HashMap<String, u32>, // Breakdown by component
}

/// Real-time collateral haircut calculator
pub struct CollateralHaircutCalculator {
    /// Assets being tracked
    assets: HashMap<u64, CollateralAsset>,
    /// Configuration per asset (or default)
    configs: HashMap<u64, HaircutConfig>,
    /// Global risk-off mode multiplier
    risk_off_multiplier: f64,
    /// Memory footprint tracking
    memory_footprint: AtomicU64,
}

impl CollateralHaircutCalculator {
    pub fn new() -> Self {
        Self {
            assets: HashMap::with_capacity(128),
            configs: HashMap::new(),
            risk_off_multiplier: 1.0,
            memory_footprint: AtomicU64::new(0),
        }
    }

    /// Add or update a collateral asset
    pub fn update_asset(
        &mut self,
        asset_id: u64,
        quantity: u128,
        price_usd: u128,
        volatility_30d: f64,
        liquidity_score: f64,
    ) {
        let notional_value = (quantity * price_usd) / FP_MULTIPLIER;
        
        let config = self.configs.get(&asset_id).copied().unwrap_or_default();
        let haircut_bps = self.calculate_haircut(volatility_30d, liquidity_score, &config);
        let haircut_amount = (notional_value * haircut_bps as u128) / 10000;
        let adjusted_value = notional_value.saturating_sub(haircut_amount);

        let asset = CollateralAsset {
            asset_id,
            quantity,
            price_usd,
            notional_value,
            volatility_30d,
            liquidity_score,
            current_haircut_bps: haircut_bps,
            adjusted_value,
        };

        self.assets.insert(asset_id, asset);
        
        self.memory_footprint.store(
            (self.assets.len() * std::mem::size_of::<CollateralAsset>()) as u64,
            Ordering::Relaxed,
        );
    }

    /// Calculate haircut for given volatility and liquidity
    fn calculate_haircut(
        &self,
        volatility: f64,
        liquidity_score: f64,
        config: &HaircutConfig,
    ) -> u32 {
        // Base haircut
        let mut haircut = config.base_haircut_bps as f64;

        // Volatility adjustment (higher vol = higher haircut)
        // Normalize volatility: 0.5 (low) to 2.0 (extreme)
        let vol_factor = (volatility / 0.5).min(4.0);
        haircut *= config.vol_scaling_factor * vol_factor;

        // Liquidity adjustment (lower liquidity = higher haircut)
        let liq_factor = if liquidity_score > 0.0 {
            1.0 / liquidity_score.min(1.0)
        } else {
            5.0 // Penalty for zero liquidity score
        };
        haircut *= config.liq_scaling_factor * liq_factor;

        // Apply global risk-off multiplier
        haircut *= self.risk_off_multiplier;

        // Clamp to min/max bounds
        haircut = haircut.max(config.min_haircut_bps as f64);
        haircut = haircut.min(config.max_haircut_bps as f64);

        haircut as u32
    }

    /// Set custom configuration for an asset
    pub fn set_config(&mut self, asset_id: u64, config: HaircutConfig) {
        self.configs.insert(asset_id, config);
        // Recalculate haircut for this asset
        if let Some(asset) = self.assets.get_mut(&asset_id) {
            let haircut_bps = self.calculate_haircut(
                asset.volatility_30d,
                asset.liquidity_score,
                &config,
            );
            asset.current_haircut_bps = haircut_bps;
            let haircut_amount = (asset.notional_value * haircut_bps as u128) / 10000;
            asset.adjusted_value = asset.notional_value.saturating_sub(haircut_amount);
        }
    }

    /// Enable/disable risk-off mode (increases all haircuts)
    pub fn set_risk_off_mode(&mut self, enabled: bool) {
        self.risk_off_multiplier = if enabled { 2.0 } else { 1.0 };
        
        // Recalculate all haircuts
        for asset in self.assets.values_mut() {
            let config = self.configs.get(&asset.asset_id).copied().unwrap_or_default();
            asset.current_haircut_bps = self.calculate_haircut(
                asset.volatility_30d,
                asset.liquidity_score,
                &config,
            );
            let haircut_amount = (asset.notional_value * asset.current_haircut_bps as u128) / 10000;
            asset.adjusted_value = asset.notional_value.saturating_sub(haircut_amount);
        }
    }

    /// Get detailed haircut breakdown for an asset
    pub fn get_haircut_breakdown(&self, asset_id: u64) -> Option<HaircutResult> {
        let asset = self.assets.get(&asset_id)?;
        let config = self.configs.get(&asset_id).copied().unwrap_or_default();
        
        let haircut_amount = asset.notional_value - asset.adjusted_value;
        
        // Calculate component breakdown
        let mut haircuts = HashMap::new();
        haircuts.insert("base".to_string(), config.base_haircut_bps);
        
        let vol_component = (config.base_haircut_bps as f64 
            * config.vol_scaling_factor 
            * (asset.volatility_30d / 0.5).min(4.0)) as u32;
        haircuts.insert("volatility".to_string(), vol_component);
        
        let liq_factor = if asset.liquidity_score > 0.0 {
            1.0 / asset.liquidity_score.min(1.0)
        } else {
            5.0
        };
        let liq_component = (config.base_haircut_bps as f64 
            * config.liq_scaling_factor 
            * liq_factor) as u32;
        haircuts.insert("liquidity".to_string(), liq_component);

        Some(HaircutResult {
            asset_id,
            original_value: asset.notional_value,
            haircut_bps: asset.current_haircut_bps,
            haircut_amount,
            adjusted_value: asset.adjusted_value,
            haircuts,
        })
    }

    /// Get total collateral value (after all haircuts)
    pub fn total_collateral_value(&self) -> u128 {
        self.assets.values().map(|a| a.adjusted_value).sum()
    }

    /// Get total unadjusted value
    pub fn total_unadjusted_value(&self) -> u128 {
        self.assets.values().map(|a| a.notional_value).sum()
    }

    /// Get total haircut amount
    pub fn total_haircut_amount(&self) -> u128 {
        self.total_unadjusted_value() - self.total_collateral_value()
    }

    /// Get average haircut across all collateral
    pub fn average_haircut_bps(&self) -> f64 {
        let total_value = self.total_unadjusted_value();
        if total_value == 0 {
            return 0.0;
        }
        
        let total_haircut = self.total_haircut_amount();
        (total_haircut as f64 / total_value as f64) * 10000.0
    }

    /// Check if collateral is sufficient for a required margin
    pub fn is_collateral_sufficient(&self, required_margin: u128) -> bool {
        self.total_collateral_value() >= required_margin
    }

    /// Get collateral deficit (if any)
    pub fn get_collateral_deficit(&self, required_margin: u128) -> u128 {
        let available = self.total_collateral_value();
        if available >= required_margin {
            0
        } else {
            required_margin - available
        }
    }

    /// Get all assets
    pub fn get_all_assets(&self) -> Vec<&CollateralAsset> {
        self.assets.values().collect()
    }

    /// Get memory footprint in bytes
    pub fn memory_footprint(&self) -> u64 {
        self.memory_footprint.load(Ordering::Relaxed)
    }

    /// Remove an asset
    pub fn remove_asset(&mut self, asset_id: u64) {
        self.assets.remove(&asset_id);
    }

    /// Clear all assets
    pub fn clear(&mut self) {
        self.assets.clear();
    }
}

impl Default for CollateralHaircutCalculator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_haircut_calculation() {
        let mut calc = CollateralHaircutCalculator::new();
        
        // Add high-quality collateral (low vol, high liquidity)
        calc.update_asset(1, 1_000_000_000, 50_000_000_000, 0.3, 0.95); // BTC-like
        
        // Add risky collateral (high vol, low liquidity)
        calc.update_asset(2, 1_000_000_000, 100_000_000, 1.5, 0.3); // Small cap altcoin
        
        let btc_result = calc.get_haircut_breakdown(1).unwrap();
        let alt_result = calc.get_haircut_breakdown(2).unwrap();
        
        // BTC should have lower haircut
        assert!(btc_result.haircut_bps < alt_result.haircut_bps);
        
        // Total collateral should be less than unadjusted
        assert!(calc.total_collateral_value() < calc.total_unadjusted_value());
        
        // Test risk-off mode
        calc.set_risk_off_mode(true);
        let btc_result_ro = calc.get_haircut_breakdown(1).unwrap();
        
        // Haircut should increase in risk-off mode
        assert!(btc_result_ro.haircut_bps > btc_result.haircut_bps);
    }
}
