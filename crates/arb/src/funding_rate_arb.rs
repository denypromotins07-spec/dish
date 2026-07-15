//! Funding Rate Arbitrage Engine
//! Cash-and-carry and funding rate arbitrage executor
//! Spots extreme funding rate divergences between spot and perps

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;

/// Funding rate data from an exchange
#[derive(Debug, Clone)]
pub struct FundingRate {
    pub exchange: String,
    pub symbol: String,
    pub funding_rate: f64,      // Annualized or per-period rate
    pub next_funding_time: u64, // Unix timestamp in milliseconds
    pub index_price: f64,
    pub mark_price: f64,
    pub basis_bps: f64,         // (mark - spot) / spot * 10000
    pub timestamp: Instant,
}

/// Hedged position for cash-and-carry
#[derive(Debug, Clone)]
pub struct HedgedPosition {
    pub id: String,
    pub symbol: String,
    pub long_exchange: String,
    pub short_exchange: String,
    pub spot_position_size: f64,
    pub perp_position_size: f64,
    pub entry_spot_price: f64,
    pub entry_perp_price: f64,
    pub expected_funding_capture: f64,
    pub opened_at: Instant,
    pub status: PositionStatus,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PositionStatus {
    Open,
    Closing,
    Closed,
    Liquidated,
}

/// Funding arbitrage opportunity
#[derive(Debug, Clone)]
pub struct FundingArbOpportunity {
    pub symbol: String,
    pub buy_exchange: String,   // Exchange with lower/negative funding
    pub sell_exchange: String,  // Exchange with higher/positive funding
    pub funding_rate_diff: f64, // Difference in funding rates
    pub annualized_return: f64, // Expected annualized return
    pub risk_score: f64,
    pub max_position: f64,
    pub timestamp: Instant,
}

/// Funding rate arbitrage engine
pub struct FundingRateArbEngine {
    funding_rates: Arc<RwLock<HashMap<String, HashMap<String, FundingRate>>>>,
    open_positions: Arc<RwLock<HashMap<String, HedgedPosition>>>,
    opportunities: Arc<RwLock<Vec<FundingArbOpportunity>>>,
    min_funding_threshold: f64,
    max_total_exposure: f64,
    enabled: bool,
}

impl FundingRateArbEngine {
    pub fn new(min_funding_threshold: f64, max_total_exposure: f64) -> Self {
        Self {
            funding_rates: Arc::new(RwLock::new(HashMap::new())),
            open_positions: Arc::new(RwLock::new(HashMap::new())),
            opportunities: Arc::new(RwLock::new(Vec::new())),
            min_funding_threshold,
            max_total_exposure,
            enabled: true,
        }
    }

    /// Process funding rate update
    pub async fn process_funding_rate(&self, rate: FundingRate) -> Option<FundingArbOpportunity> {
        if !self.enabled {
            return None;
        }

        let mut rates = self.funding_rates.write().await;
        
        rates
            .entry(rate.symbol.clone())
            .or_insert_with(HashMap::new)
            .insert(rate.exchange.clone(), rate.clone());

        // Check for arbitrage opportunity
        if let Some(symbol_rates) = rates.get(&rate.symbol) {
            if symbol_rates.len() >= 2 {
                return self.check_funding_arbitrage(&rate.symbol, symbol_rates).await;
            }
        }

        None
    }

    /// Check for funding rate arbitrage across exchanges
    async fn check_funding_arbitrage(
        &self,
        symbol: &str,
        rates: &HashMap<String, FundingRate>,
    ) -> Option<FundingArbOpportunity> {
        let mut highest_rate: Option<(&String, &FundingRate)> = None;
        let mut lowest_rate: Option<(&String, &FundingRate)> = None;

        for (exchange, rate) in rates.iter() {
            if highest_rate.is_none() || rate.funding_rate > highest_rate.unwrap().1.funding_rate {
                highest_rate = Some((exchange, rate));
            }
            if lowest_rate.is_none() || rate.funding_rate < lowest_rate.unwrap().1.funding_rate {
                lowest_rate = Some((exchange, rate));
            }
        }

        if let (Some((high_ex, high_rate)), Some((low_ex, low_rate))) = (highest_rate, lowest_rate) {
            if high_ex != low_ex {
                let rate_diff = high_rate.funding_rate - low_rate.funding_rate;
                
                // Convert to annualized (assuming 8-hour funding periods, 3 per day, ~1095 per year)
                let annualized_return = rate_diff * 3.0 * 365.0 * 100.0;

                if rate_diff.abs() > self.min_funding_threshold {
                    let opportunity = FundingArbOpportunity {
                        symbol: symbol.to_string(),
                        buy_exchange: low_ex.clone(),      // Buy on low/negative funding
                        sell_exchange: high_ex.clone(),    // Sell on high/positive funding
                        funding_rate_diff: rate_diff,
                        annualized_return,
                        risk_score: self.calculate_risk(high_rate, low_rate),
                        max_position: self.calculate_max_position(high_rate, low_rate),
                        timestamp: Instant::now(),
                    };

                    // Store opportunity
                    let mut opps = self.opportunities.write().await;
                    opps.push(opportunity.clone());
                    
                    if opps.len() > 1000 {
                        opps.remove(0);
                    }

                    return Some(opportunity);
                }
            }
        }

        None
    }

    /// Calculate risk score for funding arb
    fn calculate_risk(&self, high: &FundingRate, low: &FundingRate) -> f64 {
        let mut risk = 0.0;

        // Basis risk - large divergence between mark and index
        let high_basis_risk = high.basis_bps.abs() / 100.0; // Normalize
        let low_basis_risk = low.basis_bps.abs() / 100.0;
        risk += (high_basis_risk + low_basis_risk) / 2.0;

        // Time until next funding - closer is better
        let now_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64;
        
        let high_time_risk = if high.next_funding_time > now_ms {
            ((high.next_funding_time - now_ms) as f64 / 3600000.0).min(8.0) / 8.0
        } else {
            0.5
        };
        
        risk += high_time_risk * 0.3;

        // Exchange concentration risk
        risk = risk.min(1.0);
        risk
    }

    /// Calculate maximum position size based on exposure limits
    fn calculate_max_position(&self, high: &FundingRate, low: &FundingRate) -> f64 {
        // Simple calculation based on available exposure
        let current_exposure = self.get_current_exposure();
        let available = self.max_total_exposure - current_exposure;
        
        // Also consider liquidity
        let max_by_liquidity = (high.index_price * 10.0).min(low.index_price * 10.0);
        
        available.min(max_by_liquidity).max(0.0)
    }

    /// Get current total exposure from open positions
    async fn get_current_exposure(&self) -> f64 {
        let positions = self.open_positions.read().await;
        positions.values()
            .filter(|p| p.status == PositionStatus::Open)
            .map(|p| p.spot_position_size * p.entry_spot_price)
            .sum()
    }

    /// Open a hedged position
    pub async fn open_hedged_position(
        &self,
        symbol: &str,
        spot_exchange: &str,
        perp_exchange: &str,
        size: f64,
        spot_price: f64,
        perp_price: f64,
        expected_funding: f64,
    ) -> String {
        let position_id = format!("FUND-{}", chrono::Utc::now().timestamp_nanos());
        
        let position = HedgedPosition {
            id: position_id.clone(),
            symbol: symbol.to_string(),
            long_exchange: spot_exchange.to_string(),
            short_exchange: perp_exchange.to_string(),
            spot_position_size: size,
            perp_position_size: size,
            entry_spot_price: spot_price,
            entry_perp_price: perp_price,
            expected_funding_capture: expected_funding,
            opened_at: Instant::now(),
            status: PositionStatus::Open,
        };

        let mut positions = self.open_positions.write().await;
        positions.insert(position_id.clone(), position);

        position_id
    }

    /// Close a hedged position
    pub async fn close_position(&self, position_id: &str) -> Result<f64, &'static str> {
        let mut positions = self.open_positions.write().await;
        
        let position = positions.get_mut(position_id)
            .ok_or("Position not found")?;
        
        if position.status != PositionStatus::Open {
            return Err("Position not open");
        }

        position.status = PositionStatus::Closing;
        
        // In production, would execute actual trades here
        // Calculate P&L
        let pnl = position.spot_position_size * (position.entry_perp_price - position.entry_spot_price);
        
        position.status = PositionStatus::Closed;
        
        Ok(pnl)
    }

    /// Get all open positions
    pub async fn get_open_positions(&self) -> Vec<HedgedPosition> {
        let positions = self.open_positions.read().await;
        positions.values()
            .filter(|p| p.status == PositionStatus::Open)
            .cloned()
            .collect()
    }

    /// Get recent opportunities
    pub async fn get_recent_opportunities(&self, limit: usize) -> Vec<FundingArbOpportunity> {
        let opps = self.opportunities.read().await;
        opps.iter().rev().take(limit).cloned().collect()
    }

    /// Enable/disable engine
    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    /// Get funding rate statistics by symbol
    pub async fn get_funding_stats(&self, symbol: &str) -> Option<FundingStats> {
        let rates = self.funding_rates.read().await;
        
        if let Some(symbol_rates) = rates.get(symbol) {
            let rates_vec: Vec<f64> = symbol_rates.values()
                .map(|r| r.funding_rate)
                .collect();
            
            if rates_vec.is_empty() {
                return None;
            }

            let avg = rates_vec.iter().sum::<f64>() / rates_vec.len() as f64;
            let variance = rates_vec.iter()
                .map(|r| (r - avg).powi(2))
                .sum::<f64>() / rates_vec.len() as f64;
            let std_dev = variance.sqrt();

            Some(FundingStats {
                symbol: symbol.to_string(),
                average_rate: avg,
                std_deviation: std_dev,
                min_rate: rates_vec.iter().cloned().fold(f64::INFINITY, f64::min),
                max_rate: rates_vec.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
                exchange_count: symbol_rates.len(),
            })
        } else {
            None
        }
    }
}

/// Funding rate statistics
#[derive(Debug, Clone)]
pub struct FundingStats {
    pub symbol: String,
    pub average_rate: f64,
    pub std_deviation: f64,
    pub min_rate: f64,
    pub max_rate: f64,
    pub exchange_count: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_funding_arb_detection() {
        let engine = FundingRateArbEngine::new(0.0001, 100000.0); // 0.01% threshold
        
        let rate1 = FundingRate {
            exchange: "binance".to_string(),
            symbol: "BTCUSDT".to_string(),
            funding_rate: 0.0001, // 0.01% per 8 hours
            next_funding_time: 0,
            index_price: 45000.0,
            mark_price: 45010.0,
            basis_bps: 2.2,
            timestamp: Instant::now(),
        };

        let rate2 = FundingRate {
            exchange: "bybit".to_string(),
            symbol: "BTCUSDT".to_string(),
            funding_rate: 0.0005, // 0.05% per 8 hours
            next_funding_time: 0,
            index_price: 45000.0,
            mark_price: 45020.0,
            basis_bps: 4.4,
            timestamp: Instant::now(),
        };

        engine.process_funding_rate(rate1).await;
        let opp = engine.process_funding_rate(rate2).await;

        assert!(opp.is_some());
        let opp = opp.unwrap();
        assert!(opp.funding_rate_diff > 0.0001);
        assert!(opp.annualized_return > 0.0);
    }
}
