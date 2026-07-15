//! Cross-Exchange Latency Arbitrage Engine
//! Monitors lead-lag relationships between exchanges
//! Executes microsecond market orders on lagging venue

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;

/// Price update from an exchange
#[derive(Debug, Clone)]
pub struct PriceUpdate {
    pub exchange: String,
    pub symbol: String,
    pub bid: f64,
    pub ask: f64,
    pub bid_size: f64,
    pub ask_size: f64,
    pub timestamp: Instant,
}

/// Lead-lag relationship tracker
#[derive(Debug, Clone)]
pub struct LeadLagTracker {
    pub leader: String,
    pub lagger: String,
    pub correlation: f64,
    pub lag_ms: u64,
    pub last_lead_move: Instant,
    pub lead_price_change: f64,
}

/// Arbitrage opportunity detected
#[derive(Debug, Clone)]
pub struct ArbOpportunity {
    pub buy_exchange: String,
    pub sell_exchange: String,
    pub symbol: String,
    pub spread_bps: f64,
    pub expected_profit: f64,
    pub risk_score: f64,
    pub timestamp: Instant,
}

/// Latency arbitrage engine
pub struct LatencyArbEngine {
    prices: Arc<RwLock<HashMap<String, HashMap<String, PriceUpdate>>>>,
    lead_lag_trackers: Arc<RwLock<HashMap<String, LeadLagTracker>>>,
    opportunities: Arc<RwLock<Vec<ArbOpportunity>>>,
    threshold_bps: f64,
    max_position_size: f64,
    enabled: bool,
}

impl LatencyArbEngine {
    pub fn new(threshold_bps: f64, max_position_size: f64) -> Self {
        Self {
            prices: Arc::new(RwLock::new(HashMap::new())),
            lead_lag_trackers: Arc::new(RwLock::new(HashMap::new())),
            opportunities: Arc::new(RwLock::new(Vec::new())),
            threshold_bps,
            max_position_size,
            enabled: true,
        }
    }

    /// Process incoming price update
    pub async fn process_price_update(&self, update: PriceUpdate) -> Option<ArbOpportunity> {
        if !self.enabled {
            return None;
        }

        let mut prices = self.prices.write().await;
        
        // Store price
        prices
            .entry(update.symbol.clone())
            .or_insert_with(HashMap::new)
            .insert(update.exchange.clone(), update.clone());

        // Check for arbitrage opportunity
        if let Some(symbol_prices) = prices.get(&update.symbol) {
            if symbol_prices.len() >= 2 {
                return self.check_arbitrage(&update.symbol, symbol_prices).await;
            }
        }

        None
    }

    /// Check for arbitrage opportunities across exchanges
    async fn check_arbitrage(
        &self,
        symbol: &str,
        prices: &HashMap<String, PriceUpdate>,
    ) -> Option<ArbOpportunity> {
        let mut best_buy: Option<(&String, &PriceUpdate)> = None;
        let mut best_sell: Option<(&String, &PriceUpdate)> = None;

        // Find best bid (sell) and ask (buy) across exchanges
        for (exchange, price) in prices.iter() {
            if best_buy.is_none() || price.ask < best_buy.unwrap().1.ask {
                best_buy = Some((exchange, price));
            }
            if best_sell.is_none() || price.bid > best_sell.unwrap().1.bid {
                best_sell = Some((exchange, price));
            }
        }

        if let (Some((buy_ex, buy_price)), Some((sell_ex, sell_price))) = (best_buy, best_sell) {
            if buy_ex != sell_ex {
                // Calculate spread in basis points
                let mid_price = (buy_price.ask + sell_price.bid) / 2.0;
                let spread = sell_price.bid - buy_price.ask;
                let spread_bps = (spread / mid_price) * 10000.0;

                if spread_bps > self.threshold_bps {
                    let opportunity = ArbOpportunity {
                        buy_exchange: buy_ex.clone(),
                        sell_exchange: sell_ex.clone(),
                        symbol: symbol.to_string(),
                        spread_bps,
                        expected_profit: spread * self.max_position_size,
                        risk_score: self.calculate_risk(buy_price, sell_price),
                        timestamp: Instant::now(),
                    };

                    // Store opportunity
                    let mut opps = self.opportunities.write().await;
                    opps.push(opportunity.clone());
                    
                    // Keep only recent opportunities
                    if opps.len() > 1000 {
                        opps.remove(0);
                    }

                    return Some(opportunity);
                }
            }
        }

        None
    }

    /// Calculate risk score for arbitrage
    fn calculate_risk(&self, buy: &PriceUpdate, sell: &PriceUpdate) -> f64 {
        let mut risk = 0.0;

        // Size risk - insufficient liquidity
        if buy.ask_size < self.max_position_size * 0.5 {
            risk += 0.3;
        }
        if sell.bid_size < self.max_position_size * 0.5 {
            risk += 0.3;
        }

        // Latency risk - stale prices
        let now = Instant::now();
        if now.duration_since(buy.timestamp).as_millis() > 100 {
            risk += 0.2;
        }
        if now.duration_since(sell.timestamp).as_millis() > 100 {
            risk += 0.2;
        }

        // Spread risk - unusually wide spread
        risk = risk.min(1.0);
        risk
    }

    /// Update lead-lag relationship
    pub async fn update_lead_lag(
        &self,
        symbol: &str,
        exchange1: &str,
        exchange2: &str,
        price_change1: f64,
        price_change2: f64,
        lag_ms: u64,
    ) {
        let key = format!("{}-{}", exchange1, exchange2);
        
        let mut trackers = self.lead_lag_trackers.write().await;
        
        let tracker = trackers.entry(key).or_insert_with(|| LeadLagTracker {
            leader: exchange1.to_string(),
            lagger: exchange2.to_string(),
            correlation: 0.0,
            lag_ms,
            last_lead_move: Instant::now(),
            lead_price_change: price_change1,
        });

        // Update correlation using EMA
        let same_direction = (price_change1 > 0.0) == (price_change2 > 0.0);
        let sample_corr = if same_direction { 1.0 } else { -1.0 };
        tracker.correlation = tracker.correlation * 0.9 + sample_corr * 0.1;

        // Determine leader based on who moved first
        if price_change1.abs() > price_change2.abs() * 1.1 {
            tracker.leader = exchange1.to_string();
            tracker.lagger = exchange2.to_string();
        } else if price_change2.abs() > price_change1.abs() * 1.1 {
            tracker.leader = exchange2.to_string();
            tracker.lagger = exchange1.to_string();
        }

        tracker.lag_ms = lag_ms;
        tracker.last_lead_move = Instant::now();
        tracker.lead_price_change = price_change1;
    }

    /// Get recent arbitrage opportunities
    pub async fn get_recent_opportunities(&self, limit: usize) -> Vec<ArbOpportunity> {
        let opps = self.opportunities.read().await;
        opps.iter().rev().take(limit).cloned().collect()
    }

    /// Enable/disable arbitrage engine
    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    /// Clear old opportunities
    pub async fn clear_old_opportunities(&self, max_age_secs: u64) {
        let cutoff = Instant::now() - Duration::from_secs(max_age_secs);
        
        let mut opps = self.opportunities.write().await;
        opps.retain(|opp| opp.timestamp > cutoff);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_arbitrage_detection() {
        let engine = LatencyArbEngine::new(10.0, 1.0); // 10 bps threshold
        
        let update1 = PriceUpdate {
            exchange: "binance".to_string(),
            symbol: "BTCUSDT".to_string(),
            bid: 44990.0,
            ask: 44995.0,
            bid_size: 10.0,
            ask_size: 10.0,
            timestamp: Instant::now(),
        };

        let update2 = PriceUpdate {
            exchange: "bybit".to_string(),
            symbol: "BTCUSDT".to_string(),
            bid: 45010.0,
            ask: 45015.0,
            bid_size: 10.0,
            ask_size: 10.0,
            timestamp: Instant::now(),
        };

        // First update
        engine.process_price_update(update1).await;
        
        // Second update should trigger opportunity
        let opp = engine.process_price_update(update2).await;
        
        assert!(opp.is_some());
        let opp = opp.unwrap();
        assert!(opp.spread_bps > 10.0);
    }
}
