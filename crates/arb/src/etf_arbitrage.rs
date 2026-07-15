//! Crypto index/ETF vs. underlying basket arbitrage logic
//! Tracks premium/discount of index tokens against spot constituents
//! Microsecond-level execution for risk-free convergence trades

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};

/// Represents a constituent asset in an index basket
#[derive(Clone, Debug)]
pub struct BasketConstituent {
    pub symbol: String,
    pub weight: f64,
    pub price: f64,
    pub quantity: f64,
}

/// ETF/Index arbitrage opportunity detector
pub struct EtfArbitrageur {
    /// Index/ETF symbol
    index_symbol: String,
    /// Constituent basket
    constituents: Vec<BasketConstituent>,
    /// Current index price
    index_price: f64,
    /// Fair value of the basket
    fair_value: f64,
    /// Premium/discount (positive = premium, negative = discount)
    premium_bps: f64,
    /// Transaction costs in bps
    transaction_cost_bps: f64,
    /// Minimum threshold for arbitrage (in bps)
    min_threshold_bps: f64,
    /// Last update timestamp (microseconds)
    last_update_us: AtomicU64,
    /// Price history for tracking convergence
    premium_history: Vec<f64>,
    max_history: usize,
}

impl EtfArbitrageur {
    pub fn new(
        index_symbol: &str,
        constituents: Vec<BasketConstituent>,
        transaction_cost_bps: f64,
        min_threshold_bps: f64,
    ) -> Self {
        Self {
            index_symbol: index_symbol.to_string(),
            constituents,
            index_price: 0.0,
            fair_value: 0.0,
            premium_bps: 0.0,
            transaction_cost_bps,
            min_threshold_bps,
            last_update_us: AtomicU64::new(0),
            premium_history: Vec::new(),
            max_history: 1000,
        }
    }

    /// Update index price and recalculate fair value
    #[inline]
    pub fn update_index_price(&mut self, price: f64, timestamp_us: u64) {
        self.index_price = price;
        self.last_update_us.store(timestamp_us, Ordering::Relaxed);
        self._recalculate();
    }

    /// Update a single constituent price
    #[inline]
    pub fn update_constituent_price(&mut self, symbol: &str, price: f64, timestamp_us: u64) {
        for constituent in &mut self.constituents {
            if constituent.symbol == symbol {
                constituent.price = price;
                self.last_update_us.store(timestamp_us, Ordering::Relaxed);
                self._recalculate();
                return;
            }
        }
    }

    /// Update all constituent prices at once (more efficient)
    pub fn update_all_prices(
        &mut self,
        index_price: f64,
        constituent_prices: &HashMap<String, f64>,
        timestamp_us: u64,
    ) {
        self.index_price = index_price;
        
        for constituent in &mut self.constituents {
            if let Some(&price) = constituent_prices.get(&constituent.symbol) {
                constituent.price = price;
            }
        }
        
        self.last_update_us.store(timestamp_us, Ordering::Relaxed);
        self._recalculate();
    }

    fn _recalculate(&mut self) {
        // Calculate fair value as weighted sum of constituent prices
        self.fair_value = self.constituents
            .iter()
            .map(|c| c.weight * c.price)
            .sum();

        // Calculate premium/discount in basis points
        if self.fair_value > 0.0 {
            self.premium_bps = (self.index_price - self.fair_value) / self.fair_value * 10000.0;
        } else {
            self.premium_bps = 0.0;
        }

        // Update history
        self.premium_history.push(self.premium_bps);
        if self.premium_history.len() > self.max_history {
            self.premium_history.remove(0);
        }
    }

    /// Check if arbitrage opportunity exists
    #[inline]
    pub fn has_opportunity(&self) -> bool {
        self.premium_bps.abs() > self.min_threshold_bps + self.transaction_cost_bps
    }

    /// Get the recommended arbitrage direction
    /// Returns: Some(ArbitrageDirection) if opportunity exists, None otherwise
    pub fn get_arbitrage_direction(&self) -> Option<ArbitrageDirection> {
        if !self.has_opportunity() {
            return None;
        }

        if self.premium_bps > 0.0 {
            // Index trading at premium: sell index, buy basket
            Some(ArbitrageDirection::SellIndexBuyBasket)
        } else {
            // Index trading at discount: buy index, sell basket
            Some(ArbitrageDirection::BuyIndexSellBasket)
        }
    }

    /// Calculate expected profit from arbitrage (in bps)
    #[inline]
    pub fn expected_profit_bps(&self) -> f64 {
        if !self.has_opportunity() {
            return 0.0;
        }
        self.premium_bps.abs() - self.transaction_cost_bps
    }

    /// Calculate optimal trade sizes for arbitrage
    pub fn calculate_trade_sizes(&self, notional: f64) -> ArbitrageTrade {
        let direction = self.get_arbitrage_direction();
        
        match direction {
            Some(ArbitrageDirection::SellIndexBuyBasket) => {
                let index_quantity = notional / self.index_price;
                let basket_trades: Vec<ConstituentTrade> = self.constituents
                    .iter()
                    .map(|c| {
                        let constituent_notional = notional * c.weight;
                        ConstituentTrade {
                            symbol: c.symbol.clone(),
                            side: Side::Buy,
                            quantity: constituent_notional / c.price,
                            notional: constituent_notional,
                        }
                    })
                    .collect();

                ArbitrageTrade {
                    direction: ArbitrageDirection::SellIndexBuyBasket,
                    index_side: Side::Sell,
                    index_quantity,
                    index_price: self.index_price,
                    basket_trades,
                    expected_profit_bps: self.expected_profit_bps(),
                }
            }
            Some(ArbitrageDirection::BuyIndexSellBasket) => {
                let index_quantity = notional / self.index_price;
                let basket_trades: Vec<ConstituentTrade> = self.constituents
                    .iter()
                    .map(|c| {
                        let constituent_notional = notional * c.weight;
                        ConstituentTrade {
                            symbol: c.symbol.clone(),
                            side: Side::Sell,
                            quantity: constituent_notional / c.price,
                            notional: constituent_notional,
                        }
                    })
                    .collect();

                ArbitrageTrade {
                    direction: ArbitrageDirection::BuyIndexSellBasket,
                    index_side: Side::Buy,
                    index_quantity,
                    index_price: self.index_price,
                    basket_trades,
                    expected_profit_bps: self.expected_profit_bps(),
                }
            }
            None => ArbitrageTrade::empty(),
        }
    }

    /// Get half-life of premium mean reversion (simplified estimate)
    pub fn estimate_mean_reversion_half_life(&self) -> f64 {
        if self.premium_history.len() < 20 {
            return f64::NAN;
        }

        // Simplified: calculate autocorrelation and derive half-life
        let mean = self.premium_history.iter().sum::<f64>() / self.premium_history.len() as f64;
        
        let mut numerator = 0.0;
        let mut denominator = 0.0;
        
        for i in 1..self.premium_history.len() {
            numerator += (self.premium_history[i] - mean) * (self.premium_history[i - 1] - mean);
            denominator += (self.premium_history[i - 1] - mean).powi(2);
        }

        if denominator < 1e-10 {
            return f64::NAN;
        }

        let autocorr = numerator / denominator;
        
        // Half-life = -ln(2) / ln(autocorr)
        if autocorr <= 0.0 || autocorr >= 1.0 {
            return f64::NAN;
        }

        -2.0_f64.ln() / autocorr.ln()
    }

    /// Get current premium in bps
    #[inline]
    pub fn premium_bps(&self) -> f64 {
        self.premium_bps
    }

    /// Get fair value
    #[inline]
    pub fn fair_value(&self) -> f64 {
        self.fair_value
    }

    /// Get last update timestamp
    #[inline]
    pub fn last_update_us(&self) -> u64 {
        self.last_update_us.load(Ordering::Relaxed)
    }
}

/// Direction of arbitrage trade
#[derive(Clone, Debug, PartialEq)]
pub enum ArbitrageDirection {
    /// Sell index, buy underlying basket
    SellIndexBuyBasket,
    /// Buy index, sell underlying basket
    BuyIndexSellBasket,
}

/// Side of a trade
#[derive(Clone, Debug, PartialEq)]
pub enum Side {
    Buy,
    Sell,
}

/// Trade for a single constituent
#[derive(Clone, Debug)]
pub struct ConstituentTrade {
    pub symbol: String,
    pub side: Side,
    pub quantity: f64,
    pub notional: f64,
}

/// Complete arbitrage trade package
#[derive(Clone, Debug)]
pub struct ArbitrageTrade {
    pub direction: ArbitrageDirection,
    pub index_side: Side,
    pub index_quantity: f64,
    pub index_price: f64,
    pub basket_trades: Vec<ConstituentTrade>,
    pub expected_profit_bps: f64,
}

impl ArbitrageTrade {
    fn empty() -> Self {
        Self {
            direction: ArbitrageDirection::SellIndexBuyBasket,
            index_side: Side::Sell,
            index_quantity: 0.0,
            index_price: 0.0,
            basket_trades: Vec::new(),
            expected_profit_bps: 0.0,
        }
    }
}

/// Multi-ETF arbitrage manager
pub struct EtfArbitrageManager {
    arbitrageurs: HashMap<String, EtfArbitrageur>,
    /// Global transaction cost estimate
    default_transaction_cost_bps: f64,
    /// Global minimum threshold
    default_min_threshold_bps: f64,
}

impl EtfArbitrageManager {
    pub fn new(default_transaction_cost_bps: f64, default_min_threshold_bps: f64) -> Self {
        Self {
            arbitrageurs: HashMap::new(),
            default_transaction_cost_bps,
            default_min_threshold_bps,
        }
    }

    pub fn add_etf(
        &mut self,
        index_symbol: &str,
        constituents: Vec<BasketConstituent>,
    ) {
        let arb = EtfArbitrageur::new(
            index_symbol,
            constituents,
            self.default_transaction_cost_bps,
            self.default_min_threshold_bps,
        );
        self.arbitrageurs.insert(index_symbol.to_string(), arb);
    }

    pub fn update_prices(
        &mut self,
        index_symbol: &str,
        index_price: f64,
        constituent_prices: &HashMap<String, f64>,
        timestamp_us: u64,
    ) {
        if let Some(arb) = self.arbitrageurs.get_mut(index_symbol) {
            arb.update_all_prices(index_price, constituent_prices, timestamp_us);
        }
    }

    /// Get all current arbitrage opportunities
    pub fn get_opportunities(&self) -> Vec<(&str, &EtfArbitrageur)> {
        self.arbitrageurs
            .iter()
            .filter(|(_, arb)| arb.has_opportunity())
            .map(|(symbol, arb)| (symbol.as_str(), arb))
            .collect()
    }

    /// Get the best arbitrage opportunity by expected profit
    pub fn get_best_opportunity(&self) -> Option<(&str, &EtfArbitrageur)> {
        self.arbitrageurs
            .iter()
            .filter(|(_, arb)| arb.has_opportunity())
            .max_by(|a, b| {
                a.1.expected_profit_bps()
                    .partial_cmp(&b.1.expected_profit_bps())
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(symbol, arb)| (symbol.as_str(), arb))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_arbitrage_detection() {
        let constituents = vec![
            BasketConstituent {
                symbol: "BTC".to_string(),
                weight: 0.6,
                price: 50000.0,
                quantity: 0.0,
            },
            BasketConstituent {
                symbol: "ETH".to_string(),
                weight: 0.4,
                price: 3000.0,
                quantity: 0.0,
            },
        ];

        let mut arb = EtfArbitrageur::new("CRYPTO_INDEX", constituents, 10.0, 50.0);
        
        // Initial fair value: 0.6 * 50000 + 0.4 * 3000 = 31200
        let mut prices = HashMap::new();
        prices.insert("BTC".to_string(), 50000.0);
        prices.insert("ETH".to_string(), 3000.0);
        
        arb.update_all_prices(31200.0, &prices, 1000);
        
        assert!((arb.fair_value() - 31200.0).abs() < 0.01);
        assert!(arb.premium_bps().abs() < 1.0);
        assert!(!arb.has_opportunity());
    }

    #[test]
    fn test_premium_detection() {
        let constituents = vec![
            BasketConstituent {
                symbol: "BTC".to_string(),
                weight: 0.5,
                price: 40000.0,
                quantity: 0.0,
            },
            BasketConstituent {
                symbol: "ETH".to_string(),
                weight: 0.5,
                price: 2000.0,
                quantity: 0.0,
            },
        ];

        let mut arb = EtfArbitrageur::new("TEST_INDEX", constituents, 5.0, 20.0);
        
        let mut prices = HashMap::new();
        prices.insert("BTC".to_string(), 40000.0);
        prices.insert("ETH".to_string(), 2000.0);
        
        // Fair value = 21000, set index at 21100 (premium)
        arb.update_all_prices(21100.0, &prices, 1000);
        
        // Premium should be ~47.6 bps
        assert!(arb.premium_bps() > 40.0);
        assert!(arb.has_opportunity());
        
        let direction = arb.get_arbitrage_direction();
        assert_eq!(direction, Some(ArbitrageDirection::SellIndexBuyBasket));
    }
}
