//! Unified Order Router
//! Normalizes order parameters across Binance, Bybit, and OKX
//! Translates generic orders to venue-specific APIs in nanoseconds

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;

/// Unified order side
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum UnifiedSide {
    Buy,
    Sell,
}

/// Unified order type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum UnifiedOrderType {
    Market,
    Limit,
    StopLimit,
    StopMarket,
}

/// Unified time in force
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum UnifiedTimeInForce {
    GTC, // Good Till Cancel
    IOC, // Immediate or Cancel
    FOK, // Fill or Kill
    GTD, // Good Till Date
    Day,
}

/// Exchange venue identifier
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Venue {
    Binance,
    Bybit,
    OKX,
}

impl Venue {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Binance => "binance",
            Self::Bybit => "bybit",
            Self::OKX => "okx",
        }
    }
}

/// Unified order request
#[derive(Debug, Clone)]
pub struct UnifiedOrder {
    pub client_order_id: String,
    pub symbol: String,
    pub venue: Venue,
    pub side: UnifiedSide,
    pub order_type: UnifiedOrderType,
    pub quantity: f64,
    pub price: Option<f64>,
    pub stop_price: Option<f64>,
    pub time_in_force: UnifiedTimeInForce,
    pub reduce_only: bool,
    pub timestamp: Instant,
}

/// Unified order response
#[derive(Debug, Clone)]
pub struct UnifiedOrderResponse {
    pub client_order_id: String,
    pub venue_order_id: String,
    pub venue: Venue,
    pub status: UnifiedOrderStatus,
    pub filled_quantity: f64,
    pub average_price: Option<f64>,
    pub timestamp: Instant,
}

/// Unified order status
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnifiedOrderStatus {
    PendingNew,
    New,
    PartiallyFilled,
    Filled,
    Rejected,
    Canceled,
    Expired,
}

/// Venue-specific order translation result
#[derive(Debug, Clone)]
pub enum VenueOrder {
    Binance(BinanceOrder),
    Bybit(BybitOrder),
    OKX(OKXOrder),
}

#[derive(Debug, Clone)]
pub struct BinanceOrder {
    pub symbol: String,
    pub side: String,
    pub order_type: String,
    pub time_in_force: Option<String>,
    pub quantity: String,
    pub price: Option<String>,
    pub new_client_order_id: String,
    pub reduce_only: Option<bool>,
}

#[derive(Debug, Clone)]
pub struct BybitOrder {
    pub category: String,
    pub symbol: String,
    pub side: String,
    pub order_type: String,
    pub qty: String,
    pub price: Option<String>,
    pub time_in_force: String,
    pub order_id: Option<String>,
    pub reduce_only: Option<bool>,
}

#[derive(Debug, Clone)]
pub struct OKXOrder {
    pub inst_id: String,
    pub td_mode: String,
    pub side: String,
    pub ord_type: String,
    pub sz: String,
    pub px: Option<String>,
    pub cl_ord_id: Option<String>,
}

/// Unified router configuration
#[derive(Clone)]
pub struct RouterConfig {
    pub default_venue: Venue,
    pub smart_routing_enabled: bool,
    pub latency_threshold_us: u64,
    pub fee_optimization_enabled: bool,
}

impl Default for RouterConfig {
    fn default() -> Self {
        Self {
            default_venue: Venue::Binance,
            smart_routing_enabled: true,
            latency_threshold_us: 100,
            fee_optimization_enabled: false,
        }
    }
}

/// Venue statistics for smart routing
#[derive(Debug, Clone, Default)]
pub struct VenueStats {
    pub latency_us: u64,
    pub success_rate: f64,
    pub maker_fee: f64,
    pub taker_fee: f64,
    pub last_update: Instant,
}

/// Unified Order Router
pub struct UnifiedRouter {
    config: RouterConfig,
    venue_stats: Arc<RwLock<HashMap<Venue, VenueStats>>>,
    symbol_mappings: Arc<RwLock<HashMap<String, HashMap<Venue, String>>>>,
    sequence_counter: u64,
}

impl UnifiedRouter {
    pub fn new(config: RouterConfig) -> Self {
        let mut venue_stats = HashMap::new();
        
        // Initialize with default stats
        venue_stats.insert(Venue::Binance, VenueStats::default());
        venue_stats.insert(Venue::Bybit, VenueStats::default());
        venue_stats.insert(Venue::OKX, VenueStats::default());
        
        Self {
            config,
            venue_stats: Arc::new(RwLock::new(venue_stats)),
            symbol_mappings: Arc::new(RwLock::new(HashMap::new())),
            sequence_counter: 0,
        }
    }

    /// Generate unique client order ID
    #[inline]
    pub fn generate_client_order_id(&mut self, prefix: &str) -> String {
        self.sequence_counter += 1;
        format!("{}{}", prefix, self.sequence_counter)
    }

    /// Register symbol mapping for a venue
    pub async fn register_symbol_mapping(
        &self,
        base_symbol: &str,
        venue: Venue,
        venue_symbol: &str,
    ) {
        let mut mappings = self.symbol_mappings.write().await;
        mappings
            .entry(base_symbol.to_string())
            .or_insert_with(HashMap::new)
            .insert(venue, venue_symbol.to_string());
    }

    /// Get venue-specific symbol
    pub async fn get_venue_symbol(&self, base_symbol: &str, venue: Venue) -> String {
        let mappings = self.symbol_mappings.read().await;
        
        if let Some(venue_map) = mappings.get(base_symbol) {
            if let Some(symbol) = venue_map.get(&venue) {
                return symbol.clone();
            }
        }
        
        // Default mapping if not registered
        match venue {
            Venue::Binance => base_symbol.to_uppercase(),
            Venue::Bybit => base_symbol.to_uppercase(),
            Venue::OKX => base_symbol.replace('/', "-"),
        }
    }

    /// Translate unified order to venue-specific format
    pub async fn translate_order(&self, order: &UnifiedOrder) -> VenueOrder {
        let venue_symbol = self.get_venue_symbol(&order.symbol, order.venue).await;
        
        match order.venue {
            Venue::Binance => self.translate_to_binance(order, &venue_symbol),
            Venue::Bybit => self.translate_to_bybit(order, &venue_symbol),
            Venue::OKX => self.translate_to_okx(order, &venue_symbol),
        }
    }

    /// Translate to Binance format
    fn translate_to_binance(&self, order: &UnifiedOrder, symbol: &str) -> VenueOrder {
        let (order_type, time_in_force) = match order.order_type {
            UnifiedOrderType::Market => ("MARKET".to_string(), None),
            UnifiedOrderType::Limit => (
                "LIMIT".to_string(),
                Some(match order.time_in_force {
                    UnifiedTimeInForce::GTC => "GTC".to_string(),
                    UnifiedTimeInForce::IOC => "IOC".to_string(),
                    UnifiedTimeInForce::FOK => "FOK".to_string(),
                    UnifiedTimeInForce::Day => "DAY".to_string(),
                    UnifiedTimeInForce::GTD => "GTC".to_string(),
                }),
            ),
            UnifiedOrderType::StopLimit => ("STOP_LOSS_LIMIT".to_string(), Some("GTC".to_string())),
            UnifiedOrderType::StopMarket => ("STOP_LOSS".to_string(), Some("GTC".to_string())),
        };

        VenueOrder::Binance(BinanceOrder {
            symbol: symbol.to_string(),
            side: match order.side {
                UnifiedSide::Buy => "BUY".to_string(),
                UnifiedSide::Sell => "SELL".to_string(),
            },
            order_type,
            time_in_force,
            quantity: order.quantity.to_string(),
            price: order.price.map(|p| p.to_string()),
            new_client_order_id: order.client_order_id.clone(),
            reduce_only: if order.reduce_only { Some(true) } else { None },
        })
    }

    /// Translate to Bybit format
    fn translate_to_bybit(&self, order: &UnifiedOrder, symbol: &str) -> VenueOrder {
        let category = if symbol.contains("USDT") && !symbol.contains("PERP") {
            "linear"
        } else if symbol.contains("2") {
            "option"
        } else {
            "spot"
        };

        VenueOrder::Bybit(BybitOrder {
            category: category.to_string(),
            symbol: symbol.to_string(),
            side: match order.side {
                UnifiedSide::Buy => "Buy".to_string(),
                UnifiedSide::Sell => "Sell".to_string(),
            },
            order_type: match order.order_type {
                UnifiedOrderType::Market => "Market".to_string(),
                UnifiedOrderType::Limit => "Limit".to_string(),
                _ => "Limit".to_string(),
            },
            qty: order.quantity.to_string(),
            price: order.price.map(|p| p.to_string()),
            time_in_force: match order.time_in_force {
                UnifiedTimeInForce::GTC => "GoodTillCancel".to_string(),
                UnifiedTimeInForce::IOC => "ImmediateOrCancel".to_string(),
                UnifiedTimeInForce::FOK => "FillOrKill".to_string(),
                _ => "GoodTillCancel".to_string(),
            },
            order_id: Some(order.client_order_id.clone()),
            reduce_only: if order.reduce_only { Some(true) } else { None },
        })
    }

    /// Translate to OKX format
    fn translate_to_okx(&self, order: &UnifiedOrder, symbol: &str) -> VenueOrder {
        let td_mode = if order.reduce_only {
            "isolated_margin"
        } else {
            "cash"
        };

        VenueOrder::OKX(OKXOrder {
            inst_id: symbol.to_string(),
            td_mode: td_mode.to_string(),
            side: match order.side {
                UnifiedSide::Buy => "buy".to_string(),
                UnifiedSide::Sell => "sell".to_string(),
            },
            ord_type: match order.order_type {
                UnifiedOrderType::Market => "market".to_string(),
                UnifiedOrderType::Limit => "limit".to_string(),
                UnifiedOrderType::StopLimit => "limit".to_string(),
                UnifiedOrderType::StopMarket => "market".to_string(),
            },
            sz: order.quantity.to_string(),
            px: order.price.map(|p| p.to_string()),
            cl_ord_id: Some(order.client_order_id.clone()),
        })
    }

    /// Select best venue using smart routing
    pub async fn select_best_venue(&self, symbol: &str) -> Venue {
        if !self.config.smart_routing_enabled {
            return self.config.default_venue;
        }

        let stats = self.venue_stats.read().await;
        
        let mut best_venue = self.config.default_venue;
        let mut best_score = f64::MAX;

        for (venue, venue_stats) in stats.iter() {
            // Score based on latency and fees
            let score = venue_stats.latency_us as f64 
                + (if self.config.fee_optimization_enabled {
                    venue_stats.taker_fee * 1000.0
                } else {
                    0.0
                });

            if score < best_score {
                best_score = score;
                best_venue = *venue;
            }
        }

        best_venue
    }

    /// Update venue statistics
    pub async fn update_venue_stats(
        &self,
        venue: Venue,
        latency_us: u64,
        success: bool,
    ) {
        let mut stats = self.venue_stats.write().await;
        
        if let Some(venue_stats) = stats.get_mut(&venue) {
            // Exponential moving average for latency
            venue_stats.latency_us = (venue_stats.latency_us * 9 + latency_us) / 10;
            
            // Update success rate
            if success {
                venue_stats.success_rate = (venue_stats.success_rate * 9.0 + 1.0) / 10.0;
            } else {
                venue_stats.success_rate = (venue_stats.success_rate * 9.0) / 10.0;
            }
            
            venue_stats.last_update = Instant::now();
        }
    }

    /// Create unified order from parameters
    pub fn create_order(
        &mut self,
        symbol: &str,
        side: UnifiedSide,
        order_type: UnifiedOrderType,
        quantity: f64,
        price: Option<f64>,
        venue: Option<Venue>,
    ) -> UnifiedOrder {
        let selected_venue = venue.unwrap_or_else(|| {
            if self.config.smart_routing_enabled {
                Venue::Binance // Would call select_best_venue in async context
            } else {
                self.config.default_venue
            }
        });

        UnifiedOrder {
            client_order_id: self.generate_client_order_id("ORD"),
            symbol: symbol.to_string(),
            venue: selected_venue,
            side,
            order_type,
            quantity,
            price,
            stop_price: None,
            time_in_force: UnifiedTimeInForce::GTC,
            reduce_only: false,
            timestamp: Instant::now(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_venue_translation() {
        let router = UnifiedRouter::new(RouterConfig::default());
        
        let order = UnifiedOrder {
            client_order_id: "TEST123".to_string(),
            symbol: "BTCUSDT".to_string(),
            venue: Venue::Binance,
            side: UnifiedSide::Buy,
            order_type: UnifiedOrderType::Limit,
            quantity: 0.001,
            price: Some(45000.0),
            stop_price: None,
            time_in_force: UnifiedTimeInForce::GTC,
            reduce_only: false,
            timestamp: Instant::now(),
        };

        let venue_order = router.translate_order(&order).await;
        
        match venue_order {
            VenueOrder::Binance(bin_order) => {
                assert_eq!(bin_order.side, "BUY");
                assert_eq!(bin_order.order_type, "LIMIT");
            }
            _ => panic!("Expected Binance order"),
        }
    }

    #[test]
    fn test_venue_as_str() {
        assert_eq!(Venue::Binance.as_str(), "binance");
        assert_eq!(Venue::Bybit.as_str(), "bybit");
        assert_eq!(Venue::OKX.as_str(), "okx");
    }
}
