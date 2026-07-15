//! Smart Order Routing engine that evaluates multiple venues for best execution.
//! Factors in real-time liquidity depth, fees, and latency to find optimal routing.

use std::sync::atomic::{AtomicF64, AtomicBool, AtomicU64, Ordering};
use std::collections::HashMap;

/// Venue representation for order routing
#[derive(Debug, Clone)]
pub struct TradingVenue {
    pub venue_id: String,
    pub venue_name: String,
    pub maker_fee_bps: f64,
    pub taker_fee_bps: f64,
    pub latency_ms: f64,
    pub reliability_score: f64, // 0.0 to 1.0
}

/// Liquidity level at a venue
#[derive(Debug, Clone)]
pub struct VenueLiquidity {
    pub venue_id: String,
    pub bid_price: f64,
    pub bid_size: f64,
    pub ask_price: f64,
    pub ask_size: f64,
    pub timestamp_ns: u64,
}

/// Route decision result
#[derive(Debug, Clone)]
pub struct RouteDecision {
    pub primary_venue: String,
    pub split_routes: Vec<RouteSplit>,
    pub expected_fill_price: f64,
    pub total_fees_bps: f64,
    pub expected_slippage_bps: f64,
    pub confidence_score: f64,
    pub routing_reason: RoutingReason,
}

#[derive(Debug, Clone)]
pub struct RouteSplit {
    pub venue_id: String,
    pub quantity: f64,
    pub percentage: f64,
    pub expected_price: f64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RoutingReason {
    BestPrice,
    LowestFees,
    HighestLiquidity,
    LatencyOptimized,
    SplitForExecution,
    FeeRebateCapture,
}

/// Smart Order Router main engine
pub struct SorEngine {
    /// Available venues
    venues: HashMap<String, TradingVenue>,
    /// Current liquidity by venue
    liquidity: HashMap<String, VenueLiquidity>,
    /// Default route (for fallback)
    default_venue: String,
    /// Enable smart splitting
    enable_splitting: AtomicBool,
    /// Minimum quantity for split
    min_split_quantity: f64,
    /// Maximum venues per order
    max_venues_per_order: usize,
    /// Fee tier discounts (volume-based)
    fee_tier_discount: AtomicF64,
}

impl SorEngine {
    /// Create new SOR engine
    pub fn new(default_venue: &str) -> Self {
        let mut venues = HashMap::new();
        
        // Add default Binance venues
        venues.insert(
            "BINANCE_SPOT".to_string(),
            TradingVenue {
                venue_id: "BINANCE_SPOT".to_string(),
                venue_name: "Binance Spot".to_string(),
                maker_fee_bps: 10.0,
                taker_fee_bps: 10.0,
                latency_ms: 15.0,
                reliability_score: 0.99,
            },
        );
        
        venues.insert(
            "BINANCE_FUTURES".to_string(),
            TradingVenue {
                venue_id: "BINANCE_FUTURES".to_string(),
                venue_name: "Binance Perpetual".to_string(),
                maker_fee_bps: 2.0,
                taker_fee_bps: 4.0,
                latency_ms: 12.0,
                reliability_score: 0.99,
            },
        );
        
        Self {
            venues,
            liquidity: HashMap::new(),
            default_venue: default_venue.to_string(),
            enable_splitting: AtomicBool::new(true),
            min_split_quantity: 1.0, // BTC equivalent
            max_venues_per_order: 3,
            fee_tier_discount: AtomicF64::new(0.0),
        }
    }

    /// Register a new venue
    pub fn add_venue(&mut self, venue: TradingVenue) {
        self.venues.insert(venue.venue_id.clone(), venue);
    }

    /// Update liquidity for a venue
    #[inline(always)]
    pub fn update_liquidity(&mut self, liquidity: VenueLiquidity) {
        self.liquidity.insert(liquidity.venue_id.clone(), liquidity);
    }

    /// Find best venue for a buy order
    pub fn find_best_buy_route(
        &self,
        quantity: f64,
        side: OrderSide,
    ) -> RouteDecision {
        if self.liquidity.is_empty() {
            return self.create_single_venue_route(&self.default_venue, quantity, side);
        }
        
        let mut best_venue = self.default_venue.clone();
        let mut best_score = f64::MIN;
        let mut best_price = f64::MAX;
        
        for (venue_id, liq) in &self.liquidity {
            let price = if side == OrderSide::Buy {
                liq.ask_price
            } else {
                liq.bid_price
            };
            
            let available = if side == OrderSide::Buy {
                liq.ask_size
            } else {
                liq.bid_size
            };
            
            // Skip if insufficient liquidity
            if available < quantity * 0.1 {
                continue;
            }
            
            // Get venue fees
            let venue = match self.venues.get(venue_id) {
                Some(v) => v,
                None => continue,
            };
            
            // Calculate effective cost (price + fees - rebates)
            let fee_rate = if side == OrderSide::Buy {
                venue.taker_fee_bps
            } else {
                venue.maker_fee_bps // Assume maker for sells
            };
            
            let discount = self.fee_tier_discount.load(Ordering::Relaxed);
            let effective_fee = fee_rate * (1.0 - discount);
            
            // Score: lower price is better, adjust for fees
            let score = -price - (effective_fee / 10000.0) * price;
            
            if score > best_score {
                best_score = score;
                best_venue = venue_id.clone();
                best_price = price;
            }
        }
        
        // Check if splitting would be beneficial
        if self.enable_splitting.load(Ordering::Relaxed) 
            && quantity > self.min_split_quantity 
        {
            return self.calculate_optimal_split(quantity, side);
        }
        
        self.create_single_venue_route(&best_venue, quantity, side)
    }

    /// Calculate optimal order split across venues
    fn calculate_optimal_split(&self, quantity: f64, side: OrderSide) -> RouteDecision {
        let mut splits: Vec<RouteSplit> = Vec::new();
        let mut remaining_qty = quantity;
        let mut total_notional = 0.0;
        let mut total_fees_bps = 0.0;
        
        // Sort venues by effective price
        let mut venue_scores: Vec<(String, f64, f64)> = Vec::new();
        
        for (venue_id, liq) in &self.liquidity {
            let price = if side == OrderSide::Buy {
                liq.ask_price
            } else {
                liq.bid_price
            };
            
            let available = if side == OrderSide::Buy {
                liq.ask_size
            } else {
                liq.bid_size
            };
            
            let venue = match self.venues.get(venue_id) {
                Some(v) => v,
                None => continue,
            };
            
            let fee_rate = if side == OrderSide::Buy {
                venue.taker_fee_bps
            } else {
                venue.maker_fee_bps
            };
            
            let effective_price = price * (1.0 + fee_rate / 10000.0);
            venue_scores.push((venue_id.clone(), effective_price, available));
        }
        
        // Sort by effective price
        venue_scores.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        
        // Allocate quantity to best venues
        let mut venues_used = 0;
        for (venue_id, eff_price, available) in venue_scores.iter().take(self.max_venues_per_order) {
            if remaining_qty <= 0.0 || venues_used >= self.max_venues_per_order {
                break;
            }
            
            let alloc_qty = remaining_qty.min(*available);
            if alloc_qty < 0.001 {
                continue;
            }
            
            let venue = self.venues.get(venue_id).unwrap();
            let fee_rate = if side == OrderSide::Buy {
                venue.taker_fee_bps
            } else {
                venue.maker_fee_bps
            };
            
            let base_price = eff_price / (1.0 + fee_rate / 10000.0);
            
            splits.push(RouteSplit {
                venue_id: venue_id.clone(),
                quantity: alloc_qty,
                percentage: alloc_qty / quantity * 100.0,
                expected_price: base_price,
            });
            
            total_notional += alloc_qty * base_price;
            total_fees_bps += fee_rate * (alloc_qty / quantity);
            remaining_qty -= alloc_qty;
            venues_used += 1;
        }
        
        // Handle any remaining quantity
        if remaining_qty > 0.001 && !splits.is_empty() {
            // Add to largest allocation
            splits[0].quantity += remaining_qty;
            splits[0].percentage = splits[0].quantity / quantity * 100.0;
        }
        
        let expected_fill_price = if !splits.is_empty() {
            total_notional / quantity
        } else {
            0.0
        };
        
        RouteDecision {
            primary_venue: splits.first().map(|s| s.venue_id.clone()).unwrap_or(self.default_venue.clone()),
            split_routes: splits,
            expected_fill_price,
            total_fees_bps,
            expected_slippage_bps: 0.0, // Would calculate based on depth
            confidence_score: 0.85,
            routing_reason: RoutingReason::SplitForExecution,
        }
    }

    /// Create single venue route
    fn create_single_venue_route(&self, venue_id: &str, quantity: f64, side: OrderSide) -> RouteDecision {
        let liq = self.liquidity.get(venue_id);
        let venue = self.venues.get(venue_id);
        
        let expected_price = match (liq, side) {
            (Some(l), OrderSide::Buy) => l.ask_price,
            (Some(l), OrderSide::Sell) => l.bid_price,
            _ => 0.0,
        };
        
        let fees_bps = match (venue, side) {
            (Some(v), OrderSide::Buy) => v.taker_fee_bps,
            (Some(v), OrderSide::Sell) => v.maker_fee_bps,
            _ => 10.0,
        };
        
        RouteDecision {
            primary_venue: venue_id.to_string(),
            split_routes: vec![RouteSplit {
                venue_id: venue_id.to_string(),
                quantity,
                percentage: 100.0,
                expected_price,
            }],
            expected_fill_price: expected_price,
            total_fees_bps: fees_bps,
            expected_slippage_bps: 0.0,
            confidence_score: 0.9,
            routing_reason: RoutingReason::BestPrice,
        }
    }

    /// Set fee tier discount based on volume
    #[inline(always)]
    pub fn set_fee_tier_discount(&self, discount_pct: f64) {
        self.fee_tier_discount.store(discount_pct.clamp(0.0, 0.5), Ordering::Relaxed);
    }

    /// Enable/disable order splitting
    #[inline(always)]
    pub fn set_splitting_enabled(&self, enabled: bool) {
        self.enable_splitting.store(enabled, Ordering::Relaxed);
    }

    /// Get all available venues
    pub fn get_venues(&self) -> Vec<&TradingVenue> {
        self.venues.values().collect()
    }
}

/// Order side
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum OrderSide {
    Buy,
    Sell,
}

/// Cross-exchange arbitrage opportunity detector
pub struct ArbitrageDetector {
    /// Price threshold for arbitrage (bps)
    min_arb_threshold_bps: f64,
    /// Last detected opportunities
    opportunities: Vec<ArbitrageOpportunity>,
}

#[derive(Debug, Clone)]
pub struct ArbitrageOpportunity {
    pub buy_venue: String,
    pub sell_venue: String,
    pub buy_price: f64,
    pub sell_price: f64,
    pub spread_bps: f64,
    pub max_quantity: f64,
    pub estimated_profit: f64,
}

impl ArbitrageDetector {
    pub fn new(min_threshold_bps: f64) -> Self {
        Self {
            min_arb_threshold_bps: min_threshold_bps,
            opportunities: Vec::new(),
        }
    }

    /// Scan for arbitrage opportunities
    pub fn scan_opportunities(&mut self, liquidity: &HashMap<String, VenueLiquidity>) -> Vec<ArbitrageOpportunity> {
        self.opportunities.clear();
        
        let venues: Vec<_> = liquidity.keys().collect();
        
        for i in 0..venues.len() {
            for j in (i + 1)..venues.len() {
                let venue_a = venues[i];
                let venue_b = venues[j];
                
                let liq_a = &liquidity[venue_a];
                let liq_b = &liquidity[venue_b];
                
                // Check A buy, B sell
                if let Some(opp) = self.check_arb_pair(venue_a, venue_b, liq_a, liq_b) {
                    self.opportunities.push(opp);
                }
                
                // Check B buy, A sell
                if let Some(opp) = self.check_arb_pair(venue_b, venue_a, liq_b, liq_a) {
                    self.opportunities.push(opp);
                }
            }
        }
        
        self.opportunities.clone()
    }

    fn check_arb_pair(
        &self,
        buy_venue: &str,
        sell_venue: &str,
        buy_liq: &VenueLiquidity,
        sell_liq: &VenueLiquidity,
    ) -> Option<ArbitrageOpportunity> {
        let buy_price = buy_liq.ask_price;
        let sell_price = sell_liq.bid_price;
        
        if buy_price <= 0.0 || sell_price <= 0.0 {
            return None;
        }
        
        let spread_bps = (sell_price - buy_price) / buy_price * 10000.0;
        
        if spread_bps < self.min_arb_threshold_bps {
            return None;
        }
        
        let max_qty = buy_liq.ask_size.min(sell_liq.bid_size);
        let estimated_profit = (sell_price - buy_price) * max_qty;
        
        Some(ArbitrageOpportunity {
            buy_venue: buy_venue.to_string(),
            sell_venue: sell_venue.to_string(),
            buy_price,
            sell_price,
            spread_bps,
            max_quantity: max_qty,
            estimated_profit,
        })
    }

    /// Get best opportunity
    pub fn get_best_opportunity(&self) -> Option<&ArbitrageOpportunity> {
        self.opportunities.iter().max_by(|a, b| a.spread_bps.partial_cmp(&b.spread_bps).unwrap_or(std::cmp::Ordering::Equal))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sor_best_route() {
        let mut sor = SorEngine::new("BINANCE_SPOT");
        
        sor.update_liquidity(VenueLiquidity {
            venue_id: "BINANCE_SPOT".to_string(),
            bid_price: 49999.0,
            bid_size: 10.0,
            ask_price: 50001.0,
            ask_size: 10.0,
            timestamp_ns: 0,
        });
        
        sor.update_liquidity(VenueLiquidity {
            venue_id: "BINANCE_FUTURES".to_string(),
            bid_price: 50000.0,
            bid_size: 20.0,
            ask_price: 50000.5,
            ask_size: 20.0,
            timestamp_ns: 0,
        });
        
        let route = sor.find_best_buy_route(5.0, OrderSide::Buy);
        
        assert!(!route.split_routes.is_empty());
        assert!(route.expected_fill_price > 0.0);
    }

    #[test]
    fn test_arbitrage_detection() {
        let mut detector = ArbitrageDetector::new(5.0); // 5 bps minimum
        
        let mut liquidity = HashMap::new();
        liquidity.insert(
            "VENUE_A".to_string(),
            VenueLiquidity {
                venue_id: "VENUE_A".to_string(),
                bid_price: 50000.0,
                bid_size: 10.0,
                ask_price: 50010.0,
                ask_size: 10.0,
                timestamp_ns: 0,
            },
        );
        liquidity.insert(
            "VENUE_B".to_string(),
            VenueLiquidity {
                venue_id: "VENUE_B".to_string(),
                bid_price: 50020.0,
                bid_size: 10.0,
                ask_price: 50030.0,
                ask_size: 10.0,
                timestamp_ns: 0,
            },
        );
        
        let opportunities = detector.scan_opportunities(&liquidity);
        
        // Should find arb: buy A at 50010, sell B at 50020
        assert!(!opportunities.is_empty());
    }
}
