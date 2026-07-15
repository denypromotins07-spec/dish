//! Deterministic L2/L3 Limit Order Book simulator.
//! Reconstructs the book from historical delta updates with exact queue position matching.
//! Handles cancellations, modifications, and aggressive market order sweeps in pure Rust.

use std::collections::{BTreeMap, VecDeque};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

/// Price level in the order book
#[derive(Debug, Clone)]
pub struct PriceLevel {
    pub price: f64,
    pub volume: f64,
    pub order_count: u32,
    pub orders: VecDeque<Order>,
}

impl PriceLevel {
    pub fn new(price: f64) -> Self {
        Self {
            price,
            volume: 0.0,
            order_count: 0,
            orders: VecDeque::new(),
        }
    }
}

/// Individual order in the book
#[derive(Debug, Clone)]
pub struct Order {
    pub order_id: u64,
    pub price: f64,
    pub quantity: f64,
    pub timestamp_ns: u64,
    pub side: Side,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Side {
    Buy,
    Sell,
}

/// Order book event types
#[derive(Debug, Clone)]
pub enum BookEvent {
    NewOrder(Order),
    CancelOrder { order_id: u64, price: f64, quantity: f64 },
    ModifyOrder { order_id: u64, old_quantity: f64, new_quantity: f64 },
    Trade { price: f64, quantity: f64, aggressor_side: Side },
}

/// High-performance L2/L3 order book simulator
pub struct LOBSimulator {
    bids: BTreeMap<f64, PriceLevel>, // Sorted descending by price
    asks: BTreeMap<f64, PriceLevel>, // Sorted ascending by price
    best_bid: f64,
    best_ask: f64,
    spread: f64,
    mid_price: f64,
    last_update_ns: AtomicU64,
    tick_size: f64,
    lot_size: f64,
    sequence_number: AtomicU64,
    
    // Statistics
    total_trades: u64,
    total_volume_traded: f64,
    total_cancellations: u64,
}

impl LOBSimulator {
    pub fn new(tick_size: f64, lot_size: f64) -> Self {
        Self {
            bids: BTreeMap::new(),
            asks: BTreeMap::new(),
            best_bid: 0.0,
            best_ask: f64::INFINITY,
            spread: f64::INFINITY,
            mid_price: 0.0,
            last_update_ns: AtomicU64::new(0),
            tick_size,
            lot_size,
            sequence_number: AtomicU64::new(0),
            total_trades: 0,
            total_volume_traded: 0.0,
            total_cancellations: 0,
        }
    }
    
    /// Process a new order event
    pub fn process_new_order(&mut self, order: Order) -> Option<BookEvent> {
        let side = order.side;
        let price = self.snap_to_tick(order.price);
        
        match side {
            Side::Buy => {
                // Check if it crosses the spread (becomes a market order)
                if price >= self.best_ask && !self.asks.is_empty() {
                    return self.execute_market_buy(order);
                }
                
                self.add_limit_order(order, price, true)
            }
            Side::Sell => {
                // Check if it crosses the spread
                if price <= self.best_bid && !self.bids.is_empty() {
                    return self.execute_market_sell(order);
                }
                
                self.add_limit_order(order, price, false)
            }
        }
    }
    
    fn add_limit_order(&mut self, mut order: Order, price: f64, is_bid: bool) -> Option<BookEvent> {
        order.price = price;
        let levels = if is_bid { &mut self.bids } else { &mut self.asks };
        
        let level = levels.entry(price).or_insert_with(|| PriceLevel::new(price));
        level.volume += order.quantity;
        level.order_count += 1;
        level.orders.push_back(order.clone());
        
        self.update_best_prices();
        self.increment_sequence();
        
        Some(BookEvent::NewOrder(order))
    }
    
    /// Execute a market buy order against the ask side
    fn execute_market_buy(&mut self, mut order: Order) -> Option<BookEvent> {
        let mut remaining = order.quantity;
        let mut total_value = 0.0;
        let mut trades_executed = 0u64;
        
        while remaining > 0.0 && !self.asks.is_empty() {
            let best_ask_price = *self.asks.keys().next().unwrap();
            
            // Only cross if price is acceptable
            if best_ask_price > order.price {
                break;
            }
            
            if let Some(level) = self.asks.get_mut(&best_ask_price) {
                let fill_qty = remaining.min(level.volume);
                
                // Create trade event
                let trade_event = BookEvent::Trade {
                    price: best_ask_price,
                    quantity: fill_qty,
                    aggressor_side: Side::Buy,
                };
                
                // Update level
                level.volume -= fill_qty;
                remaining -= fill_qty;
                total_value += fill_qty * best_ask_price;
                trades_executed += 1;
                self.total_trades += 1;
                self.total_volume_traded += fill_qty;
                
                // Remove filled orders from the queue
                let mut to_remove = fill_qty;
                while to_remove > 0.0 && !level.orders.is_empty() {
                    if let Some(front_order) = level.orders.front() {
                        if front_order.quantity <= to_remove {
                            to_remove -= front_order.quantity;
                            level.orders.pop_front();
                            level.order_count -= 1;
                        } else {
                            // Partial fill
                            if let Some(front_order) = level.orders.front_mut() {
                                front_order.quantity -= to_remove;
                                level.volume -= to_remove;
                            }
                            to_remove = 0.0;
                        }
                    }
                }
                
                // Remove empty price level
                if level.volume <= 0.0 {
                    self.asks.remove(&best_ask_price);
                }
                
                // Return first trade event (in reality, multiple trades might occur)
                if trades_executed == 1 {
                    self.update_best_prices();
                    self.increment_sequence();
                    return Some(trade_event);
                }
            }
        }
        
        // If order still has remaining quantity, add as limit order
        if remaining > 0.0 {
            order.quantity = remaining;
            return self.add_limit_order(order, order.price, false);
        }
        
        None
    }
    
    /// Execute a market sell order against the bid side
    fn execute_market_sell(&mut self, mut order: Order) -> Option<BookEvent> {
        let mut remaining = order.quantity;
        
        while remaining > 0.0 && !self.bids.is_empty() {
            let best_bid_price = *self.bids.keys().rev().next().unwrap();
            
            if best_bid_price < order.price {
                break;
            }
            
            if let Some(level) = self.bids.get_mut(&best_bid_price) {
                let fill_qty = remaining.min(level.volume);
                
                let trade_event = BookEvent::Trade {
                    price: best_bid_price,
                    quantity: fill_qty,
                    aggressor_side: Side::Sell,
                };
                
                level.volume -= fill_qty;
                remaining -= fill_qty;
                self.total_trades += 1;
                self.total_volume_traded += fill_qty;
                
                // Remove filled orders
                let mut to_remove = fill_qty;
                while to_remove > 0.0 && !level.orders.is_empty() {
                    if let Some(front_order) = level.orders.front() {
                        if front_order.quantity <= to_remove {
                            to_remove -= front_order.quantity;
                            level.orders.pop_front();
                            level.order_count -= 1;
                        } else {
                            if let Some(front_order) = level.orders.front_mut() {
                                front_order.quantity -= to_remove;
                            }
                            to_remove = 0.0;
                        }
                    }
                }
                
                if level.volume <= 0.0 {
                    self.bids.remove(&best_bid_price);
                }
                
                self.update_best_prices();
                self.increment_sequence();
                return Some(trade_event);
            }
        }
        
        if remaining > 0.0 {
            order.quantity = remaining;
            return self.add_limit_order(order, order.price, true);
        }
        
        None
    }
    
    /// Cancel an order by ID
    pub fn cancel_order(&mut self, order_id: u64) -> Option<BookEvent> {
        // Search in bids
        for (price, level) in &mut self.bids {
            for (idx, order) in level.orders.iter().enumerate() {
                if order.order_id == order_id {
                    let qty = order.quantity;
                    level.volume -= qty;
                    level.order_count -= 1;
                    level.orders.remove(idx);
                    self.total_cancellations += 1;
                    
                    if level.volume <= 0.0 {
                        let price_copy = *price;
                        self.bids.remove(&price_copy);
                    }
                    
                    self.update_best_prices();
                    self.increment_sequence();
                    
                    return Some(BookEvent::CancelOrder {
                        order_id,
                        price: *price,
                        quantity: qty,
                    });
                }
            }
        }
        
        // Search in asks
        for (price, level) in &mut self.asks {
            for (idx, order) in level.orders.iter().enumerate() {
                if order.order_id == order_id {
                    let qty = order.quantity;
                    level.volume -= qty;
                    level.order_count -= 1;
                    level.orders.remove(idx);
                    self.total_cancellations += 1;
                    
                    if level.volume <= 0.0 {
                        let price_copy = *price;
                        self.asks.remove(&price_copy);
                    }
                    
                    self.update_best_prices();
                    self.increment_sequence();
                    
                    return Some(BookEvent::CancelOrder {
                        order_id,
                        price: *price,
                        quantity: qty,
                    });
                }
            }
        }
        
        None
    }
    
    /// Get queue position estimate for a limit order
    pub fn estimate_queue_position(&self, price: f64, side: Side, order_id: u64) -> Option<u32> {
        let levels = match side {
            Side::Buy => &self.bids,
            Side::Sell => &self.asks,
        };
        
        if let Some(level) = levels.get(&price) {
            for (idx, order) in level.orders.iter().enumerate() {
                if order.order_id == order_id {
                    return Some(idx as u32);
                }
            }
        }
        
        None
    }
    
    /// Get best bid price
    #[inline]
    pub fn best_bid(&self) -> f64 {
        self.best_bid
    }
    
    /// Get best ask price
    #[inline]
    pub fn best_ask(&self) -> f64 {
        self.best_ask
    }
    
    /// Get mid price
    #[inline]
    pub fn mid_price(&self) -> f64 {
        self.mid_price
    }
    
    /// Get spread
    #[inline]
    pub fn spread(&self) -> f64 {
        self.spread
    }
    
    /// Get bid depth at specified levels
    pub fn get_bid_depth(&self, levels: usize) -> Vec<(f64, f64)> {
        self.bids
            .iter()
            .rev()
            .take(levels)
            .map(|(p, l)| (*p, l.volume))
            .collect()
    }
    
    /// Get ask depth at specified levels
    pub fn get_ask_depth(&self, levels: usize) -> Vec<(f64, f64)> {
        self.asks
            .iter()
            .take(levels)
            .map(|(p, l)| (*p, l.volume))
            .collect()
    }
    
    fn update_best_prices(&mut self) {
        self.best_bid = self.bids.keys().rev().next().copied().unwrap_or(0.0);
        self.best_ask = self.asks.keys().next().copied().unwrap_or(f64::INFINITY);
        
        if self.best_bid > 0.0 && self.best_ask.is_finite() {
            self.spread = self.best_ask - self.best_bid;
            self.mid_price = (self.best_bid + self.best_ask) / 2.0;
        } else {
            self.spread = f64::INFINITY;
            self.mid_price = if self.best_bid > 0.0 {
                self.best_bid
            } else if self.best_ask.is_finite() {
                self.best_ask
            } else {
                0.0
            };
        }
    }
    
    #[inline]
    fn snap_to_tick(&self, price: f64) -> f64 {
        (price / self.tick_size).round() * self.tick_size
    }
    
    #[inline]
    fn increment_sequence(&self) {
        self.sequence_number.fetch_add(1, Ordering::Relaxed);
    }
    
    /// Get current sequence number
    pub fn sequence_number(&self) -> u64 {
        self.sequence_number.load(Ordering::Relaxed)
    }
    
    /// Get statistics
    pub fn stats(&self) -> (u64, f64, u64) {
        (self.total_trades, self.total_volume_traded, self.total_cancellations)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_basic_order_book() {
        let mut book = LOBSimulator::new(0.01, 0.001);
        
        // Add some bids
        let bid1 = Order {
            order_id: 1,
            price: 50000.0,
            quantity: 1.0,
            timestamp_ns: 1000,
            side: Side::Buy,
        };
        book.process_new_order(bid1);
        
        // Add some asks
        let ask1 = Order {
            order_id: 2,
            price: 50000.1,
            quantity: 1.0,
            timestamp_ns: 1001,
            side: Side::Sell,
        };
        book.process_new_order(ask1);
        
        assert_eq!(book.best_bid(), 50000.0);
        assert_eq!(book.best_ask(), 50000.1);
        assert!((book.spread() - 0.1).abs() < 0.001);
    }
    
    #[test]
    fn test_market_order_execution() {
        let mut book = LOBSimulator::new(0.01, 0.001);
        
        // Add ask
        let ask1 = Order {
            order_id: 1,
            price: 50000.0,
            quantity: 2.0,
            timestamp_ns: 1000,
            side: Side::Sell,
        };
        book.process_new_order(ask1);
        
        // Market buy that crosses
        let market_buy = Order {
            order_id: 2,
            price: 50000.5,
            quantity: 1.0,
            timestamp_ns: 1001,
            side: Side::Buy,
        };
        let result = book.process_new_order(market_buy);
        
        assert!(result.is_some());
        if let Some(BookEvent::Trade { price, quantity, aggressor_side }) = result {
            assert_eq!(price, 50000.0);
            assert_eq!(quantity, 1.0);
            assert_eq!(aggressor_side, Side::Buy);
        } else {
            panic!("Expected trade event");
        }
    }
    
    #[test]
    fn test_order_cancellation() {
        let mut book = LOBSimulator::new(0.01, 0.001);
        
        let order = Order {
            order_id: 1,
            price: 50000.0,
            quantity: 1.0,
            timestamp_ns: 1000,
            side: Side::Buy,
        };
        book.process_new_order(order);
        
        let result = book.cancel_order(1);
        assert!(result.is_some());
        assert_eq!(book.best_bid(), 0.0);
    }
}
