//! Position Rebuilder - Emergency state rebuilder from exchange trade history.
//! Reconstructs portfolio state, average entry prices, and open orders from raw fills.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::collections::HashMap;

/// Rebuilt position record
#[derive(Debug, Clone)]
pub struct RebuiltPosition {
    pub symbol: String,
    pub side: PositionSide,
    pub quantity: f64,
    pub avg_entry_price: f64,
    pub total_cost: f64,
    pub realized_pnl: f64,
    pub fill_count: u32,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PositionSide {
    Long,
    Short,
    Flat,
}

/// Trade fill record from exchange
#[derive(Debug, Clone)]
pub struct TradeFill {
    pub trade_id: String,
    pub order_id: String,
    pub symbol: String,
    pub side: OrderSide,
    pub price: f64,
    pub quantity: f64,
    pub commission: f64,
    pub commission_asset: String,
    pub timestamp_ms: u64,
    pub is_maker: bool,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum OrderSide {
    Buy,
    Sell,
}

/// Rebuild configuration
#[derive(Debug, Clone)]
pub struct RebuilderConfig {
    pub max_history_days: u32,      // Maximum days of history to fetch
    pub batch_size: u32,            // Number of fills per batch
    pub parallel_fetch: bool,       // Enable parallel fetching
    pub verify_checksum: bool,      // Verify rebuild checksum
}

impl Default for RebuilderConfig {
    fn default() -> Self {
        Self {
            max_history_days: 30,
            batch_size: 1000,
            parallel_fetch: true,
            verify_checksum: true,
        }
    }
}

/// Rebuild result
#[derive(Debug, Clone)]
pub struct RebuildResult {
    pub positions: HashMap<String, RebuiltPosition>,
    pub total_realized_pnl: f64,
    pub total_fills_processed: u64,
    pub rebuild_duration_ms: u64,
    pub checksum: [u8; 32],
    pub errors: Vec<String>,
}

/// Lock-free Position Rebuilder
pub struct PositionRebuilder {
    config: RebuilderConfig,
    rebuilds_performed: AtomicU64,
    successful_rebuilds: AtomicU64,
    failed_rebuilds: AtomicU64,
    total_fills_processed: AtomicU64,
    active: AtomicBool,
}

impl PositionRebuilder {
    pub fn new(config: RebuilderConfig) -> Self {
        Self {
            config,
            rebuilds_performed: AtomicU64::new(0),
            successful_rebuilds: AtomicU64::new(0),
            failed_rebuilds: AtomicU64::new(0),
            total_fills_processed: AtomicU64::new(0),
            active: AtomicBool::new(true),
        }
    }

    /// Rebuild positions from exchange trade history
    pub fn rebuild<T: TradeHistoryApi>(&self, api: &T, venue: &str) -> Result<RebuildResult, String> {
        if !self.active.load(Ordering::Relaxed) {
            return Ok(RebuildResult {
                positions: HashMap::new(),
                total_realized_pnl: 0.0,
                total_fills_processed: 0,
                rebuild_duration_ms: 0,
                checksum: [0u8; 32],
                errors: vec!["Rebuilder not active".to_string()],
            });
        }

        self.rebuilds_performed.fetch_add(1, Ordering::Relaxed);
        let start = std::time::Instant::now();

        // Fetch all trade fills from exchange
        let fills = match self.fetch_all_fills(api, venue) {
            Ok(f) => f,
            Err(e) => {
                self.failed_rebuilds.fetch_add(1, Ordering::Relaxed);
                return Err(e);
            }
        };

        self.total_fills_processed.fetch_add(fills.len() as u64, Ordering::Relaxed);

        // Process fills to rebuild positions
        let positions = self.process_fills(&fills);

        // Calculate totals
        let total_realized_pnl: f64 = positions.values().map(|p| p.realized_pnl).sum();

        // Generate checksum
        let checksum = self.compute_rebuild_checksum(&positions);

        let duration_ms = start.elapsed().as_millis() as u64;

        self.successful_rebuilds.fetch_add(1, Ordering::Relaxed);

        Ok(RebuildResult {
            positions,
            total_realized_pnl,
            total_fills_processed: fills.len() as u64,
            rebuild_duration_ms: duration_ms,
            checksum,
            errors: vec![],
        })
    }

    /// Fetch all trade fills from exchange API
    fn fetch_all_fills<T: TradeHistoryApi>(&self, api: &T, venue: &str) -> Result<Vec<TradeFill>, String> {
        let mut all_fills = Vec::new();
        let mut offset = 0;

        loop {
            let batch = api.get_trade_history(
                venue,
                self.config.batch_size,
                offset,
            )?;

            if batch.is_empty() {
                break;
            }

            all_fills.extend(batch);
            offset += self.config.batch_size as u64;

            // Safety limit
            if offset > self.config.batch_size as u64 * 1000 {
                break;
            }
        }

        Ok(all_fills)
    }

    /// Process fills to reconstruct positions
    fn process_fills(&self, fills: &[TradeFill]) -> HashMap<String, RebuiltPosition> {
        let mut positions: HashMap<String, RebuiltPosition> = HashMap::new();

        for fill in fills {
            let entry = positions.entry(fill.symbol.clone()).or_insert_with(|| {
                RebuiltPosition {
                    symbol: fill.symbol.clone(),
                    side: PositionSide::Flat,
                    quantity: 0.0,
                    avg_entry_price: 0.0,
                    total_cost: 0.0,
                    realized_pnl: 0.0,
                    fill_count: 0,
                }
            });

            self.apply_fill(entry, fill);
        }

        // Remove flat positions
        positions.retain(|_, p| p.side != PositionSide::Flat);

        positions
    }

    /// Apply a single fill to a position
    fn apply_fill(&self, position: &mut RebuiltPosition, fill: &TradeFill) {
        position.fill_count += 1;

        let fill_value = fill.price * fill.quantity;
        let fill_commission = fill.commission;

        match (position.side, fill.side) {
            // Opening or adding to long position
            (PositionSide::Flat | PositionSide::Long, OrderSide::Buy) => {
                let total_qty = position.quantity + fill.quantity;
                if total_qty > 0.0 {
                    position.avg_entry_price = (
                        (position.avg_entry_price * position.quantity) + fill_value
                    ) / total_qty;
                }
                position.quantity = total_qty;
                position.total_cost += fill_value + fill_commission;
                position.side = PositionSide::Long;
            }

            // Reducing or closing long position
            (PositionSide::Long, OrderSide::Sell) => {
                if fill.quantity >= position.quantity {
                    // Position closed or flipped
                    let close_qty = position.quantity;
                    let pnl = (fill.price - position.avg_entry_price) * close_qty - fill_commission;
                    position.realized_pnl += pnl;
                    
                    let remaining = fill.quantity - close_qty;
                    if remaining > 0.0 {
                        // Flipped to short
                        position.side = PositionSide::Short;
                        position.quantity = remaining;
                        position.avg_entry_price = fill.price;
                        position.total_cost = fill_value + fill_commission;
                    } else {
                        // Fully closed
                        position.side = PositionSide::Flat;
                        position.quantity = 0.0;
                        position.avg_entry_price = 0.0;
                        position.total_cost = 0.0;
                    }
                } else {
                    // Partially reduced
                    let pnl = (fill.price - position.avg_entry_price) * fill.quantity - fill_commission;
                    position.realized_pnl += pnl;
                    position.quantity -= fill.quantity;
                    position.total_cost -= fill_value;
                }
            }

            // Opening or adding to short position
            (PositionSide::Flat | PositionSide::Short, OrderSide::Sell) => {
                let total_qty = position.quantity + fill.quantity;
                if total_qty > 0.0 {
                    position.avg_entry_price = (
                        (position.avg_entry_price * position.quantity) + fill_value
                    ) / total_qty;
                }
                position.quantity = total_qty;
                position.total_cost += fill_value + fill_commission;
                position.side = PositionSide::Short;
            }

            // Reducing or closing short position
            (PositionSide::Short, OrderSide::Buy) => {
                if fill.quantity >= position.quantity {
                    // Position closed or flipped
                    let close_qty = position.quantity;
                    let pnl = (position.avg_entry_price - fill.price) * close_qty - fill_commission;
                    position.realized_pnl += pnl;
                    
                    let remaining = fill.quantity - close_qty;
                    if remaining > 0.0 {
                        // Flipped to long
                        position.side = PositionSide::Long;
                        position.quantity = remaining;
                        position.avg_entry_price = fill.price;
                        position.total_cost = fill_value + fill_commission;
                    } else {
                        // Fully closed
                        position.side = PositionSide::Flat;
                        position.quantity = 0.0;
                        position.avg_entry_price = 0.0;
                        position.total_cost = 0.0;
                    }
                } else {
                    // Partially reduced
                    let pnl = (position.avg_entry_price - fill.price) * fill.quantity - fill_commission;
                    position.realized_pnl += pnl;
                    position.quantity -= fill.quantity;
                    position.total_cost -= fill_value;
                }
            }
        }
    }

    /// Compute checksum of rebuilt state
    fn compute_rebuild_checksum(&self, positions: &HashMap<String, RebuiltPosition>) -> [u8; 32] {
        use sha2::{Sha256, Digest};
        let mut hasher = Sha256::new();
        
        // Sort symbols for deterministic ordering
        let mut symbols: Vec<&String> = positions.keys().collect();
        symbols.sort();
        
        for symbol in symbols {
            if let Some(pos) = positions.get(symbol) {
                hasher.update(symbol.as_bytes());
                hasher.update(&pos.side as *const PositionSide as usize.to_le_bytes());
                hasher.update(&pos.quantity.to_le_bytes());
                hasher.update(&pos.avg_entry_price.to_le_bytes());
            }
        }
        
        hasher.finalize().into()
    }

    /// Get statistics
    pub fn get_stats(&self) -> RebuilderStats {
        RebuilderStats {
            rebuilds_performed: self.rebuilds_performed.load(Ordering::Relaxed),
            successful_rebuilds: self.successful_rebuilds.load(Ordering::Relaxed),
            failed_rebuilds: self.failed_rebuilds.load(Ordering::Relaxed),
            total_fills_processed: self.total_fills_processed.load(Ordering::Relaxed),
        }
    }

    /// Set active state
    pub fn set_active(&self, active: bool) {
        self.active.store(active, Ordering::Relaxed);
    }
}

#[derive(Debug, Clone, Copy)]
pub struct RebuilderStats {
    pub rebuilds_performed: u64,
    pub successful_rebuilds: u64,
    pub failed_rebuilds: u64,
    pub total_fills_processed: u64,
}

/// Trade history API trait
pub trait TradeHistoryApi {
    fn get_trade_history(
        &self,
        venue: &str,
        limit: u32,
        offset: u64,
    ) -> Result<Vec<TradeFill>, String>;
}

#[cfg(test)]
mod tests {
    use super::*;

    struct MockTradeApi;
    
    impl TradeHistoryApi for MockTradeApi {
        fn get_trade_history(
            &self,
            _venue: &str,
            limit: u32,
            _offset: u64,
        ) -> Result<Vec<TradeFill>, String> {
            // Return some mock fills
            Ok(vec![
                TradeFill {
                    trade_id: "1".to_string(),
                    order_id: "o1".to_string(),
                    symbol: "BTCUSDT".to_string(),
                    side: OrderSide::Buy,
                    price: 50000.0,
                    quantity: 0.1,
                    commission: 0.5,
                    commission_asset: "USDT".to_string(),
                    timestamp_ms: 1000000,
                    is_maker: false,
                },
            ])
        }
    }

    #[test]
    fn test_position_rebuild() {
        let config = RebuilderConfig::default();
        let rebuilder = PositionRebuilder::new(config);
        let api = MockTradeApi;
        
        let result = rebuilder.rebuild(&api, "binance").unwrap();
        assert!(result.positions.contains_key("BTCUSDT"));
        assert_eq!(result.total_fills_processed, 1);
    }

    #[test]
    fn test_apply_fill_long() {
        let config = RebuilderConfig::default();
        let rebuilder = PositionRebuilder::new(config);
        
        let mut position = RebuiltPosition {
            symbol: "BTCUSDT".to_string(),
            side: PositionSide::Flat,
            quantity: 0.0,
            avg_entry_price: 0.0,
            total_cost: 0.0,
            realized_pnl: 0.0,
            fill_count: 0,
        };
        
        let fill = TradeFill {
            trade_id: "1".to_string(),
            order_id: "o1".to_string(),
            symbol: "BTCUSDT".to_string(),
            side: OrderSide::Buy,
            price: 50000.0,
            quantity: 0.1,
            commission: 0.5,
            commission_asset: "USDT".to_string(),
            timestamp_ms: 1000000,
            is_maker: false,
        };
        
        rebuilder.apply_fill(&mut position, &fill);
        
        assert_eq!(position.side, PositionSide::Long);
        assert_eq!(position.quantity, 0.1);
        assert_eq!(position.avg_entry_price, 50000.0);
    }
}
