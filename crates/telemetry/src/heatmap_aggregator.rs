//! High-performance order book heatmap aggregator.
//! Aggregates raw L2/L3 liquidity into price/time buckets using strictly bounded 2D arrays.
//! Generates the data matrix required for frontend "Order Book Heatmap" visualization.

use std::collections::BTreeMap;

/// Fixed-size heatmap grid with strict memory bounds
pub struct HeatmapGrid {
    /// Number of price bins (rows)
    pub rows: usize,
    /// Number of time bins (columns)
    pub cols: usize,
    /// Price bin size in ticks
    pub price_bin_size: u64,
    /// Time bin size in milliseconds
    pub time_bin_size_ms: u64,
    /// Base price for binning (center of grid)
    pub base_price: u64,
    /// 2D grid storing cumulative volume (bid_volume, ask_volume) per cell
    /// Stored as flat array for cache efficiency: row * cols + col
    pub bid_grid: Vec<u64>,
    pub ask_grid: Vec<u64>,
    /// Timestamp of the oldest bin in each column
    pub time_bins: Vec<u64>,
}

impl HeatmapGrid {
    /// Create a new heatmap grid with fixed dimensions
    pub fn new(
        rows: usize,
        cols: usize,
        price_bin_size: u64,
        time_bin_size_ms: u64,
        base_price: u64,
    ) -> Self {
        let total_cells = rows * cols;
        
        Self {
            rows,
            cols,
            price_bin_size,
            time_bin_size_ms,
            base_price,
            bid_grid: vec![0; total_cells],
            ask_grid: vec![0; total_cells],
            time_bins: vec![0; cols],
        }
    }

    /// Calculate price bin index from absolute price
    #[inline]
    pub fn price_to_row(&self, price: u64) -> Option<usize> {
        let price_offset = price as i128 - self.base_price as i128;
        let row_offset = price_offset / self.price_bin_size as i128;
        let row = (self.rows as i128 / 2) + row_offset;
        
        if row >= 0 && row < self.rows as i128 {
            Some(row as usize)
        } else {
            None // Price outside grid bounds
        }
    }

    /// Calculate time bin index from timestamp
    #[inline]
    pub fn time_to_col(&self, timestamp_ms: u64) -> usize {
        let relative_time = timestamp_ms % (self.time_bin_size_ms * self.cols as u64);
        (relative_time / self.time_bin_size_ms) as usize
    }

    /// Add a bid liquidity event to the heatmap
    #[inline]
    pub fn add_bid(&mut self, price: u64, volume: u64, timestamp_ms: u64) {
        if let Some(row) = self.price_to_row(price) {
            let col = self.time_to_col(timestamp_ms);
            let idx = row * self.cols + col;
            
            if idx < self.bid_grid.len() {
                self.bid_grid[idx] = self.bid_grid[idx].saturating_add(volume);
            }
        }
    }

    /// Add an ask liquidity event to the heatmap
    #[inline]
    pub fn add_ask(&mut self, price: u64, volume: u64, timestamp_ms: u64) {
        if let Some(row) = self.price_to_row(price) {
            let col = self.time_to_col(timestamp_ms);
            let idx = row * self.cols + col;
            
            if idx < self.ask_grid.len() {
                self.ask_grid[idx] = self.ask_grid[idx].saturating_add(volume);
            }
        }
    }

    /// Remove liquidity from the heatmap (for cancellations)
    #[inline]
    pub fn remove_liquidity(&mut self, price: u64, volume: u64, is_bid: bool, timestamp_ms: u64) {
        if let Some(row) = self.price_to_row(price) {
            let col = self.time_to_col(timestamp_ms);
            let idx = row * self.cols + col;
            
            if is_bid && idx < self.bid_grid.len() {
                self.bid_grid[idx] = self.bid_grid[idx].saturating_sub(volume);
            } else if !is_bid && idx < self.ask_grid.len() {
                self.ask_grid[idx] = self.ask_grid[idx].saturating_sub(volume);
            }
        }
    }

    /// Get the cell value at a specific row and column
    #[inline]
    pub fn get_cell(&self, row: usize, col: usize) -> Option<(u64, u64)> {
        if row >= self.rows || col >= self.cols {
            return None;
        }
        
        let idx = row * self.cols + col;
        Some((self.bid_grid[idx], self.ask_grid[idx]))
    }

    /// Get entire grid as serialized data for frontend
    pub fn to_snapshot(&self) -> HeatmapSnapshot {
        HeatmapSnapshot {
            rows: self.rows,
            cols: self.cols,
            base_price: self.base_price,
            price_bin_size: self.price_bin_size,
            time_bin_size_ms: self.time_bin_size_ms,
            bid_data: self.bid_grid.clone(),
            ask_data: self.ask_grid.clone(),
            timestamp_ms: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_millis() as u64,
        }
    }

    /// Reset a specific time column (rotate buffer)
    pub fn reset_column(&mut self, col: usize) {
        for row in 0..self.rows {
            let idx = row * self.cols + col;
            self.bid_grid[idx] = 0;
            self.ask_grid[idx] = 0;
        }
        self.time_bins[col] = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64;
    }

    /// Clear entire grid
    pub fn clear(&mut self) {
        self.bid_grid.fill(0);
        self.ask_grid.fill(0);
    }
}

/// Snapshot of heatmap data for serialization to frontend
#[derive(Debug, Clone, serde::Serialize)]
pub struct HeatmapSnapshot {
    pub rows: usize,
    pub cols: usize,
    pub base_price: u64,
    pub price_bin_size: u64,
    pub time_bin_size_ms: u64,
    pub bid_data: Vec<u64>,
    pub ask_data: Vec<u64>,
    pub timestamp_ms: u64,
}

/// Streaming heatmap aggregator with automatic time rotation
pub struct HeatmapAggregator {
    grid: HeatmapGrid,
    last_rotation_check: u64,
    rotation_interval_ms: u64,
}

impl HeatmapAggregator {
    /// Create a new heatmap aggregator
    pub fn new(
        rows: usize,
        cols: usize,
        price_bin_size: u64,
        time_bin_size_ms: u64,
        base_price: u64,
    ) -> Self {
        Self {
            grid: HeatmapGrid::new(rows, cols, price_bin_size, time_bin_size_ms, base_price),
            last_rotation_check: 0,
            rotation_interval_ms: time_bin_size_ms,
        }
    }

    /// Process a batch of order book updates
    pub fn process_updates(&mut self, updates: &[OrderBookUpdate]) {
        let now_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64;

        // Check for time column rotation
        if now_ms - self.last_rotation_check >= self.rotation_interval_ms {
            self.rotate_time_columns(now_ms);
            self.last_rotation_check = now_ms;
        }

        // Apply all updates
        for update in updates {
            match update {
                OrderBookUpdate::BidAdd { price, volume, timestamp } => {
                    self.grid.add_bid(*price, *volume, *timestamp);
                }
                OrderBookUpdate::AskAdd { price, volume, timestamp } => {
                    self.grid.add_ask(*price, *volume, *timestamp);
                }
                OrderBookUpdate::BidCancel { price, volume, timestamp } => {
                    self.grid.remove_liquidity(*price, *volume, true, *timestamp);
                }
                OrderBookUpdate::AskCancel { price, volume, timestamp } => {
                    self.grid.remove_liquidity(*price, *volume, false, *timestamp);
                }
            }
        }
    }

    /// Rotate time columns (sliding window)
    fn rotate_time_columns(&mut self, now_ms: u64) {
        let current_col = self.grid.time_to_col(now_ms);
        let prev_col = if current_col == 0 { self.grid.cols - 1 } else { current_col - 1 };
        
        // Reset the column that's sliding out of the window
        self.grid.reset_column(prev_col);
    }

    /// Get current heatmap snapshot
    pub fn get_snapshot(&self) -> HeatmapSnapshot {
        self.grid.to_snapshot()
    }

    /// Update base price (recenter grid)
    pub fn recenter(&mut self, new_base_price: u64) {
        self.grid.base_price = new_base_price;
        self.grid.clear();
    }
}

/// Order book update types
#[derive(Debug, Clone)]
pub enum OrderBookUpdate {
    BidAdd { price: u64, volume: u64, timestamp: u64 },
    AskAdd { price: u64, volume: u64, timestamp: u64 },
    BidCancel { price: u64, volume: u64, timestamp: u64 },
    AskCancel { price: u64, volume: u64, timestamp: u64 },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_heatmap_grid_creation() {
        let grid = HeatmapGrid::new(100, 60, 100, 1000, 50000);
        
        assert_eq!(grid.rows, 100);
        assert_eq!(grid.cols, 60);
        assert_eq!(grid.bid_grid.len(), 100 * 60);
        assert_eq!(grid.ask_grid.len(), 100 * 60);
    }

    #[test]
    fn test_price_to_row_mapping() {
        let grid = HeatmapGrid::new(100, 60, 100, 1000, 50000);
        
        // Base price should map to center row
        assert_eq!(grid.price_to_row(50000), Some(50));
        
        // Price above base
        assert_eq!(grid.price_to_row(50100), Some(51));
        
        // Price below base
        assert_eq!(grid.price_to_row(49900), Some(49));
        
        // Price outside bounds
        assert!(grid.price_to_row(55000).is_none());
    }

    #[test]
    fn test_add_liquidity() {
        let mut grid = HeatmapGrid::new(100, 60, 100, 1000, 50000);
        
        grid.add_bid(50000, 1000, 1000000);
        grid.add_ask(50100, 500, 1000000);
        
        let (bid_vol, ask_vol) = grid.get_cell(50, 0).unwrap();
        assert_eq!(bid_vol, 1000);
        assert_eq!(ask_vol, 0);
        
        let (bid_vol, ask_vol) = grid.get_cell(51, 0).unwrap();
        assert_eq!(bid_vol, 0);
        assert_eq!(ask_vol, 500);
    }

    #[test]
    fn test_snapshot_generation() {
        let mut grid = HeatmapGrid::new(10, 5, 100, 1000, 50000);
        grid.add_bid(50000, 100, 1000000);
        
        let snapshot = grid.to_snapshot();
        
        assert_eq!(snapshot.rows, 10);
        assert_eq!(snapshot.cols, 5);
        assert!(!snapshot.bid_data.is_empty());
    }
}
