//! Footprint and volume delta compressor for efficient UI rendering.
//! Compresses tick-by-tick data into clustered nodes, allowing frontend to render
//! complex volume profiles and CVD charts without receiving 10,000+ messages per second.

use std::collections::BTreeMap;

/// Compressed footprint node representing a cluster of ticks
#[derive(Debug, Clone, serde::Serialize)]
pub struct FootprintNode {
    pub price: u64,
    pub bid_volume: u64,
    pub ask_volume: u64,
    pub trade_count: u32,
    pub open_price: u64,
    pub close_price: u64,
    pub high_price: u64,
    pub low_price: u64,
    pub start_timestamp_ms: u64,
    pub end_timestamp_ms: u64,
    pub cumulative_delta: i64, // bid_volume - ask_volume (signed)
}

/// Volume profile at a specific price level
#[derive(Debug, Clone, Default)]
pub struct VolumeProfileLevel {
    pub total_bid_volume: u64,
    pub total_ask_volume: u64,
    pub trade_count: u32,
    pub buy_initiated: u64,
    pub sell_initiated: u64,
}

/// Compressor for footprint and CVD data
pub struct FootprintCompressor {
    /// Price-ordered map of volume profile levels
    volume_profiles: BTreeMap<u64, VolumeProfileLevel>,
    /// Clustered footprint nodes for the current period
    footprint_nodes: Vec<FootprintNode>,
    /// Cumulative Volume Delta (CVD) running total
    cvd_running_total: i64,
    /// Current CVD snapshot for streaming
    cvd_snapshot: Vec<CvdPoint>,
    /// Maximum number of footprint nodes to retain (memory bound)
    max_nodes: usize,
    /// Price clustering granularity (ticks)
    price_cluster_size: u64,
}

/// CVD point for time-series streaming
#[derive(Debug, Clone, serde::Serialize)]
pub struct CvdPoint {
    pub timestamp_ms: u64,
    pub value: i64,
    pub delta: i64,
}

impl FootprintCompressor {
    /// Create a new footprint compressor with strict memory bounds
    pub fn new(max_nodes: usize, price_cluster_size: u64) -> Self {
        Self {
            volume_profiles: BTreeMap::new(),
            footprint_nodes: Vec::with_capacity(max_nodes),
            cvd_running_total: 0,
            cvd_snapshot: Vec::with_capacity(1000),
            max_nodes,
            price_cluster_size,
        }
    }

    /// Process a single trade/tick
    #[inline]
    pub fn process_tick(
        &mut self,
        price: u64,
        volume: u64,
        is_buyer_initiated: bool,
        timestamp_ms: u64,
    ) {
        // Update volume profile
        let cluster_price = (price / self.price_cluster_size) * self.price_cluster_size;
        
        let level = self.volume_profiles.entry(cluster_price).or_default();
        level.total_bid_volume = level.total_bid_volume.saturating_add(if is_buyer_initiated { volume } else { 0 });
        level.total_ask_volume = level.total_ask_volume.saturating_add(if !is_buyer_initiated { volume } else { 0 });
        level.trade_count += 1;
        
        if is_buyer_initiated {
            level.buy_initiated = level.buy_initiated.saturating_add(volume);
        } else {
            level.sell_initiated = level.sell_initiated.saturating_add(volume);
        }

        // Update CVD
        let delta = if is_buyer_initiated {
            volume as i64
        } else {
            -(volume as i64)
        };
        self.cvd_running_total += delta;

        // Limit CVD snapshot size
        if self.cvd_snapshot.len() >= 1000 {
            self.cvd_snapshot.remove(0);
        }
        self.cvd_snapshot.push(CvdPoint {
            timestamp_ms,
            value: self.cvd_running_total,
            delta,
        });
    }

    /// Build footprint nodes from accumulated volume profiles
    pub fn build_footprint_nodes(&mut self) {
        self.footprint_nodes.clear();
        
        let now_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64;

        for (price, level) in &self.volume_profiles {
            if level.trade_count > 0 {
                let node = FootprintNode {
                    price: *price,
                    bid_volume: level.total_bid_volume,
                    ask_volume: level.total_ask_volume,
                    trade_count: level.trade_count,
                    open_price: *price,
                    close_price: *price,
                    high_price: *price + self.price_cluster_size,
                    low_price: *price,
                    start_timestamp_ms: now_ms,
                    end_timestamp_ms: now_ms,
                    cumulative_delta: (level.buy_initiated as i64) - (level.sell_initiated as i64),
                };
                
                self.footprint_nodes.push(node);
            }
        }

        // Enforce memory bound
        if self.footprint_nodes.len() > self.max_nodes {
            // Keep most active nodes (by volume)
            self.footprint_nodes.sort_by(|a, b| {
                let vol_a = a.bid_volume + a.ask_volume;
                let vol_b = b.bid_volume + b.ask_volume;
                vol_b.cmp(&vol_a)
            });
            self.footprint_nodes.truncate(self.max_nodes);
        }
    }

    /// Get compressed footprint nodes for frontend
    pub fn get_footprint_nodes(&self) -> &[FootprintNode] {
        &self.footprint_nodes
    }

    /// Get current CVD snapshot
    pub fn get_cvd_snapshot(&self) -> &[CvdPoint] {
        &self.cvd_snapshot
    }

    /// Get volume profile as sorted vector
    pub fn get_volume_profile(&self) -> Vec<(u64, VolumeProfileLevel)> {
        self.volume_profiles.iter().map(|(k, v)| (*k, v.clone())).collect()
    }

    /// Get Point of Control (POC) - price with highest volume
    pub fn get_poc(&self) -> Option<u64> {
        self.volume_profiles
            .iter()
            .max_by_key(|(_, v)| v.total_bid_volume + v.total_ask_volume)
            .map(|(price, _)| *price)
    }

    /// Get Value Area (70% of volume around POC)
    pub fn get_value_area(&self) -> Option<(u64, u64)> {
        if self.volume_profiles.is_empty() {
            return None;
        }

        let total_volume: u64 = self.volume_profiles.values()
            .map(|v| v.total_bid_volume + v.total_ask_volume)
            .sum();
        
        let target_volume = (total_volume as f64 * 0.70) as u64;
        
        let mut sorted_levels: Vec<_> = self.volume_profiles.iter().collect();
        sorted_levels.sort_by_key(|(_, v)| v.price);

        let poc = self.get_poc()?;
        let mut accumulated_volume = 0u64;
        let mut lower_bound = poc;
        let mut upper_bound = poc;

        // Simple approximation: expand from POC until we have 70%
        for (price, level) in &sorted_levels {
            let vol = level.total_bid_volume + level.total_ask_volume;
            accumulated_volume += vol;
            
            if **price < lower_bound {
                lower_bound = **price;
            }
            if **price > upper_bound {
                upper_bound = **price;
            }
            
            if accumulated_volume >= target_volume {
                break;
            }
        }

        Some((lower_bound, upper_bound))
    }

    /// Reset all accumulated data
    pub fn reset(&mut self) {
        self.volume_profiles.clear();
        self.footprint_nodes.clear();
        self.cvd_running_total = 0;
        self.cvd_snapshot.clear();
    }

    /// Reset only volume profiles (keep CVD history)
    pub fn reset_profiles(&mut self) {
        self.volume_profiles.clear();
        self.footprint_nodes.clear();
    }
}

/// Streaming footprint data for real-time updates
#[derive(Debug, Clone, serde::Serialize)]
pub struct FootprintStream {
    pub price: u64,
    pub volume: u64,
    pub delta: i64,
    pub is_buyer_initiated: bool,
    pub timestamp_ms: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compressor_creation() {
        let compressor = FootprintCompressor::new(500, 100);
        
        assert_eq!(compressor.max_nodes, 500);
        assert_eq!(compressor.price_cluster_size, 100);
        assert_eq!(compressor.cvd_running_total, 0);
    }

    #[test]
    fn test_process_tick() {
        let mut compressor = FootprintCompressor::new(500, 100);
        
        // Process buyer-initiated trade
        compressor.process_tick(50050, 100, true, 1000000);
        
        assert_eq!(compressor.cvd_running_total, 100);
        assert_eq!(compressor.cvd_snapshot.len(), 1);
        
        // Process seller-initiated trade
        compressor.process_tick(50050, 50, false, 1000001);
        
        assert_eq!(compressor.cvd_running_total, 50);
    }

    #[test]
    fn test_volume_profile_clustering() {
        let mut compressor = FootprintCompressor::new(500, 100);
        
        // Multiple trades in same cluster
        compressor.process_tick(50010, 100, true, 1000000);
        compressor.process_tick(50050, 200, false, 1000001);
        compressor.process_tick(50090, 150, true, 1000002);
        
        let profile = compressor.get_volume_profile();
        
        // All should be in same cluster (50000)
        assert_eq!(profile.len(), 1);
        assert_eq!(profile[0].0, 50000);
    }

    #[test]
    fn test_footprint_node_generation() {
        let mut compressor = FootprintCompressor::new(500, 100);
        
        compressor.process_tick(50050, 100, true, 1000000);
        compressor.process_tick(50150, 200, false, 1000001);
        
        compressor.build_footprint_nodes();
        
        let nodes = compressor.get_footprint_nodes();
        assert_eq!(nodes.len(), 2);
    }

    #[test]
    fn test_poc_calculation() {
        let mut compressor = FootprintCompressor::new(500, 100);
        
        compressor.process_tick(50050, 100, true, 1000000);
        compressor.process_tick(50150, 500, false, 1000001);
        compressor.process_tick(50250, 200, true, 1000002);
        
        let poc = compressor.get_poc();
        
        // POC should be at 50100 (highest volume cluster)
        assert_eq!(poc, Some(50100));
    }
}
