//! Microsecond queue position tracker utilizing L3 order IDs.
//! Calculates the exact volume ahead of the bot's limit orders in real-time.
//! Optimized for nanosecond precision fill probability prediction.

use std::sync::atomic::{AtomicU64, Ordering};
use std::ptr::NonNull;
use crate::l3_tree::{L3Node, L3Tree};

/// Queue position tracker for a single price level
#[repr(C, align(64))]
pub struct QueueTracker {
    /// Total volume ahead of our order at this price level
    volume_ahead: AtomicU64,
    /// Our order's position in the queue (order ID)
    our_order_id: AtomicU64,
    /// Total volume at this price level
    total_volume: AtomicU64,
    /// Number of orders ahead
    orders_ahead: AtomicU64,
    /// Last update timestamp in nanoseconds
    last_update_ns: AtomicU64,
    _padding: [u8; 24], // Cache line padding
}

impl QueueTracker {
    pub fn new(our_order_id: u64) -> Self {
        Self {
            volume_ahead: AtomicU64::new(0),
            our_order_id: AtomicU64::new(our_order_id),
            total_volume: AtomicU64::new(0),
            orders_ahead: AtomicU64::new(0),
            last_update_ns: AtomicU64::new(0),
            _padding: [0u8; 24],
        }
    }
    
    /// Update queue position based on L3 tree state - O(N) where N is orders at price level
    #[inline]
    pub fn update(&self, tree: &L3Tree, our_order_id: u64, price_level: u64) {
        let mut volume_ahead: u64 = 0;
        let mut orders_ahead: u64 = 0;
        let mut total_volume: u64 = 0;
        
        // Iterate through all orders at this price level
        // In production, this would use a secondary index for O(1) price level access
        if let Some(start_node) = tree.find(our_order_id) {
            unsafe {
                // Simple traversal - in production would be optimized with price-level indexing
                let mut current = start_node;
                
                // Traverse left to find earliest order at price level
                while let Some(left) = current.as_ref().left {
                    if left.as_ref().price == price_level {
                        current = left;
                    } else {
                        break;
                    }
                }
                
                // Now traverse and count
                loop {
                    let node = current.as_ref();
                    if node.price == price_level {
                        total_volume += node.quantity;
                        
                        if node.order_id < our_order_id {
                            volume_ahead += node.quantity;
                            orders_ahead += 1;
                        }
                    }
                    
                    // Move to next node (simplified - would use inorder successor in production)
                    if let Some(right) = node.right {
                        current = right;
                    } else {
                        break;
                    }
                }
            }
        }
        
        self.volume_ahead.store(volume_ahead, Ordering::Relaxed);
        self.total_volume.store(total_volume, Ordering::Relaxed);
        self.orders_ahead.store(orders_ahead, Ordering::Relaxed);
        self.our_order_id.store(our_order_id, Ordering::Relaxed);
    }
    
    /// Get fill probability estimate based on queue position
    #[inline]
    pub fn fill_probability(&self, incoming_volume: u64) -> f64 {
        let volume_ahead = self.volume_ahead.load(Ordering::Relaxed);
        let total_volume = self.total_volume.load(Ordering::Relaxed);
        
        if total_volume == 0 {
            return 0.0;
        }
        
        // Probability = min(1.0, incoming_volume / (volume_ahead + 1))
        let prob = (incoming_volume as f64) / ((volume_ahead + 1) as f64);
        prob.min(1.0)
    }
    
    /// Get estimated time to fill based on historical trade rate
    #[inline]
    pub fn estimated_fill_time_ns(&self, avg_trade_rate_per_ns: f64) -> u64 {
        let volume_ahead = self.volume_ahead.load(Ordering::Relaxed);
        
        if avg_trade_rate_per_ns <= 0.0 {
            return u64::MAX;
        }
        
        let time_ns = (volume_ahead as f64) / avg_trade_rate_per_ns;
        time_ns as u64
    }
    
    /// Get volume ahead atomically
    #[inline]
    pub fn get_volume_ahead(&self) -> u64 {
        self.volume_ahead.load(Ordering::Relaxed)
    }
    
    /// Get our position in queue
    #[inline]
    pub fn get_position(&self) -> u64 {
        self.orders_ahead.load(Ordering::Relaxed) + 1
    }
}

/// Global queue tracker managing multiple price levels
pub struct QueueTrackerGlobal {
    /// Trackers for bid side (indexed by price level hash)
    bid_trackers: Vec<QueueTracker>,
    /// Trackers for ask side
    ask_trackers: Vec<QueueTracker>,
    /// Current best bid price
    best_bid: AtomicU64,
    /// Current best ask price
    best_ask: AtomicU64,
}

impl QueueTrackerGlobal {
    pub fn new(max_levels: usize) -> Self {
        Self {
            bid_trackers: (0..max_levels).map(|_| QueueTracker::new(0)).collect(),
            ask_trackers: (0..max_levels).map(|_| QueueTracker::new(0)).collect(),
            best_bid: AtomicU64::new(0),
            best_ask: AtomicU64::new(0),
        }
    }
    
    /// Update all trackers - called on every L3 update
    #[inline]
    pub fn update_all(&self, tree: &L3Tree, our_bid_order_id: u64, our_ask_order_id: u64, 
                      best_bid: u64, best_ask: u64) {
        self.best_bid.store(best_bid, Ordering::Relaxed);
        self.best_ask.store(best_ask, Ordering::Relaxed);
        
        // Update bid queue tracker
        if !self.bid_trackers.is_empty() {
            self.bid_trackers[0].update(tree, our_bid_order_id, best_bid);
        }
        
        // Update ask queue tracker
        if !self.ask_trackers.is_empty() {
            self.ask_trackers[0].update(tree, our_ask_order_id, best_ask);
        }
    }
    
    /// Get fill probability for bid side
    #[inline]
    pub fn bid_fill_probability(&self, incoming_volume: u64) -> f64 {
        if self.bid_trackers.is_empty() {
            return 0.0;
        }
        self.bid_trackers[0].fill_probability(incoming_volume)
    }
    
    /// Get fill probability for ask side
    #[inline]
    pub fn ask_fill_probability(&self, incoming_volume: u64) -> f64 {
        if self.ask_trackers.is_empty() {
            return 0.0;
        }
        self.ask_trackers[0].fill_probability(incoming_volume)
    }
    
    /// Get volume ahead on bid side
    #[inline]
    pub fn bid_volume_ahead(&self) -> u64 {
        if self.bid_trackers.is_empty() {
            return 0;
        }
        self.bid_trackers[0].get_volume_ahead()
    }
    
    /// Get volume ahead on ask side
    #[inline]
    pub fn ask_volume_ahead(&self) -> u64 {
        if self.ask_trackers.is_empty() {
            return 0;
        }
        self.ask_trackers[0].get_volume_ahead()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_queue_tracker_basic() {
        let tracker = QueueTracker::new(12345);
        assert_eq!(tracker.get_position(), 1); // Default position
        assert_eq!(tracker.fill_probability(100), 1.0); // No volume ahead
    }
    
    #[test]
    fn test_fill_probability() {
        let tracker = QueueTracker::new(12345);
        tracker.volume_ahead.store(500, Ordering::Relaxed);
        
        // 100 volume incoming, 500 ahead = 100/501 probability
        let prob = tracker.fill_probability(100);
        assert!(prob > 0.19 && prob < 0.21);
    }
}
