//! Probabilistic Queue Position Estimator for Limit Orders
//! Analyzes historical fill rates, order book cancellations, and market order aggressiveness

use std::sync::atomic::{AtomicF64, AtomicU64, AtomicUsize, Ordering};
use std::time::{SystemTime, UNIX_EPOCH, Duration};

/// Order book queue state at a specific price level
pub struct QueueState {
    /// Total volume at price level
    pub total_volume: AtomicF64,
    /// Volume ahead of our position (estimated)
    pub volume_ahead: AtomicF64,
    /// Our position in queue (estimated)
    pub our_position: AtomicF64,
    /// Recent fill rate (fills per second)
    pub fill_rate: AtomicF64,
    /// Recent cancellation rate (cancels per second)
    pub cancel_rate: AtomicF64,
    /// Market order aggression rate (volume per second)
    pub aggression_rate: AtomicF64,
}

impl QueueState {
    pub fn new() -> Self {
        Self {
            total_volume: AtomicF64::new(0.0),
            volume_ahead: AtomicF64::new(0.0),
            our_position: AtomicF64::new(0.0),
            fill_rate: AtomicF64::new(0.0),
            cancel_rate: AtomicF64::new(0.0),
            aggression_rate: AtomicF64::new(0.0),
        }
    }

    #[inline]
    pub fn update_total_volume(&self, vol: f64) {
        self.total_volume.store(vol, Ordering::Relaxed);
    }

    #[inline]
    pub fn update_volume_ahead(&self, vol: f64) {
        self.volume_ahead.store(vol, Ordering::Relaxed);
    }

    #[inline]
    pub fn update_our_position(&self, pos: f64) {
        self.our_position.store(pos, Ordering::Relaxed);
    }

    #[inline]
    pub fn update_fill_rate(&self, rate: f64) {
        self.fill_rate.store(rate, Ordering::Relaxed);
    }

    #[inline]
    pub fn update_cancel_rate(&self, rate: f64) {
        self.cancel_rate.store(rate, Ordering::Relaxed);
    }

    #[inline]
    pub fn update_aggression_rate(&self, rate: f64) {
        self.aggression_rate.store(rate, Ordering::Relaxed);
    }
}

impl Default for QueueState {
    fn default() -> Self {
        Self::new()
    }
}

/// Queue position estimator with microsecond precision
pub struct QueuePositionEstimator {
    /// Bid queue state
    pub bid_queue: QueueState,
    /// Ask queue state
    pub ask_queue: QueueState,
    /// Order placement timestamp (nanoseconds)
    pub order_time_ns: AtomicU64,
    /// Order size
    pub order_size: AtomicF64,
    /// Estimated time to fill (milliseconds)
    pub estimated_fill_ms: AtomicF64,
    /// Fill probability (0-1)
    pub fill_probability: AtomicF64,
    /// Last update timestamp
    pub last_update_ns: AtomicU64,
}

impl QueuePositionEstimator {
    pub fn new() -> Self {
        let now_ns = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64;
        
        Self {
            bid_queue: QueueState::new(),
            ask_queue: QueueState::new(),
            order_time_ns: AtomicU64::new(now_ns),
            order_size: AtomicF64::new(0.0),
            estimated_fill_ms: AtomicF64::new(f64::MAX),
            fill_probability: AtomicF64::new(0.0),
            last_update_ns: AtomicU64::new(now_ns),
        }
    }

    /// Update with new order book data for bid side
    #[inline]
    pub fn update_bid_queue(
        &self,
        total_volume: f64,
        volume_ahead: f64,
        our_size: f64,
    ) {
        self.bid_queue.update_total_volume(total_volume);
        self.bid_queue.update_volume_ahead(volume_ahead);
        self.order_size.store(our_size, Ordering::Relaxed);
        
        // Estimate our position as fraction through the queue
        let position_fraction = if total_volume > 0.0 {
            volume_ahead / total_volume
        } else {
            1.0
        };
        self.bid_queue.update_our_position(position_fraction);
        
        self.update_estimates(Side::Bid);
    }

    /// Update with new order book data for ask side
    #[inline]
    pub fn update_ask_queue(
        &self,
        total_volume: f64,
        volume_ahead: f64,
        our_size: f64,
    ) {
        self.ask_queue.update_total_volume(total_volume);
        self.ask_queue.update_volume_ahead(volume_ahead);
        self.order_size.store(our_size, Ordering::Relaxed);
        
        let position_fraction = if total_volume > 0.0 {
            volume_ahead / total_volume
        } else {
            1.0
        };
        self.ask_queue.update_our_position(position_fraction);
        
        self.update_estimates(Side::Ask);
    }

    /// Update fill/cancel/aggression rates
    #[inline]
    pub fn update_rates(
        &self,
        side: Side,
        fill_rate: f64,
        cancel_rate: f64,
        aggression_rate: f64,
    ) {
        match side {
            Side::Bid => {
                self.bid_queue.update_fill_rate(fill_rate);
                self.bid_queue.update_cancel_rate(cancel_rate);
                self.bid_queue.update_aggression_rate(aggression_rate);
            }
            Side::Ask => {
                self.ask_queue.update_fill_rate(fill_rate);
                self.ask_queue.update_cancel_rate(cancel_rate);
                self.ask_queue.update_aggression_rate(aggression_rate);
            }
        }
        self.update_estimates(side);
    }

    /// Calculate estimated time to fill and probability
    #[inline]
    fn update_estimates(&self, side: Side) {
        let queue = match side {
            Side::Bid => &self.bid_queue,
            Side::Ask => &self.ask_queue,
        };

        let volume_ahead = queue.volume_ahead.load(Ordering::Relaxed);
        let fill_rate = queue.fill_rate.load(Ordering::Relaxed);
        let cancel_rate = queue.cancel_rate.load(Ordering::Relaxed);
        let aggression_rate = queue.aggression_rate.load(Ordering::Relaxed);

        // Net consumption rate = fills + cancels ahead + market aggression
        // Cancellations ahead help us move forward in queue
        let net_consumption = fill_rate + cancel_rate * 0.5 + aggression_rate;

        // Estimated time to fill (ms)
        let est_ms = if net_consumption > 0.0 {
            (volume_ahead / net_consumption) * 1000.0
        } else {
            f64::MAX
        };

        // Fill probability based on position and rates
        // Higher cancel rate = better chance (people ahead cancel)
        // Higher aggression = better chance (market orders eat queue)
        let position = queue.our_position.load(Ordering::Relaxed);
        let base_prob = 1.0 - position; // Closer to front = higher probability
        
        // Adjust for rate factors
        let rate_factor = (fill_rate + aggression_rate) / (fill_rate + aggression_rate + 1.0);
        let cancel_factor = (cancel_rate / (cancel_rate + 1.0)) * 0.3; // Cancels help somewhat
        
        let prob = (base_prob * rate_factor + cancel_factor).clamp(0.0, 1.0);

        match side {
            Side::Bid => {
                self.estimated_fill_ms.store(est_ms.min(300000.0), Ordering::Relaxed); // Cap at 5 min
                self.fill_probability.store(prob, Ordering::Relaxed);
            }
            Side::Ask => {
                // For simplicity, store ask estimates separately if needed
                // Currently using shared fields - in production would split
            }
        }

        self.last_update_ns.store(
            SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos() as u64,
            Ordering::Relaxed,
        );
    }

    /// Get estimated fill time in milliseconds
    #[inline]
    pub fn get_estimated_fill_ms(&self) -> f64 {
        self.estimated_fill_ms.load(Ordering::Relaxed)
    }

    /// Get fill probability (0-1)
    #[inline]
    pub fn get_fill_probability(&self) -> f64 {
        self.fill_probability.load(Ordering::Relaxed)
    }

    /// Check if order is likely to fill within timeout
    #[inline]
    pub fn likely_to_fill(&self, timeout_ms: f64) -> bool {
        let est = self.get_estimated_fill_ms();
        let prob = self.get_fill_probability();
        est < timeout_ms && prob > 0.3
    }

    /// Get queue position as fraction (0 = front, 1 = back)
    #[inline]
    pub fn get_queue_position(&self, side: Side) -> f64 {
        match side {
            Side::Bid => self.bid_queue.our_position.load(Ordering::Relaxed),
            Side::Ask => self.ask_queue.our_position.load(Ordering::Relaxed),
        }
    }

    /// Reset for new order
    #[inline]
    pub fn reset(&self, side: Side) {
        let now_ns = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64;
        
        self.order_time_ns.store(now_ns, Ordering::Relaxed);
        self.estimated_fill_ms.store(f64::MAX, Ordering::Relaxed);
        self.fill_probability.store(0.0, Ordering::Relaxed);
        
        match side {
            Side::Bid => {
                self.bid_queue.volume_ahead.store(0.0, Ordering::Relaxed);
                self.bid_queue.our_position.store(0.0, Ordering::Relaxed);
            }
            Side::Ask => {
                self.ask_queue.volume_ahead.store(0.0, Ordering::Relaxed);
                self.ask_queue.our_position.store(0.0, Ordering::Relaxed);
            }
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Side {
    Bid,
    Ask,
}

/// Queue analysis result
#[derive(Clone, Copy, Debug)]
pub struct QueueAnalysis {
    pub queue_position: f64,
    pub volume_ahead: f64,
    pub estimated_fill_ms: f64,
    pub fill_probability: f64,
    pub recommended_action: QueueAction,
    pub timestamp_ns: u64,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum QueueAction {
    Hold,          // Keep order in queue
    Cancel,        // Cancel and rebook
    Aggress,       // Hit the other side
    AdjustPrice,   // Move price to improve position
}

impl QueueAnalysis {
    pub fn from_estimator(estimator: &QueuePositionEstimator, side: Side) -> Self {
        let position = estimator.get_queue_position(side);
        let est_ms = estimator.get_estimated_fill_ms();
        let prob = estimator.get_fill_probability();
        
        let action = if prob < 0.2 && est_ms > 10000.0 {
            QueueAction::Cancel
        } else if prob > 0.8 && est_ms < 100.0 {
            QueueAction::Hold
        } else if position > 0.7 {
            QueueAction::AdjustPrice
        } else {
            QueueAction::Hold
        };

        Self {
            queue_position: position,
            volume_ahead: match side {
                Side::Bid => estimator.bid_queue.volume_ahead.load(Ordering::Relaxed),
                Side::Ask => estimator.ask_queue.volume_ahead.load(Ordering::Relaxed),
            },
            estimated_fill_ms: est_ms,
            fill_probability: prob,
            recommended_action: action,
            timestamp_ns: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos() as u64,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_queue_position_basic() {
        let estimator = QueuePositionEstimator::new();
        
        // Simulate being halfway through queue
        estimator.update_bid_queue(1000.0, 500.0, 100.0);
        estimator.update_rates(Side::Bid, 50.0, 20.0, 30.0);
        
        let position = estimator.get_queue_position(Side::Bid);
        assert!((position - 0.5).abs() < 0.01);
        
        let prob = estimator.get_fill_probability();
        assert!(prob > 0.0 && prob <= 1.0);
    }

    #[test]
    fn test_high_fill_probability() {
        let estimator = QueuePositionEstimator::new();
        
        // Near front of queue with high activity
        estimator.update_bid_queue(1000.0, 50.0, 100.0);
        estimator.update_rates(Side::Bid, 100.0, 50.0, 80.0);
        
        let prob = estimator.get_fill_probability();
        assert!(prob > 0.7); // Should be high probability
        
        let est_ms = estimator.get_estimated_fill_ms();
        assert!(est_ms < 2000.0); // Should fill soon
    }

    #[test]
    fn test_low_fill_probability() {
        let estimator = QueuePositionEstimator::new();
        
        // Back of queue with low activity
        estimator.update_bid_queue(1000.0, 900.0, 100.0);
        estimator.update_rates(Side::Bid, 1.0, 0.5, 0.5);
        
        let prob = estimator.get_fill_probability();
        assert!(prob < 0.3); // Low probability
        
        let est_ms = estimator.get_estimated_fill_ms();
        assert!(est_ms > 50000.0); // Long wait expected
    }

    #[test]
    fn test_queue_analysis_recommendation() {
        let estimator = QueuePositionEstimator::new();
        
        // Bad position scenario
        estimator.update_bid_queue(1000.0, 950.0, 50.0);
        estimator.update_rates(Side::Bid, 0.5, 0.1, 0.1);
        
        let analysis = QueueAnalysis::from_estimator(&estimator, Side::Bid);
        assert_eq!(analysis.recommended_action, QueueAction::Cancel);
    }
}
