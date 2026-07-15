//! Real-time Spoofing and Layering Detection Algorithm
//! Analyzes order book flicker and cancellation rates to detect fake liquidity

use std::sync::atomic::{AtomicF64, AtomicU64, AtomicBool, Ordering};
use std::time::{SystemTime, UNIX_EPOCH, Duration};
use std::collections::VecDeque;

/// Order book event for tracking
#[derive(Clone, Copy, Debug)]
pub struct OrderBookEvent {
    pub price: f64,
    pub size: f64,
    pub is_bid: bool,
    pub event_type: OrderBookEventType,
    pub timestamp_ns: u64,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum OrderBookEventType {
    Add,
    Modify,
    Cancel,
    Trade,
}

/// Spoofing detection metrics
pub struct SpoofingMetrics {
    /// Order add rate (per second)
    pub add_rate: AtomicF64,
    /// Order cancel rate (per second)
    pub cancel_rate: AtomicF64,
    /// Average order lifetime (ms)
    pub avg_lifetime_ms: AtomicF64,
    /// Cancellation ratio (cancels / adds)
    pub cancel_ratio: AtomicF64,
    /// Price flicker count
    pub flicker_count: AtomicU64,
    /// Large order ratio at key levels
    pub large_order_ratio: AtomicF64,
}

impl SpoofingMetrics {
    pub fn new() -> Self {
        Self {
            add_rate: AtomicF64::new(0.0),
            cancel_rate: AtomicF64::new(0.0),
            avg_lifetime_ms: AtomicF64::new(1000.0),
            cancel_ratio: AtomicF64::new(0.0),
            flicker_count: AtomicU64::new(0),
            large_order_ratio: AtomicF64::new(0.0),
        }
    }
}

impl Default for SpoofingMetrics {
    fn default() -> Self {
        Self::new()
    }
}

/// Active order tracking for lifetime analysis
struct ActiveOrder {
    price: f64,
    size: f64,
    is_bid: bool,
    created_ns: u64,
}

/// Spoofing detection engine
pub struct SpoofingDetector {
    /// Recent order book events
    pub events: VecDeque<OrderBookEvent>,
    /// Max events to track
    pub max_events: usize,
    /// Metrics
    pub metrics: SpoofingMetrics,
    /// Active orders being tracked
    active_orders: Vec<ActiveOrder>,
    /// Pending cancels (orders that appeared and disappeared quickly)
    pending_cancels: VecDeque<(f64, u64)>, // (price, timestamp)
    /// Spoofing detected flag
    pub spoofing_detected: AtomicBool,
    /// Last detection timestamp
    pub last_detection_ns: AtomicU64,
    /// Detection threshold (cancel ratio)
    pub cancel_ratio_threshold: AtomicF64,
    /// Flicker threshold (events per second)
    pub flicker_threshold: AtomicU64,
    /// Enabled flag
    pub enabled: AtomicBool,
}

impl SpoofingDetector {
    pub fn new(max_events: usize) -> Self {
        let now_ns = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64;
        
        Self {
            events: VecDeque::with_capacity(max_events),
            max_events,
            metrics: SpoofingMetrics::new(),
            active_orders: Vec::with_capacity(100),
            pending_cancels: VecDeque::with_capacity(50),
            spoofing_detected: AtomicBool::new(false),
            last_detection_ns: AtomicU64::new(0),
            cancel_ratio_threshold: AtomicF64::new(0.7),
            flicker_threshold: AtomicU64::new(100),
            enabled: AtomicBool::new(true),
        }
    }

    /// Process an order book event
    #[inline]
    pub fn process_event(&mut self, event: OrderBookEvent) -> SpoofingAlert {
        if !self.enabled.load(Ordering::Relaxed) {
            return SpoofingAlert::None;
        }

        self.events.push_back(event);
        if self.events.len() > self.max_events {
            self.events.pop_front();
        }

        match event.event_type {
            OrderBookEventType::Add => {
                self.active_orders.push(ActiveOrder {
                    price: event.price,
                    size: event.size,
                    is_bid: event.is_bid,
                    created_ns: event.timestamp_ns,
                });
            }
            OrderBookEventType::Cancel => {
                // Find matching order and calculate lifetime
                if let Some(pos) = self.active_orders.iter().position(|o| {
                    (o.price - event.price).abs() < 0.0001 && o.is_bid == event.is_bid
                }) {
                    let order = self.active_orders.remove(pos);
                    let lifetime_ms = (event.timestamp_ns - order.created_ns) as f64 / 1_000_000.0;
                    
                    // Track very short-lived orders as potential spoofing
                    if lifetime_ms < 100.0 {
                        self.pending_cancels.push_back((order.price, event.timestamp_ns));
                        self.metrics.flicker_count.fetch_add(1, Ordering::Relaxed);
                    }
                    
                    self.update_lifetime_stats(lifetime_ms);
                }
            }
            _ => {}
        }

        self.update_metrics();
        self.check_for_spoofing()
    }

    /// Update lifetime statistics
    #[inline]
    fn update_lifetime_stats(&self, lifetime_ms: f64) {
        let current = self.metrics.avg_lifetime_ms.load(Ordering::Relaxed);
        let alpha = 0.1; // EMA smoothing
        let updated = alpha * lifetime_ms + (1.0 - alpha) * current;
        self.metrics.avg_lifetime_ms.store(updated, Ordering::Relaxed);
    }

    /// Update rate metrics
    #[inline]
    fn update_metrics(&self) {
        if self.events.len() < 2 { return; }
        
        let first_ts = self.events.front().map(|e| e.timestamp_ns).unwrap_or(0);
        let last_ts = self.events.back().map(|e| e.timestamp_ns).unwrap_or(0);
        
        if last_ts <= first_ts { return; }
        
        let duration_sec = (last_ts - first_ts) as f64 / 1_000_000_000.0;
        if duration_sec < 0.1 { return; }
        
        let mut adds = 0u64;
        let mut cancels = 0u64;
        
        for event in &self.events {
            match event.event_type {
                OrderBookEventType::Add => adds += 1,
                OrderBookEventType::Cancel => cancels += 1,
                _ => {}
            }
        }
        
        let add_rate = adds as f64 / duration_sec;
        let cancel_rate = cancels as f64 / duration_sec;
        let cancel_ratio = if adds > 0 { cancels as f64 / adds as f64 } else { 0.0 };
        
        self.metrics.add_rate.store(add_rate, Ordering::Relaxed);
        self.metrics.cancel_rate.store(cancel_rate, Ordering::Relaxed);
        self.metrics.cancel_ratio.store(cancel_ratio, Ordering::Relaxed);
    }

    /// Check for spoofing patterns
    #[inline]
    fn check_for_spoofing(&self) -> SpoofingAlert {
        let now_ns = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64;
        
        let cancel_ratio = self.metrics.cancel_ratio.load(Ordering::Relaxed);
        let avg_lifetime = self.metrics.avg_lifetime_ms.load(Ordering::Relaxed);
        let flicker_count = self.metrics.flicker_count.load(Ordering::Relaxed);
        let threshold = self.cancel_ratio_threshold.load(Ordering::Relaxed);
        let flicker_thresh = self.flicker_threshold.load(Ordering::Relaxed);
        
        // Calculate recent flicker rate (last second)
        let one_sec_ago = now_ns.saturating_sub(1_000_000_000);
        let recent_flickers = self.pending_cancels.iter()
            .filter(|(_, ts)| *ts > one_sec_ago)
            .count() as u64;
        
        let mut alert_level = 0;
        
        // High cancel ratio
        if cancel_ratio > threshold {
            alert_level += 1;
        }
        
        // Very short average lifetime
        if avg_lifetime < 50.0 {
            alert_level += 1;
        }
        
        // High flicker rate
        if recent_flickers > flicker_thresh {
            alert_level += 1;
        }
        
        let alert = match alert_level {
            3 => {
                self.spoofing_detected.store(true, Ordering::Relaxed);
                self.last_detection_ns.store(now_ns, Ordering::Relaxed);
                SpoofingAlert::HighConfidence
            }
            2 => {
                self.spoofing_detected.store(true, Ordering::Relaxed);
                self.last_detection_ns.store(now_ns, Ordering::Relaxed);
                SpoofingAlert::MediumConfidence
            }
            1 => SpoofingAlert::LowConfidence,
            _ => {
                self.spoofing_detected.store(false, Ordering::Relaxed);
                SpoofingAlert::None
            }
        };
        
        // Clean old pending cancels
        while let Some((_, ts)) = self.pending_cancels.front() {
            if *ts < one_sec_ago {
                self.pending_cancels.pop_front();
            } else {
                break;
            }
        }
        
        alert
    }

    /// Get spoofing detection status
    #[inline]
    pub fn is_spoofing_detected(&self) -> bool {
        self.spoofing_detected.load(Ordering::Relaxed)
    }

    /// Get recommended action based on detection
    #[inline]
    pub fn get_recommended_action(&self) -> SpoofingAction {
        if !self.spoofing_detected.load(Ordering::Relaxed) {
            return SpoofingAction::Continue;
        }
        
        let now_ns = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64;
        let last_ns = self.last_detection_ns.load(Ordering::Relaxed);
        let elapsed_ms = (now_ns - last_ns) as f64 / 1_000_000.0;
        
        if elapsed_ms < 500.0 {
            SpoofingAction::HaltQuoting
        } else if elapsed_ms < 2000.0 {
            SpoofingAction::WidenSpreads
        } else {
            SpoofingAction::ReduceSize
        }
    }

    /// Reset detector state
    #[inline]
    pub fn reset(&mut self) {
        self.events.clear();
        self.active_orders.clear();
        self.pending_cancels.clear();
        self.spoofing_detected.store(false, Ordering::Relaxed);
        self.metrics = SpoofingMetrics::new();
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum SpoofingAlert {
    None,
    LowConfidence,
    MediumConfidence,
    HighConfidence,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum SpoofingAction {
    Continue,
    WidenSpreads,
    ReduceSize,
    HaltQuoting,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_spoofing_detection() {
        let mut detector = SpoofingDetector::new(100);
        
        let now_ns = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64;
        
        // Simulate spoofing pattern: many adds followed by quick cancels
        for i in 0..20 {
            let add_event = OrderBookEvent {
                price: 100.0 + (i as f64 * 0.01),
                size: 100.0,
                is_bid: true,
                event_type: OrderBookEventType::Add,
                timestamp_ns: now_ns + i * 1_000_000,
            };
            detector.process_event(add_event);
        }
        
        // Quick cancels
        for i in 0..18 {
            let cancel_event = OrderBookEvent {
                price: 100.0 + (i as f64 * 0.01),
                size: 100.0,
                is_bid: true,
                event_type: OrderBookEventType::Cancel,
                timestamp_ns: now_ns + 25_000_000 + i * 1_000_000,
            };
            let alert = detector.process_event(cancel_event);
            
            if i > 10 {
                assert!(alert != SpoofingAlert::None);
            }
        }
    }

    #[test]
    fn test_normal_market_behavior() {
        let mut detector = SpoofingDetector::new(100);
        
        let now_ns = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64;
        
        // Normal behavior: adds stay, occasional cancels
        for i in 0..10 {
            let add_event = OrderBookEvent {
                price: 100.0 + (i as f64 * 0.01),
                size: 50.0,
                is_bid: true,
                event_type: OrderBookEventType::Add,
                timestamp_ns: now_ns + i * 100_000_000,
            };
            detector.process_event(add_event);
        }
        
        // Only a few cancels after reasonable time
        for i in 0..2 {
            let cancel_event = OrderBookEvent {
                price: 100.0 + (i as f64 * 0.01),
                size: 50.0,
                is_bid: true,
                event_type: OrderBookEventType::Cancel,
                timestamp_ns: now_ns + 2_000_000_000 + i * 100_000_000,
            };
            detector.process_event(cancel_event);
        }
        
        assert!(!detector.is_spoofing_detected());
    }
}
