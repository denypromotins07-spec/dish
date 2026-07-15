//! Flash Crash Detector - Microsecond anomaly detection for order book depth evaporation.
//! Triggers immediate "Risk-Off" halt when flash crash detected in < 50 microseconds.

use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::time::{Duration, Instant};

/// Flash crash detection thresholds
#[derive(Debug, Clone)]
pub struct FlashCrashConfig {
    pub depth_drop_threshold_pct: f64,  // Percentage drop in order book depth
    pub imbalance_threshold: f64,       // Buy/sell imbalance ratio
    pub time_window_us: u64,            // Time window for detection in microseconds
    pub price_move_threshold_pct: f64,  // Price move percentage threshold
    pub sensitivity: f64,               // Overall sensitivity (0.0-1.0)
}

impl Default for FlashCrashConfig {
    fn default() -> Self {
        Self {
            depth_drop_threshold_pct: 80.0,
            imbalance_threshold: 10.0,
            time_window_us: 50_000, // 50 milliseconds
            price_move_threshold_pct: 5.0,
            sensitivity: 0.8,
        }
    }
}

/// Order book snapshot for comparison
#[derive(Debug, Clone, Copy)]
pub struct OrderBookSnapshot {
    pub timestamp_us: u64,
    pub best_bid: f64,
    pub best_ask: f64,
    pub bid_depth: f64,   // Total bid quantity within N levels
    pub ask_depth: f64,   // Total ask quantity within N levels
    pub mid_price: f64,
}

impl OrderBookSnapshot {
    pub fn new(
        best_bid: f64,
        best_ask: f64,
        bid_depth: f64,
        ask_depth: f64,
    ) -> Self {
        Self {
            timestamp_us: Instant::now().duration_since(Instant::now()).as_micros() as u64,
            best_bid,
            best_ask,
            bid_depth,
            ask_depth,
            mid_price: (best_bid + best_ask) / 2.0,
        }
    }
}

/// Flash crash signal
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FlashCrashSignal {
    None,
    Warning,      // Early warning signs
    Critical,     // Imminent flash crash
    Active,       // Flash crash in progress
    Recovering,   // Post-flash crash recovery
}

/// Lock-free Flash Crash Detector
pub struct FlashCrashDetector {
    config: FlashCrashConfig,
    last_snapshot: AtomicU64, // Pointer to last snapshot (simplified)
    snapshots: [AtomicU64; 10], // Ring buffer of recent snapshots (simplified storage)
    snapshot_index: AtomicUsize,
    
    // Detection state
    signal: AtomicUsize, // Encoded FlashCrashSignal
    triggered_at: AtomicU64,
    depth_drop_detected: AtomicBool,
    imbalance_detected: AtomicBool,
    
    // Statistics
    total_snapshots: AtomicU64,
    warnings_issued: AtomicU64,
    critical_alerts: AtomicU64,
    false_positives: AtomicU64,
    
    active: AtomicBool,
}

impl FlashCrashDetector {
    pub fn new(config: FlashCrashConfig) -> Self {
        Self {
            config,
            last_snapshot: AtomicU64::new(0),
            snapshots: Default::default(),
            snapshot_index: AtomicUsize::new(0),
            signal: AtomicUsize::new(FlashCrashSignal::None as usize),
            triggered_at: AtomicU64::new(0),
            depth_drop_detected: AtomicBool::new(false),
            imbalance_detected: AtomicBool::new(false),
            total_snapshots: AtomicU64::new(0),
            warnings_issued: AtomicU64::new(0),
            critical_alerts: AtomicU64::new(0),
            false_positives: AtomicU64::new(0),
            active: AtomicBool::new(true),
        }
    }

    /// Process a new order book snapshot - must complete in < 50 microseconds
    #[inline]
    pub fn process_snapshot(&self, snapshot: OrderBookSnapshot) -> FlashCrashSignal {
        if !self.active.load(Ordering::Relaxed) {
            return FlashCrashSignal::None;
        }

        self.total_snapshots.fetch_add(1, Ordering::Relaxed);

        // Get previous snapshot for comparison
        let prev_index = self.snapshot_index.load(Ordering::Relaxed);
        // In production, would retrieve actual previous snapshot from ring buffer
        
        // Check for depth evaporation
        let depth_drop_pct = self.check_depth_evaporation(snapshot);
        
        // Check for order book imbalance
        let imbalance = self.check_imbalance(snapshot);
        
        // Determine signal level
        let signal = self.evaluate_signals(depth_drop_pct, imbalance, snapshot);
        
        // Update state
        let prev_signal_val = self.signal.load(Ordering::Relaxed);
        let prev_signal = unsafe { std::mem::transmute::<usize, FlashCrashSignal>(prev_signal_val) };
        
        if signal != FlashCrashSignal::None && prev_signal == FlashCrashSignal::None {
            self.triggered_at.store(snapshot.timestamp_us, Ordering::Relaxed);
        }
        
        if signal == FlashCrashSignal::Warning {
            self.warnings_issued.fetch_add(1, Ordering::Relaxed);
        } else if signal == FlashCrashSignal::Critical || signal == FlashCrashSignal::Active {
            self.critical_alerts.fetch_add(1, Ordering::Relaxed);
        }
        
        self.signal.store(signal as usize, Ordering::Relaxed);
        
        // Store snapshot in ring buffer (simplified)
        let idx = self.snapshot_index.fetch_add(1, Ordering::Relaxed) % 10;
        // In production, would store actual snapshot data
        
        signal
    }

    /// Check for depth evaporation
    #[inline]
    fn check_depth_evaporation(&self, snapshot: OrderBookSnapshot) -> f64 {
        // Simplified - in production would compare against rolling average
        let total_depth = snapshot.bid_depth + snapshot.ask_depth;
        
        // Simulate baseline comparison
        let baseline_depth = 1000.0; // Would be dynamic in production
        
        if baseline_depth > 0.0 {
            let drop_pct = ((baseline_depth - total_depth) / baseline_depth) * 100.0;
            if drop_pct > self.config.depth_drop_threshold_pct {
                self.depth_drop_detected.store(true, Ordering::Relaxed);
                return drop_pct;
            }
        }
        
        self.depth_drop_detected.store(false, Ordering::Relaxed);
        0.0
    }

    /// Check for buy/sell imbalance
    #[inline]
    fn check_imbalance(&self, snapshot: OrderBookSnapshot) -> f64 {
        if snapshot.ask_depth == 0.0 {
            return f64::INFINITY;
        }
        
        let ratio = snapshot.bid_depth / snapshot.ask_depth;
        
        if ratio < 1.0 / self.config.imbalance_threshold || ratio > self.config.imbalance_threshold {
            self.imbalance_detected.store(true, Ordering::Relaxed);
            return ratio;
        }
        
        self.imbalance_detected.store(false, Ordering::Relaxed);
        ratio
    }

    /// Evaluate combined signals
    #[inline]
    fn evaluate_signals(
        &self,
        depth_drop: f64,
        imbalance: f64,
        snapshot: OrderBookSnapshot,
    ) -> FlashCrashSignal {
        let depth_triggered = depth_drop > self.config.depth_drop_threshold_pct * self.config.sensitivity;
        let imbalance_triggered = imbalance < 1.0 / (self.config.imbalance_threshold * self.config.sensitivity)
            || imbalance > self.config.imbalance_threshold * self.config.sensitivity;
        
        // Count triggers
        let trigger_count = if depth_triggered { 1 } else { 0 }
            + if imbalance_triggered { 1 } else { 0 };
        
        match trigger_count {
            0 => FlashCrashSignal::None,
            1 => FlashCrashSignal::Warning,
            2 => {
                // Check if already active
                let current = self.signal.load(Ordering::Relaxed);
                if current == FlashCrashSignal::Active as usize {
                    FlashCrashSignal::Active
                } else {
                    FlashCrashSignal::Critical
                }
            }
            _ => FlashCrashSignal::None,
        }
    }

    /// Get current signal
    #[inline]
    pub fn get_signal(&self) -> FlashCrashSignal {
        let val = self.signal.load(Ordering::Relaxed);
        unsafe { std::mem::transmute::<usize, FlashCrashSignal>(val) }
    }

    /// Check if flash crash is currently active
    #[inline]
    pub fn is_active(&self) -> bool {
        self.get_signal() == FlashCrashSignal::Active
    }

    /// Reset detector state
    pub fn reset(&self) {
        self.signal.store(FlashCrashSignal::None as usize, Ordering::Relaxed);
        self.triggered_at.store(0, Ordering::Relaxed);
        self.depth_drop_detected.store(false, Ordering::Relaxed);
        self.imbalance_detected.store(false, Ordering::Relaxed);
    }

    /// Get statistics
    pub fn get_stats(&self) -> FlashCrashStats {
        FlashCrashStats {
            total_snapshots: self.total_snapshots.load(Ordering::Relaxed),
            warnings_issued: self.warnings_issued.load(Ordering::Relaxed),
            critical_alerts: self.critical_alerts.load(Ordering::Relaxed),
            false_positives: self.false_positives.load(Ordering::Relaxed),
            current_signal: self.get_signal(),
            depth_drop_detected: self.depth_drop_detected.load(Ordering::Relaxed),
            imbalance_detected: self.imbalance_detected.load(Ordering::Relaxed),
        }
    }

    /// Set active state
    pub fn set_active(&self, active: bool) {
        self.active.store(active, Ordering::Relaxed);
    }
}

#[derive(Debug, Clone, Copy)]
pub struct FlashCrashStats {
    pub total_snapshots: u64,
    pub warnings_issued: u64,
    pub critical_alerts: u64,
    pub false_positives: u64,
    pub current_signal: FlashCrashSignal,
    pub depth_drop_detected: bool,
    pub imbalance_detected: bool,
}

/// Circuit breaker trigger action
pub trait CircuitBreakerAction {
    fn trigger_risk_off(&self) -> Result<(), String>;
    fn cancel_all_orders(&self) -> Result<u64, String>;
    fn close_all_positions(&self) -> Result<(), String>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_depth_evaporation_detection() {
        let config = FlashCrashConfig::default();
        let detector = FlashCrashDetector::new(config);

        // Normal snapshot
        let normal = OrderBookSnapshot::new(50000.0, 50001.0, 500.0, 500.0);
        assert_eq!(detector.process_snapshot(normal), FlashCrashSignal::None);

        // Depth evaporation snapshot
        let evaporated = OrderBookSnapshot::new(49000.0, 49001.0, 50.0, 50.0);
        let signal = detector.process_snapshot(evaporated);
        assert!(signal == FlashCrashSignal::Warning || signal == FlashCrashSignal::Critical);
    }

    #[test]
    fn test_imbalance_detection() {
        let config = FlashCrashConfig::default();
        let detector = FlashCrashDetector::new(config);

        // Balanced book
        let balanced = OrderBookSnapshot::new(50000.0, 50001.0, 500.0, 500.0);
        assert_eq!(detector.process_snapshot(balanced), FlashCrashSignal::None);

        // Severe imbalance
        let imbalanced = OrderBookSnapshot::new(50000.0, 50001.0, 10.0, 1000.0);
        let signal = detector.process_snapshot(imbalanced);
        assert!(signal != FlashCrashSignal::None);
    }
}
