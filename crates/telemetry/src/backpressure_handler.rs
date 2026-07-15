//! Critical backpressure handler for telemetry streaming.
//! Aggressively drops non-critical UI packets when frontend lags or buffers fill,
//! ensuring the live trading engine NEVER experiences latency spikes or memory bloat.

use std::sync::atomic::{AtomicUsize, AtomicBool, Ordering};
use std::time::{Duration, Instant};

/// Priority levels for telemetry messages
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum TelemetryPriority {
    Critical = 0,    // Never drop (PnL, positions, errors)
    High = 1,        // Drop only under extreme pressure (orders, fills)
    Medium = 2,      // Drop under moderate pressure (ticks, book updates)
    Low = 3,         // First to drop (heatmaps, analytics, decorative)
}

/// Backpressure configuration with strict thresholds
pub struct BackpressureConfig {
    /// Memory usage threshold (bytes) to trigger mild backpressure
    pub mild_threshold_bytes: usize,
    /// Memory usage threshold to trigger aggressive backpressure
    pub aggressive_threshold_bytes: usize,
    /// Maximum queue depth before dropping starts
    pub max_queue_depth: usize,
    /// Minimum interval between critical messages (ns)
    pub critical_min_interval_ns: u64,
}

impl Default for BackpressureConfig {
    fn default() -> Self {
        Self {
            mild_threshold_bytes: 50 * 1024 * 1024,      // 50MB
            aggressive_threshold_bytes: 100 * 1024 * 1024, // 100MB
            max_queue_depth: 2048,
            critical_min_interval_ns: 1_000_000, // 1ms minimum between critical msgs
        }
    }
}

/// State of the backpressure system
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackpressureState {
    Normal,
    Mild,
    Aggressive,
    Critical,
}

/// Backpressure handler with atomic state management
pub struct BackpressureHandler {
    config: BackpressureConfig,
    current_state: AtomicUsize, // BackpressureState as usize
    dropped_low: AtomicUsize,
    dropped_medium: AtomicUsize,
    dropped_high: AtomicUsize,
    last_critical_send: AtomicUsize, // Timestamp in ns
    is_active: AtomicBool,
}

impl BackpressureHandler {
    /// Create a new backpressure handler
    pub fn new(config: BackpressureConfig) -> Self {
        Self {
            config,
            current_state: AtomicUsize::new(BackpressureState::Normal as usize),
            dropped_low: AtomicUsize::new(0),
            dropped_medium: AtomicUsize::new(0),
            dropped_high: AtomicUsize::new(0),
            last_critical_send: AtomicUsize::new(0),
            is_active: AtomicBool::new(true),
        }
    }

    /// Update backpressure state based on current queue depth and memory usage
    pub fn update_state(&self, queue_depth: usize, memory_bytes: usize) {
        let new_state = if memory_bytes >= self.config.aggressive_threshold_bytes 
            || queue_depth >= self.config.max_queue_depth 
        {
            BackpressureState::Aggressive
        } else if memory_bytes >= self.config.mild_threshold_bytes 
            || queue_depth >= self.config.max_queue_depth / 2 
        {
            BackpressureState::Mild
        } else {
            BackpressureState::Normal
        };

        self.current_state.store(new_state as usize, Ordering::Relaxed);
        
        if !matches!(new_state, BackpressureState::Normal) {
            self.is_active.store(true, Ordering::Relaxed);
        }
    }

    /// Check if a message should be sent based on its priority and current state
    pub fn should_send(&self, priority: TelemetryPriority) -> bool {
        if !self.is_active.load(Ordering::Relaxed) {
            return true;
        }

        let state = unsafe { std::mem::transmute::<usize, BackpressureState>(
            self.current_state.load(Ordering::Relaxed)
        )};

        match state {
            BackpressureState::Normal => true,
            BackpressureState::Mild => priority <= TelemetryPriority::Medium,
            BackpressureState::Aggressive => priority <= TelemetryPriority::High,
            BackpressureState::Critical => priority == TelemetryPriority::Critical,
        }
    }

    /// Record a dropped message for metrics
    pub fn record_drop(&self, priority: TelemetryPriority) {
        match priority {
            TelemetryPriority::Low => {
                self.dropped_low.fetch_add(1, Ordering::Relaxed);
            }
            TelemetryPriority::Medium => {
                self.dropped_medium.fetch_add(1, Ordering::Relaxed);
            }
            TelemetryPriority::High => {
                self.dropped_high.fetch_add(1, Ordering::Relaxed);
            }
            TelemetryPriority::Critical => {
                // Critical messages should never be dropped - log warning
                tracing::warn!("CRITICAL: Critical telemetry message was dropped!");
            }
        }
    }

    /// Check rate limit for critical messages
    pub fn check_critical_rate_limit(&self) -> bool {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos() as usize;
        
        let last = self.last_critical_send.load(Ordering::Relaxed);
        
        if now - last < self.config.critical_min_interval_ns as usize {
            return false;
        }

        self.last_critical_send.store(now, Ordering::Relaxed);
        true
    }

    /// Get current backpressure state
    pub fn state(&self) -> BackpressureState {
        unsafe { std::mem::transmute::<usize, BackpressureState>(
            self.current_state.load(Ordering::Relaxed)
        )}
    }

    /// Get drop statistics
    pub fn get_drop_stats(&self) -> (usize, usize, usize) {
        (
            self.dropped_low.load(Ordering::Relaxed),
            self.dropped_medium.load(Ordering::Relaxed),
            self.dropped_high.load(Ordering::Relaxed),
        )
    }

    /// Reset backpressure state (call when conditions normalize)
    pub fn reset(&self) {
        self.current_state.store(BackpressureState::Normal as usize, Ordering::Relaxed);
        self.is_active.store(false, Ordering::Relaxed);
    }

    /// Check if system is currently under backpressure
    pub fn is_under_pressure(&self) -> bool {
        self.state() != BackpressureState::Normal
    }
}

/// Helper function to determine message priority based on type
pub fn classify_message_priority(msg_type: &str) -> TelemetryPriority {
    match msg_type {
        "error" | "liquidation_warning" | "margin_call" | "pnl_update" | "position_update" => {
            TelemetryPriority::Critical
        }
        "order_fill" | "order_cancel" | "order_new" | "execution_report" => {
            TelemetryPriority::High
        }
        "tick" | "book_update" | "trade" | "spread_update" => {
            TelemetryPriority::Medium
        }
        "heatmap" | "analytics" | "footprint" | "decorative" => {
            TelemetryPriority::Low
        }
        _ => TelemetryPriority::Medium,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_backpressure_state_transitions() {
        let config = BackpressureConfig::default();
        let handler = BackpressureHandler::new(config);

        assert_eq!(handler.state(), BackpressureState::Normal);

        // Trigger mild backpressure
        handler.update_state(1024, 60 * 1024 * 1024);
        assert_eq!(handler.state(), BackpressureState::Mild);

        // Trigger aggressive backpressure
        handler.update_state(3000, 150 * 1024 * 1024);
        assert_eq!(handler.state(), BackpressureState::Aggressive);

        // Reset
        handler.update_state(100, 10 * 1024 * 1024);
        assert_eq!(handler.state(), BackpressureState::Normal);
    }

    #[test]
    fn test_should_send_logic() {
        let config = BackpressureConfig::default();
        let handler = BackpressureHandler::new(config);

        // Normal state - all priorities allowed
        handler.update_state(100, 10 * 1024 * 1024);
        assert!(handler.should_send(TelemetryPriority::Critical));
        assert!(handler.should_send(TelemetryPriority::High));
        assert!(handler.should_send(TelemetryPriority::Medium));
        assert!(handler.should_send(TelemetryPriority::Low));

        // Mild state - Low dropped
        handler.update_state(1024, 60 * 1024 * 1024);
        assert!(handler.should_send(TelemetryPriority::Critical));
        assert!(handler.should_send(TelemetryPriority::High));
        assert!(handler.should_send(TelemetryPriority::Medium));
        assert!(!handler.should_send(TelemetryPriority::Low));

        // Aggressive state - Medium and Low dropped
        handler.update_state(3000, 150 * 1024 * 1024);
        assert!(handler.should_send(TelemetryPriority::Critical));
        assert!(handler.should_send(TelemetryPriority::High));
        assert!(!handler.should_send(TelemetryPriority::Medium));
        assert!(!handler.should_send(TelemetryPriority::Low));
    }

    #[test]
    fn test_priority_classification() {
        assert_eq!(classify_message_priority("error"), TelemetryPriority::Critical);
        assert_eq!(classify_message_priority("order_fill"), TelemetryPriority::High);
        assert_eq!(classify_message_priority("tick"), TelemetryPriority::Medium);
        assert_eq!(classify_message_priority("heatmap"), TelemetryPriority::Low);
    }
}
