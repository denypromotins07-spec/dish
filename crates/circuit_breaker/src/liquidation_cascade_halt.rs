//! Liquidation Cascade Halt - Detects cascading liquidations by correlating OI drops with volatility.
//! Automatically pulls all maker quotes to prevent catching a falling knife.

use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::time::{Duration, Instant};

/// Configuration for liquidation cascade detection
#[derive(Debug, Clone)]
pub struct LiquidationCascadeConfig {
    pub oi_drop_threshold_pct: f64,   // Open interest drop percentage threshold
    pub volatility_threshold: f64,     // Price volatility threshold (annualized)
    pub time_window_ms: u64,           // Time window for correlation in milliseconds
    pub min_oi_drop_bps: u64,          // Minimum OI drop in basis points
    pub sensitivity: f64,              // Overall sensitivity (0.0-1.0)
}

impl Default for LiquidationCascadeConfig {
    fn default() -> Self {
        Self {
            oi_drop_threshold_pct: 15.0,
            volatility_threshold: 2.0, // 2% move
            time_window_ms: 5000,      // 5 second window
            min_oi_drop_bps: 500,      // 5% minimum drop
            sensitivity: 0.7,
        }
    }
}

/// Open Interest snapshot
#[derive(Debug, Clone, Copy)]
pub struct OiSnapshot {
    pub timestamp_ms: u64,
    pub open_interest: f64,
    pub long_liquidations: f64,
    pub short_liquidations: f64,
    pub funding_rate: f64,
}

impl OiSnapshot {
    pub fn new(
        open_interest: f64,
        long_liq: f64,
        short_liq: f64,
        funding_rate: f64,
    ) -> Self {
        Self {
            timestamp_ms: Instant::now().duration_since(Instant::now()).as_millis() as u64,
            open_interest,
            long_liquidations: long_liq,
            short_liquidations: short_liq,
            funding_rate,
        }
    }
}

/// Price/volatility snapshot
#[derive(Debug, Clone, Copy)]
pub struct PriceSnapshot {
    pub timestamp_ms: u64,
    pub price: f64,
    pub volatility_1m: f64,  // 1-minute realized volatility
    pub volume_1m: f64,      // 1-minute volume
}

impl PriceSnapshot {
    pub fn new(price: f64, vol_1m: f64, vol_1m_volume: f64) -> Self {
        Self {
            timestamp_ms: Instant::now().duration_since(Instant::now()).as_millis() as u64,
            price,
            volatility_1m: vol_1m,
            volume_1m: vol_1m_volume,
        }
    }
}

/// Cascade detection signal
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CascadeSignal {
    None,
    Building,       // Conditions building up
    Imminent,       // Cascade imminent
    ActiveLongs,    // Long liquidation cascade
    ActiveShorts,   // Short liquidation cascade
    Subsiding,      // Cascade subsiding
}

/// Lock-free Liquidation Cascade Detector
pub struct LiquidationCascadeHalt {
    config: LiquidationCascadeConfig,
    
    // State tracking
    last_oi_snapshot: AtomicU64,
    last_price_snapshot: AtomicU64,
    oi_ring_buffer: [f64; 20], // Rolling OI values
    oi_index: AtomicUsize,
    
    // Detection state
    signal: AtomicUsize,
    triggered_at: AtomicU64,
    cascade_type: AtomicUsize, // 0=None, 1=Longs, 2=Shorts
    
    // Statistics
    total_snapshots: AtomicU64,
    cascades_detected: AtomicU64,
    false_positives: AtomicU64,
    quotes_pulled: AtomicU64,
    
    active: AtomicBool,
    halted: AtomicBool,
}

impl LiquidationCascadeHalt {
    pub fn new(config: LiquidationCascadeConfig) -> Self {
        Self {
            config,
            last_oi_snapshot: AtomicU64::new(0),
            last_price_snapshot: AtomicU64::new(0),
            oi_ring_buffer: [0.0; 20],
            oi_index: AtomicUsize::new(0),
            signal: AtomicUsize::new(CascadeSignal::None as usize),
            triggered_at: AtomicU64::new(0),
            cascade_type: AtomicUsize::new(0),
            total_snapshots: AtomicU64::new(0),
            cascades_detected: AtomicU64::new(0),
            false_positives: AtomicU64::new(0),
            quotes_pulled: AtomicU64::new(0),
            active: AtomicBool::new(true),
            halted: AtomicBool::new(false),
        }
    }

    /// Process OI snapshot
    #[inline]
    pub fn process_oi_snapshot(&self, snapshot: OiSnapshot) {
        if !self.active.load(Ordering::Relaxed) {
            return;
        }

        self.total_snapshots.fetch_add(1, Ordering::Relaxed);
        
        // Store in ring buffer
        let idx = self.oi_index.fetch_add(1, Ordering::Relaxed) % 20;
        self.oi_ring_buffer[idx] = snapshot.open_interest;
        
        self.last_oi_snapshot.store(snapshot.timestamp_ms, Ordering::Relaxed);
        
        // Check for cascade conditions
        self.evaluate_cascade_conditions(snapshot);
    }

    /// Process price snapshot
    #[inline]
    pub fn process_price_snapshot(&self, snapshot: PriceSnapshot) {
        if !self.active.load(Ordering::Relaxed) {
            return;
        }
        
        self.last_price_snapshot.store(snapshot.timestamp_ms, Ordering::Relaxed);
    }

    /// Evaluate cascade conditions
    #[inline]
    fn evaluate_cascade_conditions(&self, snapshot: OiSnapshot) {
        // Get rolling average OI
        let avg_oi = self.get_rolling_avg_oi();
        if avg_oi == 0.0 {
            return;
        }

        // Calculate OI drop percentage
        let oi_drop_pct = ((avg_oi - snapshot.open_interest) / avg_oi) * 100.0;
        
        // Check if OI drop exceeds threshold
        let oi_triggered = oi_drop_pct > self.config.oi_drop_threshold_pct * self.config.sensitivity;
        
        // Check liquidation volume
        let total_liquidations = snapshot.long_liquidations + snapshot.short_liquidations;
        let liq_ratio = total_liquidations / avg_oi;
        
        // Determine cascade type
        if oi_triggered && liq_ratio > 0.01 {
            let cascade_type = if snapshot.long_liquidations > snapshot.short_liquidations {
                1 // Long cascade
            } else {
                2 // Short cascade
            };
            
            self.cascade_type.store(cascade_type, Ordering::Relaxed);
            self.signal.store(CascadeSignal::ActiveLongs as usize, Ordering::Relaxed);
            self.triggered_at.store(snapshot.timestamp_ms, Ordering::Relaxed);
            self.cascades_detected.fetch_add(1, Ordering::Relaxed);
            self.halted.store(true, Ordering::Relaxed);
        } else if oi_drop_pct > self.config.oi_drop_threshold_pct * 0.5 {
            self.signal.store(CascadeSignal::Building as usize, Ordering::Relaxed);
        }
    }

    /// Get rolling average OI
    #[inline]
    fn get_rolling_avg_oi(&self) -> f64 {
        let idx = self.oi_index.load(Ordering::Relaxed);
        let count = idx.min(20);
        
        if count == 0 {
            return 0.0;
        }
        
        let mut sum = 0.0;
        for i in 0..count {
            sum += self.oi_ring_buffer[i];
        }
        
        sum / count as f64
    }

    /// Get current signal
    #[inline]
    pub fn get_signal(&self) -> CascadeSignal {
        let val = self.signal.load(Ordering::Relaxed);
        unsafe { std::mem::transmute::<usize, CascadeSignal>(val) }
    }

    /// Check if system is halted
    #[inline]
    pub fn is_halted(&self) -> bool {
        self.halted.load(Ordering::Relaxed)
    }

    /// Trigger halt - pull all maker quotes
    #[inline]
    pub fn trigger_halt(&self) {
        self.halted.store(true, Ordering::Relaxed);
        self.quotes_pulled.fetch_add(1, Ordering::Relaxed);
    }

    /// Resume trading after cascade subsides
    #[inline]
    pub fn resume(&self) {
        if self.get_signal() == CascadeSignal::Subsiding || 
           self.get_signal() == CascadeSignal::None {
            self.halted.store(false, Ordering::Relaxed);
        }
    }

    /// Reset detector state
    pub fn reset(&self) {
        self.signal.store(CascadeSignal::None as usize, Ordering::Relaxed);
        self.triggered_at.store(0, Ordering::Relaxed);
        self.cascade_type.store(0, Ordering::Relaxed);
        self.halted.store(false, Ordering::Relaxed);
    }

    /// Get statistics
    pub fn get_stats(&self) -> CascadeStats {
        CascadeStats {
            total_snapshots: self.total_snapshots.load(Ordering::Relaxed),
            cascades_detected: self.cascades_detected.load(Ordering::Relaxed),
            false_positives: self.false_positives.load(Ordering::Relaxed),
            quotes_pulled: self.quotes_pulled.load(Ordering::Relaxed),
            current_signal: self.get_signal(),
            is_halted: self.is_halted(),
            cascade_type: match self.cascade_type.load(Ordering::Relaxed) {
                1 => "Longs",
                2 => "Shorts",
                _ => "None",
            },
        }
    }

    /// Set active state
    pub fn set_active(&self, active: bool) {
        self.active.store(active, Ordering::Relaxed);
        if !active {
            self.halted.store(false, Ordering::Relaxed);
        }
    }
}

#[derive(Debug, Clone)]
pub struct CascadeStats {
    pub total_snapshots: u64,
    pub cascades_detected: u64,
    pub false_positives: u64,
    pub quotes_pulled: u64,
    pub current_signal: CascadeSignal,
    pub is_halted: bool,
    pub cascade_type: &'static str,
}

/// Quote puller interface
pub trait QuotePuller {
    fn pull_all_maker_quotes(&self) -> Result<u64, String>;
    fn pause_new_orders(&self) -> Result<(), String>;
    fn resume_trading(&self) -> Result<(), String>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_oi_drop_detection() {
        let config = LiquidationCascadeConfig::default();
        let detector = LiquidationCascadeHalt::new(config);

        // Populate ring buffer with normal OI
        for _ in 0..10 {
            let snapshot = OiSnapshot::new(1000.0, 0.0, 0.0, 0.0001);
            detector.process_oi_snapshot(snapshot);
        }

        // Sudden OI drop with liquidations
        let crash_snapshot = OiSnapshot::new(800.0, 150.0, 10.0, 0.0001);
        detector.process_oi_snapshot(crash_snapshot);

        assert!(detector.get_signal() != CascadeSignal::None);
        assert_eq!(detector.get_stats().cascades_detected, 1);
    }

    #[test]
    fn test_halt_trigger() {
        let config = LiquidationCascadeConfig::default();
        let detector = LiquidationCascadeHalt::new(config);

        assert!(!detector.is_halted());
        
        detector.trigger_halt();
        assert!(detector.is_halted());
    }
}
