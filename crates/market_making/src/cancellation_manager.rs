//! Aggressive, zero-latency order cancellation logic.
//! Instantly pulls quotes via the fastest available API route if toxic flow or latency arbitrage is detected.
//! Optimized for AMD Ryzen AI 5 with lock-free atomic operations.

use std::sync::atomic::{AtomicU64, AtomicBool, AtomicUsize, Ordering};
use std::time::Instant;

/// Cancellation reason codes
#[repr(u8)]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum CancellationReason {
    ToxicFlow = 0,
    LatencyArbitrage = 1,
    InventoryLimit = 2,
    SpreadViolation = 3,
    ManualOverride = 4,
    SystemHalt = 5,
}

/// Cancellation statistics
#[repr(C, align(64))]
pub struct CancellationStats {
    /// Total cancellations sent
    total_cancellations: AtomicU64,
    /// Cancellations due to toxic flow
    toxic_flow_cancels: AtomicU64,
    /// Cancellations due to latency arb
    latency_arb_cancels: AtomicU64,
    /// Average cancellation latency (nanoseconds)
    avg_latency_ns: AtomicU64,
    /// Last cancellation timestamp
    last_cancel_ns: AtomicU64,
    _padding: [u8; 24],
}

impl CancellationStats {
    pub fn new() -> Self {
        Self {
            total_cancellations: AtomicU64::new(0),
            toxic_flow_cancels: AtomicU64::new(0),
            latency_arb_cancels: AtomicU64::new(0),
            avg_latency_ns: AtomicU64::new(0),
            last_cancel_ns: AtomicU64::new(0),
            _padding: [0u8; 24],
        }
    }
    
    #[inline]
    pub fn record_cancel(&self, reason: CancellationReason, latency_ns: u64) {
        self.total_cancellations.fetch_add(1, Ordering::Relaxed);
        
        match reason {
            CancellationReason::ToxicFlow => {
                self.toxic_flow_cancels.fetch_add(1, Ordering::Relaxed);
            }
            CancellationReason::LatencyArbitrage => {
                self.latency_arb_cancels.fetch_add(1, Ordering::Relaxed);
            }
            _ => {}
        }
        
        // Update running average latency
        let current_avg = self.avg_latency_ns.load(Ordering::Relaxed);
        let total = self.total_cancellations.load(Ordering::Relaxed);
        let new_avg = ((current_avg * (total - 1)) + latency_ns) / total;
        self.avg_latency_ns.store(new_avg, Ordering::Relaxed);
        
        self.last_cancel_ns.store(latency_ns, Ordering::Relaxed);
    }
    
    #[inline]
    pub fn get_total_cancellations(&self) -> u64 {
        self.total_cancellations.load(Ordering::Relaxed)
    }
    
    #[inline]
    pub fn get_toxic_flow_ratio(&self) -> f64 {
        let total = self.total_cancellations.load(Ordering::Relaxed);
        if total == 0 {
            return 0.0;
        }
        let toxic = self.toxic_flow_cancels.load(Ordering::Relaxed);
        toxic as f64 / total as f64
    }
}

/// Main cancellation manager
#[repr(C, align(64))]
pub struct CancellationManager {
    /// Are cancellations enabled
    enabled: AtomicBool,
    /// Toxic flow threshold (VPIN > this triggers cancel)
    toxic_vpin_threshold: AtomicU64, // Fixed point: value * 1000
    /// Latency threshold (ns) for arbitrage detection
    latency_threshold_ns: AtomicU64,
    /// Max cancellations per second
    max_cancels_per_sec: AtomicU64,
    /// Current cancel count (resets every second)
    current_cancel_count: AtomicU64,
    /// Last reset timestamp
    last_reset_ns: AtomicU64,
    /// Statistics
    stats: CancellationStats,
    /// Emergency halt flag
    emergency_halt: AtomicBool,
    _padding: [u8; 15],
}

unsafe impl Send for CancellationManager {}
unsafe impl Sync for CancellationManager {}

impl CancellationManager {
    pub fn new(toxic_vpin_threshold: f64, latency_threshold_ms: u64) -> Self {
        Self {
            enabled: AtomicBool::new(true),
            toxic_vpin_threshold: AtomicU64::new((toxic_vpin_threshold * 1000.0) as u64),
            latency_threshold_ns: AtomicU64::new(latency_threshold_ms * 1_000_000),
            max_cancels_per_sec: AtomicU64::new(1000), // Reasonable limit
            current_cancel_count: AtomicU64::new(0),
            last_reset_ns: AtomicU64::new(0),
            stats: CancellationStats::new(),
            emergency_halt: AtomicBool::new(false),
            _padding: [0u8; 15],
        }
    }
    
    /// Check if cancellation should be triggered - O(1) lock-free check
    #[inline]
    pub fn should_cancel(&self, vpin: f64, message_latency_ns: u64) -> Option<CancellationReason> {
        if !self.enabled.load(Ordering::Relaxed) || self.emergency_halt.load(Ordering::Relaxed) {
            return None;
        }
        
        // Rate limiting check
        let current_count = self.current_cancel_count.load(Ordering::Relaxed);
        let max_rate = self.max_cancels_per_sec.load(Ordering::Relaxed);
        if current_count >= max_rate {
            return None; // Rate limited
        }
        
        // Toxic flow detection
        let vpin_threshold = self.toxic_vpin_threshold.load(Ordering::Relaxed);
        let vpin_scaled = (vpin * 1000.0) as u64;
        if vpin_scaled > vpin_threshold {
            return Some(CancellationReason::ToxicFlow);
        }
        
        // Latency arbitrage detection
        let latency_threshold = self.latency_threshold_ns.load(Ordering::Relaxed);
        if message_latency_ns > latency_threshold {
            return Some(CancellationReason::LatencyArbitrage);
        }
        
        None
    }
    
    /// Execute cancellation - records stats and increments counter
    #[inline]
    pub fn execute_cancel(&self, reason: CancellationReason, latency_ns: u64) -> bool {
        if self.emergency_halt.load(Ordering::Relaxed) {
            return false;
        }
        
        // Rate limit check with reset
        let now_ns = Instant::now().duration_since(Instant::now()).as_nanos() as u64; // Placeholder
        let last_reset = self.last_reset_ns.load(Ordering::Relaxed);
        
        if now_ns - last_reset > 1_000_000_000 {
            // Reset counter every second
            self.current_cancel_count.store(0, Ordering::Relaxed);
            self.last_reset_ns.store(now_ns, Ordering::Relaxed);
        }
        
        let current_count = self.current_cancel_count.fetch_add(1, Ordering::Relaxed);
        if current_count >= self.max_cancels_per_sec.load(Ordering::Relaxed) {
            self.current_cancel_count.fetch_sub(1, Ordering::Relaxed);
            return false; // Rate limited
        }
        
        // Record statistics
        self.stats.record_cancel(reason, latency_ns);
        
        true
    }
    
    /// Emergency halt - stops all quoting immediately
    #[inline]
    pub fn emergency_halt_all(&self) {
        self.emergency_halt.store(true, Ordering::SeqCst);
        self.enabled.store(false, Ordering::SeqCst);
    }
    
    /// Resume normal operation
    #[inline]
    pub fn resume(&self) {
        self.emergency_halt.store(false, Ordering::SeqCst);
        self.enabled.store(true, Ordering::SeqCst);
        self.current_cancel_count.store(0, Ordering::Relaxed);
    }
    
    /// Get cancellation statistics
    #[inline]
    pub fn get_stats(&self) -> &CancellationStats {
        &self.stats
    }
    
    /// Update toxic flow threshold
    #[inline]
    pub fn set_toxic_threshold(&self, vpin: f64) {
        self.toxic_vpin_threshold.store((vpin * 1000.0) as u64, Ordering::Relaxed);
    }
    
    /// Check if in emergency halt
    #[inline]
    pub fn is_halted(&self) -> bool {
        self.emergency_halt.load(Ordering::Relaxed)
    }
}

/// Batch cancellation for multiple orders
pub struct BatchCancelRequest {
    order_ids: Vec<u64>,
    reason: CancellationReason,
}

impl BatchCancelRequest {
    pub fn new(reason: CancellationReason) -> Self {
        Self {
            order_ids: Vec::new(),
            reason,
        }
    }
    
    pub fn add_order(&mut self, order_id: u64) {
        self.order_ids.push(order_id);
    }
    
    pub fn len(&self) -> usize {
        self.order_ids.len()
    }
    
    pub fn is_empty(&self) -> bool {
        self.order_ids.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_cancellation_manager_basic() {
        let mgr = CancellationManager::new(0.5, 100); // 0.5 VPIN, 100ms latency
        
        // Normal conditions - no cancel
        assert!(mgr.should_cancel(0.3, 50_000_000).is_none());
        
        // Toxic flow - should trigger
        let reason = mgr.should_cancel(0.7, 50_000_000);
        assert_eq!(reason, Some(CancellationReason::ToxicFlow));
    }
    
    #[test]
    fn test_emergency_halt() {
        let mgr = CancellationManager::new(0.5, 100);
        assert!(!mgr.is_halted());
        
        mgr.emergency_halt_all();
        assert!(mgr.is_halted());
        
        mgr.resume();
        assert!(!mgr.is_halted());
    }
}
