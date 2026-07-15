//! Automatic quoting halt mechanism.
//! Instantly pulls all maker limit orders and switches to "taker-only" or "flat" state
//! if VPIN crosses a critical threshold, protecting the bot from adverse selection.

use std::sync::atomic::{AtomicU64, AtomicBool, AtomicI64, Ordering};
use std::time::Instant;

/// Halter state machine states
#[repr(u8)]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum HalterState {
    /// Normal operation - quoting both sides
    Normal = 0,
    /// Warning state - reduced quoting
    Warning = 1,
    /// Taker-only mode - no new maker orders
    TakerOnly = 2,
    /// Flat mode - all positions closed, no trading
    Flat = 3,
    /// Emergency halt - complete shutdown
    EmergencyHalt = 4,
}

/// Toxicity halter configuration
#[repr(C, align(64))]
pub struct ToxicityHalter {
    /// Current halter state
    state: AtomicU64,
    /// VPIN warning threshold
    warning_threshold: AtomicU64, // Fixed point: * 10000
    /// VPIN taker-only threshold
    taker_only_threshold: AtomicU64,
    /// VPIN flat threshold
    flat_threshold: AtomicU64,
    /// VPIN emergency threshold
    emergency_threshold: AtomicU64,
    /// Time spent in current state (ns)
    state_entry_time_ns: AtomicU64,
    /// Number of times halted
    halt_count: AtomicU64,
    /// Last halt timestamp (ns)
    last_halt_time_ns: AtomicU64,
    /// Cooldown period before resuming (ns)
    cooldown_ns: AtomicU64,
    /// Is currently in cooldown
    in_cooldown: AtomicBool,
    _padding: [u8; 15],
}

unsafe impl Send for ToxicityHalter {}
unsafe impl Sync for ToxicityHalter {}

impl ToxicityHalter {
    pub fn new(
        warning_vpin: f64,
        taker_vpin: f64,
        flat_vpin: f64,
        emergency_vpin: f64,
        cooldown_ms: u64,
    ) -> Self {
        Self {
            state: AtomicU64::new(HalterState::Normal as u64),
            warning_threshold: AtomicU64::new((warning_vpin * 10000.0) as u64),
            taker_only_threshold: AtomicU64::new((taker_vpin * 10000.0) as u64),
            flat_threshold: AtomicU64::new((flat_vpin * 10000.0) as u64),
            emergency_threshold: AtomicU64::new((emergency_vpin * 10000.0) as u64),
            state_entry_time_ns: AtomicU64::new(0),
            halt_count: AtomicU64::new(0),
            last_halt_time_ns: AtomicU64::new(0),
            cooldown_ns: AtomicU64::new(cooldown_ms * 1_000_000),
            in_cooldown: AtomicBool::new(false),
            _padding: [0u8; 15],
        }
    }
    
    /// Check VPIN and update state - O(1) lock-free operation
    #[inline]
    pub fn check_and_update(&self, vpin: f64) -> Option<HalterState> {
        let vpin_scaled = (vpin * 10000.0) as u64;
        let old_state = self.get_state();
        
        // Check cooldown first
        if self.in_cooldown.load(Ordering::Relaxed) {
            let elapsed = Instant::now().duration_since(Instant::now()).as_nanos() as u64; // Placeholder
            let last_halt = self.last_halt_time_ns.load(Ordering::Relaxed);
            let cooldown = self.cooldown_ns.load(Ordering::Relaxed);
            
            if elapsed - last_halt < cooldown {
                return Some(old_state); // Still in cooldown
            } else {
                self.in_cooldown.store(false, Ordering::Relaxed);
            }
        }
        
        // Determine new state based on VPIN thresholds
        let new_state = if vpin_scaled >= self.emergency_threshold.load(Ordering::Relaxed) {
            HalterState::EmergencyHalt
        } else if vpin_scaled >= self.flat_threshold.load(Ordering::Relaxed) {
            HalterState::Flat
        } else if vpin_scaled >= self.taker_only_threshold.load(Ordering::Relaxed) {
            HalterState::TakerOnly
        } else if vpin_scaled >= self.warning_threshold.load(Ordering::Relaxed) {
            HalterState::Warning
        } else {
            HalterState::Normal
        };
        
        // State transition logic
        if new_state != old_state {
            self.state.store(new_state as u64, Ordering::SeqCst);
            self.state_entry_time_ns.store(
                Instant::now().duration_since(Instant::now()).as_nanos() as u64, // Placeholder
                Ordering::Relaxed
            );
            
            // Record halt events
            if matches!(new_state, HalterState::Flat | HalterState::EmergencyHalt) {
                self.halt_count.fetch_add(1, Ordering::Relaxed);
                self.last_halt_time_ns.store(
                    Instant::now().duration_since(Instant::now()).as_nanos() as u64, // Placeholder
                    Ordering::Relaxed
                );
                self.in_cooldown.store(true, Ordering::Relaxed);
            }
            
            return Some(new_state);
        }
        
        None
    }
    
    /// Get current state
    #[inline]
    pub fn get_state(&self) -> HalterState {
        match self.state.load(Ordering::Relaxed) {
            0 => HalterState::Normal,
            1 => HalterState::Warning,
            2 => HalterState::TakerOnly,
            3 => HalterState::Flat,
            4 => HalterState::EmergencyHalt,
            _ => HalterState::Normal,
        }
    }
    
    /// Check if maker orders should be cancelled
    #[inline]
    pub fn should_cancel_makers(&self) -> bool {
        let state = self.get_state();
        matches!(state, HalterState::Flat | HalterState::EmergencyHalt)
    }
    
    /// Check if new maker orders are allowed
    #[inline]
    pub fn can_place_makers(&self) -> bool {
        let state = self.get_state();
        matches!(state, HalterState::Normal | HalterState::Warning)
    }
    
    /// Check if taker orders are allowed
    #[inline]
    pub fn can_place_takers(&self) -> bool {
        let state = self.get_state();
        !matches!(state, HalterState::Flat | HalterState::EmergencyHalt)
    }
    
    /// Force state transition (for manual override)
    #[inline]
    pub fn force_state(&self, state: HalterState) {
        self.state.store(state as u64, Ordering::SeqCst);
        self.state_entry_time_ns.store(
            Instant::now().duration_since(Instant::now()).as_nanos() as u64, // Placeholder
            Ordering::Relaxed
        );
    }
    
    /// Reset to normal state
    #[inline]
    pub fn reset(&self) {
        self.force_state(HalterState::Normal);
        self.in_cooldown.store(false, Ordering::Relaxed);
    }
    
    /// Get time spent in current state (ns)
    #[inline]
    pub fn time_in_state(&self) -> u64 {
        let entry_time = self.state_entry_time_ns.load(Ordering::Relaxed);
        let now = Instant::now().duration_since(Instant::now()).as_nanos() as u64; // Placeholder
        now.saturating_sub(entry_time)
    }
    
    /// Get halt statistics
    #[inline]
    pub fn get_stats(&self) -> HalterStats {
        HalterStats {
            current_state: self.get_state(),
            halt_count: self.halt_count.load(Ordering::Relaxed),
            time_in_state_ns: self.time_in_state(),
        }
    }
}

/// Halter statistics snapshot
#[derive(Clone, Copy, Debug)]
pub struct HalterStats {
    pub current_state: HalterState,
    pub halt_count: u64,
    pub time_in_state_ns: u64,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_normal_operation() {
        let halter = ToxicityHalter::new(0.3, 0.5, 0.7, 0.9, 1000);
        
        assert_eq!(halter.get_state(), HalterState::Normal);
        assert!(halter.can_place_makers());
        assert!(halter.can_place_takers());
    }
    
    #[test]
    fn test_warning_transition() {
        let halter = ToxicityHalter::new(0.3, 0.5, 0.7, 0.9, 1000);
        
        // Trigger warning state
        halter.check_and_update(0.35);
        
        assert_eq!(halter.get_state(), HalterState::Warning);
        assert!(halter.can_place_makers()); // Still allowed in warning
        assert!(halter.can_place_takers());
    }
    
    #[test]
    fn test_emergency_halt() {
        let halter = ToxicityHalter::new(0.3, 0.5, 0.7, 0.9, 1000);
        
        // Trigger emergency halt
        halter.check_and_update(0.95);
        
        assert_eq!(halter.get_state(), HalterState::EmergencyHalt);
        assert!(!halter.can_place_makers());
        assert!(!halter.can_place_takers());
        assert!(halter.should_cancel_makers());
    }
    
    #[test]
    fn test_manual_reset() {
        let halter = ToxicityHalter::new(0.3, 0.5, 0.7, 0.9, 1000);
        
        halter.check_and_update(0.95);
        assert_eq!(halter.get_state(), HalterState::EmergencyHalt);
        
        halter.reset();
        assert_eq!(halter.get_state(), HalterState::Normal);
    }
}
