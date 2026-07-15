//! Microsecond Wash Sale rule detector.
//! Prevents closing and re-opening the same asset within the restricted window.
//! Dynamically adjusts cost basis of new lots to comply with tax laws.

use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH, Duration};

/// Default wash sale window in days (US tax law: 30 days before or after)
const DEFAULT_WASH_SALE_WINDOW_DAYS: u64 = 30;

/// Seconds in a day
const SECONDS_PER_DAY: u64 = 86400;

/// Represents a wash sale event that must be tracked
#[derive(Debug, Clone)]
pub struct WashSaleEvent {
    pub instrument_id: u64,
    pub loss_realized: i128, // In quote currency * 1e8
    pub wash_sale_timestamp_ns: u64,
    pub disallowed_loss: i128,
    pub adjustment_applied: bool,
}

/// Tracks wash sale violations and manages cost basis adjustments
pub struct WashSaleDetector {
    /// Map of instrument_id to list of wash sale events
    wash_sale_events: HashMap<u64, Vec<WashSaleEvent>>,
    /// Map of instrument_id to timestamp when wash sale window expires
    wash_sale_blacklist: HashMap<u64, u64>, // instrument_id -> safe_after_ns
    /// Accumulated disallowed losses per instrument (to add to new cost basis)
    accumulated_disallowed_losses: HashMap<u64, u128>,
    /// Wash sale window duration in nanoseconds
    wash_window_ns: u64,
    /// Memory footprint tracking
    memory_footprint_bytes: u64,
}

impl WashSaleDetector {
    pub fn new(wash_window_days: Option<u64>) -> Self {
        let days = wash_window_days.unwrap_or(DEFAULT_WASH_SALE_WINDOW_DAYS);
        let wash_window_ns = days * SECONDS_PER_DAY * 1_000_000_000 * 2; // 2x for before+after
        
        Self {
            wash_sale_events: HashMap::new(),
            wash_sale_blacklist: HashMap::new(),
            accumulated_disallowed_losses: HashMap::new(),
            wash_window_ns,
            memory_footprint_bytes: 0,
        }
    }

    /// Get current timestamp in nanoseconds
    #[inline]
    fn now_ns() -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64
    }

    /// Record a realized loss and check for potential wash sale
    /// Returns true if this is a wash sale violation
    pub fn record_loss(
        &mut self,
        instrument_id: u64,
        quantity: i64,
        realized_loss: i128, // Negative value for loss
        timestamp_ns: Option<u64>,
    ) -> bool {
        // Only track actual losses
        if realized_loss >= 0 {
            return false;
        }

        let ts = timestamp_ns.unwrap_or_else(|| Self::now_ns());
        
        // Check if we're in a wash sale window for this instrument
        if let Some(&safe_after) = self.wash_sale_blacklist.get(&instrument_id) {
            if ts < safe_after {
                // This is a wash sale - record it
                let event = WashSaleEvent {
                    instrument_id,
                    loss_realized: realized_loss,
                    wash_sale_timestamp_ns: ts,
                    disallowed_loss: realized_loss.abs(),
                    adjustment_applied: false,
                };

                self.wash_sale_events
                    .entry(instrument_id)
                    .or_insert_with(|| Vec::with_capacity(64))
                    .push(event.clone());

                // Accumulate the disallowed loss for future cost basis adjustment
                *self.accumulated_disallowed_losses.entry(instrument_id).or_insert(0) += 
                    realized_loss.abs() as u128;

                // Extend the wash sale window
                let new_safe_after = ts + (DEFAULT_WASH_SALE_WINDOW_DAYS * SECONDS_PER_DAY * 1_000_000_000);
                self.wash_sale_blacklist.insert(instrument_id, new_safe_after);

                self.memory_footprint_bytes += std::mem::size_of::<WashSaleEvent>() as u64;

                return true;
            }
        }

        // Not a wash sale, but start tracking window from this loss
        let safe_after = ts + (DEFAULT_WASH_SALE_WINDOW_DAYS * SECONDS_PER_DAY * 1_000_000_000);
        self.wash_sale_blacklist.insert(instrument_id, safe_after);

        false
    }

    /// Check if purchasing an instrument would trigger a wash sale
    pub fn would_trigger_wash_sale(&self, instrument_id: u64, timestamp_ns: Option<u64>) -> bool {
        let ts = timestamp_ns.unwrap_or_else(|| Self::now_ns());
        
        if let Some(&safe_after) = self.wash_sale_blacklist.get(&instrument_id) {
            return ts < safe_after;
        }
        
        false
    }

    /// Calculate adjusted cost basis for a new purchase after a wash sale
    /// Returns the additional amount to add to the cost basis
    pub fn get_cost_basis_adjustment(&self, instrument_id: u64) -> u128 {
        *self.accumulated_disallowed_losses.get(&instrument_id).unwrap_or(&0)
    }

    /// Apply cost basis adjustment and clear accumulated losses
    /// Call this when you've created a new lot with the adjusted basis
    pub fn apply_cost_basis_adjustment(&mut self, instrument_id: u64, adjustment_amount: u128) {
        if let Some(accumulated) = self.accumulated_disallowed_losses.get_mut(&instrument_id) {
            if *accumulated >= adjustment_amount {
                *accumulated -= adjustment_amount;
            } else {
                *accumulated = 0;
            }
        }
    }

    /// Mark all wash sale events for an instrument as having applied adjustments
    pub fn mark_adjustments_applied(&mut self, instrument_id: u64) {
        if let Some(events) = self.wash_sale_events.get_mut(&instrument_id) {
            for event in events.iter_mut() {
                event.adjustment_applied = true;
            }
        }
    }

    /// Check if an instrument has any pending (unadjusted) wash sale losses
    pub fn has_pending_wash_sales(&self, instrument_id: u64) -> bool {
        self.accumulated_disallowed_losses
            .get(&instrument_id)
            .map(|&v| v > 0)
            .unwrap_or(false)
    }

    /// Get all wash sale events for an instrument
    pub fn get_wash_sale_events(&self, instrument_id: u64) -> &[WashSaleEvent] {
        self.wash_sale_events
            .get(&instrument_id)
            .map(|v| v.as_slice())
            .unwrap_or(&[])
    }

    /// Clear expired wash sale blacklists (call periodically)
    pub fn cleanup_expired(&mut self) -> usize {
        let now = Self::now_ns();
        let mut cleared = 0;

        self.wash_sale_blacklist.retain(|_, &safe_after| {
            if now >= safe_after {
                cleared += 1;
                false
            } else {
                true
            }
        });

        cleared
    }

    /// Get total disallowed losses across all instruments
    pub fn total_disallowed_losses(&self) -> u128 {
        self.accumulated_disallowed_losses.values().sum()
    }

    /// Get memory footprint in bytes
    pub fn memory_footprint(&self) -> u64 {
        self.memory_footprint_bytes
    }

    /// Get count of tracked wash sale events
    pub fn event_count(&self) -> usize {
        self.wash_sale_events.values().map(|v| v.len()).sum()
    }
}

impl Default for WashSaleDetector {
    fn default() -> Self {
        Self::new(None)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wash_sale_detection() {
        let mut detector = WashSaleDetector::new(Some(30));
        
        // Record a loss
        let is_wash = detector.record_loss(1, 100, -500_000_000, Some(1000)); // $5 loss
        assert!(!is_wash); // First loss is not a wash sale
        
        // Try to buy back within wash sale window
        let would_trigger = detector.would_trigger_wash_sale(1, Some(2000));
        assert!(would_trigger);
        
        // Record another loss during wash sale window (this becomes a wash sale)
        let is_wash = detector.record_loss(1, 100, -300_000_000, Some(2000));
        assert!(is_wash); // This IS a wash sale
        
        // Check accumulated disallowed losses
        assert_eq!(detector.get_cost_basis_adjustment(1), 300_000_000);
        
        // Check total disallowed
        assert_eq!(detector.total_disallowed_losses(), 300_000_000);
    }

    #[test]
    fn test_wash_sale_cleanup() {
        let mut detector = WashSaleDetector::new(Some(1)); // 1-day window for testing
        
        // Record a loss
        detector.record_loss(1, 100, -500_000_000, Some(1000));
        
        // Cleanup should not remove recent entries
        let cleared = detector.cleanup_expired();
        assert_eq!(cleared, 0);
        
        // Simulate time passing (in real code, wait or mock time)
        // For this test, manually insert an old entry
        detector.wash_sale_blacklist.insert(2, 100); // Very old timestamp
        
        let cleared = detector.cleanup_expired();
        assert!(cleared >= 1);
    }
}
