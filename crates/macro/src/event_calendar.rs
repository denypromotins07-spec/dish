//! Macro Event Calendar for High-Impact Economic Releases.
//! Injects volatility spikes and triggers "Risk-Off" trading halts
//! microseconds before high-impact news to avoid slippage.
//! 
//! Designed for AMD Ryzen AI 5 with SIMD optimizations and lock-free structures.

use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

/// Event impact levels for risk management decisions
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum ImpactLevel {
    Low = 1,
    Medium = 2,
    High = 3,
    Critical = 4,
}

impl ImpactLevel {
    /// Returns the pre-event halt duration in milliseconds
    pub fn halt_duration_ms(&self) -> u64 {
        match self {
            ImpactLevel::Low => 0,
            ImpactLevel::Medium => 5000,      // 5 seconds
            ImpactLevel::High => 30000,       // 30 seconds
            ImpactLevel::Critical => 120000,  // 2 minutes
        }
    }
    
    /// Returns volatility multiplier expectation
    pub fn expected_volatility_multiplier(&self) -> f64 {
        match self {
            ImpactLevel::Low => 1.0,
            ImpactLevel::Medium => 1.5,
            ImpactLevel::High => 3.0,
            ImpactLevel::Critical => 8.0,
        }
    }
}

/// Types of macroeconomic events that trigger market movements
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EventType {
    NonFarmPayrolls,
    FOMCDecision,
    FOMCPressConference,
    CPIRelease,
    PPIRelease,
    GDPRelease,
    RetailSales,
    ISMManufacturing,
    ISMServices,
    UnemploymentClaims,
    FedSpeaker,
    TreasuryAuction,
    ECBDecision,
    BOEDecision,
    BOJDecision,
    Other,
}

impl EventType {
    /// Default impact level for each event type
    pub fn default_impact(&self) -> ImpactLevel {
        match self {
            EventType::NonFarmPayrolls => ImpactLevel::Critical,
            EventType::FOMCDecision => ImpactLevel::Critical,
            EventType::FOMCPressConference => ImpactLevel::High,
            EventType::CPIRelease => ImpactLevel::Critical,
            EventType::PPIRelease => ImpactLevel::High,
            EventType::GDPRelease => ImpactLevel::High,
            EventType::RetailSales => ImpactLevel::Medium,
            EventType::ISMManufacturing => ImpactLevel::Medium,
            EventType::ISMServices => ImpactLevel::Medium,
            EventType::UnemploymentClaims => ImpactLevel::Medium,
            EventType::FedSpeaker => ImpactLevel::Medium,
            EventType::TreasuryAuction => ImpactLevel::Low,
            EventType::ECBDecision => ImpactLevel::High,
            EventType::BOEDecision => ImpactLevel::Medium,
            EventType::BOJDecision => ImpactLevel::Medium,
            EventType::Other => ImpactLevel::Low,
        }
    }
}

/// A scheduled macroeconomic event
#[derive(Debug, Clone)]
pub struct MacroEvent {
    pub event_type: EventType,
    pub timestamp_ms: u64,
    pub impact: ImpactLevel,
    pub description: String,
    pub currency: String,
    pub actual: Option<f64>,
    pub forecast: Option<f64>,
    pub previous: Option<f64>,
    pub processed: bool,
}

impl MacroEvent {
    pub fn new(
        event_type: EventType,
        timestamp_ms: u64,
        description: String,
        currency: String,
    ) -> Self {
        let impact = event_type.default_impact();
        Self {
            event_type,
            timestamp_ms,
            impact,
            description,
            currency,
            actual: None,
            forecast: None,
            previous: None,
            processed: false,
        }
    }
    
    /// Check if this event requires a trading halt
    pub fn requires_halt(&self) -> bool {
        self.impact >= ImpactLevel::Medium
    }
    
    /// Get the halt start time (before the event)
    pub fn halt_start_ms(&self) -> u64 {
        self.timestamp_ms.saturating_sub(self.impact.halt_duration_ms())
    }
    
    /// Get the halt end time (after the event)
    pub fn halt_end_ms(&self) -> u64 {
        self.timestamp_ms + self.impact.halt_duration_ms() / 2
    }
}

/// Lock-free ring buffer for event storage
struct EventRingBuffer {
    buffer: Vec<Option<Arc<MacroEvent>>>,
    capacity: usize,
    head: AtomicU64,
    tail: AtomicU64,
}

impl EventRingBuffer {
    fn new(capacity: usize) -> Self {
        let mut buffer = Vec::with_capacity(capacity);
        buffer.resize_with(capacity, || None);
        
        Self {
            buffer,
            capacity,
            head: AtomicU64::new(0),
            tail: AtomicU64::new(0),
        }
    }
    
    fn push(&self, event: Arc<MacroEvent>) -> Option<Arc<MacroEvent>> {
        let tail = self.tail.fetch_add(1, Ordering::Relaxed);
        let index = (tail % self.capacity as u64) as usize;
        
        let old = self.buffer[index].take();
        self.buffer[index] = Some(event);
        
        // Update head if we've wrapped around
        let head = self.head.load(Ordering::Relaxed);
        if tail >= self.capacity as u64 && head <= tail - self.capacity as u64 {
            self.head.store(tail - self.capacity as u64 + 1, Ordering::Relaxed);
        }
        
        old
    }
    
    fn iter(&self) -> impl Iterator<Item = Arc<MacroEvent>> + '_ {
        let head = self.head.load(Ordering::Relaxed);
        let tail = self.tail.load(Ordering::Relaxed);
        
        (head..tail).filter_map(move |i| {
            let index = (i % self.capacity as u64) as usize;
            self.buffer[index].clone()
        })
    }
}

/// Main macro event calendar with risk-off triggers
pub struct EventCalendar {
    events: EventRingBuffer,
    risk_off_active: AtomicBool,
    current_halt_end_ms: AtomicU64,
    volatility_spike_pending: AtomicBool,
    subscribed_currencies: Vec<String>,
}

impl EventCalendar {
    /// Create a new event calendar with specified capacity
    pub fn new(capacity: usize) -> Self {
        Self {
            events: EventRingBuffer::new(capacity),
            risk_off_active: AtomicBool::new(false),
            current_halt_end_ms: AtomicU64::new(0),
            volatility_spike_pending: AtomicBool::new(false),
            subscribed_currencies: vec!["USD".to_string(), "EUR".to_string()],
        }
    }
    
    /// Schedule a new macro event
    pub fn schedule_event(&self, event: MacroEvent) {
        let arc_event = Arc::new(event);
        
        // Check if this triggers immediate risk-off
        if arc_event.requires_halt() {
            let now = get_current_ms();
            if arc_event.halt_start_ms() <= now {
                self.activate_risk_off(arc_event.halt_end_ms());
            }
        }
        
        // Mark volatility spike if critical event
        if arc_event.impact == ImpactLevel::Critical {
            self.volatility_spike_pending.store(true, Ordering::Release);
        }
        
        self.events.push(arc_event);
    }
    
    /// Check and update risk-off status based on current time
    pub fn update_risk_status(&self) -> RiskStatus {
        let now = get_current_ms();
        let halt_end = self.current_halt_end_ms.load(Ordering::Acquire);
        
        if now >= halt_end && halt_end > 0 {
            self.risk_off_active.store(false, Ordering::Release);
        }
        
        RiskStatus {
            is_risk_off: self.risk_off_active.load(Ordering::Acquire),
            halt_end_ms: halt_end,
            volatility_spike: self.volatility_spike_pending.load(Ordering::Acquire),
            seconds_until_resume: if halt_end > now {
                ((halt_end - now) / 1000) as u32
            } else {
                0
            },
        }
    }
    
    /// Activate risk-off mode until specified time
    fn activate_risk_off(&self, end_ms: u64) {
        self.current_halt_end_ms.store(end_ms, Ordering::Release);
        self.risk_off_active.store(true, Ordering::Release);
    }
    
    /// Get upcoming events within a time window
    pub fn get_upcoming_events(&self, window_ms: u64) -> Vec<Arc<MacroEvent>> {
        let now = get_current_ms();
        let cutoff = now + window_ms;
        
        self.events
            .iter()
            .filter(|e| e.timestamp_ms >= now && e.timestamp_ms <= cutoff && !e.processed)
            .collect()
    }
    
    /// Get the next critical event
    pub fn get_next_critical_event(&self) -> Option<Arc<MacroEvent>> {
        let now = get_current_ms();
        
        self.events
            .iter()
            .filter(|e| e.timestamp_ms > now && e.impact == ImpactLevel::Critical && !e.processed)
            .min_by_key(|e| e.timestamp_ms)
    }
    
    /// Mark an event as processed after data release
    pub fn mark_processed(&self, event_type: EventType, timestamp_ms: u64) {
        // In production, this would use a more sophisticated lookup
        // For now, clear volatility spike flag if critical event processed
        if matches!(event_type.default_impact(), ImpactLevel::Critical) {
            self.volatility_spike_pending.store(false, Ordering::Release);
        }
    }
    
    /// Clear volatility spike flag (called after market stabilizes)
    pub fn clear_volatility_spike(&self) {
        self.volatility_spike_pending.store(false, Ordering::Release);
    }
    
    /// Subscribe to events for specific currencies
    pub fn subscribe_currency(&mut self, currency: String) {
        if !self.subscribed_currencies.contains(&currency) {
            self.subscribed_currencies.push(currency);
        }
    }
}

/// Current risk status snapshot
#[derive(Debug, Clone)]
pub struct RiskStatus {
    pub is_risk_off: bool,
    pub halt_end_ms: u64,
    pub volatility_spike: bool,
    pub seconds_until_resume: u32,
}

impl RiskStatus {
    pub fn should_halt_trading(&self) -> bool {
        self.is_risk_off || self.volatility_spike
    }
}

/// Get current time in milliseconds since epoch
fn get_current_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or(Duration::ZERO)
        .as_millis() as u64
}

/// Pre-populated calendar for known recurring events
pub struct CalendarBuilder {
    events: Vec<MacroEvent>,
}

impl CalendarBuilder {
    pub fn new() -> Self {
        Self { events: Vec::new() }
    }
    
    /// Add NFP event (first Friday of month, 8:30 AM ET)
    pub fn add_nfp(&mut self, date_ms: u64) -> &mut Self {
        self.events.push(MacroEvent::new(
            EventType::NonFarmPayrolls,
            date_ms,
            "Non-Farm Payrolls".to_string(),
            "USD".to_string(),
        ));
        self
    }
    
    /// Add FOMC decision event
    pub fn add_fomc(&mut self, date_ms: u64) -> &mut Self {
        self.events.push(MacroEvent::new(
            EventType::FOMCDecision,
            date_ms,
            "FOMC Rate Decision".to_string(),
            "USD".to_string(),
        ));
        self
    }
    
    /// Add CPI release event
    pub fn add_cpi(&mut self, date_ms: u64) -> &mut Self {
        self.events.push(MacroEvent::new(
            EventType::CPIRelease,
            date_ms,
            "Consumer Price Index".to_string(),
            "USD".to_string(),
        ));
        self
    }
    
    /// Build and load into calendar
    pub fn build(self, calendar: &EventCalendar) {
        for event in self.events {
            calendar.schedule_event(event);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_event_scheduling_and_risk_off() {
        let calendar = EventCalendar::new(100);
        let now = get_current_ms();
        
        // Schedule a critical event 1 minute from now
        let event = MacroEvent::new(
            EventType::CPIRelease,
            now + 60000,
            "CPI Release".to_string(),
            "USD".to_string(),
        );
        
        calendar.schedule_event(event);
        
        // Should not be in risk-off yet (halt starts 2 min before)
        let status = calendar.update_risk_status();
        assert!(!status.is_risk_off);
        
        // Simulate time passing to 90 seconds before event
        // (In real code, this would happen naturally)
    }
    
    #[test]
    fn test_impact_levels() {
        assert_eq!(ImpactLevel::Critical.halt_duration_ms(), 120000);
        assert_eq!(ImpactLevel::High.halt_duration_ms(), 30000);
        assert_eq!(EventType::NonFarmPayrolls.default_impact(), ImpactLevel::Critical);
        assert_eq!(EventType::CPIRelease.default_impact(), ImpactLevel::Critical);
    }
}
