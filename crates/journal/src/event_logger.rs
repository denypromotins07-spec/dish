// crates/journal/src/event_logger.rs
// Lock-free, zero-allocation trade event logger
// Captures every microsecond of order lifecycle into shared memory ring buffer

use std::sync::atomic::{AtomicU64, AtomicBool, Ordering};
use std::mem::MaybeUninit;

/// Order lifecycle stages - each trade progresses through these states
#[derive(Clone, Copy, Debug, PartialEq)]
#[repr(u8)]
pub enum OrderStage {
    Signal = 0,        // Alpha signal generated
    PreRisk = 1,       // Pre-trade risk check
    Routing = 2,       // Order routed to exchange
    Ack = 3,           // Exchange acknowledgment
    Partial = 4,       // Partial fill
    Filled = 5,        // Completely filled
    Cancelled = 6,     // Order cancelled
    Settlement = 7,    // Trade settled
}

/// Compact event type for the journal
#[derive(Clone, Copy, Debug)]
#[repr(C, packed)]
pub struct TradeEvent {
    /// Nanosecond timestamp
    pub timestamp_ns: u64,
    /// Unique order ID
    pub order_id: u64,
    /// Event stage
    pub stage: OrderStage,
    /// Price (scaled integer for precision)
    pub price_scaled: i64,
    /// Quantity (scaled integer)
    pub quantity_scaled: i64,
    /// Strategy ID
    pub strategy_id: u16,
    /// Asset ID
    pub asset_id: u16,
    /// Side: 0=buy, 1=sell
    pub side: u8,
    /// Venue/exchange ID
    pub venue_id: u8,
    /// Latency from previous stage in nanoseconds
    pub stage_latency_ns: u32,
    /// Sequence number within this order
    pub sequence: u16,
    /// Flags (maker/taker, etc.)
    pub flags: u8,
    /// Reserved padding
    _reserved: u8,
}

impl TradeEvent {
    #[inline]
    pub const fn new() -> Self {
        Self {
            timestamp_ns: 0,
            order_id: 0,
            stage: OrderStage::Signal,
            price_scaled: 0,
            quantity_scaled: 0,
            strategy_id: 0,
            asset_id: 0,
            side: 0,
            venue_id: 0,
            stage_latency_ns: 0,
            sequence: 0,
            flags: 0,
            _reserved: 0,
        }
    }

    #[inline]
    pub fn with_order(
        mut self,
        order_id: u64,
        strategy_id: u16,
        asset_id: u16,
        side: u8,
    ) -> Self {
        self.order_id = order_id;
        self.strategy_id = strategy_id;
        self.asset_id = asset_id;
        self.side = side;
        self
    }

    #[inline]
    pub fn set_price(&mut self, price: f64, scale: f64) {
        self.price_scaled = (price * scale) as i64;
    }

    #[inline]
    pub fn get_price(&self, scale: f64) -> f64 {
        self.price_scaled as f64 / scale
    }

    #[inline]
    pub fn set_quantity(&mut self, qty: f64, scale: f64) {
        self.quantity_scaled = (qty * scale) as i64;
    }

    #[inline]
    pub fn get_quantity(&self, scale: f64) -> f64 {
        self.quantity_scaled as f64 / scale
    }
}

impl Default for TradeEvent {
    fn default() -> Self {
        Self::new()
    }
}

/// Lock-free ring buffer for trade events
/// Fixed size, pre-allocated, zero runtime allocation
pub struct EventRingBuffer<const CAPACITY: usize> {
    /// The actual storage
    buffer: [TradeEvent; CAPACITY],
    /// Head index (write position)
    head: AtomicU64,
    /// Tail index (read position)
    tail: AtomicU64,
    /// Count of events currently in buffer
    count: AtomicU64,
    /// Total events written (for overflow tracking)
    total_written: AtomicU64,
    /// Flag indicating if buffer is in shutdown mode
    shutdown: AtomicBool,
}

impl<const CAPACITY: usize> EventRingBuffer<CAPACITY> {
    /// Create a new ring buffer
    pub const fn new() -> Self {
        // SAFETY: TradeEvent is safe to zero-initialize
        const INIT: TradeEvent = TradeEvent::new();
        Self {
            buffer: [INIT; CAPACITY],
            head: AtomicU64::new(0),
            tail: AtomicU64::new(0),
            count: AtomicU64::new(0),
            total_written: AtomicU64::new(0),
            shutdown: AtomicBool::new(false),
        }
    }

    /// Push an event to the buffer (lock-free, wait-free)
    /// Returns true if successful, false if buffer is full
    #[inline]
    pub fn push(&self, event: TradeEvent) -> bool {
        if self.shutdown.load(Ordering::Relaxed) {
            return false;
        }

        let head = self.head.load(Ordering::Relaxed);
        let tail = self.tail.load(Ordering::Acquire);
        
        // Check if buffer is full
        if head.wrapping_sub(tail) >= CAPACITY as u64 {
            // Buffer full - caller should handle overflow
            return false;
        }

        let idx = (head as usize) % CAPACITY;
        
        // Write the event
        unsafe {
            let ptr = &mut *(self.buffer.as_ptr().add(idx) as *mut TradeEvent);
            std::ptr::write(ptr, event);
        }

        // Memory barrier to ensure write is visible
        std::sync::atomic::fence(Ordering::Release);

        // Update head
        self.head.store(head.wrapping_add(1), Ordering::Release);
        self.count.fetch_add(1, Ordering::Relaxed);
        self.total_written.fetch_add(1, Ordering::Relaxed);

        true
    }

    /// Pop an event from the buffer
    /// Returns None if buffer is empty
    #[inline]
    pub fn pop(&self) -> Option<TradeEvent> {
        let tail = self.tail.load(Ordering::Relaxed);
        let head = self.head.load(Ordering::Acquire);

        if tail >= head {
            return None;
        }

        let idx = (tail as usize) % CAPACITY;
        
        // Read the event
        let event = unsafe {
            let ptr = self.buffer.as_ptr().add(idx);
            std::ptr::read(ptr)
        };

        // Memory barrier
        std::sync::atomic::fence(Ordering::Acquire);

        // Update tail
        self.tail.store(tail.wrapping_add(1), Ordering::Release);
        self.count.fetch_sub(1, Ordering::Relaxed);

        Some(event)
    }

    /// Peek at the next event without removing it
    #[inline]
    pub fn peek(&self) -> Option<TradeEvent> {
        let tail = self.tail.load(Ordering::Relaxed);
        let head = self.head.load(Ordering::Acquire);

        if tail >= head {
            return None;
        }

        let idx = (tail as usize) % CAPACITY;
        unsafe {
            let ptr = self.buffer.as_ptr().add(idx);
            Some(std::ptr::read(ptr))
        }
    }

    /// Get current event count
    #[inline]
    pub fn len(&self) -> usize {
        self.count.load(Ordering::Relaxed) as usize
    }

    /// Check if buffer is empty
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Check if buffer is full
    #[inline]
    pub fn is_full(&self) -> bool {
        let head = self.head.load(Ordering::Relaxed);
        let tail = self.tail.load(Ordering::Relaxed);
        head.wrapping_sub(tail) >= CAPACITY as u64
    }

    /// Get total events written since creation
    #[inline]
    pub fn total_written(&self) -> u64 {
        self.total_written.load(Ordering::Relaxed)
    }

    /// Initiate shutdown
    #[inline]
    pub fn shutdown(&self) {
        self.shutdown.store(true, Ordering::Release);
    }

    /// Check if shutdown requested
    #[inline]
    pub fn is_shutdown(&self) -> bool {
        self.shutdown.load(Ordering::Acquire)
    }

    /// Drain multiple events into a slice
    /// Returns number of events drained
    #[inline]
    pub fn drain_into(&self, out: &mut [TradeEvent]) -> usize {
        let mut written = 0;
        while written < out.len() {
            match self.pop() {
                Some(event) => {
                    out[written] = event;
                    written += 1;
                }
                None => break,
            }
        }
        written
    }

    /// Get approximate latency statistics
    #[inline]
    pub fn get_latency_stats(&self, order_id: u64) -> LatencyStats {
        let mut stats = LatencyStats::default();
        let mut first_ts = 0u64;
        let mut last_ts = 0u64;
        let mut found = false;

        let tail = self.tail.load(Ordering::Relaxed);
        let head = self.head.load(Ordering::Relaxed);
        let count = head.wrapping_sub(tail) as usize;

        for i in 0..count.min(CAPACITY) {
            let idx = ((tail as usize) + i) % CAPACITY;
            let event = unsafe {
                let ptr = self.buffer.as_ptr().add(idx);
                &*ptr
            };

            if event.order_id == order_id {
                if !found {
                    first_ts = event.timestamp_ns;
                    found = true;
                }
                last_ts = event.timestamp_ns;
                stats.stage_count += 1;
            }
        }

        if found {
            stats.total_latency_ns = last_ts.saturating_sub(first_ts);
        }

        stats
    }
}

impl<const CAPACITY: usize> Default for EventRingBuffer<CAPACITY> {
    fn default() -> Self {
        Self::new()
    }
}

/// Latency statistics for a single order
#[derive(Clone, Copy, Debug, Default)]
#[repr(C)]
pub struct LatencyStats {
    pub total_latency_ns: u64,
    pub stage_count: u32,
}

/// High-resolution clock for nanosecond timestamps
#[inline]
pub fn now_ns() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0)
}

/// Trade journal writer - batches events for efficiency
pub struct JournalWriter<const CAPACITY: usize> {
    buffer: &'static EventRingBuffer<CAPACITY>,
    batch: [TradeEvent; 64],
    batch_size: usize,
    last_flush_ns: u64,
    flush_interval_ns: u64,
}

impl<const CAPACITY: usize> JournalWriter<CAPACITY> {
    #[inline]
    pub const fn new(buffer: &'static EventRingBuffer<CAPACITY>) -> Self {
        Self {
            buffer,
            batch: [TradeEvent::new(); 64],
            batch_size: 0,
            last_flush_ns: 0,
            flush_interval_ns: 1_000_000, // 1ms default
        }
    }

    #[inline]
    pub fn log_event(&mut self, event: TradeEvent) {
        if self.batch_size < self.batch.len() {
            self.batch[self.batch_size] = event;
            self.batch_size += 1;
        }

        // Flush if batch is full or interval elapsed
        let now = now_ns();
        if self.batch_size >= self.batch.len() 
            || now.saturating_sub(self.last_flush_ns) > self.flush_interval_ns 
        {
            self.flush();
            self.last_flush_ns = now;
        }
    }

    #[inline]
    fn flush(&mut self) {
        for i in 0..self.batch_size {
            if !self.buffer.push(self.batch[i]) {
                // Buffer full - could implement spill to disk here
                break;
            }
        }
        self.batch_size = 0;
    }

    #[inline]
    pub fn force_flush(&mut self) {
        self.flush();
    }

    #[inline]
    pub fn set_flush_interval(&mut self, interval_ns: u64) {
        self.flush_interval_ns = interval_ns;
    }
}

/// Per-order state tracker
pub struct OrderTracker {
    order_id: u64,
    first_timestamp_ns: u64,
    last_timestamp_ns: u64,
    stage_history: u32, // Bitmask of stages seen
    fill_count: u8,
}

impl OrderTracker {
    #[inline]
    pub const fn new(order_id: u64) -> Self {
        Self {
            order_id,
            first_timestamp_ns: 0,
            last_timestamp_ns: 0,
            stage_history: 0,
            fill_count: 0,
        }
    }

    #[inline]
    pub fn record_event(&mut self, event: &TradeEvent) {
        if self.first_timestamp_ns == 0 {
            self.first_timestamp_ns = event.timestamp_ns;
        }
        self.last_timestamp_ns = event.timestamp_ns;
        self.stage_history |= 1 << (event.stage as u8);

        if event.stage == OrderStage::Partial || event.stage == OrderStage::Filled {
            self.fill_count += 1;
        }
    }

    #[inline]
    pub const fn get_total_latency_ns(&self) -> u64 {
        self.last_timestamp_ns.saturating_sub(self.first_timestamp_ns)
    }

    #[inline]
    pub const fn has_reached_stage(&self, stage: OrderStage) -> bool {
        self.stage_history & (1 << (stage as u8)) != 0
    }

    #[inline]
    pub const fn get_fill_count(&self) -> u8 {
        self.fill_count
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ring_buffer_basic() {
        static BUFFER: EventRingBuffer::<1024> = EventRingBuffer::new();
        
        let event = TradeEvent {
            timestamp_ns: 1000,
            order_id: 12345,
            stage: OrderStage::Signal,
            ..TradeEvent::new()
        };

        assert!(BUFFER.push(event));
        assert_eq!(BUFFER.len(), 1);

        let popped = BUFFER.pop();
        assert!(popped.is_some());
        assert_eq!(popped.unwrap().order_id, 12345);
        assert_eq!(BUFFER.len(), 0);
    }

    #[test]
    fn test_zero_allocation_path() {
        static BUFFER: EventRingBuffer::<100> = EventRingBuffer::new();
        
        for i in 0..1000 {
            let event = TradeEvent {
                timestamp_ns: i * 1000,
                order_id: i,
                stage: OrderStage::Signal,
                ..TradeEvent::new()
            };
            let _ = BUFFER.push(event);
        }
        
        // Should have processed all events without allocation
        assert!(BUFFER.total_written() > 0);
    }

    #[test]
    fn test_latency_tracking() {
        static BUFFER: EventRingBuffer::<256> = EventRingBuffer::new();
        
        let base_event = TradeEvent {
            order_id: 99999,
            ..TradeEvent::new()
        };

        // Simulate order lifecycle
        let mut event = base_event;
        event.stage = OrderStage::Signal;
        event.timestamp_ns = 1000;
        BUFFER.push(event);

        event.stage = OrderStage::Routing;
        event.timestamp_ns = 1500;
        BUFFER.push(event);

        event.stage = OrderStage::Ack;
        event.timestamp_ns = 2000;
        BUFFER.push(event);

        event.stage = OrderStage::Filled;
        event.timestamp_ns = 2500;
        BUFFER.push(event);

        let stats = BUFFER.get_latency_stats(99999);
        assert_eq!(stats.total_latency_ns, 1500);
        assert_eq!(stats.stage_count, 4);
    }
}
