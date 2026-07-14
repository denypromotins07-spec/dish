//! Lock-Free Event Bus with Crossbeam Channels and Ring Buffers
//! 
//! This module implements microsecond-latency event routing between the network layer
//! and the strategy engine using lock-free crossbeam channels and SPSC ring buffers.
//! Optimized for AMD Ryzen AI 5 cache topology.

use crossbeam::channel::{bounded, unbounded, Receiver, Sender, TrySendError};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;

/// Event types for the trading system
#[derive(Debug, Clone)]
pub enum Event {
    /// Market tick data from exchange
    Tick {
        symbol: String,
        timestamp_ns: u64,
        bid_price: f64,
        ask_price: f64,
        bid_size: f64,
        ask_size: f64,
    },
    /// Order book update
    OrderBookUpdate {
        symbol: String,
        timestamp_ns: u64,
        bids: Vec<(f64, f64)>,
        asks: Vec<(f64, f64)>,
    },
    /// New order request
    NewOrder {
        order_id: u64,
        symbol: String,
        side: OrderSide,
        order_type: OrderType,
        price: Option<f64>,
        quantity: f64,
        timestamp_ns: u64,
    },
    /// Order execution confirmation
    OrderFilled {
        order_id: u64,
        fill_id: u64,
        symbol: String,
        side: OrderSide,
        price: f64,
        quantity: f64,
        timestamp_ns: u64,
    },
    /// Order cancellation
    OrderCancelled {
        order_id: u64,
        symbol: String,
        timestamp_ns: u64,
    },
    /// System control event
    SystemControl(SystemControl),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OrderSide {
    Buy,
    Sell,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OrderType {
    Market,
    Limit,
    StopLoss,
    StopLimit,
}

#[derive(Debug, Clone)]
pub enum SystemControl {
    Start,
    Pause,
    Resume,
    Shutdown,
    EmergencyStop,
}

/// High-priority channel for critical events (orders, fills)
pub struct PriorityChannel {
    sender: Sender<Event>,
    receiver: Receiver<Event>,
    capacity: usize,
}

impl PriorityChannel {
    pub fn new(capacity: usize) -> Self {
        let (sender, receiver) = bounded(capacity);
        Self {
            sender,
            receiver,
            capacity,
        }
    }
    
    #[inline]
    pub fn send(&self, event: Event) -> Result<(), TrySendError<Event>> {
        self.sender.try_send(event)
    }
    
    #[inline]
    pub fn recv(&self) -> Result<Event, crossbeam::channel::TryRecvError> {
        self.receiver.try_recv()
    }
    
    pub fn len(&self) -> usize {
        self.receiver.len()
    }
    
    pub fn is_empty(&self) -> bool {
        self.receiver.is_empty()
    }
}

/// Low-priority channel for non-critical events (market data, logs)
pub struct StandardChannel {
    sender: Sender<Event>,
    receiver: Receiver<Event>,
}

impl StandardChannel {
    pub fn new(capacity: usize) -> Self {
        let (sender, receiver) = bounded(capacity);
        Self { sender, receiver }
    }
    
    #[inline]
    pub fn send(&self, event: Event) -> Result<(), TrySendError<Event>> {
        self.sender.try_send(event)
    }
    
    #[inline]
    pub fn recv(&self) -> Result<Event, crossbeam::channel::TryRecvError> {
        self.receiver.try_recv()
    }
}

/// Ultra-low latency SPSC (Single Producer Single Consumer) ring buffer
/// For tick data where allocation must be avoided
pub struct TickRingBuffer {
    buffer: Box<[Option<TickData>]>,
    capacity: usize,
    head: AtomicU64,
    tail: AtomicU64,
    overflow_count: AtomicU64,
}

#[derive(Debug, Clone)]
pub struct TickData {
    pub symbol: [u8; 12], // Fixed-size symbol storage (e.g., "BTCUSDT")
    pub timestamp_ns: u64,
    pub bid_price: f64,
    pub ask_price: f64,
    pub bid_size: f64,
    pub ask_size: f64,
    pub sequence: u64,
}

unsafe impl Send for TickRingBuffer {}
unsafe impl Sync for TickRingBuffer {}

impl TickRingBuffer {
    pub fn new(capacity: usize) -> Self {
        let mut buffer = Vec::with_capacity(capacity);
        for _ in 0..capacity {
            buffer.push(None);
        }
        
        Self {
            buffer: buffer.into_boxed_slice(),
            capacity,
            head: AtomicU64::new(0),
            tail: AtomicU64::new(0),
            overflow_count: AtomicU64::new(0),
        }
    }
    
    /// Push tick data to the ring buffer (lock-free, SPSC)
    #[inline]
    pub fn push(&self, tick: TickData) -> bool {
        let tail = self.tail.load(Ordering::Relaxed);
        let next_tail = (tail + 1) % self.capacity as u64;
        
        // Check if buffer is full
        if next_tail == self.head.load(Ordering::Acquire) {
            self.overflow_count.fetch_add(1, Ordering::Relaxed);
            return false;
        }
        
        unsafe {
            self.buffer[tail as usize] = Some(tick);
        }
        
        self.tail.store(next_tail, Ordering::Release);
        true
    }
    
    /// Pop tick data from the ring buffer (lock-free, SPSC)
    #[inline]
    pub fn pop(&self) -> Option<TickData> {
        let head = self.head.load(Ordering::Relaxed);
        
        if head == self.tail.load(Ordering::Acquire) {
            return None;
        }
        
        let tick = unsafe { self.buffer[head as usize].take() };
        
        let next_head = (head + 1) % self.capacity as u64;
        self.head.store(next_head, Ordering::Release);
        
        tick
    }
    
    pub fn len(&self) -> usize {
        let head = self.head.load(Ordering::Relaxed);
        let tail = self.tail.load(Ordering::Relaxed);
        if tail >= head {
            (tail - head) as usize
        } else {
            self.capacity - (head - tail) as usize
        }
    }
    
    pub fn is_empty(&self) -> bool {
        self.head.load(Ordering::Relaxed) == self.tail.load(Ordering::Relaxed)
    }
    
    pub fn overflow_count(&self) -> u64 {
        self.overflow_count.load(Ordering::Relaxed)
    }
}

/// Central event bus coordinating all channels
pub struct EventBus {
    priority_channel: Arc<PriorityChannel>,
    standard_channel: Arc<StandardChannel>,
    tick_buffer: Arc<TickRingBuffer>,
    running: AtomicBool,
    events_processed: AtomicU64,
    started_at: Instant,
}

impl EventBus {
    pub fn new(
        priority_capacity: usize,
        standard_capacity: usize,
        tick_buffer_capacity: usize,
    ) -> Self {
        Self {
            priority_channel: Arc::new(PriorityChannel::new(priority_capacity)),
            standard_channel: Arc::new(StandardChannel::new(standard_capacity)),
            tick_buffer: Arc::new(TickRingBuffer::new(tick_buffer_capacity)),
            running: AtomicBool::new(false),
            events_processed: AtomicU64::new(0),
            started_at: Instant::now(),
        }
    }
    
    /// Send high-priority event (orders, fills)
    #[inline]
    pub fn send_priority(&self, event: Event) -> Result<(), TrySendError<Event>> {
        self.priority_channel.send(event)
    }
    
    /// Send standard event (market data)
    #[inline]
    pub fn send_standard(&self, event: Event) -> Result<(), TrySendError<Event>> {
        self.standard_channel.send(event)
    }
    
    /// Push tick data to ring buffer
    #[inline]
    pub fn push_tick(&self, tick: TickData) -> bool {
        self.tick_buffer.push(tick)
    }
    
    /// Receive from priority channel
    #[inline]
    pub fn recv_priority(&self) -> Result<Event, crossbeam::channel::TryRecvError> {
        let event = self.priority_channel.recv();
        if event.is_ok() {
            self.events_processed.fetch_add(1, Ordering::Relaxed);
        }
        event
    }
    
    /// Receive from standard channel
    #[inline]
    pub fn recv_standard(&self) -> Result<Event, crossbeam::channel::TryRecvError> {
        let event = self.standard_channel.recv();
        if event.is_ok() {
            self.events_processed.fetch_add(1, Ordering::Relaxed);
        }
        event
    }
    
    /// Pop tick from ring buffer
    #[inline]
    pub fn pop_tick(&self) -> Option<TickData> {
        let tick = self.tick_buffer.pop();
        if tick.is_some() {
            self.events_processed.fetch_add(1, Ordering::Relaxed);
        }
        tick
    }
    
    /// Get shared reference to tick buffer for direct access
    pub fn tick_buffer(&self) -> Arc<TickRingBuffer> {
        Arc::clone(&self.tick_buffer)
    }
    
    /// Start the event bus
    pub fn start(&self) {
        self.running.store(true, Ordering::Release);
    }
    
    /// Stop the event bus
    pub fn stop(&self) {
        self.running.store(false, Ordering::Release);
    }
    
    pub fn is_running(&self) -> bool {
        self.running.load(Ordering::Acquire)
    }
    
    /// Get bus statistics
    pub fn stats(&self) -> EventBusStats {
        EventBusStats {
            priority_queue_len: self.priority_channel.len(),
            standard_queue_len: self.standard_channel.len(),
            tick_buffer_len: self.tick_buffer.len(),
            tick_overflows: self.tick_buffer.overflow_count(),
            total_events_processed: self.events_processed.load(Ordering::Relaxed),
            uptime_secs: self.started_at.elapsed().as_secs_f64(),
            events_per_second: self.events_processed.load(Ordering::Relaxed) as f64 
                / self.started_at.elapsed().as_secs_f64(),
        }
    }
}

/// Event bus statistics for monitoring
#[derive(Debug, Clone)]
pub struct EventBusStats {
    pub priority_queue_len: usize,
    pub standard_queue_len: usize,
    pub tick_buffer_len: usize,
    pub tick_overflows: u64,
    pub total_events_processed: u64,
    pub uptime_secs: f64,
    pub events_per_second: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_priority_channel() {
        let channel = PriorityChannel::new(100);
        let event = Event::Tick {
            symbol: "BTCUSDT".to_string(),
            timestamp_ns: 1234567890,
            bid_price: 50000.0,
            ask_price: 50001.0,
            bid_size: 1.5,
            ask_size: 2.0,
        };
        
        channel.send(event.clone()).unwrap();
        let received = channel.recv().unwrap();
        
        match received {
            Event::Tick { symbol, .. } => assert_eq!(symbol, "BTCUSDT"),
            _ => panic!("Wrong event type"),
        }
    }
    
    #[test]
    fn test_tick_ring_buffer() {
        let buffer = TickRingBuffer::new(1024);
        
        let tick = TickData {
            symbol: *b"BTCUSDT\0\0\0\0\0",
            timestamp_ns: 1234567890,
            bid_price: 50000.0,
            ask_price: 50001.0,
            bid_size: 1.5,
            ask_size: 2.0,
            sequence: 1,
        };
        
        assert!(buffer.push(tick.clone()));
        assert_eq!(buffer.len(), 1);
        
        let popped = buffer.pop().unwrap();
        assert_eq!(popped.timestamp_ns, tick.timestamp_ns);
        assert!(buffer.is_empty());
    }
    
    #[test]
    fn test_event_bus() {
        let bus = EventBus::new(100, 1000, 4096);
        bus.start();
        
        let tick = TickData {
            symbol: *b"ETHUSDT\0\0\0\0\0",
            timestamp_ns: 9876543210,
            bid_price: 3000.0,
            ask_price: 3000.5,
            bid_size: 5.0,
            ask_size: 4.0,
            sequence: 1,
        };
        
        assert!(bus.push_tick(tick));
        let popped = bus.pop_tick().unwrap();
        assert_eq!(popped.timestamp_ns, 9876543210);
        
        let stats = bus.stats();
        assert_eq!(stats.total_events_processed, 1);
    }
}
