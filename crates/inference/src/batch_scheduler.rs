//! Micro-batching scheduler for optimal inference throughput.
//! Groups incoming tick events into optimal batch sizes to maximize AMD GPU/NPU utilization
//! without introducing unacceptable latency.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicUsize, AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use std::thread;

/// Configuration for the batch scheduler
#[derive(Debug, Clone)]
pub struct BatchSchedulerConfig {
    /// Target batch size for optimal GPU utilization
    pub target_batch_size: usize,
    /// Minimum batch size before forcing execution
    pub min_batch_size: usize,
    /// Maximum wait time for batch accumulation (latency budget)
    pub max_wait_time_ms: u64,
    /// Maximum queue size before dropping old events
    pub max_queue_size: usize,
    /// Enable dynamic batch sizing based on load
    pub dynamic_batch_sizing: bool,
    /// Minimum inter-batch interval (throughput control)
    pub min_batch_interval_ms: u64,
}

impl Default for BatchSchedulerConfig {
    fn default() -> Self {
        Self {
            target_batch_size: 32,      // Optimal for most GPU workloads
            min_batch_size: 8,          // Don't wait too long for small batches
            max_wait_time_ms: 1,        // 1ms latency budget
            max_queue_size: 1024,       // Prevent memory bloat
            dynamic_batch_sizing: true, // Adapt to load
            min_batch_interval_ms: 0,   // No minimum interval by default
        }
    }
}

/// A single tick event awaiting batching
#[derive(Debug, Clone)]
pub struct TickEvent {
    pub timestamp: Instant,
    pub symbol: String,
    pub data: Vec<f32>,
    pub priority: u8,  // Higher = more urgent
    pub correlation_id: u64,
}

impl TickEvent {
    pub fn new(symbol: &str, data: Vec<f32>, priority: u8) -> Self {
        Self {
            timestamp: Instant::now(),
            symbol: symbol.to_string(),
            data,
            priority,
            correlation_id: 0,
        }
    }
    
    /// Check if this event has exceeded its latency budget
    pub fn is_stale(&self, max_age_ms: u64) -> bool {
        self.timestamp.elapsed().as_millis() as u64 > max_age_ms
    }
}

/// A ready-to-process batch of tick events
#[derive(Debug, Clone)]
pub struct TickBatch {
    pub events: Vec<TickEvent>,
    pub created_at: Instant,
    pub batch_id: u64,
    pub symbols: Vec<String>,
}

impl TickBatch {
    pub fn new(events: Vec<TickEvent>, batch_id: u64) -> Self {
        let symbols: Vec<String> = events.iter().map(|e| e.symbol.clone()).collect();
        Self {
            events,
            created_at: Instant::now(),
            batch_id,
            symbols,
        }
    }
    
    /// Get the age of the oldest event in the batch
    pub fn oldest_event_age_ms(&self) -> u128 {
        self.events.first()
            .map(|e| e.timestamp.elapsed().as_millis())
            .unwrap_or(0)
    }
}

/// Statistics for monitoring scheduler performance
#[derive(Debug, Clone, Default)]
pub struct SchedulerStats {
    pub total_events_received: u64,
    pub total_batches_created: u64,
    pub total_events_dropped: u64,
    pub avg_batch_size: f64,
    pub avg_latency_us: f64,
    pub max_latency_us: f64,
    pub current_queue_depth: usize,
    pub batches_per_second: f64,
}

/// Micro-batching scheduler for tick events
pub struct BatchScheduler {
    config: BatchSchedulerConfig,
    is_running: AtomicBool,
    batch_counter: AtomicUsize,
    
    // Event queue
    event_queue: Arc<std::sync::Mutex<VecDeque<TickEvent>>>,
    
    // High-priority queue (for urgent events)
    priority_queue: Arc<std::sync::Mutex<VecDeque<TickEvent>>>,
    
    // Statistics
    stats: Arc<std::sync::RwLock<SchedulerStats>>,
    
    // Last batch creation time (for rate limiting)
    last_batch_time: Arc<std::sync::Mutex<Option<Instant>>>,
    
    // Dynamic batch size tracking
    current_target_batch: AtomicUsize,
}

impl BatchScheduler {
    /// Create a new batch scheduler
    pub fn new(config: BatchSchedulerConfig) -> Self {
        let initial_target = config.target_batch_size;
        
        Self {
            config,
            is_running: AtomicBool::new(false),
            batch_counter: AtomicUsize::new(0),
            event_queue: Arc::new(std::sync::Mutex::new(
                VecDeque::with_capacity(config.max_queue_size)
            )),
            priority_queue: Arc::new(std::sync::Mutex::new(
                VecDeque::with_capacity(config.max_queue_size / 4)
            )),
            stats: Arc::new(std::sync::RwLock::new(SchedulerStats::default())),
            last_batch_time: Arc::new(std::sync::Mutex::new(None)),
            current_target_batch: AtomicUsize::new(initial_target),
        }
    }
    
    /// Submit a tick event for batching
    pub fn submit(&self, event: TickEvent) -> Result<(), &'static str> {
        if !self.is_running.load(Ordering::Relaxed) {
            return Err("Scheduler not running");
        }
        
        // Update stats
        {
            let mut stats = self.stats.write().unwrap();
            stats.total_events_received += 1;
        }
        
        // Route to appropriate queue based on priority
        if event.priority >= 5 {
            // High priority
            let mut pq = self.priority_queue.lock().unwrap();
            if pq.len() >= pq.capacity() {
                // Drop oldest high-priority event if full
                pq.pop_front();
                let mut stats = self.stats.write().unwrap();
                stats.total_events_dropped += 1;
            }
            pq.push_back(event);
        } else {
            // Normal priority
            let mut eq = self.event_queue.lock().unwrap();
            if eq.len() >= eq.capacity() {
                // Drop oldest event if full
                eq.pop_front();
                let mut stats = self.stats.write().unwrap();
                stats.total_events_dropped += 1;
            }
            eq.push_back(event);
        }
        
        Ok(())
    }
    
    /// Start the scheduler background thread
    pub fn start(&self) -> std::thread::JoinHandle<()> {
        self.is_running.store(true, Ordering::Relaxed);
        
        let event_queue = Arc::clone(&self.event_queue);
        let priority_queue = Arc::clone(&self.priority_queue);
        let stats = Arc::clone(&self.stats);
        let last_batch_time = Arc::clone(&self.last_batch_time);
        let config = self.config.clone();
        let batch_counter = &self.batch_counter;
        let current_target = &self.current_target_batch;
        
        thread::spawn(move || {
            let mut batch_count = 0u64;
            let mut last_stats_time = Instant::now();
            
            while !event_queue.lock().unwrap().is_empty() || 
                  !priority_queue.lock().unwrap().is_empty() {
                
                // Check if we should create a batch
                if let Some(batch) = Self::try_create_batch(
                    &event_queue,
                    &priority_queue,
                    &config,
                    &last_batch_time,
                    current_target.load(Ordering::Relaxed),
                ) {
                    // Update timing
                    *last_batch_time.lock().unwrap() = Some(Instant::now());
                    
                    // Update stats
                    {
                        let mut s = stats.write().unwrap();
                        s.total_batches_created += 1;
                        batch_count += 1;
                        
                        // Update average batch size
                        let total = s.total_batches_created as f64;
                        s.avg_batch_size = (s.avg_batch_size * (total - 1.0) + 
                                           batch.events.len() as f64) / total;
                        
                        // Update latency
                        let latency_us = batch.oldest_event_age_ms() as f64 * 1000.0;
                        s.max_latency_us = s.max_latency_us.max(latency_us);
                        s.avg_latency_us = (s.avg_latency_us * (total - 1.0) + latency_us) / total;
                    }
                    
                    // Here you would send the batch to the inference engine
                    // For now, we just track it
                    batch_counter.fetch_add(1, Ordering::Relaxed);
                }
                
                // Update throughput stats periodically
                if last_stats_time.elapsed().as_secs() >= 1 {
                    let elapsed = last_stats_time.elapsed().as_secs_f64();
                    let mut s = stats.write().unwrap();
                    s.batches_per_second = batch_count as f64 / elapsed;
                    s.current_queue_depth = event_queue.lock().unwrap().len() + 
                                           priority_queue.lock().unwrap().len();
                    batch_count = 0;
                    last_stats_time = Instant::now();
                }
                
                // Small sleep to prevent busy-waiting
                thread::sleep(Duration::from_micros(10));
            }
            
            self.is_running.store(false, Ordering::Relaxed);
        })
    }
    
    /// Try to create a batch from available events
    fn try_create_batch(
        event_queue: &std::sync::Mutex<VecDeque<TickEvent>>,
        priority_queue: &std::sync::Mutex<VecDeque<TickEvent>>,
        config: &BatchSchedulerConfig,
        last_batch_time: &std::sync::Mutex<Option<Instant>>,
        target_batch: usize,
    ) -> Option<TickBatch> {
        let now = Instant::now();
        
        // Check rate limiting
        {
            let lbt = last_batch_time.lock().unwrap();
            if let Some(last) = *lbt {
                if config.min_batch_interval_ms > 0 {
                    let elapsed_ms = last.elapsed().as_millis() as u64;
                    if elapsed_ms < config.min_batch_interval_ms {
                        return None;
                    }
                }
            }
        }
        
        // Count available events
        let eq = event_queue.lock().unwrap();
        let pq = priority_queue.lock().unwrap();
        let total_available = eq.len() + pq.len();
        
        if total_available == 0 {
            return None;
        }
        
        // Check if we have enough events or exceeded time budget
        let should_batch = total_available >= config.min_batch_size ||
                          total_available >= target_batch ||
                          pq.iter().any(|e| e.is_stale(config.max_wait_time_ms)) ||
                          (eq.front().map(|e| e.is_stale(config.max_wait_time_ms)).unwrap_or(false));
        
        if !should_batch {
            return None;
        }
        
        // Collect events for batch (priority first)
        let mut events = Vec::with_capacity(total_available.min(target_batch));
        
        // Take all priority events first
        while let Some(event) = pq.front() {
            if events.len() >= target_batch {
                break;
            }
            if let Some(e) = pq.pop_front() {
                events.push(e);
            }
        }
        
        // Fill with normal events
        while let Some(event) = eq.front() {
            if events.len() >= target_batch {
                break;
            }
            if let Some(e) = eq.pop_front() {
                events.push(e);
            }
        }
        
        drop(eq);
        drop(pq);
        
        if events.is_empty() {
            return None;
        }
        
        Some(TickBatch::new(events, 0))
    }
    
    /// Stop the scheduler
    pub fn stop(&self) {
        self.is_running.store(false, Ordering::Relaxed);
    }
    
    /// Check if scheduler is running
    pub fn is_running(&self) -> bool {
        self.is_running.load(Ordering::Relaxed)
    }
    
    /// Get current statistics
    pub fn get_stats(&self) -> SchedulerStats {
        self.stats.read().unwrap().clone()
    }
    
    /// Adjust target batch size dynamically
    pub fn adjust_batch_size(&self, new_target: usize) {
        let clamped = new_target.clamp(self.config.min_batch_size, 128);
        self.current_target_batch.store(clamped, Ordering::Relaxed);
    }
    
    /// Clear all pending events
    pub fn clear(&self) {
        self.event_queue.lock().unwrap().clear();
        self.priority_queue.lock().unwrap().clear();
    }
}

/// Builder for creating batch schedulers with fluent API
pub struct BatchSchedulerBuilder {
    config: BatchSchedulerConfig,
}

impl BatchSchedulerBuilder {
    pub fn new() -> Self {
        Self {
            config: BatchSchedulerConfig::default(),
        }
    }
    
    pub fn target_batch_size(mut self, size: usize) -> Self {
        self.config.target_batch_size = size;
        self
    }
    
    pub fn min_batch_size(mut self, size: usize) -> Self {
        self.config.min_batch_size = size;
        self
    }
    
    pub fn max_wait_time_ms(mut self, ms: u64) -> Self {
        self.config.max_wait_time_ms = ms;
        self
    }
    
    pub fn max_queue_size(mut self, size: usize) -> Self {
        self.config.max_queue_size = size;
        self
    }
    
    pub fn min_batch_interval_ms(mut self, ms: u64) -> Self {
        self.config.min_batch_interval_ms = ms;
        self
    }
    
    pub fn build(self) -> BatchScheduler {
        BatchScheduler::new(self.config)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_scheduler_creation() {
        let scheduler = BatchSchedulerBuilder::new()
            .target_batch_size(32)
            .min_batch_size(8)
            .max_wait_time_ms(1)
            .build();
        
        assert!(!scheduler.is_running());
    }
    
    #[test]
    fn test_event_submission() {
        let scheduler = BatchScheduler::new(BatchSchedulerConfig::default());
        scheduler.is_running.store(true, Ordering::Relaxed);
        
        let event = TickEvent::new("BTC/USD", vec![1.0, 2.0, 3.0], 3);
        let result = scheduler.submit(event);
        
        assert!(result.is_ok());
        
        let stats = scheduler.get_stats();
        assert_eq!(stats.total_events_received, 1);
    }
    
    #[test]
    fn test_batch_creation_logic() {
        let config = BatchSchedulerConfig {
            target_batch_size: 4,
            min_batch_size: 2,
            max_wait_time_ms: 100,
            ..Default::default()
        };
        
        let mut queue = VecDeque::new();
        
        // Add events
        for i in 0..3 {
            queue.push_back(TickEvent::new("TEST", vec![i as f32], 0));
        }
        
        // Should not create batch yet (below min_batch_size threshold for fresh events)
        let mutex_queue = std::sync::Mutex::new(queue);
        let priority_queue = std::sync::Mutex::new(VecDeque::new());
        let last_batch = std::sync::Mutex::new(None::<Instant>);
        
        // Force batch by making events stale
        thread::sleep(Duration::from_millis(150));
        
        let batch = BatchScheduler::try_create_batch(
            &mutex_queue,
            &priority_queue,
            &config,
            &last_batch,
            4,
        );
        
        assert!(batch.is_some());
        assert_eq!(batch.unwrap().events.len(), 3);
    }
}
