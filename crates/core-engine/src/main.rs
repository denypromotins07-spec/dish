//! Core Engine Main Entry Point
//! 
//! This module binds the memory pool and event bus, applying strict thread affinity
//! to pin threads to specific AMD L3 cache slices for minimal latency.

mod memory_pool;
mod event_bus;
mod hardware_bind;

use memory_pool::{MemoryPool, MemoryPoolConfig};
use event_bus::{EventBus, Event, OrderSide, OrderType, TickData};
use hardware_bind::ThreadAffinity;

use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};
use std::sync::atomic::{AtomicBool, Ordering};

/// Core engine configuration loaded from core_config.toml
#[derive(Debug)]
pub struct EngineConfig {
    pub network_io_cores: Vec<usize>,
    pub event_engine_cores: Vec<usize>,
    pub strategy_cores: Vec<usize>,
    pub order_book_pool_size_mb: usize,
    pub tick_data_pool_size_mb: usize,
    pub priority_channel_capacity: usize,
    pub standard_channel_capacity: usize,
    pub tick_buffer_capacity: usize,
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            network_io_cores: vec![0, 1],
            event_engine_cores: vec![2, 3],
            strategy_cores: vec![4, 5],
            order_book_pool_size_mb: 512,
            tick_data_pool_size_mb: 1024,
            priority_channel_capacity: 10000,
            standard_channel_capacity: 100000,
            tick_buffer_capacity: 65536,
        }
    }
}

/// Main trading engine
pub struct TradingEngine {
    config: EngineConfig,
    memory_pool: Arc<MemoryPool>,
    event_bus: Arc<EventBus>,
    running: AtomicBool,
}

impl TradingEngine {
    /// Create a new trading engine with the given configuration
    pub fn new(config: EngineConfig) -> Option<Self> {
        // Calculate slab count based on configured pool sizes (reduced for 10GB limit)
        let slab_size = 4096;
        let order_book_slabs = (config.order_book_pool_size_mb * 1024 * 1024) / slab_size;
        
        // Ensure we don't exceed memory limits (max 512MB for order book pool)
        let order_book_slabs = order_book_slabs.min(131072); // Cap at 512MB
        
        let pool_config = MemoryPoolConfig {
            slab_size,
            num_slabs: order_book_slabs,
            alignment: 64, // Cache line alignment
        };
        
        let memory_pool = Arc::new(MemoryPool::new(pool_config)?);
        
        let event_bus = Arc::new(EventBus::new(
            config.priority_channel_capacity,
            config.standard_channel_capacity,
            config.tick_buffer_capacity,
        ));
        
        Some(Self {
            config,
            memory_pool,
            event_bus,
            running: AtomicBool::new(false),
        })
    }
    
    /// Start the trading engine
    pub fn start(&self) {
        self.running.store(true, Ordering::Release);
        self.event_bus.start();
        
        println!("[ENGINE] Starting Trading Engine...");
        println!("[ENGINE] Memory Pool: {} MB", self.memory_pool.stats().pool_size_mb);
        println!("[ENGINE] Event Bus initialized");
        
        // Spawn network I/O thread pinned to cores 0-1
        let network_cores = self.config.network_io_cores.clone();
        let event_bus_rx = Arc::clone(&self.event_bus);
        let mem_pool = Arc::clone(&self.memory_pool);
        let _network_thread = thread::spawn(move || {
            if let Err(e) = ThreadAffinity::pin_to_cpus(&network_cores) {
                eprintln!("[NETWORK] Failed to pin thread: {}", e);
            }
            println!("[NETWORK] Thread pinned to cores: {:?}", network_cores);
            network_io_loop(event_bus_rx, mem_pool);
        });
        
        // Spawn event processing thread pinned to cores 2-3
        let event_cores = self.config.event_engine_cores.clone();
        let event_bus_tx = Arc::clone(&self.event_bus);
        let _event_thread = thread::spawn(move || {
            if let Err(e) = ThreadAffinity::pin_to_cpus(&event_cores) {
                eprintln!("[EVENT] Failed to pin thread: {}", e);
            }
            println!("[EVENT] Thread pinned to cores: {:?}", event_cores);
            event_processing_loop(event_bus_tx);
        });
        
        // Spawn strategy thread pinned to cores 4-5
        let strategy_cores = self.config.strategy_cores.clone();
        let event_bus_strategy = Arc::clone(&self.event_bus);
        let _strategy_thread = thread::spawn(move || {
            if let Err(e) = ThreadAffinity::pin_to_cpus(&strategy_cores) {
                eprintln!("[STRATEGY] Failed to pin thread: {}", e);
            }
            println!("[STRATEGY] Thread pinned to cores: {:?}", strategy_cores);
            strategy_loop(event_bus_strategy);
        });
        
        println!("[ENGINE] All threads started");
    }
    
    /// Stop the trading engine gracefully
    pub fn stop(&self) {
        println!("[ENGINE] Stopping Trading Engine...");
        self.running.store(false, Ordering::Release);
        self.event_bus.stop();
        
        // Log final statistics
        let pool_stats = self.memory_pool.stats();
        let bus_stats = self.event_bus.stats();
        
        println!("[ENGINE] === Final Statistics ===");
        println!("[ENGINE] Memory Pool:");
        println!("  - Total Allocations: {}", pool_stats.total_allocations);
        println!("  - Peak Usage: {}", pool_stats.peak_usage);
        println!("  - Pool Size: {} MB", pool_stats.pool_size_mb);
        println!("[ENGINE] Event Bus:");
        println!("  - Events Processed: {}", bus_stats.total_events_processed);
        println!("  - Events/sec: {:.2}", bus_stats.events_per_second);
        println!("  - Tick Overflows: {}", bus_stats.tick_overflows);
    }
    
    pub fn is_running(&self) -> bool {
        self.running.load(Ordering::Acquire)
    }
    
    /// Get engine statistics
    pub fn stats(&self) -> EngineStats {
        EngineStats {
            memory_pool: self.memory_pool.stats(),
            event_bus: self.event_bus.stats(),
            uptime_secs: self.event_bus.stats().uptime_secs,
        }
    }
}

/// Engine statistics
#[derive(Debug)]
pub struct EngineStats {
    pub memory_pool: memory_pool::PoolStats,
    pub event_bus: event_bus::EventBusStats,
    pub uptime_secs: f64,
}

/// Network I/O loop - receives WebSocket data and pushes to event bus
fn network_io_loop(event_bus: Arc<EventBus>, _memory_pool: Arc<MemoryPool>) {
    println!("[NETWORK] Network I/O loop started");
    let mut tick_count: u64 = 0;
    
    while true { // In production, check a shutdown flag
        // Simulate receiving tick data from exchange
        tick_count += 1;
        
        let tick = TickData {
            symbol: *b"BTCUSDT\0\0\0\0\0",
            timestamp_ns: get_timestamp_ns(),
            bid_price: 50000.0 + (tick_count as f64 * 0.01),
            ask_price: 50000.5 + (tick_count as f64 * 0.01),
            bid_size: 1.5,
            ask_size: 2.0,
            sequence: tick_count,
        };
        
        if !event_bus.push_tick(tick) {
            eprintln!("[NETWORK] Tick buffer full, dropping tick");
        }
        
        // Small delay to simulate realistic tick rate
        thread::sleep(Duration::from_micros(100));
        
        if tick_count % 10000 == 0 {
            let stats = event_bus.stats();
            println!("[NETWORK] Ticks sent: {}, Buffer len: {}, Overflows: {}", 
                     tick_count, stats.tick_buffer_len, stats.tick_overflows);
        }
    }
}

/// Event processing loop - processes events from channels
fn event_processing_loop(event_bus: Arc<EventBus>) {
    println!("[EVENT] Event processing loop started");
    let mut processed: u64 = 0;
    
    while true {
        // Process priority events first (orders, fills)
        while let Ok(event) = event_bus.recv_priority() {
            processed += 1;
            match event {
                Event::NewOrder { order_id, symbol, .. } => {
                    // Process order
                    if processed % 1000 == 0 {
                        println!("[EVENT] Processed order {} for {}", order_id, symbol);
                    }
                }
                _ => {}
            }
        }
        
        // Process standard events
        while let Ok(event) = event_bus.recv_standard() {
            processed += 1;
            match event {
                Event::OrderBookUpdate { symbol, .. } => {
                    if processed % 1000 == 0 {
                        println!("[EVENT] Updated order book for {}", symbol);
                    }
                }
                _ => {}
            }
        }
        
        // Yield to avoid busy-waiting
        thread::yield_now();
    }
}

/// Strategy loop - consumes ticks and generates orders
fn strategy_loop(event_bus: Arc<EventBus>) {
    println!("[STRATEGY] Strategy loop started");
    let mut order_counter: u64 = 0;
    
    while true {
        if let Some(tick) = event_bus.pop_tick() {
            // Simple mock strategy: generate order every 100 ticks
            if tick.sequence % 100 == 0 {
                order_counter += 1;
                let order_event = Event::NewOrder {
                    order_id: order_counter,
                    symbol: "BTCUSDT".to_string(),
                    side: OrderSide::Buy,
                    order_type: OrderType::Limit,
                    price: Some(tick.bid_price),
                    quantity: 0.1,
                    timestamp_ns: get_timestamp_ns(),
                };
                
                if let Err(_) = event_bus.send_priority(order_event) {
                    eprintln!("[STRATEGY] Priority channel full, dropping order");
                }
            }
        } else {
            thread::yield_now();
        }
    }
}

/// Get current timestamp in nanoseconds
#[inline]
fn get_timestamp_ns() -> u64 {
    use std::time::SystemTime;
    SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos() as u64
}

fn main() {
    println!("=== Crypto Trading Bot - Core Engine ===");
    println!("Hardware Target: AMD Ryzen AI 5 (Zen 4)");
    println!("Memory Limit: 14GB");
    println!();
    
    let config = EngineConfig::default();
    
    match TradingEngine::new(config) {
        Some(engine) => {
            engine.start();
            
            // Run for demonstration (in production, run until shutdown signal)
            thread::sleep(Duration::from_secs(5));
            
            engine.stop();
        }
        None => {
            eprintln!("[ERROR] Failed to initialize trading engine");
            std::process::exit(1);
        }
    }
    
    println!();
    println!("=== Engine Shutdown Complete ===");
}
