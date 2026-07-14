"""
Ingestor Main Entry Point for Binance Market Data.

Spawns WebSocket clients, reconstructs order books, and pushes
normalized Nautilus OrderBookDelta and TradeTick events to the
Rust core event bus for downstream processing.
"""

use std::sync::Arc;
use std::time::Duration;

use tokio::signal;
use tokio::sync::broadcast;
use tracing::{error, info, Level};
use tracing_subscriber::FmtSubscriber;

mod binance_ws_client;
mod orderbook_reconstructor;

use binance_ws_client::{BinanceWsClient, BinanceWsConfig, StreamType};
use orderbook_reconstructor::OrderBookReconstructor;

// Re-export core engine components (assumed to exist from Stage 1)
// In production, these would be separate crate dependencies
use core_engine::event_bus::EventBus;
use core_engine::memory_pool::MemoryPool;

/// Configuration for the ingestor
#[derive(Debug, Clone)]
pub struct IngestorConfig {
    pub symbols: Vec<String>,
    pub is_testnet: bool,
    pub order_book_depth: u8,
    pub subscribe_trades: bool,
    pub subscribe_klines: bool,
}

impl Default for IngestorConfig {
    fn default() -> Self {
        Self {
            symbols: vec!["BTCUSDT".to_string(), "ETHUSDT".to_string()],
            is_testnet: true,
            order_book_depth: 20,
            subscribe_trades: true,
            subscribe_klines: false,
        }
    }
}

/// Main ingestor service coordinating all data ingestion
pub struct MarketDataIngestor {
    config: IngestorConfig,
    event_bus: Arc<EventBus>,
    memory_pool: Arc<MemoryPool>,
    reconstructor: Arc<OrderBookReconstructor>,
    shutdown_tx: broadcast::Sender<()>,
}

impl MarketDataIngestor {
    /// Create a new market data ingestor
    pub fn new(
        config: IngestorConfig,
        event_bus: Arc<EventBus>,
        memory_pool: Arc<MemoryPool>,
    ) -> Self {
        let (shutdown_tx, _) = broadcast::channel(1);
        
        Self {
            config,
            event_bus,
            memory_pool,
            reconstructor: Arc::new(OrderBookReconstructor::new()),
            shutdown_tx,
        }
    }

    /// Run the ingestor service
    pub async fn run(&self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        info!(
            "Starting MarketDataIngestor with {} symbols",
            self.config.symbols.len()
        );

        let mut handles = Vec::new();

        // Spawn WebSocket client for each symbol
        for symbol in &self.config.symbols {
            let mut streams = Vec::new();

            // Add order book stream
            streams.push(StreamType::OrderBookDepth(self.config.order_book_depth));

            // Add trade stream
            if self.config.subscribe_trades {
                streams.push(StreamType::Trade);
            }

            // Add kline stream
            if self.config.subscribe_klines {
                streams.push(StreamType::Kline);
            }

            let ws_config = BinanceWsConfig {
                symbol: symbol.clone(),
                streams,
                is_testnet: self.config.is_testnet,
                ..Default::default()
            };

            let client = BinanceWsClient::new(
                ws_config,
                self.event_bus.clone(),
                self.memory_pool.clone(),
            );

            let reconstructor = self.reconstructor.clone();
            let mut shutdown_rx = self.shutdown_tx.subscribe();

            let handle = tokio::spawn(async move {
                info!("Starting WebSocket client for {}", symbol);

                // Run client with shutdown handling
                tokio::select! {
                    result = client.run() => {
                        if let Err(e) = result {
                            error!("WebSocket client error for {}: {}", symbol, e);
                        }
                    }
                    _ = shutdown_rx.recv() => {
                        info!("Shutdown signal received for {}", symbol);
                    }
                }
            });

            handles.push(handle);
        }

        // Wait for shutdown signal
        signal::ctrl_c().await?;
        info!("Received Ctrl+C, shutting down...");

        // Signal all clients to shutdown
        let _ = self.shutdown_tx.send(());

        // Wait for all clients to finish
        for handle in handles {
            let _ = handle.await;
        }

        // Shutdown reconstructor
        self.reconstructor.shutdown();

        // Print statistics
        let (updates, stale, gaps) = self.reconstructor.get_stats();
        info!(
            "Ingestor statistics: {} updates, {} stale, {} gaps",
            updates, stale, gaps
        );

        Ok(())
    }
}

/// Initialize logging with performance-oriented settings
fn init_logging() {
    let subscriber = FmtSubscriber::builder()
        .with_max_level(Level::INFO)
        .with_thread_ids(true)
        .with_thread_names(true)
        .with_target(false)
        .with_file(false)
        .with_line_number(false)
        .with_timer(tracing_subscriber::fmt::time::uptime())
        .finish();

    tracing::subscriber::set_global_default(subscriber)
        .expect("Failed to set tracing subscriber");
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    // Initialize logging
    init_logging();

    info!("=== Binance Market Data Ingestor ===");
    info!("Target: AMD Ryzen AI 5 / AMD Radeon GPU");
    info!("Memory Limit: 14GB system RAM");

    // Create core components
    let event_bus = Arc::new(EventBus::new(4096)); // 4K event capacity
    let memory_pool = Arc::new(MemoryPool::new());

    // Configure ingestor
    let config = IngestorConfig {
        symbols: vec![
            "BTCUSDT".to_string(),
            "ETHUSDT".to_string(),
            "SOLUSDT".to_string(),
        ],
        is_testnet: true, // Set to false for production
        order_book_depth: 20,
        subscribe_trades: true,
        subscribe_klines: false,
    };

    // Create and run ingestor
    let ingestor = MarketDataIngestor::new(config, event_bus, memory_pool);

    match ingestor.run().await {
        Ok(_) => {
            info!("Ingestor shutdown complete");
        }
        Err(e) => {
            error!("Ingestor error: {}", e);
            return Err(e);
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ingestor_config_default() {
        let config = IngestorConfig::default();
        assert!(config.is_testnet);
        assert_eq!(config.symbols.len(), 2);
        assert!(config.subscribe_trades);
    }

    #[tokio::test]
    async fn test_ingestor_creation() {
        let config = IngestorConfig::default();
        let event_bus = Arc::new(EventBus::new(100));
        let memory_pool = Arc::new(MemoryPool::new());

        let ingestor = MarketDataIngestor::new(config, event_bus, memory_pool);
        
        assert_eq!(ingestor.config.symbols.len(), 2);
    }
}
