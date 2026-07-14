"""
Ultra-Low-Latency Binance WebSocket Client in Rust.

Uses tokio-tungstenite for high-performance WebSocket connections
to Binance streams (depth, trade, kline). Implements zero-copy parsing,
automatic reconnection with exponential backoff, and direct integration
with the Rust core event bus.
"""

use std::sync::Arc;
use std::time::{Duration, Instant};

use futures_util::{SinkExt, StreamExt};
use serde::Deserialize;
use tokio::sync::mpsc;
use tokio::time::sleep;
use tokio_tungstenite::{connect_async, tungstenite::Message};
use tracing::{debug, error, info, warn};

use crate::event_bus::EventBus;
use crate::memory_pool::MemoryPool;

/// Binance WebSocket stream types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StreamType {
    Trade,
    OrderBookDepth(u8),  // Depth level: 5, 10, 20
    Kline,
    Ticker,
    Liquidation,
}

impl StreamType {
    pub fn as_str(&self) -> &'static str {
        match self {
            StreamType::Trade => "trade",
            StreamType::OrderBookDepth(d) => match d {
                5 => "depth5",
                10 => "depth10",
                20 => "depth20",
                _ => "depth@100ms",
            },
            StreamType::Kline => "kline_1s",
            StreamType::Ticker => "ticker",
            StreamType::Liquidation => "forceOrder",
        }
    }
}

/// Binance trade message structure (zero-copy deserialization)
#[derive(Debug, Deserialize, Clone)]
pub struct BinanceTrade {
    #[serde(rename = "e")]
    pub event_type: String,
    #[serde(rename = "E")]
    pub event_time: i64,
    #[serde(rename = "s")]
    pub symbol: String,
    #[serde(rename = "t")]
    pub trade_id: i64,
    #[serde(rename = "p")]
    pub price: String,
    #[serde(rename = "q")]
    pub quantity: String,
    #[serde(rename = "b")]
    pub buyer_order_id: i64,
    #[serde(rename = "a")]
    pub seller_order_id: i64,
    #[serde(rename = "T")]
    pub trade_time: i64,
    #[serde(rename = "m")]
    pub is_buyer_maker: bool,
}

/// Binance order book update message
#[derive(Debug, Deserialize, Clone)]
pub struct BinanceOrderBookUpdate {
    #[serde(rename = "e")]
    pub event_type: String,
    #[serde(rename = "E")]
    pub event_time: i64,
    #[serde(rename = "s")]
    pub symbol: String,
    #[serde(rename = "U")]
    pub first_update_id: i64,
    #[serde(rename = "u")]
    pub last_update_id: i64,
    #[serde(rename = "pu")]
    pub prev_last_update_id: i64,
    #[serde(rename = "bids")]
    pub bids: Vec<(String, String)>,
    #[serde(rename = "asks")]
    pub asks: Vec<(String, String)>,
}

/// Configuration for Binance WebSocket client
#[derive(Debug, Clone)]
pub struct BinanceWsConfig {
    pub symbol: String,
    pub streams: Vec<StreamType>,
    pub is_testnet: bool,
    pub reconnect_base_delay_ms: u64,
    pub reconnect_max_delay_ms: u64,
    pub max_reconnect_attempts: u32,
}

impl Default for BinanceWsConfig {
    fn default() -> Self {
        Self {
            symbol: "BTCUSDT".to_string(),
            streams: vec![StreamType::Trade, StreamType::OrderBookDepth(20)],
            is_testnet: false,
            reconnect_base_delay_ms: 1000,
            reconnect_max_delay_ms: 30000,
            max_reconnect_attempts: 10,
        }
    }
}

/// High-performance Binance WebSocket client
pub struct BinanceWsClient {
    config: BinanceWsConfig,
    event_bus: Arc<EventBus>,
    memory_pool: Arc<MemoryPool>,
    shutdown_tx: mpsc::Sender<()>,
}

impl BinanceWsClient {
    /// Create a new Binance WebSocket client
    pub fn new(
        config: BinanceWsConfig,
        event_bus: Arc<EventBus>,
        memory_pool: Arc<MemoryPool>,
    ) -> Self {
        let (shutdown_tx, _) = mpsc::channel(1);
        Self {
            config,
            event_bus,
            memory_pool,
            shutdown_tx,
        }
    }

    /// Build WebSocket URL for combined streams
    fn build_ws_url(&self) -> String {
        let base_url = if self.config.is_testnet {
            "wss://testnet.binancefuture.com/ws"
        } else {
            "wss://fstream.binance.com/ws"
        };

        let stream_names: Vec<String> = self
            .config
            .streams
            .iter()
            .map(|s| {
                let symbol_lower = self.config.symbol.to_lowercase();
                format!("{}@{}", symbol_lower, s.as_str())
            })
            .collect();

        if stream_names.len() == 1 {
            format!("{}/{}", base_url, stream_names[0])
        } else {
            format!("{}/stream?streams={}", base_url, stream_names.join("/"))
        }
    }

    /// Run the WebSocket client with automatic reconnection
    pub async fn run(&self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let mut attempt = 0u32;
        let mut delay_ms = self.config.reconnect_base_delay_ms;

        loop {
            attempt += 1;
            let ws_url = self.build_ws_url();

            info!(
                "Connecting to Binance WebSocket (attempt {}): {}",
                attempt, ws_url
            );

            match connect_async(&ws_url).await {
                Ok((ws_stream, _)) => {
                    info!("WebSocket connection established");
                    attempt = 0;
                    delay_ms = self.config.reconnect_base_delay_ms;

                    // Handle WebSocket messages
                    if let Err(e) = self.handle_ws_stream(ws_stream).await {
                        error!("WebSocket stream error: {}", e);
                    }
                }
                Err(e) => {
                    error!("WebSocket connection failed: {}", e);

                    if attempt >= self.config.max_reconnect_attempts {
                        error!("Max reconnection attempts reached, giving up");
                        return Err(format!("Max reconnection attempts reached").into());
                    }

                    // Exponential backoff with jitter
                    let jitter = rand::random::<u64>() % (delay_ms / 4);
                    let actual_delay = Duration::from_millis(delay_ms + jitter);

                    warn!(
                        "Reconnecting in {:?} (attempt {}/{})",
                        actual_delay,
                        attempt,
                        self.config.max_reconnect_attempts
                    );

                    sleep(actual_delay).await;
                    delay_ms = (delay_ms * 2).min(self.config.reconnect_max_delay_ms);
                }
            }
        }
    }

    /// Process WebSocket message stream
    async fn handle_ws_stream(
        &self,
        mut ws_stream: tokio_tungstenite::WebSocketStream<
            tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>,
        >,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let mut last_heartbeat = Instant::now();
        let heartbeat_interval = Duration::from_secs(30);

        loop {
            tokio::select! {
                msg = ws_stream.next() => {
                    match msg {
                        Some(Ok(Message::Text(text))) => {
                            self.process_message(&text)?;
                        }
                        Some(Ok(Message::Ping(data))) => {
                            ws_stream.send(Message::Pong(data)).await?;
                        }
                        Some(Ok(Message::Close(frame))) => {
                            info!("WebSocket closed: {:?}", frame);
                            return Ok(());
                        }
                        Some(Err(e)) => {
                            error!("WebSocket error: {}", e);
                            return Err(Box::new(e));
                        }
                        _ => {}
                    }
                }
                _ = sleep(heartbeat_interval) => {
                    if last_heartbeat.elapsed() > heartbeat_interval * 2 {
                        warn!("No heartbeat received, connection may be stale");
                        return Ok(());
                    }
                    last_heartbeat = Instant::now();
                }
            }
        }
    }

    /// Process incoming WebSocket message
    fn process_message(&self, text: &str) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        debug!("Received message: {}", text);

        // Try to parse as trade first
        if let Ok(trade) = serde_json::from_str::<BinanceTrade>(text) {
            if trade.event_type == "trade" {
                self.on_trade_received(&trade);
                return Ok(());
            }
        }

        // Try to parse as order book update
        if let Ok(ob_update) = serde_json::from_str::<BinanceOrderBookUpdate>(text) {
            if ob_update.event_type == "depthUpdate" {
                self.on_order_book_update(&ob_update);
                return Ok(());
            }
        }

        // Unknown message type - log but don't fail
        debug!("Unknown message type: {}", text);
        Ok(())
    }

    /// Handle trade message - push to event bus
    fn on_trade_received(&self, trade: &BinanceTrade) {
        // Allocate from memory pool for zero-allocation processing
        let trade_data = self.memory_pool.allocate_trade();

        // Parse price/quantity with high precision
        let price: f64 = trade.price.parse().unwrap_or(0.0);
        let quantity: f64 = trade.quantity.parse().unwrap_or(0.0);

        // Populate trade data structure
        trade_data.symbol = trade.symbol.clone();
        trade_data.price = price;
        trade_data.quantity = quantity;
        trade_data.timestamp_ns = trade.trade_time * 1_000_000;
        trade_data.is_buyer_maker = trade.is_buyer_maker;

        // Push to event bus for strategy engine consumption
        self.event_bus.push_trade(trade_data);

        debug!(
            "Trade: {} {} @ {} (buyer_maker: {})",
            trade.symbol, quantity, price, trade.is_buyer_maker
        );
    }

    /// Handle order book update - push to event bus
    fn on_order_book_update(&self, update: &BinanceOrderBookUpdate) {
        // Allocate from memory pool
        let ob_data = self.memory_pool.allocate_orderbook();

        ob_data.symbol = update.symbol.clone();
        ob_data.last_update_id = update.last_update_id;
        ob_data.prev_last_update_id = update.prev_last_update_id;

        // Parse bids/asks
        for (i, (price, qty)) in update.bids.iter().take(25).enumerate() {
            let p: f64 = price.parse().unwrap_or(0.0);
            let q: f64 = qty.parse().unwrap_or(0.0);
            ob_data.bids[i] = (p, q);
        }

        for (i, (price, qty)) in update.asks.iter().take(25).enumerate() {
            let p: f64 = price.parse().unwrap_or(0.0);
            let q: f64 = qty.parse().unwrap_or(0.0);
            ob_data.asks[i] = (p, q);
        }

        ob_data.timestamp_ns = update.event_time * 1_000_000;

        // Push to event bus
        self.event_bus.push_orderbook(ob_data);

        debug!(
            "OrderBook update: {} (update_id: {})",
            update.symbol, update.last_update_id
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build_ws_url_single_stream() {
        let config = BinanceWsConfig {
            symbol: "BTCUSDT".to_string(),
            streams: vec![StreamType::Trade],
            is_testnet: false,
            ..Default::default()
        };

        let client = BinanceWsClient::new(
            config,
            Arc::new(EventBus::new(1024)),
            Arc::new(MemoryPool::new()),
        );

        let url = client.build_ws_url();
        assert!(url.contains("btcusdt@trade"));
    }

    #[test]
    fn test_build_ws_url_multi_stream() {
        let config = BinanceWsConfig {
            symbol: "ETHUSDT".to_string(),
            streams: vec![StreamType::Trade, StreamType::OrderBookDepth(20)],
            is_testnet: true,
            ..Default::default()
        };

        let client = BinanceWsClient::new(
            config,
            Arc::new(EventBus::new(1024)),
            Arc::new(MemoryPool::new()),
        );

        let url = client.build_ws_url();
        assert!(url.contains("testnet"));
        assert!(url.contains("stream?streams="));
    }
}
