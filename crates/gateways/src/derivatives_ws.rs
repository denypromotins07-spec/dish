//! High-performance WebSocket adapter for crypto options exchanges
//! Handles massive delta updates for options order books and maps to Nautilus event bus

use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::mpsc;
use tokio_tungstenite::{connect_async, WebSocketStream};
use futures_util::{stream::SplitSink, SinkExt, StreamExt};
use serde::{Deserialize, Serialize};

/// Options order book level
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct OptionsOrderBookLevel {
    pub strike: f64,
    pub expiry: String,
    pub option_type: String, // "call" or "put"
    pub bid_price: f64,
    pub bid_size: f64,
    pub ask_price: f64,
    pub ask_size: f64,
    pub mark_iv: f64,
    pub timestamp_ns: u64,
}

/// Delta update for options chain
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct OptionsChainDelta {
    pub symbol: String,
    pub updates: Vec<OptionsOrderBookLevel>,
    pub sequence: u64,
    pub is_snapshot: bool,
}

/// Parsed message from derivatives exchange
#[derive(Debug, Clone)]
pub enum DerivativesMessage {
    ChainDelta(OptionsChainDelta),
    FundingRate { symbol: String, rate: f64, timestamp_ns: u64 },
    OpenInterest { symbol: String, oi: f64, oi_usd: f64, timestamp_ns: u64 },
    Liquidation { symbol: String, side: String, quantity: f64, price: f64, value_usd: f64 },
    Heartbeat(u64),
}

/// WebSocket adapter state
pub struct DerivativesWsAdapter {
    /// WebSocket URL
    url: String,
    /// Message sender channel
    tx: mpsc::Sender<DerivativesMessage>,
    /// Subscription state
    subscriptions: HashMap<String, Vec<String>>,
    /// Sequence tracking per symbol
    sequences: HashMap<String, u64>,
    /// Reconnect attempts
    reconnect_count: u32,
    /// Max reconnect attempts
    max_reconnects: u32,
}

impl DerivativesWsAdapter {
    pub fn new(url: &str, tx: mpsc::Sender<DerivativesMessage>) -> Self {
        Self {
            url: url.to_string(),
            tx,
            subscriptions: HashMap::new(),
            sequences: HashMap::new(),
            reconnect_count: 0,
            max_reconnects: 10,
        }
    }

    /// Subscribe to options chain updates
    pub fn subscribe_options_chain(&mut self, underlying: &str) {
        self.subscriptions
            .entry("options".to_string())
            .or_insert_with(Vec::new)
            .push(underlying.to_string());
    }

    /// Subscribe to funding rate updates
    pub fn subscribe_funding_rates(&mut self, symbols: &[&str]) {
        self.subscriptions
            .entry("funding".to_string())
            .or_insert_with(Vec::new)
            .extend(symbols.iter().map(|s| s.to_string()));
    }

    /// Subscribe to open interest updates
    pub fn subscribe_open_interest(&mut self, symbols: &[&str]) {
        self.subscriptions
            .entry("oi".to_string())
            .or_insert_with(Vec::new)
            .extend(symbols.iter().map(|s| s.to_string()));
    }

    /// Subscribe to liquidation updates
    pub fn subscribe_liquidations(&mut self, symbols: &[&str]) {
        self.subscriptions
            .entry("liquidations".to_string())
            .or_insert_with(Vec::new)
            .extend(symbols.iter().map(|s| s.to_string()));
    }

    /// Build subscription message for exchange
    fn build_subscribe_message(&self) -> serde_json::Value {
        let mut params = Vec::new();

        for (channel, symbols) in &self.subscriptions {
            for symbol in symbols {
                params.push(serde_json::json!({
                    "channel": channel,
                    "symbol": symbol
                }));
            }
        }

        serde_json::json!({
            "op": "subscribe",
            "args": params
        })
    }

    /// Run the WebSocket connection with automatic reconnection
    pub async fn run(&mut self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        while self.reconnect_count < self.max_reconnects {
            match self.connect_and_run().await {
                Ok(_) => {
                    self.reconnect_count = 0; // Reset on successful long-lived connection
                }
                Err(e) => {
                    self.reconnect_count += 1;
                    eprintln!(
                        "WebSocket error (attempt {}): {}. Reconnecting...",
                        self.reconnect_count, e
                    );
                    
                    if self.reconnect_count < self.max_reconnects {
                        tokio::time::sleep(tokio::time::Duration::from_secs(
                            (self.reconnect_count as u64).min(30),
                        )).await;
                    }
                }
            }
        }

        Err("Max reconnection attempts exceeded".into())
    }

    /// Connect and run WebSocket loop
    async fn connect_and_run(&mut self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let (ws_stream, _) = connect_async(&self.url).await?;
        let (mut write, mut read) = ws_stream.split();

        // Send subscription message
        let subscribe_msg = self.build_subscribe_message();
        write.send(tokio_tungstenite::Message::Text(subscribe_msg.to_string())).await?;

        // Message processing loop
        while let Some(msg) = read.next().await {
            match msg {
                Ok(tokio_tungstenite::Message::Text(text)) => {
                    if let Some(parsed) = self.parse_message(&text) {
                        if self.tx.send(parsed).await.is_err() {
                            return Err("Channel closed".into());
                        }
                    }
                }
                Ok(tokio_tungstenite::Message::Ping(data)) => {
                    write.send(tokio_tungstenite::Message::Pong(data)).await?;
                }
                Ok(tokio_tungstenite::Message::Close(_)) => {
                    break;
                }
                Err(e) => {
                    return Err(Box::new(e));
                }
                _ => {}
            }
        }

        Ok(())
    }

    /// Parse incoming WebSocket message
    fn parse_message(&mut self, text: &str) -> Option<DerivativesMessage> {
        let value: serde_json::Value = serde_json::from_str(text).ok()?;

        // Check for heartbeat
        if let Some(op) = value.get("op").and_then(|v| v.as_str()) {
            if op == "heartbeat" {
                let ts = value.get("ts").and_then(|v| v.as_u64()).unwrap_or(0);
                return Some(DerivativesMessage::Heartbeat(ts));
            }
        }

        // Check for options chain delta (Deribit-style)
        if let Some(params) = value.get("params") {
            if let Some(data) = params.get("data") {
                if let Some(updates) = data.as_array() {
                    let parsed_updates: Vec<OptionsOrderBookLevel> = updates
                        .iter()
                        .filter_map(|u| serde_json::from_value(u.clone()).ok())
                        .collect();

                    if !parsed_updates.is_empty() {
                        let symbol = value
                            .get("params")
                            .and_then(|p| p.get("instrument"))
                            .and_then(|i| i.as_str())
                            .unwrap_or("UNKNOWN")
                            .to_string();

                        let seq = value
                            .get("params")
                            .and_then(|p| p.get("change_id"))
                            .and_then(|c| c.as_u64())
                            .unwrap_or(0);

                        return Some(DerivativesMessage::ChainDelta(OptionsChainDelta {
                            symbol,
                            updates: parsed_updates,
                            sequence: seq,
                            is_snapshot: false,
                        }));
                    }
                }
            }
        }

        // Check for funding rate update
        if let Some(funding) = value.get("funding_rate") {
            if let Some(rate) = funding.as_f64() {
                let symbol = value.get("symbol").and_then(|v| v.as_str()).unwrap_or("").to_string();
                let ts = value.get("timestamp").and_then(|v| v.as_u64()).unwrap_or(0);
                return Some(DerivativesMessage::FundingRate { symbol, rate, timestamp_ns: ts });
            }
        }

        // Check for open interest update
        if let Some(oi) = value.get("open_interest") {
            if let Some(oi_val) = oi.as_f64() {
                let symbol = value.get("symbol").and_then(|v| v.as_str()).unwrap_or("").to_string();
                let oi_usd = value.get("open_interest_usd").and_then(|v| v.as_f64()).unwrap_or(0.0);
                let ts = value.get("timestamp").and_then(|v| v.as_u64()).unwrap_or(0);
                return Some(DerivativesMessage::OpenInterest { symbol, oi: oi_val, oi_usd, timestamp_ns: ts });
            }
        }

        // Check for liquidation update
        if let Some(liq) = value.get("liquidation") {
            let symbol = liq.get("symbol").and_then(|v| v.as_str()).unwrap_or("").to_string();
            let side = liq.get("side").and_then(|v| v.as_str()).unwrap_or("").to_string();
            let quantity = liq.get("quantity").and_then(|v| v.as_f64()).unwrap_or(0.0);
            let price = liq.get("price").and_then(|v| v.as_f64()).unwrap_or(0.0);
            let value_usd = liq.get("value_usd").and_then(|v| v.as_f64()).unwrap_or(0.0);
            
            return Some(DerivativesMessage::Liquidation { symbol, side, quantity, price, value_usd });
        }

        None
    }

    /// Get current sequence number for a symbol
    pub fn get_sequence(&self, symbol: &str) -> u64 {
        *self.sequences.get(symbol).unwrap_or(&0)
    }

    /// Update sequence number
    pub fn update_sequence(&mut self, symbol: &str, seq: u64) {
        self.sequences.insert(symbol.to_string(), seq);
    }
}

/// Nautilus event bus bridge for derivatives messages
pub struct NautilusDerivativesBridge {
    rx: mpsc::Receiver<DerivativesMessage>,
    /// Event count for monitoring
    event_count: u64,
}

impl NautilusDerivativesBridge {
    pub fn new(rx: mpsc::Receiver<DerivativesMessage>) -> Self {
        Self { rx, event_count: 0 }
    }

    /// Process messages and forward to Nautilus
    pub async fn process_messages<F>(&mut self, mut handler: F) -> Result<(), Box<dyn std::error::Error>>
    where
        F: FnMut(DerivativesMessage) -> futures_util::future::BoxFuture<'static, ()>,
    {
        use futures_util::future::FutureExt;

        while let Some(msg) = self.rx.recv().await {
            self.event_count += 1;
            handler(msg).await;
        }

        Ok(())
    }

    /// Get event count
    pub fn event_count(&self) -> u64 {
        self.event_count
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_heartbeat() {
        let json = r#"{"op": "heartbeat", "ts": 1234567890}"#;
        
        let (tx, _rx) = mpsc::channel(100);
        let mut adapter = DerivativesWsAdapter::new("ws://test", tx);
        
        let msg = adapter.parse_message(json);
        assert!(matches!(msg, Some(DerivativesMessage::Heartbeat(_))));
    }

    #[test]
    fn test_subscription_building() {
        let (tx, _rx) = mpsc::channel(100);
        let mut adapter = DerivativesWsAdapter::new("ws://test", tx);
        
        adapter.subscribe_options_chain("BTC");
        adapter.subscribe_funding_rates(&["BTC-PERP"]);
        
        let sub_msg = adapter.build_subscribe_message();
        assert!(sub_msg.get("op").is_some());
    }
}
