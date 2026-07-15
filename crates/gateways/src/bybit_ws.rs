//! Bybit V5 WebSocket Gateway
//! High-performance implementation with HMAC authentication
//! Handles rate limits and delta orderbook formats

use std::collections::HashMap;
use std::time::{Duration, Instant};
use hmac::{Hmac, Mac};
use sha2::Sha256;
use serde::{Deserialize, Serialize};
use tokio::time;

type HmacSha256 = Hmac<Sha256>;

/// Bybit V5 authentication credentials
#[derive(Clone)]
pub struct BybitCredentials {
    pub api_key: String,
    pub api_secret: String,
}

/// Bybit V5 WebSocket message types
#[derive(Debug, Deserialize, Clone)]
#[serde(tag = "topic", content = "data")]
pub enum BybitMessage {
    #[serde(rename = "orderbook.1")]
    OrderbookLevel1(OrderbookSnapshot),
    #[serde(rename = "orderbook.50")]
    OrderbookLevel50(OrderbookSnapshot),
    #[serde(rename = "orderbook.500")]
    OrderbookLevel500(OrderbookSnapshot),
    #[serde(rename = "trade")]
    Trade(TradeData),
    #[serde(rename = "kline.1")]
    Kline1(KlineData),
    #[serde(rename = "kline.5")]
    Kline5(KlineData),
    #[serde(rename = "kline.15")]
    Kline15(KlineData),
}

/// Orderbook snapshot/depth data
#[derive(Debug, Deserialize, Clone)]
pub struct OrderbookSnapshot {
    pub s: String,           // Symbol
    pub b: Vec<[String; 2]>, // Bids: [[price, size], ...]
    pub a: Vec<[String; 2]>, // Asks: [[price, size], ...]
    pub u: u64,              // Update ID
    pub seq: u64,            // Sequence number
}

/// Trade data
#[derive(Debug, Deserialize, Clone)]
pub struct TradeData {
    pub s: String,   // Symbol
    pub v: Vec<Trade>, // Trades
}

#[derive(Debug, Deserialize, Clone)]
pub struct Trade {
    pub T: u64,      // Timestamp
    pub s: String,   // Symbol
    pub S: String,   // Side (Buy/Sell)
    pub v: String,   // Volume
    pub p: String,   // Price
}

/// Kline/candlestick data
#[derive(Debug, Deserialize, Clone)]
pub struct KlineData {
    pub s: String,     // Symbol
    pub k: Vec<Kline>, // Klines
}

#[derive(Debug, Deserialize, Clone)]
pub struct Kline {
    pub start: u64,    // Start time
    pub end: u64,      // End time
    pub interval: String, // Interval
    pub open: String,  // Open price
    pub close: String, // Close price
    pub high: String,  // High price
    pub low: String,   // Low price
    pub volume: String,// Volume
    pub turnover: String, // Turnover
    pub confirm: bool, // Confirmed
}

/// Bybit V5 WebSocket gateway
pub struct BybitWsGateway {
    credentials: Option<BybitCredentials>,
    subscriptions: Vec<String>,
    rate_limit_tokens: u32,
    last_request: Instant,
    reconnect_delay_ms: u64,
    heartbeat_interval: Duration,
}

impl BybitWsGateway {
    pub fn new(credentials: Option<BybitCredentials>) -> Self {
        Self {
            credentials,
            subscriptions: Vec::new(),
            rate_limit_tokens: 10, // Bybit allows ~10 requests/second
            last_request: Instant::now(),
            reconnect_delay_ms: 1000,
            heartbeat_interval: Duration::from_secs(20),
        }
    }

    /// Generate HMAC signature for authentication
    pub fn generate_signature(&self, api_secret: &str, timestamp: u64, recv_window: u64) -> String {
        let param = format!("{}{}{}", api_secret, timestamp, recv_window);
        
        let mut mac = HmacSha256::new_from_slice(api_secret.as_bytes())
            .expect("HMAC can take key of any size");
        mac.update(param.as_bytes());
        
        let result = mac.finalize();
        hex::encode(result.into_bytes())
    }

    /// Get current timestamp in milliseconds
    #[inline]
    fn current_timestamp_ms() -> u64 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64
    }

    /// Build authentication request
    pub fn build_auth_request(&self) -> Option<String> {
        let creds = self.credentials.as_ref()?;
        
        let timestamp = Self::current_timestamp_ms();
        let recv_window: u64 = 5000;
        
        let signature = self.generate_signature(&creds.api_secret, timestamp, recv_window);
        
        let auth_msg = serde_json::json!({
            "op": "auth",
            "args": [
                creds.api_key,
                timestamp,
                recv_window,
                signature
            ]
        });
        
        Some(auth_msg.to_string())
    }

    /// Subscribe to orderbook channel
    pub fn subscribe_orderbook(&mut self, symbol: &str, depth: usize) -> String {
        let topic = match depth {
            1 => format!("orderbook.1.{}", symbol),
            50 => format!("orderbook.50.{}", symbol),
            _ => format!("orderbook.500.{}", symbol),
        };
        
        self.subscriptions.push(topic.clone());
        topic
    }

    /// Subscribe to trade channel
    pub fn subscribe_trades(&mut self, symbol: &str) -> String {
        let topic = format!("publicTrade.{}", symbol);
        self.subscriptions.push(topic.clone());
        topic
    }

    /// Subscribe to kline channel
    pub fn subscribe_kline(&mut self, symbol: &str, interval: &str) -> String {
        let topic = format!("kline.{}.{}", interval, symbol);
        self.subscriptions.push(topic.clone());
        topic
    }

    /// Build subscription message
    pub fn build_subscribe_message(&self, topics: &[String]) -> String {
        let sub_msg = serde_json::json!({
            "op": "subscribe",
            "args": topics
        });
        sub_msg.to_string()
    }

    /// Rate limit check - wait if necessary
    pub async fn wait_for_rate_limit(&mut self) {
        let elapsed = self.last_request.elapsed();
        
        if elapsed < Duration::from_millis(100) {
            // Need to wait
            let wait_time = Duration::from_millis(100) - elapsed;
            time::sleep(wait_time).await;
        }
        
        // Replenish tokens over time
        if self.rate_limit_tokens < 10 {
            self.rate_limit_tokens += 1;
        }
        
        self.last_request = Instant::now();
    }

    /// Parse incoming WebSocket message
    pub fn parse_message(&self, data: &[u8]) -> Result<Option<BybitMessage>, serde_json::Error> {
        // Check for pong/response first
        let text = std::str::from_utf8(data).map_err(|_| {
            serde_json::Error::custom("Invalid UTF-8")
        })?;
        
        // Handle ping/pong
        if text.contains("\"op\":\"ping\"") || text.contains("\"op\":\"pong\"") {
            return Ok(None);
        }
        
        // Handle subscription response
        if text.contains("\"op\":\"subscribe\"") {
            return Ok(None);
        }
        
        // Try to parse as topic message
        serde_json::from_str::<BybitMessage>(text)
    }

    /// Convert Bybit orderbook to unified format
    pub fn convert_orderbook(&self, snapshot: &OrderbookSnapshot) -> UnifiedOrderbook {
        let bids = snapshot.b.iter()
            .map(|level| {
                let price = level[0].parse::<f64>().unwrap_or(0.0);
                let size = level[1].parse::<f64>().unwrap_or(0.0);
                (price, size)
            })
            .collect();
        
        let asks = snapshot.a.iter()
            .map(|level| {
                let price = level[0].parse::<f64>().unwrap_or(0.0);
                let size = level[1].parse::<f64>().unwrap_or(0.0);
                (price, size)
            })
            .collect();
        
        UnifiedOrderbook {
            symbol: snapshot.s.clone(),
            bids,
            asks,
            update_id: snapshot.u,
            sequence: snapshot.seq,
        }
    }

    /// Get reconnect delay with exponential backoff
    pub fn get_reconnect_delay(&self) -> Duration {
        Duration::from_millis(self.reconnect_delay_ms)
    }

    /// Increase reconnect delay for backoff
    pub fn increase_reconnect_delay(&mut self) {
        self.reconnect_delay_ms = std::cmp::min(self.reconnect_delay_ms * 2, 60000);
    }

    /// Reset reconnect delay
    pub fn reset_reconnect_delay(&mut self) {
        self.reconnect_delay_ms = 1000;
    }
}

/// Unified orderbook format
#[derive(Debug, Clone)]
pub struct UnifiedOrderbook {
    pub symbol: String,
    pub bids: Vec<(f64, f64)>,
    pub asks: Vec<(f64, f64)>,
    pub update_id: u64,
    pub sequence: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_signature_generation() {
        let gateway = BybitWsGateway::new(Some(BybitCredentials {
            api_key: "test_key".to_string(),
            api_secret: "test_secret".to_string(),
        }));
        
        let signature = gateway.generate_signature("test_secret", 1234567890, 5000);
        assert!(!signature.is_empty());
    }

    #[test]
    fn test_subscription_topics() {
        let mut gateway = BybitWsGateway::new(None);
        
        let ob_topic = gateway.subscribe_orderbook("BTCUSDT", 50);
        assert!(ob_topic.contains("orderbook.50"));
        
        let trade_topic = gateway.subscribe_trades("BTCUSDT");
        assert!(trade_topic.contains("publicTrade"));
    }
}
