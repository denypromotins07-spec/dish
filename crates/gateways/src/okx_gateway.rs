//! OKX REST and WebSocket Gateway
//! Handles demo/trading account routing, instrument ID mapping
//! Distinct order execution semantics

use std::collections::HashMap;
use std::time::{Duration, Instant};
use hmac::{Hmac, Mac};
use sha2::Sha256;
use base64::{Engine as _, engine::general_purpose::STANDARD as BASE64};
use serde::{Deserialize, Serialize};
use tokio::time;

type HmacSha256 = Hmac<Sha256>;

/// OKX API credentials
#[derive(Clone)]
pub struct OkxCredentials {
    pub api_key: String,
    pub api_secret: String,
    pub passphrase: String,
}

/// OKX account type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OkxAccountType {
    Demo,
    Live,
}

/// OKX instrument types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
pub enum OkxInstType {
    #[serde(rename = "SPOT")]
    Spot,
    #[serde(rename = "SWAP")]
    Swap,
    #[serde(rename = "FUTURES")]
    Futures,
    #[serde(rename = "OPTION")]
    Option,
}

/// OKX order side
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
pub enum OkxSide {
    #[serde(rename = "buy")]
    Buy,
    #[serde(rename = "sell")]
    Sell,
}

/// OKX position side
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
pub enum OkxPosSide {
    #[serde(rename = "long")]
    Long,
    #[serde(rename = "short")]
    Short,
    #[serde(rename = "net")]
    Net,
}

/// OKX order type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
pub enum OkxOrderType {
    #[serde(rename = "market")]
    Market,
    #[serde(rename = "limit")]
    Limit,
    #[serde(rename = "post_only")]
    PostOnly,
    #[serde(rename = "fok")]
    FOK,
    #[serde(rename = "ioc")]
    IOC,
}

/// OKX order status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
pub enum OkxOrderState {
    #[serde(rename = "live")]
    Live,
    #[serde(rename = "partially_filled")]
    PartiallyFilled,
    #[serde(rename = "filled")]
    Filled,
    #[serde(rename = "canceled")]
    Canceled,
    #[serde(rename = "mmp_canceled")]
    MmpCanceled,
}

/// OKX instrument ID (exchange-specific format)
#[derive(Debug, Clone)]
pub struct OkxInstrumentId {
    pub base_currency: String,
    pub quote_currency: String,
    pub inst_type: OkxInstType,
    pub expiry: Option<String>,
    pub strike: Option<String>,
    pub option_type: Option<String>,
}

impl OkxInstrumentId {
    /// Convert to OKX instrument ID string format
    pub fn to_okx_id(&self) -> String {
        match self.inst_type {
            OkxInstType::Spot => {
                format!("{}-{}", self.base_currency, self.quote_currency)
            }
            OkxInstType::Swap => {
                format!("{}-{}-SWAP", self.base_currency, self.quote_currency)
            }
            OkxInstType::Futures => {
                if let Some(expiry) = &self.expiry {
                    format!("{}-{}-{}", self.base_currency, self.quote_currency, expiry)
                } else {
                    format!("{}-{}-FUTURES", self.base_currency, self.quote_currency)
                }
            }
            OkxInstType::Option => {
                format!(
                    "{}-{}-{}-{}-{}",
                    self.base_currency,
                    self.quote_currency,
                    self.expiry.as_deref().unwrap_or(""),
                    self.strike.as_deref().unwrap_or(""),
                    self.option_type.as_deref().unwrap_or("C")
                )
            }
        }
    }

    /// Parse from OKX instrument ID string
    pub fn from_okx_id(inst_id: &str) -> Option<Self> {
        let parts: Vec<&str> = inst_id.split('-').collect();
        
        if parts.len() < 2 {
            return None;
        }

        let inst_type = if parts.len() >= 3 {
            match parts[2] {
                "SWAP" => OkxInstType::Swap,
                "FUTURES" | _ if parts.len() == 3 => OkxInstType::Futures,
                _ => OkxInstType::Spot,
            }
        } else {
            OkxInstType::Spot
        };

        Some(Self {
            base_currency: parts[0].to_string(),
            quote_currency: parts.get(1).map(|s| s.to_string()).unwrap_or_default(),
            inst_type,
            expiry: parts.get(2).map(|s| s.to_string()),
            strike: None,
            option_type: None,
        })
    }
}

/// OKX order request
#[derive(Debug, Serialize)]
pub struct OkxPlaceOrderRequest {
    #[serde(rename = "instId")]
    pub inst_id: String,
    #[serde(rename = "tdMode")]
    pub td_mode: String,
    #[serde(rename = "side")]
    pub side: OkxSide,
    #[serde(rename = "posSide", skip_serializing_if = "Option::is_none")]
    pub pos_side: Option<OkxPosSide>,
    #[serde(rename = "ordType")]
    pub ord_type: OkxOrderType,
    #[serde(rename = "sz")]
    pub sz: String,
    #[serde(rename = "px", skip_serializing_if = "Option::is_none")]
    pub px: Option<String>,
    #[serde(rename = "ccy", skip_serializing_if = "Option::is_none")]
    pub ccy: Option<String>,
    #[serde(rename = "reduceOnly", skip_serializing_if = "Option::is_none")]
    pub reduce_only: Option<bool>,
    #[serde(rename = "tgtCcy", skip_serializing_if = "Option::is_none")]
    pub tgt_ccy: Option<String>,
    #[serde(rename = "banAmend", skip_serializing_if = "Option::is_none")]
    pub ban_amend: Option<bool>,
    #[serde(rename = "quickMgnType", skip_serializing_if = "Option::is_none")]
    pub quick_mgn_type: Option<String>,
    #[serde(rename = "clOrdId", skip_serializing_if = "Option::is_none")]
    pub cl_ord_id: Option<String>,
    #[serde(rename = "tag", skip_serializing_if = "Option::is_none")]
    pub tag: Option<String>,
}

/// OKX cancel order request
#[derive(Debug, Serialize)]
pub struct OkxCancelOrderRequest {
    #[serde(rename = "instId")]
    pub inst_id: String,
    #[serde(rename = "ordId", skip_serializing_if = "Option::is_none")]
    pub ord_id: Option<String>,
    #[serde(rename = "clOrdId", skip_serializing_if = "Option::is_none")]
    pub cl_ord_id: Option<String>,
}

/// OKX WebSocket message
#[derive(Debug, Deserialize, Clone)]
pub struct OkxWsMessage {
    #[serde(rename = "arg")]
    pub arg: OkxWsArg,
    pub data: Vec<serde_json::Value>,
}

#[derive(Debug, Deserialize, Clone)]
pub struct OkxWsArg {
    pub channel: String,
    #[serde(rename = "instId")]
    pub inst_id: Option<String>,
    #[serde(rename = "instFamily")]
    pub inst_family: Option<String>,
}

/// OKX Gateway
pub struct OkxGateway {
    credentials: Option<OkxCredentials>,
    account_type: OkxAccountType,
    base_url: String,
    ws_url: String,
    rate_limit_tokens: u32,
    last_request: Instant,
    instrument_cache: HashMap<String, OkxInstrumentId>,
}

impl OkxGateway {
    pub fn new(credentials: Option<OkxCredentials>, account_type: OkxAccountType) -> Self {
        let (base_url, ws_url) = match account_type {
            OkxAccountType::Demo => (
                "https://www.okx.com",
                "wss://ws.okx.com:8443/ws/v5/public?brokerId=9999",
            ),
            OkxAccountType::Live => (
                "https://www.okx.com",
                "wss://ws.okx.com:8443/ws/v5/private",
            ),
        };

        Self {
            credentials,
            account_type,
            base_url: base_url.to_string(),
            ws_url: ws_url.to_string(),
            rate_limit_tokens: 20, // OKX allows ~20 requests/second for some endpoints
            last_request: Instant::now(),
            instrument_cache: HashMap::new(),
        }
    }

    /// Generate OKX signature
    pub fn generate_signature(
        &self,
        timestamp: &str,
        method: &str,
        request_path: &str,
        body: &str,
    ) -> String {
        let message = format!("{}{}{}{}", timestamp, method, request_path, body);
        
        let mut mac = HmacSha256::new_from_slice(self.credentials.as_ref().unwrap().api_secret.as_bytes())
            .expect("HMAC can take key of any size");
        mac.update(message.as_bytes());
        
        let result = mac.finalize();
        BASE64.encode(result.into_bytes())
    }

    /// Get current timestamp in ISO format
    pub fn get_timestamp_iso() -> String {
        chrono::Utc::now().format("%Y-%m-%dT%H:%M:%S.%fZ").to_string()
    }

    /// Build authentication headers for REST request
    pub fn build_auth_headers(
        &self,
        method: &str,
        request_path: &str,
        body: &str,
    ) -> Option<HashMap<String, String>> {
        let creds = self.credentials.as_ref()?;
        
        let timestamp = Self::get_timestamp_iso();
        let signature = self.generate_signature(&timestamp, method, request_path, body);
        
        let mut headers = HashMap::new();
        headers.insert("OK-ACCESS-KEY".to_string(), creds.api_key.clone());
        headers.insert("OK-ACCESS-SIGN".to_string(), signature);
        headers.insert("OK-ACCESS-TIMESTAMP".to_string(), timestamp);
        headers.insert("OK-ACCESS-PASSPHRASE".to_string(), creds.passphrase.clone());
        headers.insert("Content-Type".to_string(), "application/json".to_string());
        
        if self.account_type == OkxAccountType::Demo {
            headers.insert("x-simulated-trading".to_string(), "1".to_string());
        }
        
        Some(headers)
    }

    /// Build WebSocket authentication message
    pub fn build_ws_auth_message(&self) -> Option<String> {
        let creds = self.credentials.as_ref()?;
        
        let timestamp = Self::get_timestamp_iso();
        let signature = self.generate_signature(&timestamp, "GET", "/users/self/verify", "");
        
        let auth_msg = serde_json::json!({
            "op": "login",
            "args": [{
                "apiKey": creds.api_key,
                "passphrase": creds.passphrase,
                "timestamp": timestamp,
                "sign": signature
            }]
        });
        
        Some(auth_msg.to_string())
    }

    /// Subscribe to orderbook channel
    pub fn subscribe_orderbook(&self, inst_id: &str) -> String {
        let sub_msg = serde_json::json!({
            "op": "subscribe",
            "args": [{
                "channel": "books",
                "instId": inst_id
            }]
        });
        sub_msg.to_string()
    }

    /// Subscribe to trades channel
    pub fn subscribe_trades(&self, inst_id: &str) -> String {
        let sub_msg = serde_json::json!({
            "op": "subscribe",
            "args": [{
                "channel": "trades",
                "instId": inst_id
            }]
        });
        sub_msg.to_string()
    }

    /// Subscribe to tickers channel
    pub fn subscribe_tickers(&self, inst_id: &str) -> String {
        let sub_msg = serde_json::json!({
            "op": "subscribe",
            "args": [{
                "channel": "tickers",
                "instId": inst_id
            }]
        });
        sub_msg.to_string()
    }

    /// Place order via REST API
    pub fn build_place_order_request(&self, request: &OkxPlaceOrderRequest) -> (String, String) {
        let body = serde_json::to_string(request).unwrap();
        let path = "/api/v5/trade/order";
        ("POST".to_string(), format!("{}{}", self.base_url, path))
    }

    /// Cancel order via REST API
    pub fn build_cancel_order_request(&self, request: &OkxCancelOrderRequest) -> (String, String) {
        let body = serde_json::to_string(request).unwrap();
        let path = "/api/v5/trade/cancel-order";
        ("DELETE".to_string(), format!("{}{}", self.base_url, path))
    }

    /// Rate limit check
    pub async fn wait_for_rate_limit(&mut self) {
        let elapsed = self.last_request.elapsed();
        
        if elapsed < Duration::from_millis(50) {
            let wait_time = Duration::from_millis(50) - elapsed;
            time::sleep(wait_time).await;
        }
        
        if self.rate_limit_tokens < 20 {
            self.rate_limit_tokens += 1;
        }
        
        self.last_request = Instant::now();
    }

    /// Cache instrument ID mapping
    pub fn cache_instrument(&mut self, symbol: &str, inst_id: OkxInstrumentId) {
        self.instrument_cache.insert(symbol.to_string(), inst_id);
    }

    /// Get instrument ID from cache or generate
    pub fn get_or_create_inst_id(&mut self, symbol: &str, inst_type: OkxInstType) -> String {
        if let Some(cached) = self.instrument_cache.get(symbol) {
            return cached.to_okx_id();
        }

        // Parse symbol (e.g., "BTC/USDT" -> BTC-USDT)
        let parts: Vec<&str> = symbol.split('/').collect();
        let inst = if parts.len() == 2 {
            OkxInstrumentId {
                base_currency: parts[0].to_string(),
                quote_currency: parts[1].to_string(),
                inst_type,
                expiry: None,
                strike: None,
                option_type: None,
            }
        } else {
            OkxInstrumentId {
                base_currency: symbol.to_string(),
                quote_currency: "USDT".to_string(),
                inst_type,
                expiry: None,
                strike: None,
                option_type: None,
            }
        };

        let id = inst.to_okx_id();
        self.instrument_cache.insert(symbol.to_string(), inst);
        id
    }

    /// Parse OKX order book data
    pub fn parse_orderbook(&self, data: &serde_json::Value) -> Option<UnifiedOrderbook> {
        let bids = data.get("bids")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|level| {
                        level.as_array().and_then(|l| {
                            let price = l.get(0)?.as_str()?.parse::<f64>().ok()?;
                            let size = l.get(1)?.as_str()?.parse::<f64>().ok()?;
                            Some((price, size))
                        })
                    })
                    .collect::<Vec<_>>()
            })?;

        let asks = data.get("asks")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|level| {
                        level.as_array().and_then(|l| {
                            let price = l.get(0)?.as_str()?.parse::<f64>().ok()?;
                            let size = l.get(1)?.as_str()?.parse::<f64>().ok()?;
                            Some((price, size))
                        })
                    })
                    .collect::<Vec<_>>()
            })?;

        let symbol = data.get("instId")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();

        Some(UnifiedOrderbook {
            symbol,
            bids,
            asks,
        })
    }
}

/// Unified orderbook format
#[derive(Debug, Clone)]
pub struct UnifiedOrderbook {
    pub symbol: String,
    pub bids: Vec<(f64, f64)>,
    pub asks: Vec<(f64, f64)>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_instrument_id_spot() {
        let inst = OkxInstrumentId {
            base_currency: "BTC".to_string(),
            quote_currency: "USDT".to_string(),
            inst_type: OkxInstType::Spot,
            expiry: None,
            strike: None,
            option_type: None,
        };
        
        assert_eq!(inst.to_okx_id(), "BTC-USDT");
    }

    #[test]
    fn test_instrument_id_swap() {
        let inst = OkxInstrumentId {
            base_currency: "BTC".to_string(),
            quote_currency: "USDT".to_string(),
            inst_type: OkxInstType::Swap,
            expiry: None,
            strike: None,
            option_type: None,
        };
        
        assert_eq!(inst.to_okx_id(), "BTC-USDT-SWAP");
    }
}
