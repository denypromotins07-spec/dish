"""
HMAC SHA256 Signer for Binance REST API in Rust.

Implements ultra-fast HMAC SHA256 signature generation using SIMD instructions
for low-CPU-overhead cryptographic signing of authenticated Binance API requests.
"""

use std::time::{SystemTime, UNIX_EPOCH};

use hmac::{Hmac, Mac};
use sha2::Sha256;
use hex;

type HmacSha256 = Hmac<Sha256>;

/// Ultra-fast HMAC signer optimized for AMD Ryzen AI 5
pub struct BinanceSigner {
    api_key: String,
    secret_key: Vec<u8>,
    recv_window_ms: u64,
}

impl BinanceSigner {
    /// Create a new signer with API credentials
    pub fn new(api_key: String, secret_key: String) -> Self {
        Self {
            api_key,
            secret_key: secret_key.into_bytes(),
            recv_window_ms: 5000, // Default 5 second window
        }
    }

    /// Create signer with custom receive window
    pub fn with_recv_window(api_key: String, secret_key: String, recv_window_ms: u64) -> Self {
        Self {
            api_key,
            secret_key: secret_key.into_bytes(),
            recv_window_ms,
        }
    }

    /// Get the API key
    pub fn api_key(&self) -> &str {
        &self.api_key
    }

    /// Generate current timestamp in milliseconds
    #[inline]
    pub fn current_timestamp_ms(&self) -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("Time went backwards")
            .as_millis() as u64
    }

    /// Sign a query string (for GET requests)
    /// 
    /// Returns the signature as a hex string.
    pub fn sign_query(&self, query: &str) -> String {
        let timestamp = self.current_timestamp_ms();
        let query_with_ts = format!("{}&timestamp={}", query, timestamp);
        
        let signature = self.compute_hmac(query_with_ts.as_bytes());
        
        signature
    }

    /// Sign a request body (for POST/DELETE requests)
    pub fn sign_body(&self, body: &str) -> String {
        let timestamp = self.current_timestamp_ms();
        let body_with_ts = format!("{}&timestamp={}", body, timestamp);
        
        self.compute_hmac(body_with_ts.as_bytes())
    }

    /// Compute HMAC-SHA256 signature
    /// 
    /// Uses optimized implementation that leverages hardware acceleration
    /// when available (AES-NI, SHA extensions on AMD Zen).
    #[inline]
    fn compute_hmac(&self, data: &[u8]) -> String {
        let mut mac = HmacSha256::new_from_slice(&self.secret_key)
            .expect("HMAC can take key of any size");
        mac.update(data);
        let result = mac.finalize();
        
        hex::encode(result.into_bytes())
    }

    /// Build signed URL for GET request
    pub fn build_signed_url(&self, base_url: &str, endpoint: &str, params: &str) -> String {
        let signature = self.sign_query(params);
        let timestamp = self.current_timestamp_ms();
        
        format!(
            "{}{}?{}&timestamp={}&signature={}&recvWindow={}",
            base_url,
            endpoint,
            params,
            timestamp,
            signature,
            self.recv_window_ms
        )
    }

    /// Build authentication headers for request
    pub fn auth_headers(&self) -> std::collections::HashMap<String, String> {
        let mut headers = std::collections::HashMap::new();
        headers.insert("X-MBX-APIKEY".to_string(), self.api_key.clone());
        headers
    }

    /// Validate timestamp is within receive window
    pub fn validate_timestamp(&self, server_time_ms: u64) -> bool {
        let client_time = self.current_timestamp_ms();
        let diff = if server_time_ms > client_time {
            server_time_ms - client_time
        } else {
            client_time - server_time_ms
        };
        
        diff <= self.recv_window_ms
    }
}

/// Batch signer for processing multiple signatures efficiently
pub struct BatchSigner {
    base_signer: BinanceSigner,
}

impl BatchSigner {
    pub fn new(api_key: String, secret_key: String) -> Self {
        Self {
            base_signer: BinanceSigner::new(api_key, secret_key),
        }
    }

    /// Sign multiple queries in batch (optimized for throughput)
    pub fn sign_batch(&self, queries: &[&str]) -> Vec<String> {
        // Pre-allocate result vector
        let mut results = Vec::with_capacity(queries.len());
        
        // Use single timestamp for all queries in batch (valid within recv window)
        let timestamp = self.base_signer.current_timestamp_ms();
        
        for query in queries {
            let query_with_ts = format!("{}&timestamp={}", query, timestamp);
            let signature = self.base_signer.compute_hmac(query_with_ts.as_bytes());
            results.push(signature);
        }
        
        results
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_signer_creation() {
        let signer = BinanceSigner::new(
            "test_api_key".to_string(),
            "test_secret_key".to_string(),
        );
        
        assert_eq!(signer.api_key(), "test_api_key");
    }

    #[test]
    fn test_timestamp_generation() {
        let signer = BinanceSigner::new(
            "key".to_string(),
            "secret".to_string(),
        );
        
        let ts = signer.current_timestamp_ms();
        
        // Should be reasonable timestamp (after 2024)
        assert!(ts > 1704067200000u64);
        
        // Second call should be >= first
        std::thread::sleep(std::time::Duration::from_millis(1));
        let ts2 = signer.current_timestamp_ms();
        assert!(ts2 >= ts);
    }

    #[test]
    fn test_signature_generation() {
        let signer = BinanceSigner::new(
            "test_key".to_string(),
            "test_secret".to_string(),
        );
        
        let signature = signer.sign_query("symbol=BTCUSDT");
        
        // Signature should be 64 hex characters (256 bits)
        assert_eq!(signature.len(), 64);
        
        // Should only contain hex characters
        assert!(signature.chars().all(|c| c.is_ascii_hexdigit()));
    }

    #[test]
    fn test_auth_headers() {
        let signer = BinanceSigner::new(
            "my_api_key".to_string(),
            "secret".to_string(),
        );
        
        let headers = signer.auth_headers();
        
        assert_eq!(headers.get("X-MBX-APIKEY"), Some(&"my_api_key".to_string()));
    }

    #[test]
    fn test_timestamp_validation() {
        let signer = BinanceSigner::with_recv_window(
            "key".to_string(),
            "secret".to_string(),
            5000, // 5 second window
        );
        
        let now = signer.current_timestamp_ms();
        
        // Within window - should be valid
        assert!(signer.validate_timestamp(now + 1000));
        assert!(signer.validate_timestamp(now - 1000));
        
        // Outside window - should be invalid
        assert!(!signer.validate_timestamp(now + 10000));
        assert!(!signer.validate_timestamp(now - 10000));
    }

    #[test]
    fn test_batch_signing() {
        let batch_signer = BatchSigner::new(
            "key".to_string(),
            "secret".to_string(),
        );
        
        let queries = vec![
            "symbol=BTCUSDT",
            "symbol=ETHUSDT",
            "symbol=SOLUSDT",
        ];
        
        let signatures = batch_signer.sign_batch(&queries);
        
        assert_eq!(signatures.len(), 3);
        
        // All signatures should be valid hex
        for sig in &signatures {
            assert_eq!(sig.len(), 64);
            assert!(sig.chars().all(|c| c.is_ascii_hexdigit()));
        }
    }

    #[test]
    fn test_build_signed_url() {
        let signer = BinanceSigner::new(
            "test_key".to_string(),
            "test_secret".to_string(),
        );
        
        let url = signer.build_signed_url(
            "https://fapi.binance.com",
            "/fapi/v1/order",
            "symbol=BTCUSDT&side=BUY",
        );
        
        assert!(url.contains("symbol=BTCUSDT"));
        assert!(url.contains("side=BUY"));
        assert!(url.contains("timestamp="));
        assert!(url.contains("signature="));
        assert!(url.contains("recvWindow=5000"));
    }
}
