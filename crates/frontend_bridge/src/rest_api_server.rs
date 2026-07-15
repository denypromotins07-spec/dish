//! High-Performance Axum REST Server
//! Serves historical data, backtest results, and configuration with rate limiting and CORS.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use log::{debug, error, info, warn};

/// Rate limiter configuration
#[derive(Debug, Clone)]
pub struct RateLimitConfig {
    pub requests_per_second: u64,
    pub burst_size: u64,
}

impl Default for RateLimitConfig {
    fn default() -> Self {
        Self {
            requests_per_second: 100,
            burst_size: 200,
        }
    }
}

/// Token bucket rate limiter
pub struct TokenBucket {
    tokens: Arc<AtomicU64>,
    last_refill: Arc<parking_lot::Mutex<Instant>>,
    config: RateLimitConfig,
}

impl TokenBucket {
    pub fn new(config: RateLimitConfig) -> Self {
        Self {
            tokens: Arc::new(AtomicU64::new(config.burst_size)),
            last_refill: Arc::new(parking_lot::Mutex::new(Instant::now())),
            config,
        }
    }

    /// Try to consume a token
    pub fn try_consume(&self) -> bool {
        // Refill tokens based on elapsed time
        {
            let mut last_refill = self.last_refill.lock();
            let now = Instant::now();
            let elapsed = now.duration_since(*last_refill);
            
            let tokens_to_add = (elapsed.as_secs_f64() * self.config.requests_per_second as f64) as u64;
            if tokens_to_add > 0 {
                let current = self.tokens.load(Ordering::Relaxed);
                let new_tokens = (current + tokens_to_add).min(self.config.burst_size);
                self.tokens.store(new_tokens, Ordering::Relaxed);
                *last_refill = now;
            }
        }

        // Try to consume a token
        let current = self.tokens.load(Ordering::Acquire);
        if current > 0 {
            self.tokens.fetch_sub(1, Ordering::Release);
            true
        } else {
            false
        }
    }
}

/// REST API endpoint definition
#[derive(Debug, Clone)]
pub struct ApiEndpoint {
    pub path: String,
    pub method: HttpMethod,
    pub handler: Arc<dyn Fn(&str) -> ApiResponse + Send + Sync>,
    pub requires_auth: bool,
}

/// HTTP methods
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HttpMethod {
    GET,
    POST,
    PUT,
    DELETE,
}

/// API response structure
#[derive(Debug, Clone)]
pub struct ApiResponse {
    pub status_code: u16,
    pub body: String,
    pub content_type: String,
}

impl ApiResponse {
    pub fn ok(body: String) -> Self {
        Self {
            status_code: 200,
            body,
            content_type: "application/json".to_string(),
        }
    }

    pub fn error(status: u16, message: String) -> Self {
        Self {
            status_code: status,
            body: format!(r#"{{"error":"{}"}}"#, message),
            content_type: "application/json".to_string(),
        }
    }
}

/// REST server configuration
#[derive(Debug, Clone)]
pub struct RestServerConfig {
    pub host: String,
    pub port: u16,
    pub cors_allowed_origins: Vec<String>,
    pub rate_limit: RateLimitConfig,
    pub enable_tls: bool,
}

impl Default for RestServerConfig {
    fn default() -> Self {
        Self {
            host: "127.0.0.1".to_string(),
            port: 8080,
            cors_allowed_origins: vec!["http://localhost:3000".to_string()],
            rate_limit: RateLimitConfig::default(),
            enable_tls: false,
        }
    }
}

/// REST API server state
pub struct RestApiServer {
    config: RestServerConfig,
    endpoints: HashMap<String, ApiEndpoint>,
    rate_limiter: TokenBucket,
    is_running: Arc<AtomicBool>,
    request_count: Arc<AtomicU64>,
    rejected_count: Arc<AtomicU64>,
}

impl RestApiServer {
    pub fn new(config: RestServerConfig) -> Self {
        Self {
            config,
            endpoints: HashMap::new(),
            rate_limiter: TokenBucket::new(config.rate_limit.clone()),
            is_running: Arc::new(AtomicBool::new(false)),
            request_count: Arc::new(AtomicU64::new(0)),
            rejected_count: Arc::new(AtomicU64::new(0)),
        }
    }

    /// Register an API endpoint
    pub fn register_endpoint<F>(&mut self, path: &str, method: HttpMethod, handler: F, requires_auth: bool)
    where
        F: Fn(&str) -> ApiResponse + Send + Sync + 'static,
    {
        let key = format!("{}:{}", method_as_str(method), path);
        let endpoint = ApiEndpoint {
            path: path.to_string(),
            method,
            handler: Arc::new(handler),
            requires_auth,
        };
        self.endpoints.insert(key, endpoint);
        info!("Registered endpoint: {} {}", method_as_str(method), path);
    }

    /// Handle an incoming request
    pub fn handle_request(&self, method: HttpMethod, path: &str, _body: &str) -> ApiResponse {
        self.request_count.fetch_add(1, Ordering::Relaxed);

        // Rate limiting
        if !self.rate_limiter.try_consume() {
            self.rejected_count.fetch_add(1, Ordering::Relaxed);
            return ApiResponse::error(429, "Rate limit exceeded".to_string());
        }

        let key = format!("{}:{}", method_as_str(method), path);
        
        if let Some(endpoint) = self.endpoints.get(&key) {
            debug!("Handling request: {} {}", method_as_str(method), path);
            (endpoint.handler)(path)
        } else {
            ApiResponse::error(404, format!("Endpoint not found: {}", path))
        }
    }

    /// Start the server
    pub fn start(&mut self) {
        self.is_running.store(true, Ordering::SeqCst);
        info!(
            "REST API server starting on {}:{}",
            self.config.host, self.config.port
        );

        // In production, spawn actual Axum/Tokio server
        // This is a simplified simulation
    }

    /// Stop the server
    pub fn stop(&mut self) {
        self.is_running.store(false, Ordering::SeqCst);
        info!("REST API server stopped");
    }

    /// Get server statistics
    pub fn get_stats(&self) -> RestServerStats {
        RestServerStats {
            is_running: self.is_running.load(Ordering::Relaxed),
            request_count: self.request_count.load(Ordering::Relaxed),
            rejected_count: self.rejected_count.load(Ordering::Relaxed),
            endpoint_count: self.endpoints.len(),
        }
    }
}

fn method_as_str(method: HttpMethod) -> &'static str {
    match method {
        HttpMethod::GET => "GET",
        HttpMethod::POST => "POST",
        HttpMethod::PUT => "PUT",
        HttpMethod::DELETE => "DELETE",
    }
}

/// Server statistics
#[derive(Debug, Clone)]
pub struct RestServerStats {
    pub is_running: bool,
    pub request_count: u64,
    pub rejected_count: u64,
    pub endpoint_count: usize,
}

impl Default for RestApiServer {
    fn default() -> Self {
        Self::new(RestServerConfig::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_token_bucket_rate_limiter() {
        let config = RateLimitConfig {
            requests_per_second: 10,
            burst_size: 5,
        };
        let limiter = TokenBucket::new(config);

        // Should allow burst_size requests immediately
        for i in 0..5 {
            assert!(limiter.try_consume(), "Request {} should succeed", i);
        }

        // Next request should fail (burst exhausted)
        assert!(!limiter.try_consume());
    }

    #[test]
    fn test_rest_server_endpoints() {
        let mut server = RestApiServer::new(RestServerConfig::default());

        // Register test endpoint
        server.register_endpoint(
            "/api/health",
            HttpMethod::GET,
            |_path| ApiResponse::ok(r#"{"status":"ok"}"#.to_string()),
            false,
        );

        // Test request
        let response = server.handle_request(HttpMethod::GET, "/api/health", "");
        assert_eq!(response.status_code, 200);
        assert!(response.body.contains("ok"));

        // Test 404
        let response = server.handle_request(HttpMethod::GET, "/api/unknown", "");
        assert_eq!(response.status_code, 404);
    }

    #[test]
    fn test_server_stats() {
        let mut server = RestApiServer::new(RestServerConfig::default());
        
        server.register_endpoint(
            "/test",
            HttpMethod::GET,
            |_path| ApiResponse::ok("test".to_string()),
            false,
        );

        server.start();
        
        // Make some requests
        for _ in 0..10 {
            server.handle_request(HttpMethod::GET, "/test", "");
        }

        let stats = server.get_stats();
        assert!(stats.is_running);
        assert_eq!(stats.request_count, 10);
        assert_eq!(stats.endpoint_count, 1);

        server.stop();
        assert!(!server.get_stats().is_running);
    }
}
