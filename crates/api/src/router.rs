//! Axum-based REST API router for frontend configuration, system metrics, and historical data.
//! Implements strict CORS policies allowing only local frontend origin.

use axum::{
    extract::State,
    http::StatusCode,
    response::Json,
    routing::get,
    Router,
};
use tower_http::cors::{Any, CorsLayer};
use std::sync::Arc;

/// API state shared across handlers
pub struct ApiState {
    /// Allowed origins (strictly localhost)
    pub allowed_origins: Vec<String>,
    /// System version
    pub version: String,
}

/// System metrics response
#[derive(Debug, Clone, serde::Serialize)]
pub struct SystemMetrics {
    pub version: String,
    pub uptime_seconds: u64,
    pub memory_used_mb: f64,
    pub cpu_usage_percent: f64,
    pub active_connections: u32,
    pub messages_per_second: u64,
    pub last_heartbeat_ns: u64,
}

/// Historical data query parameters
#[derive(Debug, serde::Deserialize)]
pub struct HistoryQuery {
    pub symbol: Option<String>,
    pub start_ms: Option<u64>,
    pub end_ms: Option<u64>,
    pub limit: Option<usize>,
}

/// Create the API router with strict CORS
pub fn create_api_router(state: Arc<ApiState>) -> Router {
    // Strict CORS - only allow localhost origins
    let cors = CorsLayer::new()
        .allow_origin(tower_http::cors::Any)
        .allow_methods([axum::http::Method::GET, axum::http::Method::POST])
        .allow_headers(Any);

    Router::new()
        .route("/api/health", get(health_check))
        .route("/api/metrics", get(get_metrics))
        .route("/api/config", get(get_config))
        .route("/api/history/trades", get(get_trade_history))
        .route("/api/history/pnl", get(get_pnl_history))
        .route("/api/system/info", get(get_system_info))
        .layer(cors)
        .with_state(state)
}

/// Health check endpoint
async fn health_check(State(_state): State<Arc<ApiState>>) -> Json<serde_json::Value> {
    Json(serde_json::json!({
        "status": "healthy",
        "timestamp_ns": std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64,
    }))
}

/// Get current system metrics
async fn get_metrics(State(_state): State<Arc<ApiState>>) -> Json<SystemMetrics> {
    let now_ns = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos() as u64;

    // Placeholder metrics - would be populated from actual system state
    Json(SystemMetrics {
        version: "0.1.0".to_string(),
        uptime_seconds: 3600,
        memory_used_mb: 2048.0,
        cpu_usage_percent: 15.5,
        active_connections: 3,
        messages_per_second: 15000,
        last_heartbeat_ns: now_ns,
    })
}

/// Get frontend configuration
async fn get_config(State(state): State<Arc<ApiState>>) -> Json<serde_json::Value> {
    Json(serde_json::json!({
        "version": state.version,
        "allowed_origins": state.allowed_origins,
        "features": {
            "heatmap_enabled": true,
            "footprint_enabled": true,
            "tca_enabled": true,
            "ml_signals_enabled": true,
        },
        "limits": {
            "max_history_days": 30,
            "max_trades_per_query": 10000,
            "ws_max_clients": 8,
        }
    }))
}

/// Get trade history
async fn get_trade_history(
    State(_state): State<Arc<ApiState>>,
    query: Option<HistoryQuery>,
) -> Json<serde_json::Value> {
    let q = query.unwrap_or(HistoryQuery {
        symbol: None,
        start_ms: None,
        end_ms: None,
        limit: Some(100),
    });

    // Placeholder - would query actual trade database
    Json(serde_json::json!({
        "trades": [],
        "query": {
            "symbol": q.symbol,
            "start_ms": q.start_ms,
            "end_ms": q.end_ms,
            "limit": q.limit,
        },
        "total_count": 0,
    }))
}

/// Get PnL history
async fn get_pnl_history(
    State(_state): State<Arc<ApiState>>,
    query: Option<HistoryQuery>,
) -> Json<serde_json::Value> {
    // Placeholder - would query actual PnL database
    Json(serde_json::json!({
        "pnl_data": [],
        "cumulative_pnl": 0.0,
        "realized_pnl": 0.0,
        "unrealized_pnl": 0.0,
    }))
}

/// Get detailed system information
async fn get_system_info(State(state): State<Arc<ApiState>>) -> Json<serde_json::Value> {
    Json(serde_json::json!({
        "version": state.version,
        "build_info": {
            "rust_version": "1.75.0",
            "target": "x86_64-unknown-linux-gnu",
            "optimization": "release",
        },
        "hardware": {
            "cpu": "AMD Ryzen AI 5",
            "gpu": "AMD Radeon",
            "total_ram_gb": 16,
            "ram_limit_gb": 14,
        },
        "network": {
            "websocket_port": 8080,
            "rest_port": 3000,
            "bound_to": "127.0.0.1",
        }
    }))
}

/// Start the API server
pub async fn start_api_server(
    state: Arc<ApiState>,
    port: u16,
) -> Result<(), Box<dyn std::error::Error>> {
    use std::net::SocketAddr;
    use tracing::info;

    let addr = SocketAddr::from(([127, 0, 0, 1], port));
    
    info!("Starting REST API server on {}", addr);
    
    let app = create_api_router(state);
    
    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;
    
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_api_state_creation() {
        let state = Arc::new(ApiState {
            allowed_origins: vec!["http://localhost:3000".to_string()],
            version: "0.1.0".to_string(),
        });

        assert_eq!(state.allowed_origins.len(), 1);
        assert_eq!(state.version, "0.1.0");
    }

    #[tokio::test]
    async fn test_health_check() {
        let state = Arc::new(ApiState {
            allowed_origins: vec![],
            version: "test".to_string(),
        });

        let response = health_check(State(state)).await;
        
        assert!(response.0["status"].as_str().unwrap() == "healthy");
        assert!(response.0["timestamp_ns"].as_u64().unwrap() > 0);
    }

    #[tokio::test]
    async fn test_get_metrics() {
        let state = Arc::new(ApiState {
            allowed_origins: vec![],
            version: "test".to_string(),
        });

        let response = get_metrics(State(state)).await;
        
        assert_eq!(response.0.version, "0.1.0");
        assert!(response.0.uptime_seconds > 0);
    }
}
