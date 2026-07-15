//! Ultra-low-latency WebSocket Telemetry Server
//! Binds strictly to 127.0.0.1 to prevent external exposure.
//! Uses Axum/Tungstenite for high-throughput broadcasting of bot state.

use axum::{
    extract::ws::{Message, WebSocket, WebSocketUpgrade},
    response::IntoResponse,
    routing::get,
    Router,
};
use tokio::sync::broadcast;
use std::net::SocketAddr;
use tracing::{info, warn};

/// Telemetry message types for frontend consumption
#[derive(Clone, Debug, serde::Serialize)]
pub struct TelemetryMessage {
    pub timestamp_ns: u64,
    pub msg_type: String,
    pub payload: serde_json::Value,
}

/// Shared state for WebSocket connections
pub struct TelemetryState {
    pub tx: broadcast::Sender<TelemetryMessage>,
}

impl Clone for TelemetryState {
    fn clone(&self) -> Self {
        Self {
            tx: self.tx.clone(),
        }
    }
}

/// WebSocket handler for real-time bot state streaming
async fn ws_handler(
    ws: WebSocketUpgrade,
    axum::extract::State(state): axum::extract::State<TelemetryState>,
) -> impl IntoResponse {
    ws.on_upgrade(|socket| handle_socket(socket, state))
}

async fn handle_socket(socket: WebSocket, state: TelemetryState) {
    let (mut sender, mut receiver) = socket.split();
    let mut rx = state.tx.subscribe();

    // Spawn task to receive messages from broadcast channel
    let send_task = tokio::spawn(async move {
        while let Ok(msg) = rx.recv().await {
            if let Ok(json) = serde_json::to_string(&msg) {
                if sender.send(Message::Text(json)).await.is_err()) {
                    break;
                }
            }
        }
    });

    // Receive messages from client (mostly for heartbeat/keepalive)
    let recv_task = tokio::spawn(async move {
        while let Some(Ok(_msg)) = receiver.next().await {
            // Handle client messages if needed
        }
    });

    tokio::select! {
        _ = send_task => {},
        _ = recv_task => {},
    }
}

/// Create the telemetry router bound to localhost
pub fn create_telemetry_router(tx: broadcast::Sender<TelemetryMessage>) -> Router {
    let state = TelemetryState { tx };

    Router::new()
        .route("/ws/telemetry", get(ws_handler))
        .with_state(state)
}

/// Start the telemetry server on localhost only
pub async fn start_telemetry_server(
    port: u16,
    tx: broadcast::Sender<TelemetryMessage>,
) -> Result<(), Box<dyn std::error::Error>> {
    let addr = SocketAddr::from(([127, 0, 0, 1], port));
    
    info!("Starting telemetry WebSocket server on {}", addr);
    
    let app = create_telemetry_router(tx);
    
    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;
    
    Ok(())
}

/// Broadcast a telemetry message to all connected clients
pub fn broadcast_telemetry(
    tx: &broadcast::Sender<TelemetryMessage>,
    msg_type: &str,
    payload: serde_json::Value,
) {
    let msg = TelemetryMessage {
        timestamp_ns: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64,
        msg_type: msg_type.to_string(),
        payload,
    };

    // Non-blocking send; drop if channel is full (backpressure)
    let _ = tx.send(msg);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_telemetry_message_serialization() {
        let msg = TelemetryMessage {
            timestamp_ns: 1234567890,
            msg_type: "test".to_string(),
            payload: serde_json::json!({"key": "value"}),
        };
        
        let json = serde_json::to_string(&msg).unwrap();
        assert!(json.contains("test"));
        assert!(json.contains("value"));
    }
}
