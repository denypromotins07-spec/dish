//! Lock-free broadcast channel manager for fanning out telemetry events.
//! Uses tokio::sync::broadcast to distribute ticks and orders to multiple UI clients
//! without blocking the main trading event loop.

use tokio::sync::broadcast;
use std::sync::Arc;
use crate::ws_server::TelemetryMessage;

/// Configuration for the broadcast channel
pub struct BroadcasterConfig {
    /// Maximum number of messages to buffer (backpressure limit)
    pub buffer_size: usize,
    /// Maximum number of concurrent UI clients
    pub max_clients: usize,
}

impl Default for BroadcasterConfig {
    fn default() -> Self {
        Self {
            buffer_size: 4096, // Strict memory bound
            max_clients: 8,
        }
    }
}

/// Lock-free channel broadcaster manager
pub struct ChannelBroadcaster {
    sender: broadcast::Sender<TelemetryMessage>,
    config: BroadcasterConfig,
    client_count: Arc<std::sync::atomic::AtomicUsize>,
}

impl ChannelBroadcaster {
    /// Create a new channel broadcaster with strict bounds
    pub fn new(config: BroadcasterConfig) -> Self {
        let (tx, _rx) = broadcast::channel(config.buffer_size);
        
        Self {
            sender: tx,
            config,
            client_count: Arc::new(std::sync::atomic::AtomicUsize::new(0)),
        }
    }

    /// Get a clone of the sender for broadcasting
    pub fn sender(&self) -> broadcast::Sender<TelemetryMessage> {
        self.sender.clone()
    }

    /// Register a new client connection
    pub fn register_client(&self) -> Result<(), &'static str> {
        let current = self.client_count.load(std::sync::atomic::Ordering::Relaxed);
        
        if current >= self.config.max_clients {
            return Err("Maximum client count reached");
        }
        
        self.client_count.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        Ok(())
    }

    /// Unregister a client connection
    pub fn unregister_client(&self) {
        self.client_count.fetch_sub(1, std::sync::atomic::Ordering::Relaxed);
    }

    /// Get current client count
    pub fn client_count(&self) -> usize {
        self.client_count.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Broadcast a tick event (non-blocking, drops if full)
    pub fn broadcast_tick(&self, payload: serde_json::Value) -> bool {
        let msg = TelemetryMessage {
            timestamp_ns: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos() as u64,
            msg_type: "tick".to_string(),
            payload,
        };

        match self.sender.send(msg) {
            Ok(_) => true,
            Err(broadcast::error::SendError(_)) => false, // Channel closed
        }
    }

    /// Broadcast an order event (non-blocking, drops if full)
    pub fn broadcast_order(&self, payload: serde_json::Value) -> bool {
        let msg = TelemetryMessage {
            timestamp_ns: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos() as u64,
            msg_type: "order".to_string(),
            payload,
        };

        match self.sender.send(msg) {
            Ok(_) => true,
            Err(broadcast::error::SendError(_)) => false,
        }
    }

    /// Broadcast a strategy state update
    pub fn broadcast_strategy(&self, payload: serde_json::Value) -> bool {
        let msg = TelemetryMessage {
            timestamp_ns: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos() as u64,
            msg_type: "strategy".to_string(),
            payload,
        };

        match self.sender.send(msg) {
            Ok(_) => true,
            Err(broadcast::error::SendError(_)) => false,
        }
    }

    /// Get subscription handle for a new client
    pub fn subscribe(&self) -> broadcast::Receiver<TelemetryMessage> {
        self.sender.subscribe()
    }

    /// Check if channel is near capacity (for backpressure signaling)
    pub fn is_near_capacity(&self, threshold: f64) -> bool {
        let len = self.sender.len();
        (len as f64 / self.config.buffer_size as f64) > threshold
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_broadcaster_creation() {
        let config = BroadcasterConfig::default();
        let broadcaster = ChannelBroadcaster::new(config);
        
        assert_eq!(broadcaster.client_count(), 0);
        assert!(!broadcaster.is_near_capacity(0.5));
    }

    #[test]
    fn test_client_registration() {
        let config = BroadcasterConfig {
            buffer_size: 1024,
            max_clients: 2,
        };
        let broadcaster = ChannelBroadcaster::new(config);
        
        assert!(broadcaster.register_client().is_ok());
        assert!(broadcaster.register_client().is_ok());
        assert!(broadcaster.register_client().is_err()); // Max reached
        
        broadcaster.unregister_client();
        assert!(broadcaster.register_client().is_ok());
    }

    #[tokio::test]
    async fn test_broadcast_message() {
        let config = BroadcasterConfig::default();
        let broadcaster = ChannelBroadcaster::new(config);
        
        let payload = serde_json::json!({"price": 50000.0});
        let success = broadcaster.broadcast_tick(payload);
        
        assert!(success);
    }
}
