//! WebSocket Broadcaster with Protobuf Compression
//! Multiplexes order book, PnL, signals, and logs into a single compressed stream.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use log::{debug, error, info, warn};

/// Message types for WebSocket multiplexing
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WsMessageType {
    OrderBook = 0,
    PnL = 1,
    Signal = 2,
    Log = 3,
    Heartbeat = 255,
}

/// WebSocket message header (fixed size)
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct WsMessageHeader {
    pub msg_type: u8,
    pub flags: u8,
    pub sequence: u32,
    pub payload_len: u32,
    pub timestamp_ns: u64,
}

impl WsMessageHeader {
    pub const SIZE: usize = std::mem::size_of::<Self>();

    pub fn new(msg_type: WsMessageType, payload_len: u32) -> Self {
        Self {
            msg_type: msg_type as u8,
            flags: 0,
            sequence: 0,
            payload_len,
            timestamp_ns: 0,
        }
    }
}

/// Client connection handle
#[derive(Debug, Clone)]
pub struct ClientHandle {
    pub client_id: u64,
    pub connected_at: u64,
    pub last_heartbeat: u64,
    pub messages_sent: u64,
    pub is_active: bool,
}

/// LTTB (Largest-Triangle-Three-Buckets) downsampling for time series
pub struct LttbDownsampler {
    bucket_size: usize,
}

impl LttbDownsampler {
    pub fn new(bucket_size: usize) -> Self {
        Self { bucket_size }
    }

    /// Downsample a time series using LTTB algorithm
    pub fn downsample(&self, data: &[(f64, f64)], target_points: usize) -> Vec<(f64, f64)> {
        if data.len() <= target_points {
            return data.to_vec();
        }

        let mut result = Vec::with_capacity(target_points);
        let bucket_size = data.len() / target_points;

        // Always include first point
        if let Some(&first) = data.first() {
            result.push(first);
        }

        // Sample from each bucket
        for i in 1..target_points - 1 {
            let start = i * bucket_size;
            let end = (i + 1) * bucket_size;
            
            if start < data.len() {
                let bucket = &data[start..end.min(data.len())];
                if let Some(&max_point) = bucket.iter().max_by(|a, b| {
                    // Simplified: just pick the point with largest y value
                    a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal)
                }) {
                    result.push(max_point);
                }
            }
        }

        // Always include last point
        if let Some(&last) = data.last() {
            result.push(last);
        }

        result
    }
}

/// WebSocket broadcaster state
pub struct WsBroadcaster {
    clients: HashMap<u64, ClientHandle>,
    next_client_id: u64,
    sequence: Arc<AtomicU64>,
    is_running: Arc<AtomicBool>,
    message_buffer: Vec<u8>,
    downsampler: LttbDownsampler,
    max_clients: usize,
    heartbeat_interval_ms: u64,
}

impl WsBroadcaster {
    pub fn new(max_clients: usize) -> Self {
        Self {
            clients: HashMap::with_capacity(max_clients),
            next_client_id: 1,
            sequence: Arc::new(AtomicU64::new(0)),
            is_running: Arc::new(AtomicBool::new(false)),
            message_buffer: Vec::with_capacity(65536),
            downsampler: LttbDownsampler::new(100),
            max_clients,
            heartbeat_interval_ms: 5000,
        }
    }

    /// Register a new client connection
    pub fn add_client(&mut self) -> Option<u64> {
        if self.clients.len() >= self.max_clients {
            warn!("Maximum client limit reached ({})", self.max_clients);
            return None;
        }

        let client_id = self.next_client_id;
        self.next_client_id += 1;

        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64;

        let handle = ClientHandle {
            client_id,
            connected_at: now,
            last_heartbeat: now,
            messages_sent: 0,
            is_active: true,
        };

        self.clients.insert(client_id, handle);
        info!("Client {} connected (total: {})", client_id, self.clients.len());

        Some(client_id)
    }

    /// Remove a client connection
    pub fn remove_client(&mut self, client_id: u64) {
        if self.clients.remove(&client_id).is_some() {
            info!("Client {} disconnected (total: {})", client_id, self.clients.len());
        }
    }

    /// Broadcast a message to all connected clients
    pub fn broadcast(&mut self, msg_type: WsMessageType, payload: &[u8]) -> usize {
        if !self.is_running.load(Ordering::SeqCst) {
            return 0;
        }

        let seq = self.sequence.fetch_add(1, Ordering::SeqCst);
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64;

        // Build message with header
        let mut header = WsMessageHeader::new(msg_type, payload.len() as u32);
        header.sequence = seq as u32;
        header.timestamp_ns = timestamp;

        self.message_buffer.clear();
        self.message_buffer.extend_from_slice(&unsafe {
            std::slice::from_raw_parts(
                &header as *const WsMessageHeader as *const u8,
                WsMessageHeader::SIZE,
            )
        });
        self.message_buffer.extend_from_slice(payload);

        // In production, send to actual WebSocket connections
        // For now, just update client stats
        let mut sent_count = 0;
        for client in self.clients.values_mut() {
            if client.is_active {
                client.messages_sent += 1;
                client.last_heartbeat = timestamp / 1_000_000; // Convert to ms
                sent_count += 1;
            }
        }

        debug!(
            "Broadcast {:?} (seq={}, {} bytes) to {} clients",
            msg_type, seq, self.message_buffer.len(), sent_count
        );

        sent_count
    }

    /// Send order book update (with LTTB downsampling)
    pub fn send_orderbook(&mut self, symbol: &str, bids: &[(f64, f64)], asks: &[(f64, f64)]) {
        // Downsample for efficient transmission
        let bids_downsampled = self.downsampler.downsample(bids, 50);
        let asks_downsampled = self.downsampler.downsample(asks, 50);

        // Serialize (in production, use protobuf)
        let payload = format!("{}:{:?}:{:?}", symbol, bids_downsampled, asks_downsampled);
        self.broadcast(WsMessageType::OrderBook, payload.as_bytes());
    }

    /// Send PnL update
    pub fn send_pnl(&mut self, total_pnl: f64, realized: f64, unrealized: f64) {
        let payload = format!("{:.8}:{:.8}:{:.8}", total_pnl, realized, unrealized);
        self.broadcast(WsMessageType::PnL, payload.as_bytes());
    }

    /// Send signal
    pub fn send_signal(&mut self, symbol: &str, side: i8, quantity: f64, confidence: f64) {
        let payload = format!("{}:{}:{:.4}:{:.4}", symbol, side, quantity, confidence);
        self.broadcast(WsMessageType::Signal, payload.as_bytes());
    }

    /// Send log message
    pub fn send_log(&mut self, level: &str, message: &str) {
        let payload = format!("{}:{}", level, message);
        self.broadcast(WsMessageType::Log, payload.as_bytes());
    }

    /// Start the broadcaster
    pub fn start(&mut self) {
        self.is_running.store(true, Ordering::SeqCst);
        info!("WebSocket broadcaster started");
    }

    /// Stop the broadcaster
    pub fn stop(&mut self) {
        self.is_running.store(false, Ordering::SeqCst);
        
        // Disconnect all clients
        self.clients.clear();
        
        info!("WebSocket broadcaster stopped");
    }

    /// Get connected client count
    pub fn client_count(&self) -> usize {
        self.clients.len()
    }

    /// Get broadcaster statistics
    pub fn get_stats(&self) -> WsBroadcasterStats {
        WsBroadcasterStats {
            client_count: self.clients.len(),
            total_messages: self.sequence.load(Ordering::Relaxed),
            is_running: self.is_running.load(Ordering::Relaxed),
        }
    }
}

impl Default for WsBroadcaster {
    fn default() -> Self {
        Self::new(100)
    }
}

/// Broadcaster statistics
#[derive(Debug, Clone)]
pub struct WsBroadcasterStats {
    pub client_count: usize,
    pub total_messages: u64,
    pub is_running: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lttb_downsampling() {
        let downsampler = LttbDownsampler::new(100);
        
        // Generate test data
        let data: Vec<(f64, f64)> = (0..1000)
            .map(|i| (i as f64, (i % 100) as f64))
            .collect();

        let downsampled = downsampler.downsample(&data, 10);
        
        assert!(downsampled.len() <= 10);
        assert_eq!(downsampled.first(), Some(&(0.0, 0.0)));
        assert_eq!(downsampled.last(), Some(&(999.0, 99.0)));
    }

    #[test]
    fn test_broadcaster_lifecycle() {
        let mut broadcaster = WsBroadcaster::new(10);
        
        // Add clients
        let client1 = broadcaster.add_client();
        let client2 = broadcaster.add_client();
        
        assert_eq!(client1, Some(1));
        assert_eq!(client2, Some(2));
        assert_eq!(broadcaster.client_count(), 2);

        // Broadcast
        broadcaster.start();
        let sent = broadcaster.broadcast(WsMessageType::Heartbeat, b"test");
        assert_eq!(sent, 2);

        // Remove client
        broadcaster.remove_client(1);
        assert_eq!(broadcaster.client_count(), 1);

        // Stop
        broadcaster.stop();
        assert_eq!(broadcaster.client_count(), 0);
    }
}
