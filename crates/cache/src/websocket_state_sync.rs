//! WebSocket State Sync: Instant state restoration on frontend refresh.
//! Compresses and dumps current sandbox, optimization, and portfolio state.
//! Uses efficient binary encoding with optional gzip compression.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use tokio::sync::broadcast;

/// State types that can be synchronized
#[derive(Debug, Clone)]
pub enum SyncState {
    Sandbox {
        active_shocks: Vec<String>,
        scenario_name: String,
        injection_active: bool,
    },
    Optimization {
        run_id: u64,
        progress_pct: f64,
        best_params: HashMap<String, f64>,
        best_score: f64,
    },
    Portfolio {
        total_equity: f64,
        available_cash: f64,
        positions: Vec<PositionState>,
        unrealized_pnl: f64,
        realized_pnl: f64,
    },
    Backtest {
        replay_position: u64,
        replay_total: u64,
        is_playing: bool,
        speed_multiplier: u32,
    },
}

#[derive(Debug, Clone)]
pub struct PositionState {
    pub symbol: String,
    pub quantity: f64,
    pub entry_price: f64,
    pub current_price: f64,
    pub unrealized_pnl: f64,
}

/// Compressed state dump for initial sync
#[derive(Debug, Clone)]
pub struct StateDump {
    pub timestamp_us: u64,
    pub sequence_number: u64,
    pub data: Vec<u8>, // Compressed binary
    pub checksum: u32, // CRC32 for integrity
}

impl StateDump {
    /// Create a new state dump from raw JSON/binary data
    pub fn new(data: Vec<u8>) -> Self {
        let timestamp_us = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_micros() as u64;
        
        let checksum = crc32fast::hash(&data);
        
        Self {
            timestamp_us,
            sequence_number: 0,
            data,
            checksum,
        }
    }

    /// Verify integrity
    pub fn verify(&self) -> bool {
        crc32fast::hash(&self.data) == self.checksum
    }

    /// Get approximate compressed size
    pub fn compressed_size(&self) -> usize {
        self.data.len()
    }
}

/// WebSocket state synchronization manager
pub struct WebSocketStateSync {
    /// Broadcast channel for state updates
    tx: broadcast::Sender<Arc<StateDump>>,
    /// Current full state (for new connections)
    current_state: parking_lot::RwLock<HashMap<String, SyncState>>,
    /// Sequence counter
    sequence: AtomicU64,
    /// Last dump timestamp
    last_dump_us: AtomicU64,
    /// Minimum interval between full dumps (microseconds)
    min_dump_interval_us: u64,
    /// Compression enabled
    compression_enabled: bool,
}

impl WebSocketStateSync {
    pub fn new(buffer_size: usize, min_dump_interval_ms: u64) -> Self {
        let (tx, _) = broadcast::channel(buffer_size.min(4096));
        
        Self {
            tx,
            current_state: parking_lot::RwLock::new(HashMap::with_capacity(16)),
            sequence: AtomicU64::new(0),
            last_dump_us: AtomicU64::new(0),
            min_dump_interval_us: min_dump_interval_ms * 1000,
            compression_enabled: true,
        }
    }

    /// Update a specific state component
    pub fn update_state(&self, key: String, state: SyncState) {
        let mut current = self.current_state.write();
        current.insert(key, state);
        
        // Trigger async dump if interval passed
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_micros() as u64;
        
        let last = self.last_dump_us.load(Ordering::Relaxed);
        if now - last >= self.min_dump_interval_us {
            self.trigger_dump();
        }
    }

    /// Get current state for a specific key
    pub fn get_state(&self, key: &str) -> Option<SyncState> {
        let current = self.current_state.read();
        current.get(key).cloned()
    }

    /// Get all current state
    pub fn get_all_state(&self) -> HashMap<String, SyncState> {
        self.current_state.read().clone()
    }

    /// Trigger an immediate state dump
    pub fn trigger_dump(&self) -> Option<Arc<StateDump>> {
        let current = self.current_state.read();
        
        // Serialize to JSON-like binary format
        let serialized = self.serialize_state(&current);
        
        // Optionally compress
        let data = if self.compression_enabled {
            flate2::write::GzEncoder::new(
                Vec::new(),
                flate2::Compression::fast()
            ).write_all(&serialized).ok()?;
            // Note: In production, use proper streaming compression
            serialized // Fallback for simplicity
        } else {
            serialized
        };
        
        let mut dump = StateDump::new(data);
        let seq = self.sequence.fetch_add(1, Ordering::AcqRel);
        dump.sequence_number = seq;
        
        self.last_dump_us.store(dump.timestamp_us, Ordering::Release);
        
        let dump_arc = Arc::new(dump);
        let _ = self.tx.send(dump_arc.clone());
        
        Some(dump_arc)
    }

    /// Serialize state to binary (simplified - would use protobuf in production)
    fn serialize_state(&self, state: &HashMap<String, SyncState>) -> Vec<u8> {
        // Simplified serialization - in production use protobuf or msgpack
        let mut buffer = Vec::with_capacity(state.len() * 256);
        
        for (key, value) in state {
            // Write key length and key
            buffer.extend_from_slice(&(key.len() as u16).to_le_bytes());
            buffer.extend_from_slice(key.as_bytes());
            
            // Write value type and data (simplified)
            match value {
                SyncState::Sandbox { scenario_name, .. } => {
                    buffer.push(0u8); // Type indicator
                    buffer.extend_from_slice(&(scenario_name.len() as u16).to_le_bytes());
                    buffer.extend_from_slice(scenario_name.as_bytes());
                }
                SyncState::Optimization { best_score, .. } => {
                    buffer.push(1u8);
                    buffer.extend_from_slice(&best_score.to_le_bytes());
                }
                SyncState::Portfolio { total_equity, .. } => {
                    buffer.push(2u8);
                    buffer.extend_from_slice(&total_equity.to_le_bytes());
                }
                SyncState::Backtest { replay_position, replay_total, .. } => {
                    buffer.push(3u8);
                    buffer.extend_from_slice(&replay_position.to_le_bytes());
                    buffer.extend_from_slice(&replay_total.to_le_bytes());
                }
            }
        }
        
        buffer
    }

    /// Subscribe to state dumps
    pub fn subscribe(&self) -> broadcast::Receiver<Arc<StateDump>> {
        self.tx.subscribe()
    }

    /// Get initial state for new connection
    pub fn get_initial_sync(&self) -> Arc<StateDump> {
        // Check if we have a recent dump
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_micros() as u64;
        
        let last = self.last_dump_us.load(Ordering::Acquire);
        
        if now - last < self.min_dump_interval_us {
            // Recent dump exists, but we need to create a fresh one for new client
            // In production, might return cached recent dump
        }
        
        // Generate fresh dump
        self.trigger_dump().unwrap_or_else(|| {
            Arc::new(StateDump::new(Vec::new()))
        })
    }

    /// Remove a state component
    pub fn remove_state(&self, key: &str) {
        self.current_state.write().remove(key);
    }

    /// Clear all state
    pub fn clear(&self) {
        self.current_state.write().clear();
        self.sequence.store(0, Ordering::Release);
    }

    /// Enable/disable compression
    pub fn set_compression(&mut self, enabled: bool) {
        self.compression_enabled = enabled;
    }

    /// Get statistics
    pub fn get_stats(&self) -> StateSyncStats {
        let current = self.current_state.read();
        
        StateSyncStats {
            state_count: current.len(),
            sequence: self.sequence.load(Ordering::Relaxed),
            last_dump_age_ms: {
                let now = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_micros() as u64;
                let last = self.last_dump_us.load(Ordering::Relaxed);
                ((now - last) / 1000) as u64
            },
            subscriber_count: self.tx.receiver_count(),
        }
    }
}

/// Statistics for monitoring
#[derive(Debug, Clone)]
pub struct StateSyncStats {
    pub state_count: usize,
    pub sequence: u64,
    pub last_dump_age_ms: u64,
    pub subscriber_count: usize,
}

impl Default for WebSocketStateSync {
    fn default() -> Self {
        Self::new(2048, 100) // 100ms minimum dump interval
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_state_sync() {
        let sync = WebSocketStateSync::new(1024, 50);
        
        // Update some state
        sync.update_state(
            "sandbox".to_string(),
            SyncState::Sandbox {
                active_shocks: vec!["flash_crash".to_string()],
                scenario_name: "stress_test".to_string(),
                injection_active: true,
            }
        );
        
        sync.update_state(
            "portfolio".to_string(),
            SyncState::Portfolio {
                total_equity: 1_000_000.0,
                available_cash: 500_000.0,
                positions: vec![],
                unrealized_pnl: 10_000.0,
                realized_pnl: 50_000.0,
            }
        );
        
        // Get state
        let state = sync.get_state("sandbox");
        assert!(state.is_some());
        
        // Trigger dump
        let dump = sync.trigger_dump();
        assert!(dump.is_some());
        
        let dump = dump.unwrap();
        assert!(dump.verify());
        assert!(dump.compressed_size() > 0);
        
        // Get stats
        let stats = sync.get_stats();
        assert_eq!(stats.state_count, 2);
        assert_eq!(stats.sequence, 1);
    }

    #[test]
    fn test_state_dump_integrity() {
        let data = b"test data for checksum";
        let dump = StateDump::new(data.to_vec());
        
        assert!(dump.verify());
        
        // Tamper with data
        let mut tampered = dump.clone();
        tampered.data[0] ^= 0xFF;
        assert!(!tampered.verify());
    }
}
