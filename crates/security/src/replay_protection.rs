//! Nonce and timestamp validator for incoming REST/WebSocket payloads.
//! Drops duplicate or delayed requests to prevent replay attacks.

use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use tokio::sync::RwLock;

/// Configuration for replay protection.
#[derive(Clone, Debug)]
pub struct ReplayProtectionConfig {
    /// Maximum allowed age of a request in seconds
    pub max_request_age_secs: u64,
    /// Maximum number of nonces to track per client
    pub max_nonces_per_client: usize,
    /// Time window for nonce tracking (seconds)
    pub nonce_window_secs: u64,
}

impl Default for ReplayProtectionConfig {
    fn default() -> Self {
        ReplayProtectionConfig {
            max_request_age_secs: 60, // 1 minute
            max_nonces_per_client: 10000,
            nonce_window_secs: 300, // 5 minutes
        }
    }
}

/// Tracks nonce usage for a single client.
struct ClientNonceTracker {
    nonces: VecDeque<(String, Instant)>,
    nonce_set: HashMap<String, Instant>,
    max_size: usize,
}

impl ClientNonceTracker {
    fn new(max_size: usize) -> Self {
        ClientNonceTracker {
            nonces: VecDeque::with_capacity(max_size),
            nonce_set: HashMap::with_capacity(max_size),
            max_size,
        }
    }

    /// Checks if a nonce has been seen before.
    fn is_duplicate(&self, nonce: &str) -> bool {
        self.nonce_set.contains_key(nonce)
    }

    /// Records a new nonce. Returns true if successfully recorded.
    fn record(&mut self, nonce: String) -> bool {
        // Check if already exists
        if self.nonce_set.contains_key(&nonce) {
            return false;
        }

        // Prune old entries if at capacity
        while self.nonces.len() >= self.max_size {
            if let Some((old_nonce, _)) = self.nonces.pop_front() {
                self.nonce_set.remove(&old_nonce);
            }
        }

        // Add new nonce
        let now = Instant::now();
        self.nonces.push_back((nonce.clone(), now));
        self.nonce_set.insert(nonce, now);

        true
    }

    /// Prunes nonces older than the specified duration.
    fn prune_older_than(&mut self, max_age: Duration) {
        let cutoff = Instant::now() - max_age;

        while let Some((_, timestamp)) = self.nonces.front() {
            if *timestamp < cutoff {
                if let Some((old_nonce, _)) = self.nonces.pop_front() {
                    self.nonce_set.remove(&old_nonce);
                }
            } else {
                break;
            }
        }
    }
}

/// Replay protection engine for API requests.
pub struct ReplayProtector {
    config: ReplayProtectionConfig,
    clients: Arc<RwLock<HashMap<String, ClientNonceTracker>>>,
}

impl ReplayProtector {
    /// Creates a new replay protector with the given configuration.
    pub fn new(config: ReplayProtectionConfig) -> Self {
        ReplayProtector {
            config,
            clients: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Validates a request's timestamp and nonce.
    pub async fn validate_request(
        &self,
        client_id: &str,
        timestamp: u64,
        nonce: &str,
    ) -> Result<(), ReplayError> {
        // Validate timestamp
        self.validate_timestamp(timestamp)?;

        // Get or create client tracker
        let mut clients = self.clients.write().await;
        let tracker = clients
            .entry(client_id.to_string())
            .or_insert_with(|| ClientNonceTracker::new(self.config.max_nonces_per_client));

        // Check for duplicate nonce
        if tracker.is_duplicate(nonce) {
            return Err(ReplayError::DuplicateNonce(nonce.to_string()));
        }

        // Record the nonce
        if !tracker.record(nonce.to_string()) {
            return Err(ReplayError::DuplicateNonce(nonce.to_string()));
        }

        Ok(())
    }

    /// Validates just the timestamp of a request.
    pub fn validate_timestamp(&self, timestamp: u64) -> Result<(), ReplayError> {
        let current_time = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();

        let age = if timestamp > current_time {
            // Request from the future (clock skew)
            timestamp - current_time
        } else {
            current_time - timestamp
        };

        if age > self.config.max_request_age_secs {
            return Err(ReplayError::ExpiredTimestamp {
                timestamp,
                age,
                max_age: self.config.max_request_age_secs,
            });
        }

        Ok(())
    }

    /// Validates a WebSocket message with sequence number.
    pub async fn validate_ws_message(
        &self,
        client_id: &str,
        seq_num: u64,
    ) -> Result<(), ReplayError> {
        // For WebSocket, we use sequence numbers instead of nonces
        let mut clients = self.clients.write().await;
        let tracker = clients
            .entry(client_id.to_string())
            .or_insert_with(|| ClientNonceTracker::new(self.config.max_nonces_per_client));

        let seq_str = format!("seq_{}", seq_num);

        if tracker.is_duplicate(&seq_str) {
            return Err(ReplayError::DuplicateSequence(seq_num));
        }

        if !tracker.record(seq_str) {
            return Err(ReplayError::DuplicateSequence(seq_num));
        }

        Ok(())
    }

    /// Periodic cleanup task to prune old nonces.
    pub async fn run_cleanup(&self) {
        let mut clients = self.clients.write().await;
        let prune_duration = Duration::from_secs(self.config.nonce_window_secs);

        for tracker in clients.values_mut() {
            tracker.prune_older_than(prune_duration);
        }

        // Remove empty trackers
        clients.retain(|_, tracker| !tracker.nonces.is_empty());
    }

    /// Gets statistics about tracked nonces.
    pub async fn get_stats(&self) -> ReplayStats {
        let clients = self.clients.read().await;
        
        let total_clients = clients.len();
        let total_nonces: usize = clients.values().map(|t| t.nonces.len()).sum();
        let avg_nonces_per_client = if total_clients > 0 {
            total_nonces / total_clients
        } else {
            0
        };

        ReplayStats {
            total_clients,
            total_nonces,
            avg_nonces_per_client,
            config: self.config.clone(),
        }
    }
}

/// Errors that can occur during replay protection validation.
#[derive(Debug, Clone)]
pub enum ReplayError {
    /// Request timestamp is too old or too far in the future
    ExpiredTimestamp {
        timestamp: u64,
        age: u64,
        max_age: u64,
    },
    /// Nonce has already been used
    DuplicateNonce(String),
    /// WebSocket sequence number has already been seen
    DuplicateSequence(u64),
    /// Missing required timestamp
    MissingTimestamp,
    /// Missing required nonce
    MissingNonce,
}

impl std::fmt::Display for ReplayError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ReplayError::ExpiredTimestamp { timestamp, age, max_age } => {
                write!(
                    f,
                    "Request timestamp expired: {} (age: {}s, max: {}s)",
                    timestamp, age, max_age
                )
            }
            ReplayError::DuplicateNonce(nonce) => {
                write!(f, "Duplicate nonce detected: {}", nonce)
            }
            ReplayError::DuplicateSequence(seq) => {
                write!(f, "Duplicate sequence number: {}", seq)
            }
            ReplayError::MissingTimestamp => write!(f, "Missing required timestamp"),
            ReplayError::MissingNonce => write!(f, "Missing required nonce"),
        }
    }
}

impl std::error::Error for ReplayError {}

/// Statistics about replay protection state.
#[derive(Debug, Clone)]
pub struct ReplayStats {
    pub total_clients: usize,
    pub total_nonces: usize,
    pub avg_nonces_per_client: usize,
    pub config: ReplayProtectionConfig,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_valid_request() {
        let config = ReplayProtectionConfig::default();
        let protector = ReplayProtector::new(config);

        let current_ts = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();

        let result = protector.validate_request("client1", current_ts, "nonce123").await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_duplicate_nonce_rejected() {
        let config = ReplayProtectionConfig::default();
        let protector = ReplayProtector::new(config);

        let current_ts = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();

        // First request should succeed
        let result1 = protector.validate_request("client1", current_ts, "nonce456").await;
        assert!(result1.is_ok());

        // Same nonce should be rejected
        let result2 = protector.validate_request("client1", current_ts, "nonce456").await;
        assert!(matches!(result2, Err(ReplayError::DuplicateNonce(_))));
    }

    #[tokio::test]
    async fn test_expired_timestamp_rejected() {
        let config = ReplayProtectionConfig {
            max_request_age_secs: 60,
            ..Default::default()
        };
        let protector = ReplayProtector::new(config);

        // Old timestamp (5 minutes ago)
        let old_ts = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs()
            - 300;

        let result = protector.validate_request("client1", old_ts, "nonce789").await;
        assert!(matches!(result, Err(ReplayError::ExpiredTimestamp { .. })));
    }

    #[tokio::test]
    async fn test_different_clients_independent() {
        let config = ReplayProtectionConfig::default();
        let protector = ReplayProtector::new(config);

        let current_ts = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();

        // Same nonce from different clients should be allowed
        let result1 = protector.validate_request("client_a", current_ts, "shared_nonce").await;
        assert!(result1.is_ok());

        let result2 = protector.validate_request("client_b", current_ts, "shared_nonce").await;
        assert!(result2.is_ok());
    }

    #[tokio::test]
    async fn test_ws_sequence_validation() {
        let config = ReplayProtectionConfig::default();
        let protector = ReplayProtector::new(config);

        // First sequence should succeed
        let result1 = protector.validate_ws_message("ws_client", 1).await;
        assert!(result1.is_ok());

        // Same sequence should fail
        let result2 = protector.validate_ws_message("ws_client", 1).await;
        assert!(matches!(result2, Err(ReplayError::DuplicateSequence(1))));

        // Next sequence should succeed
        let result3 = protector.validate_ws_message("ws_client", 2).await;
        assert!(result3.is_ok());
    }
}
