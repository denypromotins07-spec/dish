//! State Parker for Emergency Fallback
//! Serializes and parks bot state to encrypted disk (LMDB) on critical failures.
//! Prevents orphaned orders during network loss or system crashes.

use anyhow::{Result, anyhow};
use lmdb::{Environment, Database, WriteFlags, Transaction};
use serde::{Serialize, Deserialize};
use std::sync::Arc;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};
use tokio::sync::RwLock;
use aes_gcm::{Aes256Gcm, Key, Nonce};
use aes_gcm::aead::{Aead, KeyInit};
use base64::{Engine as _, engine::general_purpose::STANDARD as BASE64};

/// Maximum state size before parking (bytes)
const MAX_STATE_SIZE: usize = 10 * 1024 * 1024; // 10MB

/// Encrypted bot state for persistence
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParkedState {
    pub timestamp_ns: u64,
    pub reason: ParkingReason,
    pub positions: Vec<ParkedPosition>,
    pub pending_orders: Vec<ParkedOrder>,
    pub strategy_state: Vec<u8>,
    pub checksum: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ParkingReason {
    NetworkLoss,
    CriticalError,
    ManualStop,
    SystemSleep,
    PowerFailure,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParkedPosition {
    pub symbol: String,
    pub side: String,  // "long" or "short"
    pub quantity: f64,
    pub entry_price: f64,
    pub unrealized_pnl: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParkedOrder {
    pub order_id: String,
    pub symbol: String,
    pub side: String,
    pub quantity: f64,
    pub price: f64,
    pub order_type: String,
}

/// State Parker for emergency persistence
pub struct StateParker {
    env: Arc<Environment>,
    db: Database,
    encryption_key: Key<Aes256Gcm>,
    park_path: PathBuf,
    is_parked: Arc<RwLock<bool>>,
}

impl StateParker {
    /// Initialize the state parker with LMDB environment
    pub fn new(park_path: PathBuf, encryption_key: &[u8; 32]) -> Result<Self> {
        // Create LMDB environment
        let env = Environment::new()
            .set_map_size(MAX_STATE_SIZE * 10)  // 100MB max
            .set_max_dbs(1)
            .open(&park_path)
            .map_err(|e| anyhow!("Failed to open LMDB: {}", e))?;
        
        // Open database
        let db = env.open_db(None)
            .map_err(|e| anyhow!("Failed to open DB: {}", e))?;
        
        Ok(Self {
            env: Arc::new(env),
            db,
            encryption_key: Key::<Aes256Gcm>::from_slice(encryption_key).clone(),
            park_path,
            is_parked: Arc::new(RwLock::new(false)),
        })
    }
    
    /// Park the current state immediately (emergency)
    pub async fn park_emergency(&self, reason: ParkingReason, state: &ParkedState) -> Result<()> {
        let mut parked = self.is_parked.write().await;
        
        if *parked {
            return Err(anyhow!("State already parked"));
        }
        
        // Serialize state
        let serialized = bincode::serialize(state)
            .map_err(|e| anyhow!("Serialization failed: {}", e))?;
        
        // Encrypt
        let encrypted = self.encrypt_data(&serialized)?;
        
        // Store in LMDB
        let txn = self.env.begin_rw_txn()
            .map_err(|e| anyhow!("Transaction failed: {}", e))?;
        
        txn.put(self.db, b"parked_state", &encrypted, WriteFlags::empty())
            .map_err(|e| anyhow!("LMDB put failed: {}", e))?;
        
        txn.commit()
            .map_err(|e| anyhow!("Commit failed: {}", e))?;
        
        // Also write to file as backup
        let backup_path = self.park_path.join("parked_state.bin");
        std::fs::write(&backup_path, &encrypted)
            .map_err(|e| anyhow!("Backup write failed: {}", e))?;
        
        *parked = true;
        
        log::warn!("State parked: reason={:?}, path={:?}", reason, self.park_path);
        Ok(())
    }
    
    /// Restore state from park
    pub async fn restore(&self) -> Result<ParkedState> {
        let txn = self.env.begin_ro_txn()
            .map_err(|e| anyhow!("Transaction failed: {}", e))?;
        
        let encrypted: Vec<u8> = txn.get(self.db, b"parked_state")
            .map_err(|e| anyhow!("Get failed: {}", e))?
            .to_vec();
        
        txn.commit()
            .map_err(|e| anyhow!("Commit failed: {}", e))?;
        
        // Decrypt
        let decrypted = self.decrypt_data(&encrypted)?;
        
        // Deserialize
        let state: ParkedState = bincode::deserialize(&decrypted)
            .map_err(|e| anyhow!("Deserialization failed: {}", e))?;
        
        // Reset parked flag
        let mut parked = self.is_parked.write().await;
        *parked = false;
        
        log::info!("State restored from park");
        Ok(state)
    }
    
    /// Check if state is currently parked
    pub async fn is_parked(&self) -> bool {
        *self.is_parked.read().await
    }
    
    /// Encrypt data using AES-256-GCM
    fn encrypt_data(&self, plaintext: &[u8]) -> Result<Vec<u8>> {
        use rand::RngCore;
        
        let nonce_bytes = rand::thread_rng().fill([0u8; 12]);
        let nonce = Nonce::from_slice(&nonce_bytes);
        
        let cipher = Aes256Gcm::new(&self.encryption_key);
        let ciphertext = cipher.encrypt(nonce, plaintext)
            .map_err(|e| anyhow!("Encryption failed: {}", e))?;
        
        // Prepend nonce to ciphertext
        let mut result = nonce_bytes.to_vec();
        result.extend_from_slice(&ciphertext);
        
        Ok(result)
    }
    
    /// Decrypt data using AES-256-GCM
    fn decrypt_data(&self, encrypted: &[u8]) -> Result<Vec<u8>> {
        if encrypted.len() < 12 {
            return Err(anyhow!("Encrypted data too short"));
        }
        
        let nonce = Nonce::from_slice(&encrypted[..12]);
        let ciphertext = &encrypted[12..];
        
        let cipher = Aes256Gcm::new(&self.encryption_key);
        let plaintext = cipher.decrypt(nonce, ciphertext)
            .map_err(|e| anyhow!("Decryption failed: {}", e))?;
        
        Ok(plaintext)
    }
    
    /// Calculate checksum for integrity verification
    fn calculate_checksum(state: &ParkedState) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        
        let mut hasher = DefaultHasher::new();
        state.timestamp_ns.hash(&mut hasher);
        state.positions.len().hash(&mut hasher);
        state.pending_orders.len().hash(&mut hasher);
        hasher.finish()
    }
}

/// Builder for creating ParkedState
pub struct ParkedStateBuilder {
    positions: Vec<ParkedPosition>,
    pending_orders: Vec<ParkedOrder>,
    strategy_state: Vec<u8>,
}

impl ParkedStateBuilder {
    pub fn new() -> Self {
        Self {
            positions: Vec::new(),
            pending_orders: Vec::new(),
            strategy_state: Vec::new(),
        }
    }
    
    pub fn add_position(mut self, position: ParkedPosition) -> Self {
        self.positions.push(position);
        self
    }
    
    pub fn add_order(mut self, order: ParkedOrder) -> Self {
        self.pending_orders.push(order);
        self
    }
    
    pub fn set_strategy_state(mut self, state: Vec<u8>) -> Self {
        self.strategy_state = state;
        self
    }
    
    pub fn build(self, reason: ParkingReason) -> ParkedState {
        let timestamp_ns = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64;
        
        let checksum = StateParker::calculate_checksum(&ParkedState {
            timestamp_ns,
            reason,
            positions: self.positions.clone(),
            pending_orders: self.pending_orders.clone(),
            strategy_state: self.strategy_state.clone(),
            checksum: 0,
        });
        
        ParkedState {
            timestamp_ns,
            reason,
            positions: self.positions,
            pending_orders: self.pending_orders,
            strategy_state: self.strategy_state,
            checksum,
        }
    }
}

impl Default for ParkedStateBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;
    
    #[tokio::test]
    async fn test_park_restore() {
        let temp_dir = TempDir::new().unwrap();
        let key = [42u8; 32];
        
        let parker = StateParker::new(temp_dir.path().to_path_buf(), &key).unwrap();
        
        let state = ParkedStateBuilder::new()
            .add_position(ParkedPosition {
                symbol: "BTCUSDT".to_string(),
                side: "long".to_string(),
                quantity: 1.5,
                entry_price: 50000.0,
                unrealized_pnl: 150.0,
            })
            .build(ParkingReason::NetworkLoss);
        
        // Park
        parker.park_emergency(ParkingReason::NetworkLoss, &state).await.unwrap();
        
        // Restore
        let restored = parker.restore().await.unwrap();
        
        assert_eq!(restored.positions.len(), 1);
        assert_eq!(restored.positions[0].symbol, "BTCUSDT");
        assert_eq!(restored.reason, ParkingReason::NetworkLoss);
    }
}
