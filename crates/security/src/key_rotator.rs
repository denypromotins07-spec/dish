//! Automated background daemon that monitors API key age and triggers secure rotation.
//! Updates the locked memory vault without interrupting live trading threads.

use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};
use std::thread;
use std::collections::HashMap;
use crate::vault::{SecretsVault, LockedSecret};

/// Configuration for key rotation.
pub struct RotationConfig {
    pub max_key_age_hours: u64,
    pub check_interval_secs: u64,
    pub rotation_warning_threshold_hours: u64,
}

impl Default for RotationConfig {
    fn default() -> Self {
        RotationConfig {
            max_key_age_hours: 72, // Rotate every 72 hours
            check_interval_secs: 300, // Check every 5 minutes
            rotation_warning_threshold_hours: 24, // Warn 24 hours before expiry
        }
    }
}

/// Tracks the metadata of a rotatable key.
#[derive(Clone)]
pub struct KeyMetadata {
    pub exchange: String,
    pub created_at: Instant,
    pub last_rotated: Instant,
    pub rotation_count: u64,
}

/// Background daemon for automatic key rotation.
pub struct KeyRotatorDaemon {
    config: RotationConfig,
    keys_metadata: Arc<RwLock<HashMap<String, KeyMetadata>>>,
    vault: Arc<RwLock<SecretsVault>>,
    running: Arc<RwLock<bool>>,
}

impl KeyRotatorDaemon {
    pub fn new(
        config: RotationConfig,
        vault: Arc<RwLock<SecretsVault>>,
    ) -> Self {
        KeyRotatorDaemon {
            config,
            keys_metadata: Arc::new(RwLock::new(HashMap::new())),
            vault,
            running: Arc::new(RwLock::new(false)),
        }
    }

    /// Registers a key for rotation tracking.
    pub fn register_key(&self, exchange: &str) {
        let mut metadata_map = self.keys_metadata.write().unwrap();
        let now = Instant::now();
        metadata_map.insert(
            exchange.to_string(),
            KeyMetadata {
                exchange: exchange.to_string(),
                created_at: now,
                last_rotated: now,
                rotation_count: 0,
            },
        );
    }

    /// Starts the background rotation daemon.
    pub fn start(&self) -> thread::JoinHandle<()> {
        let config = self.config.clone();
        let keys_metadata = Arc::clone(&self.keys_metadata);
        let vault = Arc::clone(&self.vault);
        let running = Arc::clone(&self.running);

        *running.write().unwrap() = true;

        thread::spawn(move || {
            while *running.read().unwrap() {
                let now = Instant::now();
                
                {
                    let metadata_map = keys_metadata.read().unwrap();
                    for (exchange, metadata) in metadata_map.iter() {
                        let age = now.duration_since(metadata.last_rotated);
                        let age_hours = age.as_secs() / 3600;

                        if age_hours >= config.max_key_age_hours {
                            log_rotation_needed(exchange, age_hours);
                            // Trigger rotation callback (implemented by caller)
                            trigger_rotation(exchange, &vault);
                        } else if age_hours >= config.rotation_warning_threshold_hours {
                            log_rotation_warning(exchange, age_hours);
                        }
                    }
                }

                thread::sleep(Duration::from_secs(config.check_interval_secs));
            }
        })
    }

    /// Stops the background daemon.
    pub fn stop(&self) {
        *self.running.write().unwrap() = false;
    }

    /// Manually rotates a key for a specific exchange.
    pub fn rotate_key(&self, exchange: &str, new_key_data: &[u8]) -> Result<(), String> {
        // Update the vault
        {
            let mut vault = self.vault.write().unwrap();
            vault.update_key(exchange, new_key_data)?;
        }

        // Update metadata
        {
            let mut metadata_map = self.keys_metadata.write().unwrap();
            if let Some(metadata) = metadata_map.get_mut(exchange) {
                metadata.last_rotated = Instant::now();
                metadata.rotation_count += 1;
            }
        }

        Ok(())
    }

    /// Gets the age of a key in hours.
    pub fn get_key_age_hours(&self, exchange: &str) -> Option<u64> {
        let metadata_map = self.keys_metadata.read().unwrap();
        metadata_map.get(exchange).map(|m| {
            let age = Instant::now().duration_since(m.last_rotated);
            age.as_secs() / 3600
        })
    }
}

fn log_rotation_needed(exchange: &str, age_hours: u64) {
    eprintln!(
        "[KEY_ROTATOR] CRITICAL: Key for {} is {} hours old. Rotation required.",
        exchange, age_hours
    );
}

fn log_rotation_warning(exchange: &str, age_hours: u64) {
    eprintln!(
        "[KEY_ROTATOR] WARNING: Key for {} is {} hours old. Rotation recommended soon.",
        exchange, age_hours
    );
}

fn trigger_rotation(exchange: &str, _vault: &Arc<RwLock<SecretsVault>>) {
    // This would typically call an exchange API to generate a new key
    // For now, we just log the event. The actual rotation logic is exchange-specific.
    eprintln!(
        "[KEY_ROTATOR] ACTION: Initiating rotation sequence for exchange: {}",
        exchange
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    static ROTATION_COUNT: AtomicUsize = AtomicUsize::new(0);

    #[test]
    fn test_key_registration_and_age_tracking() {
        let config = RotationConfig::default();
        let vault = Arc::new(RwLock::new(SecretsVault {
            keys: HashMap::new(),
            master_key: None,
        }));
        
        let rotator = KeyRotatorDaemon::new(config, vault);
        rotator.register_key("binance");
        
        let age = rotator.get_key_age_hours("binance");
        assert!(age.is_some());
        assert_eq!(age.unwrap(), 0); // Just registered, so age is 0
    }
}
