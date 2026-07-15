//! Config Hot-Reloader: Watches for strategy config changes from UI.
//! Applies parameter updates atomically using Read-Copy-Update (RCU) patterns.
//! Zero-downtime configuration updates without race conditions.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use parking_lot::{RwLock, RwLockReadGuard, RwLockWriteGuard};

/// Strategy configuration with version tracking
#[derive(Debug, Clone)]
pub struct StrategyConfig {
    pub name: String,
    pub version: u64,
    pub parameters: HashMap<String, f64>,
    pub risk_limits: HashMap<String, f64>,
    pub enabled: bool,
    pub last_updated_us: u64,
}

impl StrategyConfig {
    pub fn new(name: String) -> Self {
        Self {
            name,
            version: 0,
            parameters: HashMap::new(),
            risk_limits: HashMap::new(),
            enabled: true,
            last_updated_us: 0,
        }
    }

    pub fn with_param(mut self, key: &str, value: f64) -> Self {
        self.parameters.insert(key.to_string(), value);
        self
    }

    pub fn with_risk_limit(mut self, key: &str, value: f64) -> Self {
        self.risk_limits.insert(key.to_string(), value);
        self
    }
}

/// RCU-protected configuration container
pub struct RcuConfig<T> {
    data: RwLock<Arc<T>>,
}

impl<T: Clone> RcuConfig<T> {
    pub fn new(initial: T) -> Self {
        Self {
            data: RwLock::new(Arc::new(initial)),
        }
    }

    /// Get a read handle (lock-free after initial acquisition)
    pub fn read(&self) -> Arc<T> {
        self.data.read().clone()
    }

    /// Update with RCU semantics (copy-on-write)
    pub fn update<F>(&self, modifier: F) -> u64
    where
        F: FnOnce(&mut T),
    {
        let mut write_guard = self.data.write();
        let mut new_data = (**write_guard).clone();
        modifier(&mut new_data);
        
        let new_arc = Arc::new(new_data);
        let old_version = (*write_guard).version().unwrap_or(0);
        *write_guard = new_arc;
        
        old_version + 1
    }
}

/// Versioned trait for configs
pub trait Versioned {
    fn version(&self) -> u64;
    fn set_version(&mut self, v: u64);
    fn touch(&mut self);
}

impl Versioned for StrategyConfig {
    fn version(&self) -> u64 {
        self.version
    }

    fn set_version(&mut self, v: u64) {
        self.version = v;
    }

    fn touch(&mut self) {
        use std::time::{SystemTime, UNIX_EPOCH};
        self.last_updated_us = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_micros() as u64;
    }
}

/// Hot-reloader managing multiple strategy configs
pub struct ConfigHotReloader {
    /// Map of strategy name to its RCU-protected config
    configs: RwLock<HashMap<String, RcuConfig<StrategyConfig>>>,
    /// Global version counter
    global_version: AtomicU64,
    /// Maximum number of strategies
    max_strategies: usize,
}

impl ConfigHotReloader {
    pub fn new(max_strategies: usize) -> Self {
        Self {
            configs: RwLock::new(HashMap::with_capacity(max_strategies.min(100))),
            global_version: AtomicU64::new(0),
            max_strategies,
        }
    }

    /// Register a new strategy config
    pub fn register(&self, config: StrategyConfig) -> bool {
        let mut configs = self.configs.write();
        
        if configs.len() >= self.max_strategies && !configs.contains_key(&config.name) {
            // Evict oldest disabled strategy if at capacity
            if let Some((name, _)) = configs.iter().find(|(_, c)| {
                c.read().enabled == false
            }) {
                let name = name.clone();
                drop(configs);
                let mut configs = self.configs.write();
                configs.remove(&name);
            } else {
                return false; // Cannot add more
            }
        }

        let rcu = RcuConfig::new(config);
        configs.insert(rcu.read().name.clone(), rcu);
        
        self.global_version.fetch_add(1, Ordering::AcqRel);
        true
    }

    /// Get a read handle to a strategy config
    pub fn get_config(&self, name: &str) -> Option<Arc<StrategyConfig>> {
        let configs = self.configs.read();
        configs.get(name).map(|rcu| rcu.read())
    }

    /// Update a strategy's parameters atomically
    pub fn update_params(
        &self,
        name: &str,
        updates: HashMap<String, f64>,
    ) -> Option<u64> {
        let configs = self.configs.read();
        let rcu = configs.get(name)?;
        drop(configs);

        Some(rcu.update(|config| {
            config.parameters.extend(updates);
            config.set_version(config.version + 1);
            config.touch();
        }))
    }

    /// Update a strategy's risk limits atomically
    pub fn update_risk_limits(
        &self,
        name: &str,
        updates: HashMap<String, f64>,
    ) -> Option<u64> {
        let configs = self.configs.read();
        let rcu = configs.get(name)?;
        drop(configs);

        Some(rcu.update(|config| {
            config.risk_limits.extend(updates);
            config.set_version(config.version + 1);
            config.touch();
        }))
    }

    /// Enable/disable a strategy
    pub fn set_enabled(&self, name: &str, enabled: bool) -> Option<u64> {
        let configs = self.configs.read();
        let rcu = configs.get(name)?;
        drop(configs);

        Some(rcu.update(|config| {
            config.enabled = enabled;
            config.set_version(config.version + 1);
            config.touch();
        }))
    }

    /// Get all strategy names
    pub fn list_strategies(&self) -> Vec<String> {
        let configs = self.configs.read();
        configs.keys().cloned().collect()
    }

    /// Get global config version
    pub fn get_global_version(&self) -> u64 {
        self.global_version.load(Ordering::Acquire)
    }

    /// Remove a strategy
    pub fn remove(&self, name: &str) -> bool {
        let mut configs = self.configs.write();
        if configs.remove(name).is_some() {
            self.global_version.fetch_add(1, Ordering::AcqRel);
            true
        } else {
            false
        }
    }

    /// Snapshot all configs (for persistence or sync)
    pub fn snapshot(&self) -> HashMap<String, StrategyConfig> {
        let configs = self.configs.read();
        configs
            .iter()
            .map(|(name, rcu)| (name.clone(), (*rcu.read()).clone()))
            .collect()
    }
}

impl Default for ConfigHotReloader {
    fn default() -> Self {
        Self::new(50)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;
    use std::time::Duration;

    #[test]
    fn test_rcu_update() {
        let reloader = ConfigHotReloader::new(10);
        
        let config = StrategyConfig::new("test".to_string())
            .with_param("lookback", 20.0)
            .with_risk_limit("max_dd", 0.05);
        
        assert!(reloader.register(config));
        
        // Read initial config
        let initial = reloader.get_config("test").unwrap();
        assert_eq!(initial.version, 0);
        assert_eq!(initial.parameters.get("lookback"), Some(&20.0));
        
        // Update params
        let mut updates = HashMap::new();
        updates.insert("threshold".to_string(), 0.02);
        
        let new_version = reloader.update_params("test", updates).unwrap();
        assert_eq!(new_version, 1);
        
        // Verify update
        let updated = reloader.get_config("test").unwrap();
        assert_eq!(updated.version, 1);
        assert_eq!(updated.parameters.get("threshold"), Some(&0.02));
        
        // Original arc should still have old data (RCU semantics)
        assert_eq!(initial.version, 0);
    }

    #[test]
    fn test_concurrent_reads() {
        let reloader = Arc::new(ConfigHotReloader::new(10));
        
        let config = StrategyConfig::new("concurrent".to_string())
            .with_param("value", 100.0);
        
        reloader.register(config);
        
        let mut handles = vec![];
        
        // Spawn multiple readers
        for i in 0..10 {
            let r = Arc::clone(&reloader);
            handles.push(thread::spawn(move || {
                for _ in 0..100 {
                    let config = r.get_config("concurrent").unwrap();
                    assert!(config.parameters.contains_key("value"));
                    thread::sleep(Duration::from_micros(10));
                }
                i
            }));
        }
        
        // Concurrent writer
        let r = Arc::clone(&reloader);
        let writer = thread::spawn(move || {
            for i in 0..10 {
                let mut updates = HashMap::new();
                updates.insert("value".to_string(), 100.0 + i);
                r.update_params("concurrent", updates);
                thread::sleep(Duration::from_micros(50));
            }
        });
        
        for h in handles {
            h.join().unwrap();
        }
        writer.join().unwrap();
    }
}
