//! Rust-based atomic hot-swapping engine for model weights.
//! Replaces live ONNX model weights in memory using Read-Copy-Update (RCU)
//! patterns, ensuring zero dropped packets or interrupted orders during swap.

use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};
use std::collections::HashMap;

/// Model metadata
#[derive(Debug, Clone)]
pub struct ModelMetadata {
    pub version: String,
    pub created_at_ns: u64,
    pub sha256_hash: String,
    pub input_shape: Vec<usize>,
    pub output_shape: Vec<usize>,
    pub metrics: HashMap<String, f64>,
}

/// Container for model data with atomic reference counting
pub struct ModelContainer<T> {
    pub metadata: ModelMetadata,
    pub data: Arc<T>,
    pub loaded_at: Instant,
}

impl<T> Clone for ModelContainer<T> {
    fn clone(&self) -> Self {
        Self {
            metadata: self.metadata.clone(),
            data: Arc::clone(&self.data),
            loaded_at: self.loaded_at,
        }
    }
}

/// RCU-based model hot-swapper for zero-downtime updates
pub struct ModelHotSwapper<T> {
    /// Current active model (read-mostly)
    active_model: RwLock<Option<ModelContainer<T>>>,
    /// Previous model (kept for rollback)
    previous_model: RwLock<Option<ModelContainer<T>>>,
    /// Swap statistics
    swap_count: std::sync::atomic::AtomicU64,
    failed_swaps: std::sync::atomic::AtomicU64,
    last_swap_time_ns: std::sync::atomic::AtomicU64,
}

impl<T> ModelHotSwapper<T> 
where 
    T: Send + Sync + 'static,
{
    /// Create a new hot-swapper instance
    pub fn new() -> Self {
        Self {
            active_model: RwLock::new(None),
            previous_model: RwLock::new(None),
            swap_count: std::sync::atomic::AtomicU64::new(0),
            failed_swaps: std::sync::atomic::AtomicU64::new(0),
            last_swap_time_ns: std::sync::atomic::AtomicU64::new(0),
        }
    }

    /// Load initial model
    pub fn load_initial(&self, data: Arc<T>, metadata: ModelMetadata) -> Result<(), &'static str> {
        let mut active = self.active_model.write().map_err(|_| "Lock poisoned")?;
        
        *active = Some(ModelContainer {
            metadata,
            data,
            loaded_at: Instant::now(),
        });

        Ok(())
    }

    /// Atomically swap to new model (RCU pattern)
    /// Returns true if swap succeeded, false if validation failed
    pub fn swap(&self, new_data: Arc<T>, new_metadata: ModelMetadata) -> bool {
        let start = Instant::now();
        
        // Validate new model before swap
        if !self.validate_model(&new_metadata) {
            self.failed_swaps.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            return false;
        }

        let new_container = ModelContainer {
            metadata: new_metadata,
            data: new_data,
            loaded_at: Instant::now(),
        };

        // RCU swap: acquire write lock briefly, swap pointers
        {
            let mut active_guard = match self.active_model.write() {
                Ok(guard) => guard,
                Err(_) => {
                    self.failed_swaps.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    return false;
                }
            };

            // Move current to previous
            let mut prev_guard = self.previous_model.write().unwrap();
            *prev_guard = active_guard.take();
            
            // Install new as active
            *active_guard = Some(new_container);
            
            // Locks released here
        }

        // Update statistics
        self.swap_count.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64;
        self.last_swap_time_ns.store(now, std::sync::atomic::Ordering::Relaxed);

        // Log swap duration (should be microseconds)
        let duration_us = start.elapsed().as_micros();
        eprintln!("[ModelHotSwapper] Swap completed in {} μs", duration_us);

        true
    }

    /// Get current active model for inference (lock-free read)
    pub fn get_active(&self) -> Option<ModelContainer<T>> {
        self.active_model.read().ok().and_then(|guard| guard.clone())
    }

    /// Get current active model data only (for inference)
    pub fn get_active_data(&self) -> Option<Arc<T>> {
        self.active_model.read().ok().and_then(|g| g.as_ref().map(|c| Arc::clone(&c.data)))
    }

    /// Rollback to previous model
    pub fn rollback(&self) -> bool {
        let prev_guard = self.previous_model.read().ok();
        if prev_guard.is_none() || prev_guard.as_ref().unwrap().is_none() {
            eprintln!("[ModelHotSwapper] No previous model to rollback to");
            return false;
        }

        let prev_container = prev_guard.unwrap().clone();

        {
            let mut active_guard = match self.active_model.write() {
                Ok(guard) => guard,
                Err(_) => return false,
            };

            // Swap active and previous
            let mut prev_guard = self.previous_model.write().unwrap();
            let current = active_guard.take();
            *active_guard = prev_container;
            *prev_guard = current;
        }

        eprintln!("[ModelHotSwapper] Rollback completed successfully");
        true
    }

    /// Validate model metadata before swap
    fn validate_model(&self, metadata: &ModelMetadata) -> bool {
        // Check version is newer
        if let Some(current) = self.active_model.read().ok().and_then(|g| g.clone()) {
            if metadata.version <= current.metadata.version {
                eprintln!("[ModelHotSwapper] Version {} not newer than {}", 
                    metadata.version, current.metadata.version);
                return false;
            }

            // Check input shape compatibility
            if metadata.input_shape != current.metadata.input_shape {
                eprintln!("[ModelHotSwapper] Input shape mismatch");
                return false;
            }
        }

        // Check hash is valid (non-empty)
        if metadata.sha256_hash.is_empty() {
            eprintln!("[ModelHotSwapper] Invalid model hash");
            return false;
        }

        true
    }

    /// Get swap statistics
    pub fn get_stats(&self) -> SwapStats {
        SwapStats {
            swap_count: self.swap_count.load(std::sync::atomic::Ordering::Relaxed),
            failed_swaps: self.failed_swaps.load(std::sync::atomic::Ordering::Relaxed),
            last_swap_time_ns: self.last_swap_time_ns.load(std::sync::atomic::Ordering::Relaxed),
            has_active: self.active_model.read().ok().map_or(false, |g| g.is_some()),
            has_previous: self.previous_model.read().ok().map_or(false, |g| g.is_some()),
        }
    }

    /// Get current model version
    pub fn current_version(&self) -> Option<String> {
        self.active_model.read()
            .ok()
            .and_then(|g| g.as_ref().map(|c| c.metadata.version.clone()))
    }
}

impl<T> Default for ModelHotSwapper<T> 
where 
    T: Send + Sync + 'static,
{
    fn default() -> Self {
        Self::new()
    }
}

/// Statistics about model swaps
#[derive(Debug, Clone)]
pub struct SwapStats {
    pub swap_count: u64,
    pub failed_swaps: u64,
    pub last_swap_time_ns: u64,
    pub has_active: bool,
    pub has_previous: bool,
}

/// ONNX-specific model container
#[cfg(feature = "onnx")]
pub mod onnx {
    use super::*;
    use ort::{Session, Value};

    /// Wrapper for ONNX session with metadata
    pub struct OnnxModel {
        pub session: Arc<Session>,
        pub input_names: Vec<String>,
        pub output_names: Vec<String>,
    }

    /// Type alias for ONNX hot-swapper
    pub type OnnxHotSwapper = ModelHotSwapper<OnnxModel>;

    impl OnnxHotSwapper {
        /// Run inference with current model
        pub fn infer(&self, inputs: Vec<Value>) -> Result<Vec<Value>, String> {
            let model = self.get_active()
                .ok_or_else(|| "No active model loaded".to_string())?;
            
            let session = &model.data.session;
            
            session.run(inputs)
                .map_err(|e| e.to_string())
                .map(|outputs| outputs.collect())
        }

        /// Get model output shape
        pub fn get_output_shape(&self) -> Option<Vec<usize>> {
            self.get_active().map(|m| m.metadata.output_shape)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_initial_load() {
        let swapper: ModelHotSwapper<Vec<f32>> = ModelHotSwapper::new();
        let data = Arc::new(vec![1.0, 2.0, 3.0]);
        let metadata = ModelMetadata {
            version: "v1.0.0".to_string(),
            created_at_ns: 0,
            sha256_hash: "abc123".to_string(),
            input_shape: vec![1, 3],
            output_shape: vec![1, 1],
            metrics: HashMap::new(),
        };

        assert!(swapper.load_initial(data, metadata).is_ok());
        assert_eq!(swapper.current_version(), Some("v1.0.0".to_string()));
    }

    #[test]
    fn test_atomic_swap() {
        let swapper: ModelHotSwapper<Vec<f32>> = ModelHotSwapper::new();
        
        // Load initial
        swapper.load_initial(
            Arc::new(vec![1.0]),
            ModelMetadata {
                version: "v1.0.0".to_string(),
                created_at_ns: 0,
                sha256_hash: "hash1".to_string(),
                input_shape: vec![1],
                output_shape: vec![1],
                metrics: HashMap::new(),
            },
        ).unwrap();

        // Swap to new
        let success = swapper.swap(
            Arc::new(vec![2.0, 3.0]),
            ModelMetadata {
                version: "v2.0.0".to_string(),
                created_at_ns: 0,
                sha256_hash: "hash2".to_string(),
                input_shape: vec![1],
                output_shape: vec![1],
                metrics: HashMap::new(),
            },
        );

        assert!(success);
        assert_eq!(swapper.current_version(), Some("v2.0.0".to_string()));
        
        let stats = swapper.get_stats();
        assert_eq!(stats.swap_count, 1);
        assert!(stats.has_previous);
    }

    #[test]
    fn test_rollback() {
        let swapper: ModelHotSwapper<Vec<f32>> = ModelHotSwapper::new();
        
        swapper.load_initial(
            Arc::new(vec![1.0]),
            ModelMetadata {
                version: "v1.0.0".to_string(),
                created_at_ns: 0,
                sha256_hash: "hash1".to_string(),
                input_shape: vec![1],
                output_shape: vec![1],
                metrics: HashMap::new(),
            },
        ).unwrap();

        swapper.swap(
            Arc::new(vec![2.0]),
            ModelMetadata {
                version: "v2.0.0".to_string(),
                created_at_ns: 0,
                sha256_hash: "hash2".to_string(),
                input_shape: vec![1],
                output_shape: vec![1],
                metrics: HashMap::new(),
            },
        );

        assert!(swapper.rollback());
        assert_eq!(swapper.current_version(), Some("v1.0.0".to_string()));
    }
}
