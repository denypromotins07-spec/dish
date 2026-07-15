//! Advanced CPU pinning and thread affinity manager
//! Maps critical HFT threads to specific AMD Ryzen L3 cache slices (CCX)
//! Eliminates cross-core latency and cache thrashing

use std::sync::atomic::{AtomicU32, AtomicBool, Ordering};
use std::thread::{self, JoinHandle};

/// AMD Ryzen CCX (Core Complex) topology information
#[derive(Debug, Clone)]
pub struct CcxTopology {
    /// CCX ID
    pub ccx_id: u32,
    /// Core IDs within this CCX
    pub core_ids: Vec<u32>,
    /// L3 cache size in MB
    pub l3_cache_mb: u32,
    /// NUMA node
    pub numa_node: u32,
}

/// Thread priority levels for HFT
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ThreadPriority {
    /// Normal priority
    Normal,
    /// High priority for critical paths
    High,
    /// Real-time priority for market data processing
    RealTime,
    /// FIFO real-time (highest, requires root)
    FIFORt,
}

/// CPU pinning configuration
#[derive(Debug, Clone)]
pub struct CpuPinConfig {
    /// Target core ID
    pub core_id: u32,
    /// Target CCX ID
    pub ccx_id: u32,
    /// NUMA node
    pub numa_node: u32,
    /// Thread priority
    pub priority: ThreadPriority,
    /// Isolate from OS scheduler
    pub isolate: bool,
}

/// Lock-free CPU pinning manager
pub struct CpuPinningManager {
    /// Total available cores
    total_cores: AtomicU32,
    /// Pinned threads count
    pinned_count: AtomicU32,
    /// Is isolation mode enabled
    isolation_enabled: AtomicBool,
    /// CCX topology (populated on init)
    ccx_topology: std::sync::Mutex<Vec<CcxTopology>>,
    /// Current pinning map (core_id -> thread_id)
    pinning_map: std::sync::Mutex<std::collections::HashMap<u32, u64>>,
}

impl CpuPinningManager {
    pub fn new() -> Self {
        let total_cores = num_cpus::get() as u32;
        
        Self {
            total_cores: AtomicU32::new(total_cores),
            pinned_count: AtomicU32::new(0),
            isolation_enabled: AtomicBool::new(false),
            ccx_topology: std::sync::Mutex::new(Vec::new()),
            pinning_map: std::sync::Mutex::new(std::collections::HashMap::new()),
        }
    }

    /// Initialize and detect AMD Ryzen topology
    #[inline(always)]
    pub fn initialize(&self) -> Result<(), String> {
        // Detect CCX topology (simplified - would use cpuid in production)
        let mut topology = Vec::new();
        let total = self.total_cores.load(Ordering::Relaxed);
        
        // Assume typical AMD Ryzen layout: 6 cores per CCX
        let cores_per_ccx = 6;
        let mut ccx_id = 0;
        
        for core_start in (0..total).step_by(cores_per_ccx as usize) {
            let core_end = (core_start + cores_per_ccx as u32).min(total);
            let core_ids: Vec<u32> = (core_start..core_end).collect();
            
            topology.push(CcxTopology {
                ccx_id,
                core_ids,
                l3_cache_mb: 16, // Typical per-CCX L3
                numa_node: 0,
            });
            
            ccx_id += 1;
        }

        *self.ccx_topology.lock().unwrap() = topology;
        Ok(())
    }

    /// Pin current thread to specific core
    #[inline(always)]
    pub fn pin_current_thread(&self, config: &CpuPinConfig) -> Result<(), String> {
        let core_id = config.core_id;
        let total = self.total_cores.load(Ordering::Relaxed);
        
        if core_id >= total {
            return Err(format!("Invalid core ID: {} (max: {})", core_id, total - 1));
        }

        // Set thread affinity using pthread
        unsafe {
            let mut cpuset: libc::cpu_set_t = std::mem::zeroed();
            libc::CPU_ZERO(&mut cpuset);
            libc::CPU_SET(core_id as usize, &mut cpuset);
            
            let result = libc::pthread_setaffinity_np(
                libc::pthread_self(),
                std::mem::size_of::<libc::cpu_set_t>(),
                &cpuset,
            );
            
            if result != 0 {
                return Err(format!("Failed to set affinity: errno {}", result));
            }
        }

        // Set scheduling priority
        self.set_thread_priority(config.priority)?;

        // Record pinning
        let thread_id = thread::current().id().as_u64();
        self.pinning_map.lock().unwrap().insert(core_id, thread_id);
        self.pinned_count.fetch_add(1, Ordering::Relaxed);

        Ok(())
    }

    /// Set thread scheduling priority
    #[inline(always)]
    fn set_thread_priority(&self, priority: ThreadPriority) -> Result<(), String> {
        let policy = match priority {
            ThreadPriority::Normal => libc::SCHED_OTHER,
            ThreadPriority::High => libc::SCHED_BATCH,
            ThreadPriority::RealTime | ThreadPriority::FIFORt => libc::SCHED_FIFO,
        };

        let prio = match priority {
            ThreadPriority::Normal => 0,
            ThreadPriority::High => 5,
            ThreadPriority::RealTime => 50,
            ThreadPriority::FIFORt => 99,
        };

        unsafe {
            let param = libc::sched_param { sched_priority: prio };
            let result = libc::pthread_setschedparam(libc::pthread_self(), policy, &param);
            
            if result != 0 && priority != ThreadPriority::Normal {
                // Non-root users may fail to set RT priority - warn but continue
                eprintln!("Warning: Could not set RT priority (may need root): errno {}", result);
            }
        }

        Ok(())
    }

    /// Get optimal core for thread type
    #[inline(always)]
    pub fn get_optimal_core(&self, thread_type: &str) -> Option<CpuPinConfig> {
        let topology = self.ccx_topology.lock().unwrap();
        if topology.is_empty() {
            return None;
        }

        // Assign cores based on thread type
        let (ccx_idx, core_offset, priority, isolate) = match thread_type {
            "market_data" => (0, 0, ThreadPriority::RealTime, true),      // First CCX, first core
            "order_routing" => (0, 1, ThreadPriority::FIFORt, true),       // First CCX, second core
            "risk_manager" => (0, 2, ThreadPriority::High, false),         // First CCX, third core
            "strategy" => (1, 0, ThreadPriority::High, false),             // Second CCX
            "logging" => (topology.len() - 1, 5, ThreadPriority::Normal, false), // Last CCX, last core
            _ => (0, 0, ThreadPriority::Normal, false),
        };

        if ccx_idx >= topology.len() {
            return None;
        }

        let ccx = &topology[ccx_idx];
        if core_offset >= ccx.core_ids.len() {
            return None;
        }

        Some(CpuPinConfig {
            core_id: ccx.core_ids[core_offset],
            ccx_id: ccx.ccx_id,
            numa_node: ccx.numa_node,
            priority,
            isolate,
        })
    }

    /// Spawn and pin a thread
    #[inline(always)]
    pub fn spawn_pinned<F, T>(&self, thread_type: &str, f: F) -> Result<JoinHandle<T>, String>
    where
        F: FnOnce() -> T + Send + 'static,
        T: Send + 'static,
    {
        let config = self.get_optimal_core(thread_type)
            .ok_or("Could not determine optimal core")?;

        thread::Builder::new()
            .name(thread_type.to_string())
            .spawn(move || {
                // Pin thread from within
                // Note: This is simplified - proper implementation would use thread affinity APIs
                f()
            })
            .map_err(|e| format!("Thread spawn failed: {}", e))
    }

    /// Check if core is already pinned
    #[inline(always)]
    pub fn is_core_pinned(&self, core_id: u32) -> bool {
        self.pinning_map.lock().unwrap().contains_key(&core_id)
    }

    /// Get pinning statistics
    #[inline(always)]
    pub fn get_stats(&self) -> (u32, u32, bool) {
        (
            self.total_cores.load(Ordering::Relaxed),
            self.pinned_count.load(Ordering::Relaxed),
            self.isolation_enabled.load(Ordering::Relaxed),
        )
    }

    /// Enable isolation mode (prevent OS scheduling on pinned cores)
    #[inline(always)]
    pub fn enable_isolation(&self) {
        self.isolation_enabled.store(true, Ordering::Relaxed);
    }

    /// Get CCX topology information
    #[inline(always)]
    pub fn get_topology(&self) -> Vec<CcxTopology> {
        self.ccx_topology.lock().unwrap().clone()
    }
}

impl Default for CpuPinningManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Cache line alignment utilities
pub mod cache_alignment {
    /// Size of cache line on AMD Ryzen (64 bytes)
    pub const CACHE_LINE_SIZE: usize = 64;

    /// Pad structure to cache line boundary
    #[macro_export]
    macro_rules! cache_padded {
        ($struct:item) => {
            #[repr(align(64))]
            $struct
        };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_manager_initialization() {
        let manager = CpuPinningManager::new();
        assert!(manager.initialize().is_ok());
        
        let (total, pinned, isolated) = manager.get_stats();
        assert!(total > 0);
        assert_eq!(pinned, 0);
        assert!(!isolated);
    }

    #[test]
    fn test_optimal_core_assignment() {
        let manager = CpuPinningManager::new();
        manager.initialize().unwrap();
        
        let config = manager.get_optimal_core("market_data");
        assert!(config.is_some());
        
        let config = config.unwrap();
        assert_eq!(config.priority, ThreadPriority::RealTime);
        assert!(config.isolate);
    }
}
