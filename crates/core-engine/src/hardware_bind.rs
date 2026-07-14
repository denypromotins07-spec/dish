//! Hardware Thread Binding Module
//! 
//! This module provides OS-specific APIs to bind critical trading threads
//! to specific CPU cores, minimizing context switching and cache misses.
//! Optimized for AMD Ryzen AI 5 topology with L3 cache awareness.

use std::fmt;
use std::io;
use libc::{self, cpu_set_t, sched_setaffinity, pthread_self, pthread_setaffinity_np};

/// Error types for thread affinity operations
#[derive(Debug)]
pub enum AffinityError {
    IoError(io::Error),
    InvalidCore(usize),
    SetAffinityFailed(i32),
    GetAffinityFailed(i32),
}

impl fmt::Display for AffinityError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AffinityError::IoError(e) => write!(f, "IO error: {}", e),
            AffinityError::InvalidCore(core) => write!(f, "Invalid core ID: {}", core),
            AffinityError::SetAffinityFailed(code) => write!(f, "setaffinity failed: {}", code),
            AffinityError::GetAffinityFailed(code) => write!(f, "getaffinity failed: {}", code),
        }
    }
}

impl std::error::Error for AffinityError {}

/// CPU core topology information for AMD Ryzen AI 5
#[derive(Debug, Clone)]
pub struct CpuTopology {
    pub physical_cores: usize,
    pub logical_threads: usize,
    pub l3_cache_slices: Vec<L3CacheSlice>,
}

#[derive(Debug, Clone)]
pub struct L3CacheSlice {
    pub slice_id: usize,
    pub cores: Vec<usize>,
    pub size_mb: usize,
}

impl Default for CpuTopology {
    fn default() -> Self {
        // AMD Ryzen AI 5 default topology
        Self {
            physical_cores: 6,
            logical_threads: 12,
            l3_cache_slices: vec![
                L3CacheSlice {
                    slice_id: 0,
                    cores: vec![0, 1, 2, 3],
                    size_mb: 8,
                },
                L3CacheSlice {
                    slice_id: 1,
                    cores: vec![4, 5, 6, 7],
                    size_mb: 8,
                },
            ],
        }
    }
}

impl CpuTopology {
    /// Detect actual CPU topology from system
    pub fn detect() -> Result<Self, AffinityError> {
        #[cfg(target_os = "linux")]
        {
            // Read from /proc/cpuinfo
            if let Ok(content) = std::fs::read_to_string("/proc/cpuinfo") {
                let mut physical_ids = std::collections::HashSet::new();
                let mut siblings = 0;
                
                for line in content.lines() {
                    if line.starts_with("physical id") {
                        if let Some(id) = line.split(':').nth(1) {
                            physical_ids.insert(id.trim().to_string());
                        }
                    }
                    if line.starts_with("siblings") {
                        if let Some(sib) = line.split(':').nth(1) {
                            siblings = sib.trim().parse().unwrap_or(0);
                        }
                    }
                }
                
                let logical_threads = num_cpus::get();
                let physical_cores = logical_threads / 2;
                
                return Ok(Self {
                    physical_cores,
                    logical_threads,
                    l3_cache_slices: Self::build_cache_slices(logical_threads),
                });
            }
        }
        
        // Fallback to default
        Ok(Self::default())
    }
    
    fn build_cache_slices(total_threads: usize) -> Vec<L3CacheSlice> {
        // Simplified cache slice detection
        // In production, would read from sysfs on Linux
        let half = total_threads / 2;
        vec![
            L3CacheSlice {
                slice_id: 0,
                cores: (0..half).collect(),
                size_mb: 8,
            },
            L3CacheSlice {
                slice_id: 1,
                cores: (half..total_threads).collect(),
                size_mb: 8,
            },
        ]
    }
    
    /// Get the optimal L3 cache slice for a given core
    pub fn get_slice_for_core(&self, core: usize) -> Option<&L3CacheSlice> {
        self.l3_cache_slices.iter().find(|slice| slice.cores.contains(&core))
    }
    
    /// Check if two cores share the same L3 cache
    pub fn cores_share_l3(&self, core1: usize, core2: usize) -> bool {
        self.l3_cache_slices.iter().any(|slice| {
            slice.cores.contains(&core1) && slice.cores.contains(&core2)
        })
    }
}

/// Thread affinity manager for binding threads to CPU cores
pub struct ThreadAffinity;

impl ThreadAffinity {
    /// Pin current thread to specific CPU cores
    pub fn pin_to_cpus(cores: &[usize]) -> Result<(), AffinityError> {
        let max_cpu = cores.iter().copied().max().unwrap_or(0);
        
        // Validate all core IDs
        let available_cpus = num_cpus::get();
        for &core in cores {
            if core >= available_cpus {
                return Err(AffinityError::InvalidCore(core));
            }
        }
        
        #[cfg(target_os = "linux")]
        {
            unsafe {
                let mut cpuset: cpu_set_t = std::mem::zeroed();
                libc::CPU_ZERO(&mut cpuset);
                
                for &core in cores {
                    libc::CPU_SET(core, &mut cpuset);
                }
                
                // Get current thread
                let tid = pthread_self();
                
                // Set affinity
                let result = pthread_setaffinity_np(
                    tid,
                    std::mem::size_of::<cpu_set_t>(),
                    &cpuset as *const _ as *const _,
                );
                
                if result != 0 {
                    return Err(AffinityError::SetAffinityFailed(result));
                }
            }
            
            return Ok(());
        }
        
        #[cfg(not(target_os = "linux"))]
        {
            // Platform not supported, return success but log warning
            eprintln!("[AFFINITY] Thread pinning not supported on this platform");
            Ok(())
        }
    }
    
    /// Pin current thread to a single CPU core
    pub fn pin_to_core(core: usize) -> Result<(), AffinityError> {
        Self::pin_to_cpus(&[core])
    }
    
    /// Get current thread's CPU affinity
    pub fn get_current_affinity() -> Result<Vec<usize>, AffinityError> {
        #[cfg(target_os = "linux")]
        {
            unsafe {
                let mut cpuset: cpu_set_t = std::mem::zeroed();
                
                let tid = pthread_self();
                let result = pthread_getaffinity_np(
                    tid,
                    std::mem::size_of::<cpu_set_t>(),
                    &mut cpuset,
                );
                
                if result != 0 {
                    return Err(AffinityError::GetAffinityFailed(result));
                }
                
                let mut cores = Vec::new();
                for i in 0..libc::CPU_SETSIZE as usize {
                    if libc::CPU_ISSET(i, &cpuset) {
                        cores.push(i);
                    }
                }
                
                return Ok(cores);
            }
        }
        
        #[cfg(not(target_os = "linux"))]
        {
            Ok(vec![])
        }
    }
    
    /// Optimal core assignment for trading components based on L3 cache topology
    pub fn get_optimal_assignment() -> TradingCoreAssignment {
        let topology = CpuTopology::default();
        
        // Assign components to cores that share L3 cache for low latency
        TradingCoreAssignment {
            network_io: vec![0, 1],           // First L3 slice
            event_processing: vec![2, 3],     // Same L3 slice as network
            strategy_engine: vec![4, 5],      // Second L3 slice
            risk_management: vec![6, 7],      // Same L3 slice as strategy
            background_tasks: vec![8, 9, 10, 11], // Remaining cores
        }
    }
}

/// Core assignment for different trading components
#[derive(Debug, Clone)]
pub struct TradingCoreAssignment {
    pub network_io: Vec<usize>,
    pub event_processing: Vec<usize>,
    pub strategy_engine: Vec<usize>,
    pub risk_management: Vec<usize>,
    pub background_tasks: Vec<usize>,
}

impl TradingCoreAssignment {
    /// Verify that related components share L3 cache
    pub fn verify_cache_locality(&self, topology: &CpuTopology) -> CacheLocalityReport {
        let network_event_shared = self.network_io.iter().all(|&c1| {
            self.event_processing.iter().all(|&c2| {
                topology.cores_share_l3(c1, c2)
            })
        });
        
        let strategy_risk_shared = self.strategy_engine.iter().all(|&c1| {
            self.risk_management.iter().all(|&c2| {
                topology.cores_share_l3(c1, c2)
            })
        });
        
        CacheLocalityReport {
            network_event_cache_local: network_event_shared,
            strategy_risk_cache_local: strategy_risk_shared,
        }
    }
}

/// Report on cache locality of component assignments
#[derive(Debug, Clone)]
pub struct CacheLocalityReport {
    pub network_event_cache_local: bool,
    pub strategy_risk_cache_local: bool,
}

/// Scoped thread binder - automatically restores affinity when dropped
pub struct ScopedThreadBinder {
    original_affinity: Vec<usize>,
}

impl ScopedThreadBinder {
    /// Create a new scoped binder that pins to specified cores
    pub fn new(cores: &[usize]) -> Result<Self, AffinityError> {
        let original = ThreadAffinity::get_current_affinity()?;
        ThreadAffinity::pin_to_cpus(cores)?;
        
        Ok(Self {
            original_affinity: original,
        })
    }
    
    /// Restore original affinity immediately
    pub fn restore(self) {
        drop(self);
    }
}

impl Drop for ScopedThreadBinder {
    fn drop(&mut self) {
        let _ = ThreadAffinity::pin_to_cpus(&self.original_affinity);
    }
}

// Required for pthread_getaffinity_np which we use in get_current_affinity
#[cfg(target_os = "linux")]
extern "C" {
    fn pthread_getaffinity_np(
        thread: libc::pthread_t,
        cpusetsize: usize,
        cpuset: *mut cpu_set_t,
    ) -> i32;
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_topology_detection() {
        let topology = CpuTopology::detect().unwrap();
        assert!(topology.physical_cores > 0);
        assert!(topology.logical_threads > 0);
        assert!(!topology.l3_cache_slices.is_empty());
    }
    
    #[test]
    fn test_core_assignment() {
        let assignment = ThreadAffinity::get_optimal_assignment();
        
        // Verify no overlap between critical and background tasks
        for &bg_core in &assignment.background_tasks {
            assert!(!assignment.network_io.contains(&bg_core));
            assert!(!assignment.event_processing.contains(&bg_core));
            assert!(!assignment.strategy_engine.contains(&bg_core));
        }
    }
    
    #[test]
    fn test_cache_locality_verification() {
        let topology = CpuTopology::default();
        let assignment = ThreadAffinity::get_optimal_assignment();
        let report = assignment.verify_cache_locality(&topology);
        
        println!("Cache locality report: {:?}", report);
    }
}
