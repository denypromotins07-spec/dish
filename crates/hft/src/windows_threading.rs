// Chapter 4, File 1: Windows Thread Affinity & Core Binding
// crates/hft/src/windows_threading.rs
// Binds critical HFT threads to specific AMD Ryzen cores using SetThreadAffinityMask

use std::sync::atomic::{AtomicBool, Ordering};
use windows::{
    Win32::System::Threading::{
        GetCurrentThread, SetThreadAffinityMask, SetThreadPriority,
        THREAD_PRIORITY_TIME_CRITICAL, THREAD_PRIORITY_HIGHEST,
        GROUP_AFFINITY,
    },
    Win32::Foundation::{HANDLE, BOOL},
};

const MAX_LOGICAL_PROCESSORS: usize = 64; // Support up to 64 logical cores

/// Thread affinity configuration for HFT workloads
pub struct ThreadAffinityConfig {
    pub core_id: usize,
    pub disable_smt: bool,
    pub priority: i32,
}

impl Default for ThreadAffinityConfig {
    fn default() -> Self {
        ThreadAffinityConfig {
            core_id: 0,
            disable_smt: true,
            priority: THREAD_PRIORITY_TIME_CRITICAL,
        }
    }
}

/// HFT Thread Manager - Manages thread affinity and priorities
pub struct HFTThreadManager {
    running: AtomicBool,
    bound_cores: Vec<usize>,
}

unsafe impl Send for HFTThreadManager {}
unsafe impl Sync for HFTThreadManager {}

impl HFTThreadManager {
    pub fn new() -> Self {
        HFTThreadManager {
            running: AtomicBool::new(false),
            bound_cores: Vec::new(),
        }
    }

    /// Bind current thread to a specific logical processor
    pub fn bind_current_thread_to_core(core_id: usize) -> Result<(), String> {
        unsafe {
            let thread_handle = GetCurrentThread();
            
            // Create affinity mask for single core
            let affinity_mask = 1usize << core_id;
            
            let result = SetThreadAffinityMask(thread_handle, affinity_mask);
            
            if result == 0 {
                return Err(format!(
                    "Failed to set thread affinity for core {}: {}",
                    core_id,
                    windows::Win32::Foundation::GetLastError().0
                ));
            }

            Ok(())
        }
    }

    /// Set current thread priority to time-critical (highest possible)
    pub fn set_time_critical_priority() -> Result<(), String> {
        unsafe {
            let thread_handle = GetCurrentThread();
            
            let result = SetThreadPriority(thread_handle, THREAD_PRIORITY_TIME_CRITICAL);
            
            if result == 0 {
                return Err(format!(
                    "Failed to set thread priority: {}",
                    windows::Win32::Foundation::GetLastError().0
                ));
            }

            Ok(())
        }
    }

    /// Configure thread for HFT execution (affinity + priority)
    pub fn configure_hft_thread(core_id: usize) -> Result<(), String> {
        // First bind to core
        Self::bind_current_thread_to_core(core_id)?;
        
        // Then set priority
        Self::set_time_critical_priority()?;
        
        log_action(&format!("[THREAD] Configured for HFT on core {}", core_id));
        Ok(())
    }

    /// Get recommended core layout for AMD Ryzen AI (Zen 4)
    /// Returns (CCD0 cores, CCD1 cores) - prefer CCD0 for latency-critical work
    pub fn get_amd_zen_core_layout() -> (Vec<usize>, Vec<usize>) {
        // AMD Ryzen AI typically has:
        // - CCD0: Cores 0-5 (preferred for latency)
        // - CCD1: Cores 6-11 (for background work)
        // SMT pairs: (0,1), (2,3), (4,5), etc.
        
        let ccd0_cores = vec![0, 2, 4]; // Physical cores on first CCD
        let ccd1_cores = vec![6, 8, 10]; // Physical cores on second CCD
        
        (ccd0_cores, ccd1_cores)
    }

    /// Disable SMT on specified core by avoiding its sibling
    /// For SMT pair (core, core+1), use only even-numbered core
    pub fn get_physical_core_without_smt(logical_core: usize) -> usize {
        // If odd, return the even sibling
        if logical_core % 2 == 1 {
            logical_core - 1
        } else {
            logical_core
        }
    }

    /// Spawn HFT thread bound to specific core
    pub fn spawn_hft_thread<F, T>(
        core_id: usize,
        f: F,
    ) -> Result<std::thread::JoinHandle<T>, String>
    where
        F: FnOnce() -> T + Send + 'static,
        T: Send + 'static,
    {
        let (tx, rx) = std::sync::mpsc::channel();

        let handle = std::thread::Builder::new()
            .name(format!("hft-core-{}", core_id))
            .spawn(move || {
                // Configure thread affinity inside the new thread
                match Self::configure_hft_thread(core_id) {
                    Ok(_) => tx.send(Ok(())).unwrap(),
                    Err(e) => tx.send(Err(e)).unwrap(),
                }

                f()
            })
            .map_err(|e| format!("Failed to spawn thread: {}", e))?;

        // Wait for affinity configuration
        rx.recv()
            .map_err(|e| format!("Channel error: {}", e))??;

        Ok(handle)
    }

    /// Start the thread manager
    pub fn start(&self) {
        self.running.store(true, Ordering::Release);
    }

    /// Stop the thread manager
    pub fn stop(&self) {
        self.running.store(false, Ordering::Release);
    }

    pub fn is_running(&self) -> bool {
        self.running.load(Ordering::Acquire)
    }

    pub fn bound_cores(&self) -> &[usize] {
        &self.bound_cores
    }
}

impl Default for HFTThreadManager {
    fn default() -> Self {
        Self::new()
    }
}

fn log_action(msg: &str) {
    println!("{}", msg);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_physical_core_calculation() {
        assert_eq!(HFTThreadManager::get_physical_core_without_smt(0), 0);
        assert_eq!(HFTThreadManager::get_physical_core_without_smt(1), 0);
        assert_eq!(HFTThreadManager::get_physical_core_without_smt(2), 2);
        assert_eq!(HFTThreadManager::get_physical_core_without_smt(3), 2);
    }

    #[test]
    fn test_core_layout() {
        let (ccd0, ccd1) = HFTThreadManager::get_amd_zen_core_layout();
        assert!(!ccd0.is_empty());
        assert!(!ccd1.is_empty());
    }

    #[test]
    fn test_manager_creation() {
        let manager = HFTThreadManager::new();
        assert!(!manager.is_running());
        assert!(manager.bound_cores().is_empty());
    }
}
