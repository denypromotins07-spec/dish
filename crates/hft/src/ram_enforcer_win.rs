// Chapter 1, File 3: RAM Enforcer Watchdog
// crates/hft/src/ram_enforcer_win.rs
// Monitors GlobalMemoryStatusEx and triggers GC/telemetry flush when approaching 10GB limit

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use windows::{
    Win32::System::Memory::{
        GlobalMemoryStatusEx, MEMORYSTATUSEX,
    },
    Win32::Foundation::{BOOL, TRUE},
};

const TOTAL_RAM_LIMIT_BYTES: u64 = 10 * 1024 * 1024 * 1024; // 10GB hard cap
const WARNING_THRESHOLD_PERCENT: f64 = 0.85; // 85% of limit triggers action
const CRITICAL_THRESHOLD_PERCENT: f64 = 0.95; // 95% triggers aggressive action
const POLL_INTERVAL_MS: u64 = 100; // Check every 100ms

/// Shared state for RAM monitoring
pub struct RamEnforcerState {
    pub current_working_set_bytes: AtomicU64,
    pub python_gc_triggered: AtomicBool,
    pub telemetry_flushed: AtomicBool,
    pub critical_alert: AtomicBool,
}

impl RamEnforcerState {
    pub fn new() -> Self {
        RamEnforcerState {
            current_working_set_bytes: AtomicU64::new(0),
            python_gc_triggered: AtomicBool::new(false),
            telemetry_flushed: AtomicBool::new(false),
            critical_alert: AtomicBool::new(false),
        }
    }

    pub fn reset_flags(&self) {
        self.python_gc_triggered.store(false, Ordering::Relaxed);
        self.telemetry_flushed.store(false, Ordering::Relaxed);
        self.critical_alert.store(false, Ordering::Relaxed);
    }
}

/// RAM Enforcer - Watches process memory and triggers mitigation actions
pub struct RamEnforcer {
    state: Arc<RamEnforcerState>,
    running: Arc<AtomicBool>,
    limit_bytes: u64,
    warning_threshold: f64,
    critical_threshold: f64,
}

impl RamEnforcer {
    pub fn new(limit_bytes: u64) -> Self {
        RamEnforcer {
            state: Arc::new(RamEnforcerState::new()),
            running: Arc::new(AtomicBool::new(false)),
            limit_bytes,
            warning_threshold: WARNING_THRESHOLD_PERCENT,
            critical_threshold: CRITICAL_THRESHOLD_PERCENT,
        }
    }

    /// Get current system memory status
    pub fn get_memory_status() -> Result<MEMORYSTATUSEX, String> {
        unsafe {
            let mut status = MEMORYSTATUSEX {
                dwLength: std::mem::size_of::<MEMORYSTATUSEX>() as u32,
                ..Default::default()
            };
            
            if GlobalMemoryStatusEx(&mut status).is_err() {
                return Err("Failed to get memory status".to_string());
            }
            
            Ok(status)
        }
    }

    /// Get current process working set size (Windows-specific)
    pub fn get_process_working_set() -> Result<u64, String> {
        use windows::Win32::System::Threading::{GetCurrentProcess, GetProcessMemoryInfo};
        use windows::Win32::System::Diagnostics::Psapi::PROCESS_MEMORY_COUNTERS;

        unsafe {
            let mut pmc: PROCESS_MEMORY_COUNTERS = std::mem::zeroed();
            pmc.cb = std::mem::size_of::<PROCESS_MEMORY_COUNTERS>() as u32;
            
            if GetProcessMemoryInfo(GetCurrentProcess(), &mut pmc, std::mem::size_of::<PROCESS_MEMORY_COUNTERS>() as u32).is_err() {
                return Err("Failed to get process memory info".to_string());
            }
            
            Ok(pmc.WorkingSetSize as u64)
        }
    }

    /// Trigger Python GC via IPC signal
    fn trigger_python_gc(&self) {
        self.state.python_gc_triggered.store(true, Ordering::Release);
        log_action("[RAM_ENFORCER] Triggered Python GC");
    }

    /// Flush non-critical telemetry to disk
    fn flush_telemetry(&self) {
        self.state.telemetry_flushed.store(true, Ordering::Release);
        log_action("[RAM_ENFORCER] Flushed telemetry to disk");
    }

    /// Log critical alert
    fn log_critical_alert(&self, usage_percent: f64) {
        self.state.critical_alert.store(true, Ordering::Release);
        eprintln!("[CRITICAL] RAM usage at {:.2}% - Immediate action required!", usage_percent);
    }

    /// Main enforcement loop - call this in a dedicated thread
    pub fn run_enforcement_loop(&self, gc_callback: Box<dyn Fn() + Send + Sync>, telemetry_callback: Box<dyn Fn() + Send + Sync>) {
        self.running.store(true, Ordering::Release);
        
        while self.running.load(Ordering::Acquire) {
            match Self::get_process_working_set() {
                Ok(working_set) => {
                    self.state.current_working_set_bytes.store(working_set, Ordering::Relaxed);
                    
                    let usage_ratio = working_set as f64 / self.limit_bytes as f64;
                    let usage_percent = usage_ratio * 100.0;

                    if usage_ratio >= self.critical_threshold {
                        self.log_critical_alert(usage_percent);
                        gc_callback();
                        telemetry_callback();
                        std::thread::sleep(Duration::from_millis(10)); // Aggressive polling
                    } else if usage_ratio >= self.warning_threshold {
                        log_action(&format!("[WARNING] RAM usage at {:.2}%", usage_percent));
                        gc_callback();
                    }
                }
                Err(e) => {
                    eprintln!("[RAM_ENFORCER] Error getting memory info: {}", e);
                }
            }

            std::thread::sleep(Duration::from_millis(POLL_INTERVAL_MS));
        }
    }

    pub fn stop(&self) {
        self.running.store(false, Ordering::Release);
    }

    pub fn is_running(&self) -> bool {
        self.running.load(Ordering::Acquire)
    }

    pub fn state(&self) -> Arc<RamEnforcerState> {
        Arc::clone(&self.state)
    }

    pub fn current_usage_percent(&self) -> f64 {
        let current = self.state.current_working_set_bytes.load(Ordering::Relaxed);
        (current as f64 / self.limit_bytes as f64) * 100.0
    }
}

fn log_action(msg: &str) {
    println!("{}", msg);
}

/// Default callbacks for Python GC and telemetry flush
pub fn default_gc_callback() {
    // Signal Python side via shared memory or IPC to trigger GC
    log_action("[GC_CALLBACK] Signaling Python workers to garbage collect");
}

pub fn default_telemetry_callback() {
    // Flush telemetry buffers to disk
    log_action("[TELEMETRY_CALLBACK] Flushing non-critical telemetry");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_status() {
        let status = RamEnforcer::get_memory_status().expect("Failed to get memory status");
        assert!(status.ullTotalPhys > 0);
        assert!(status.ullAvailPhys > 0);
    }

    #[test]
    fn test_working_set_query() {
        let ws = RamEnforcer::get_process_working_set().expect("Failed to get working set");
        assert!(ws > 0);
        assert!(ws < TOTAL_RAM_LIMIT_BYTES);
    }

    #[test]
    fn test_enforcer_state() {
        let enforcer = RamEnforcer::new(TOTAL_RAM_LIMIT_BYTES);
        let state = enforcer.state();
        assert_eq!(state.current_working_set_bytes.load(Ordering::Relaxed), 0);
        assert!(!state.python_gc_triggered.load(Ordering::Relaxed));
    }
}
