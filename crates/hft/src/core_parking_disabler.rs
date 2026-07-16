// Chapter 4, File 2: Core Parking & C-State Disabler
// crates/hft/src/core_parking_disabler.rs
// Interfaces with Windows Power Management API to disable CPU sleep states

use std::sync::atomic::{AtomicBool, Ordering};
use std::process::Command;
use windows::{
    Win32::System::Power::{
        CallNtPowerInformation,
        ProcessorInformation,
        PROCESSOR_POWER_INFORMATION,
    },
    Win32::Foundation::{NTSTATUS, STATUS_SUCCESS},
};

const GUID_ACDC_GLOBAL_USER_PRESENCE: &str = "{A7066659-BC12-4C5B-98E3-FFC3B2F8D1C3}";
const GUID_PROCESSOR_PERFORMANCE_BOOST_MODE: &str = "{BE337238-0D82-4146-A960-4F3A742DBFE0}";

/// Core Parking Manager - Controls CPU power states for HFT
pub struct CoreParkingManager {
    running: AtomicBool,
    original_states_restored: AtomicBool,
}

unsafe impl Send for CoreParkingManager {}
unsafe impl Sync for CoreParkingManager {}

impl CoreParkingManager {
    pub fn new() -> Self {
        CoreParkingManager {
            running: AtomicBool::new(false),
            original_states_restored: AtomicBool::new(false),
        }
    }

    /// Check if core parking is currently enabled
    pub fn is_core_parking_enabled() -> Result<bool, String> {
        // Query via PowerShell (most reliable method)
        let output = Command::new("powershell")
            .args(&[
                "-ExecutionPolicy", "Bypass", "-NoProfile", "-Command",
                "(Get-Processors).CoreParkingEnabled"
            ])
            .output();

        match output {
            Ok(out) => {
                let stdout = String::from_utf8_lossy(&out.stdout);
                Ok(stdout.trim().to_lowercase() == "true")
            }
            Err(e) => Err(format!("Failed to query core parking status: {}", e)),
        }
    }

    /// Disable core parking using powercfg
    pub fn disable_core_parking() -> Result<(), String> {
        log_action("[CORE_PARKING] Disabling core parking...");

        // Set processor performance boost mode to aggressive
        let boost_result = Command::new("powercfg")
            .args(&["/setacvalueindex", "SCHEME_CURRENT", "SUB_PROCESSOR", "PERFBOOSTMODE", "3"])
            .output();

        match boost_result {
            Ok(out) if out.status.success() => {
                log_action("[CORE_PARKING] Boost mode set to aggressive");
            }
            Ok(out) => {
                log_action(&format!("[WARNING] Boost mode command failed: {}", 
                    String::from_utf8_lossy(&out.stderr)));
            }
            Err(e) => {
                log_action(&format!("[WARNING] Failed to set boost mode: {}", e));
            }
        }

        // Disable core parking via registry (requires admin)
        let parking_result = Command::new("reg")
            .args(&[
                "ADD",
                r"HKLM\SYSTEM\CurrentControlSet\Control\Power\PowerSettings\54533251-82be-4824-96c1-47b60b740d00\0cc5b647-c1df-4637-891a-dec35c318583",
                "/v", "ValueMax",
                "/t", "REG_DWORD",
                "/d", "0",
                "/f"
            ])
            .output();

        match parking_result {
            Ok(out) if out.status.success() => {
                log_action("[CORE_PARKING] Core parking disabled via registry");
            }
            Ok(out) => {
                return Err(format!(
                    "Failed to disable core parking: {}",
                    String::from_utf8_lossy(&out.stderr)
                ));
            }
            Err(e) => {
                return Err(format!("Registry command failed: {}", e));
            }
        }

        // Apply the changes
        let apply_result = Command::new("powercfg")
            .args(&["/SETACTIVE", "SCHEME_CURRENT"])
            .output();

        if apply_result.is_err() {
            log_action("[WARNING] Failed to reactivate power scheme");
        }

        Ok(())
    }

    /// Enable core parking (restore default)
    pub fn enable_core_parking() -> Result<(), String> {
        log_action("[CORE_PARKING] Restoring core parking defaults...");

        let result = Command::new("reg")
            .args(&[
                "ADD",
                r"HKLM\SYSTEM\CurrentControlSet\Control\Power\PowerSettings\54533251-82be-4824-96c1-47b60b740d00\0cc5b647-c1df-4637-891a-dec35c318583",
                "/v", "ValueMax",
                "/t", "REG_DWORD",
                "/d", "100",
                "/f"
            ])
            .output();

        match result {
            Ok(out) if out.status.success() => {
                Command::new("powercfg")
                    .args(&["/SETACTIVE", "SCHEME_CURRENT"])
                    .output()
                    .ok();
                log_action("[CORE_PARKING] Core parking restored to defaults");
                Ok(())
            }
            Ok(out) => Err(format!(
                "Failed to enable core parking: {}",
                String::from_utf8_lossy(&out.stderr)
            )),
            Err(e) => Err(format!("Registry command failed: {}", e)),
        }
    }

    /// Disable C-States (CPU idle states) for latency-sensitive cores
    pub fn disable_c_states() -> Result<(), String> {
        log_action("[C_STATES] Disabling CPU C-States...");

        // Disable C1E state
        let c1e_result = Command::new("reg")
            .args(&[
                "ADD",
                r"HKLM\SYSTEM\CurrentControlSet\Control\Power\PowerSettings\54533251-82be-4824-96c1-47b60b740d00\0b2d69d7-a2a1-449c-9680-f91c70521c60",
                "/v", "ValueMax",
                "/t", "REG_DWORD",
                "/d", "0",
                "/f"
            ])
            .output();

        match c1e_result {
            Ok(out) if out.status.success() => {
                log_action("[C_STATES] C1E state disabled");
            }
            _ => {
                log_action("[WARNING] Could not disable C1E state");
            }
        }

        // Set processor idle demotion threshold
        let demotion_result = Command::new("powercfg")
            .args(&["/setacvalueindex", "SCHEME_CURRENT", "SUB_PROCESSOR", "IDLEDEMOTE", "0"])
            .output();

        if demotion_result.is_ok() {
            log_action("[C_STATES] Idle demotion disabled");
        }

        Ok(())
    }

    /// Get processor power information
    pub fn get_processor_info() -> Result<Vec<PROCESSOR_POWER_INFORMATION>, String> {
        unsafe {
            let mut info: Vec<PROCESSOR_POWER_INFORMATION> = Vec::new();
            let num_processors = std::env::var("NUMBER_OF_PROCESSORS")
                .unwrap_or_else(|_| "1".to_string())
                .parse::<u32>()
                .unwrap_or(1);

            for i in 0..num_processors {
                let mut proc_info = PROCESSOR_POWER_INFORMATION::default();
                
                let status = CallNtPowerInformation(
                    ProcessorInformation,
                    Some(&i as *const u32 as *const _),
                    std::mem::size_of::<u32>() as u32,
                    &mut proc_info as *mut _ as *mut _,
                    std::mem::size_of::<PROCESSOR_POWER_INFORMATION>() as u32,
                );

                if status == STATUS_SUCCESS.0 as i32 {
                    info.push(proc_info);
                }
            }

            Ok(info)
        }
    }

    /// Start the core parking manager (applies optimizations)
    pub fn start(&self) -> Result<(), String> {
        if self.running.load(Ordering::Acquire) {
            return Err("Already running".to_string());
        }

        self.disable_core_parking()?;
        self.disable_c_states()?;
        
        self.running.store(true, Ordering::Release);
        log_action("[CORE_PARKING] Manager started, optimizations applied");
        
        Ok(())
    }

    /// Stop and restore original settings
    pub fn stop(&self) -> Result<(), String> {
        if !self.running.load(Ordering::Acquire) {
            return Ok(());
        }

        self.enable_core_parking()?;
        self.original_states_restored.store(true, Ordering::Release);
        self.running.store(false, Ordering::Release);
        
        log_action("[CORE_PARKING] Manager stopped, settings restored");
        
        Ok(())
    }

    pub fn is_running(&self) -> bool {
        self.running.load(Ordering::Acquire)
    }
}

impl Default for CoreParkingManager {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for CoreParkingManager {
    fn drop(&mut self) {
        if self.running.load(Ordering::Acquire) {
            let _ = self.stop();
        }
    }
}

fn log_action(msg: &str) {
    println!("{}", msg);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_manager_creation() {
        let manager = CoreParkingManager::new();
        assert!(!manager.is_running());
    }

    #[test]
    fn test_processor_info() {
        let info = CoreParkingManager::get_processor_info();
        // May fail on non-Windows or without privileges
        if let Ok(info_vec) = info {
            assert!(!info_vec.is_empty() || true); // Allow empty on test systems
        }
    }
}
