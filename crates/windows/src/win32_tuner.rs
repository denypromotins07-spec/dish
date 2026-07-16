//! Windows-specific low-latency tuning module
//! 
//! This module provides Win32 API bindings for:
//! - Locking working set in physical RAM (preventing paging)
//! - Setting thread priority to TIME_CRITICAL
//! - Pinning threads to specific AMD Ryzen logical cores
//! 
//! Target: AMD Ryzen AI 5 with 16GB total RAM, strict 10GB system limit

use std::ffi::c_void;
use std::io;
use std::ptr;
use windows_sys::Win32::Foundation::{BOOL, HANDLE, TRUE};
use windows_sys::Win32::System::Memory::{
    SetProcessWorkingSetSize, VirtualLock, MEM_COMMIT, PAGE_READWRITE,
};
use windows_sys::Win32::System::Threading::{
    GetCurrentProcess, GetCurrentThread, SetThreadPriority, THREAD_PRIORITY_TIME_CRITICAL,
};
use windows_sys::Win32::System::SystemServices::{VER_PLATFORM_WIN32_NT};
use windows_sys::Win32::System::Diagnostics::ToolHelp::{
    CreateToolhelp32Snapshot, Thread32First, Thread32Next, TH32CS_SNAPTHREAD, THREADENTRY32,
};

/// Locks the process working set to prevent Windows from paging memory to disk.
/// This is critical for maintaining microsecond latency in high-frequency trading.
/// 
/// # Arguments
/// * `min_working_set` - Minimum bytes to keep in physical RAM
/// * `max_working_set` - Maximum bytes allowed in physical RAM
/// 
/// # Returns
/// * `Ok(())` on success
/// * `Err(io::Error)` on failure
pub fn lock_working_set(min_working_set: usize, max_working_set: usize) -> io::Result<()> {
    unsafe {
        let process = GetCurrentProcess();
        let result = SetProcessWorkingSetSize(
            process,
            min_working_set,
            max_working_set,
        );
        
        if result == 0 {
            return Err(io::Error::last_os_error());
        }
        
        // Enable "Lock Pages in Memory" privilege if available
        // Note: Requires SeLockMemoryPrivilege via Local Security Policy
        enable_lock_pages_privilege()?;
        
        Ok(())
    }
}

/// Enables the SeLockMemoryPrivilege required for VirtualLock
fn enable_lock_pages_privilege() -> io::Result<()> {
    use windows_sys::Win32::Security::*;
    use windows_sys::Win32::System::Threading::*;
    
    unsafe {
        let mut token_handle: HANDLE = 0;
        let mut luid: LUID = LUID { LowPart: 0, HighPart: 0 };
        
        // Open process token
        if OpenProcessToken(GetCurrentProcess(), TOKEN_ADJUST_PRIVILEGES | TOKEN_QUERY, &mut token_handle) == 0 {
            return Err(io::Error::last_os_error());
        }
        
        // Lookup privilege value for SeLockMemoryPrivilege
        if LookupPrivilegeValueA(ptr::null(), b"SeLockMemoryPrivilege\0".as_ptr() as *const i8, &mut luid) == 0 {
            CloseHandle(token_handle);
            return Err(io::Error::last_os_error());
        }
        
        // Adjust privileges
        let mut tkp: TOKEN_PRIVILEGES = TOKEN_PRIVILEGES {
            PrivilegeCount: 1,
            Privileges: [LUID_AND_ATTRIBUTES {
                Luid: luid,
                Attributes: SE_PRIVILEGE_ENABLED,
            }],
        };
        
        if AdjustTokenPrivileges(token_handle, FALSE, &mut tkp, 0, ptr::null_mut(), ptr::null_mut()) == 0 {
            CloseHandle(token_handle);
            return Err(io::Error::last_os_error());
        }
        
        CloseHandle(token_handle);
        Ok(())
    }
}

/// Sets the current thread priority to TIME_CRITICAL for ultra-low latency
/// This gives the thread the highest possible priority on Windows
pub fn set_thread_time_critical() -> io::Result<()> {
    unsafe {
        let thread = GetCurrentThread();
        let result = SetThreadPriority(thread, THREAD_PRIORITY_TIME_CRITICAL);
        
        if result == 0 {
            return Err(io::Error::last_os_error());
        }
        
        Ok(())
    }
}

/// Pins the current thread to a specific logical core on AMD Ryzen AI 5
/// 
/// # Arguments
/// * `core_id` - The logical core ID (0-11 for 6-core/12-thread Ryzen AI 5)
pub fn pin_thread_to_core(core_id: usize) -> io::Result<()> {
    use windows_sys::Win32::System::Threading::SetThreadAffinityMask;
    
    unsafe {
        let thread = GetCurrentThread();
        let affinity_mask: usize = 1 << core_id;
        
        let result = SetThreadAffinityMask(thread, affinity_mask);
        
        if result == 0 {
            return Err(io::Error::last_os_error());
        }
        
        Ok(())
    }
}

/// Configures optimal CPU affinity for the entire process
/// Distributes work across AMD Ryzen AI 5 cores while avoiding OS overhead
pub fn configure_process_affinity() -> io::Result<()> {
    use windows_sys::Win32::System::Threading::SetProcessAffinityMask;
    
    unsafe {
        let process = GetCurrentProcess();
        // Use all available logical cores (adjust mask based on actual core count)
        // For Ryzen AI 5 (6 cores, 12 threads): 0xFFF = all 12 logical processors
        let affinity_mask: usize = 0xFFF;
        
        let result = SetProcessAffinityMask(process, affinity_mask);
        
        if result == 0 {
            return Err(io::Error::last_os_error());
        }
        
        Ok(())
    }
}

/// Virtual locks a memory region to ensure it stays in physical RAM
/// 
/// # Safety
/// The pointer must be valid and the size must not exceed the allocated region
pub unsafe fn lock_memory_region(ptr: *mut c_void, size: usize) -> io::Result<()> {
    let result = VirtualLock(ptr, size);
    
    if result == 0 {
        return Err(io::Error::last_os_error());
    }
    
    Ok(())
}

/// Initializes the Windows low-latency environment
/// Call this at application startup
pub fn init_low_latency_env() -> io::Result<()> {
    // Lock 8GB of working set (leaving room for OS and other processes within 10GB limit)
    lock_working_set(4 * 1024 * 1024 * 1024, 8 * 1024 * 1024 * 1024)?;
    
    // Configure process-wide CPU affinity
    configure_process_affinity()?;
    
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    #[ignore] // Requires admin privileges
    fn test_lock_working_set() {
        let result = lock_working_set(1024 * 1024 * 1024, 2 * 1024 * 1024 * 1024);
        assert!(result.is_ok());
    }
    
    #[test]
    fn test_set_thread_time_critical() {
        let result = set_thread_time_critical();
        assert!(result.is_ok());
    }
}
