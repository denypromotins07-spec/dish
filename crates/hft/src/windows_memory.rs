// Chapter 1, File 1: Windows Memory Locking & Large Pages
// crates/hft/src/windows_memory.rs
// Replaces Linux mlock with VirtualLock and SeLockMemoryPrivilege for Large Pages

use std::ptr;
use std::ffi::c_void;
use windows::{
    Win32::System::Memory::{
        VirtualLock, VirtualUnlock, VirtualAlloc, VirtualFree,
        MEM_COMMIT, MEM_RESERVE, MEM_LARGE_PAGES, PAGE_READWRITE,
        MEMORY_BASIC_INFORMATION, VirtualQuery,
    },
    Win32::System::Threading::{
        GetCurrentProcess, OpenProcessToken,
        LookupPrivilegeValueW, AdjustTokenPrivileges,
        TOKEN_PRIVILEGES, LUID, TOKEN_ADJUST_PRIVILEGES, TOKEN_QUERY,
    },
    Win32::Foundation::{HANDLE, BOOL, TRUE, FALSE, CloseHandle},
    Win32::Security::{SE_LOCK_MEMORY_NAME, PRIVILEGE_SET_ALL_NECESSARY},
};

const LARGE_PAGE_SIZE: usize = 2 * 1024 * 1024; // 2MB large pages on x64

/// Enables the SeLockMemoryPrivilege required for allocating large pages
pub fn enable_lock_memory_privilege() -> Result<(), String> {
    unsafe {
        let mut token_handle: HANDLE = HANDLE::default();
        if OpenProcessToken(GetCurrentProcess(), TOKEN_ADJUST_PRIVILEGES | TOKEN_QUERY, &mut token_handle).is_err() {
            return Err("Failed to open process token".to_string());
        }

        let mut luid: LUID = LUID::default();
        let privilege_name_wide: Vec<u16> = SE_LOCK_MEMORY_NAME.encode_utf16().chain(Some(0)).collect();
        
        if LookupPrivilegeValueW(None, &privilege_name_wide[0], &mut luid).is_err() {
            CloseHandle(token_handle).ok();
            return Err("Failed to lookup privilege value".to_string());
        }

        let tp = TOKEN_PRIVILEGES {
            PrivilegeCount: 1,
            Privileges: [windows::Win32::System::Threading::LUID_AND_ATTRIBUTES {
                Luid: luid,
                Attributes: windows::Win32::System::Threading::SE_PRIVILEGE_ENABLED,
            }],
        };

        if AdjustTokenPrivileges(token_handle, FALSE, Some(&tp as *const _ as *const c_void), 0, None, None).is_err() {
            CloseHandle(token_handle).ok();
            return Err("Failed to adjust token privileges".to_string());
        }

        CloseHandle(token_handle).ok();
        Ok(())
    }
}

/// Allocates memory using Windows Large Pages (2MB) for reduced TLB misses
pub fn allocate_large_pages(size: usize) -> Result<*mut u8, String> {
    unsafe {
        // Round up to nearest large page boundary
        let aligned_size = ((size + LARGE_PAGE_SIZE - 1) / LARGE_PAGE_SIZE) * LARGE_PAGE_SIZE;
        
        let ptr = VirtualAlloc(
            None,
            aligned_size,
            MEM_COMMIT | MEM_RESERVE | MEM_LARGE_PAGES,
            PAGE_READWRITE,
        );

        if ptr.is_null() {
            return Err(format!("VirtualAlloc failed with error: {}", windows::Win32::Foundation::GetLastError().0));
        }

        Ok(ptr as *mut u8)
    }
}

/// Locks existing memory pages into physical RAM (prevents paging to disk)
pub fn lock_memory_to_ram(ptr: *mut u8, size: usize) -> Result<(), String> {
    unsafe {
        if VirtualLock(ptr as *const c_void, size).is_err() {
            return Err(format!("VirtualLock failed: {}", windows::Win32::Foundation::GetLastError().0));
        }
        Ok(())
    }
}

/// Unlocks memory pages, allowing them to be paged out
pub fn unlock_memory(ptr: *mut u8, size: usize) -> Result<(), String> {
    unsafe {
        if VirtualUnlock(ptr as *const c_void, size).is_ok() {
            Ok(())
        } else {
            Err(format!("VirtualUnlock failed: {}", windows::Win32::Foundation::GetLastError().0))
        }
    }
}

/// Frees large page memory
pub fn free_large_pages(ptr: *mut u8) -> Result<(), String> {
    unsafe {
        if VirtualFree(ptr as *mut c_void, 0, MEM_RELEASE).is_err() {
            return Err(format!("VirtualFree failed: {}", windows::Win32::Foundation::GetLastError().0));
        }
        Ok(())
    }
}

/// HFTMemoryArena - Manages pinned large page memory for order book and execution engine
pub struct HFTMemoryArena {
    ptr: *mut u8,
    size: usize,
    locked: bool,
}

unsafe impl Send for HFTMemoryArena {}
unsafe impl Sync for HFTMemoryArena {}

impl HFTMemoryArena {
    pub fn new(size: usize) -> Result<Self, String> {
        enable_lock_memory_privilege()?;
        let ptr = allocate_large_pages(size)?;
        let arena = HFTMemoryArena {
            ptr,
            size,
            locked: false,
        };
        Ok(arena)
    }

    pub fn lock(&mut self) -> Result<(), String> {
        if !self.locked {
            lock_memory_to_ram(self.ptr, self.size)?;
            self.locked = true;
        }
        Ok(())
    }

    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.size) }
    }

    pub fn as_ptr(&self) -> *const u8 {
        self.ptr
    }

    pub fn size(&self) -> usize {
        self.size
    }
}

impl Drop for HFTMemoryArena {
    fn drop(&mut self) {
        if self.locked {
            let _ = unlock_memory(self.ptr, self.size);
        }
        let _ = free_large_pages(self.ptr);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_large_page_allocation() {
        let mut arena = HFTMemoryArena::new(4 * 1024 * 1024).expect("Failed to allocate");
        arena.lock().expect("Failed to lock");
        let slice = arena.as_mut_slice();
        assert_eq!(slice.len(), 4 * 1024 * 1024);
    }
}
