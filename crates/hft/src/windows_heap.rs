// Chapter 1, File 2: Custom Windows Heap Allocator
// crates/hft/src/windows_heap.rs
// Uses HeapCreate with HEAP_NO_SERIALIZE and custom memory arenas

use std::ptr;
use std::ffi::c_void;
use windows::{
    Win32::System::Heap::{
        HeapCreate, HeapDestroy, HeapAlloc, HeapFree,
        HEAP_GENERATE_EXCEPTIONS, HEAP_NO_SERIALIZE, HEAP_ZERO_MEMORY,
    },
    Win32::Foundation::{HANDLE, BOOL},
};

const ARENA_BLOCK_SIZE: usize = 64 * 1024 * 1024; // 64MB blocks
const MAX_ARENA_COUNT: usize = 16;

/// Custom heap wrapper for HFT tick ingestion - bypasses standard allocator overhead
pub struct HFTheap {
    handle: HANDLE,
    allocated_bytes: usize,
    allocation_count: usize,
}

unsafe impl Send for HFTheap {}
unsafe impl Sync for HFTheap {}

impl HFTheap {
    /// Creates a non-serialized heap for single-threaded high-frequency allocations
    pub fn create() -> Result<Self, String> {
        unsafe {
            let handle = HeapCreate(HEAP_GENERATE_EXCEPTIONS | HEAP_NO_SERIALIZE, 0, 0);
            if handle.is_invalid() || handle.0 == 0 {
                return Err("Failed to create heap".to_string());
            }
            Ok(HFTheap {
                handle,
                allocated_bytes: 0,
                allocation_count: 0,
            })
        }
    }

    /// Allocates memory from the custom heap without serialization overhead
    pub fn alloc(&mut self, size: usize) -> Result<*mut u8, String> {
        unsafe {
            let ptr = HeapAlloc(self.handle, HEAP_NO_SERIALIZE, size);
            if ptr.is_null() {
                return Err(format!("HeapAlloc failed: {}", windows::Win32::Foundation::GetLastError().0));
            }
            self.allocated_bytes += size;
            self.allocation_count += 1;
            Ok(ptr as *mut u8)
        }
    }

    /// Allocates zero-initialized memory
    pub fn alloc_zeroed(&mut self, size: usize) -> Result<*mut u8, String> {
        unsafe {
            let ptr = HeapAlloc(self.handle, HEAP_NO_SERIALIZE | HEAP_ZERO_MEMORY, size);
            if ptr.is_null() {
                return Err(format!("HeapAlloc failed: {}", windows::Win32::Foundation::GetLastError().0));
            }
            self.allocated_bytes += size;
            self.allocation_count += 1;
            Ok(ptr as *mut u8)
        }
    }

    /// Frees memory back to the custom heap
    pub fn free(&mut self, ptr: *mut u8, size: usize) -> Result<(), String> {
        unsafe {
            if HeapFree(self.handle, HEAP_NO_SERIALIZE, ptr as *mut c_void).is_err() {
                return Err(format!("HeapFree failed: {}", windows::Win32::Foundation::GetLastError().0));
            }
            self.allocated_bytes -= size;
            self.allocation_count -= 1;
            Ok(())
        }
    }

    pub fn allocated_bytes(&self) -> usize {
        self.allocated_bytes
    }

    pub fn allocation_count(&self) -> usize {
        self.allocation_count
    }
}

impl Drop for HFTheap {
    fn drop(&mut self) {
        unsafe {
            let _ = HeapDestroy(self.handle);
        }
    }
}

/// MemoryArena - Pre-allocated block pool for tick data
pub struct MemoryArena {
    blocks: Vec<*mut u8>,
    current_block: usize,
    current_offset: usize,
    block_size: usize,
}

unsafe impl Send for MemoryArena {}
unsafe impl Sync for MemoryArena {}

impl MemoryArena {
    pub fn new(initial_blocks: usize, block_size: usize) -> Result<Self, String> {
        let mut arena = MemoryArena {
            blocks: Vec::with_capacity(initial_blocks),
            current_block: 0,
            current_offset: 0,
            block_size,
        };

        for _ in 0..initial_blocks {
            let block = unsafe {
                let ptr = windows::Win32::System::Memory::VirtualAlloc(
                    None,
                    block_size,
                    windows::Win32::System::Memory::MEM_COMMIT | windows::Win32::System::Memory::MEM_RESERVE,
                    windows::Win32::System::Memory::PAGE_READWRITE,
                );
                if ptr.is_null() {
                    return Err("Failed to allocate arena block".to_string());
                }
                ptr as *mut u8
            };
            arena.blocks.push(block);
        }

        Ok(arena)
    }

    /// Fast bump allocation - no individual frees, reset entire arena at once
    pub fn alloc(&mut self, size: usize) -> Option<*mut u8> {
        if self.current_block >= self.blocks.len() {
            return None;
        }

        if self.current_offset + size > self.block_size {
            self.current_block += 1;
            self.current_offset = 0;
            if self.current_block >= self.blocks.len() {
                return None;
            }
        }

        let ptr = unsafe {
            self.blocks[self.current_block].add(self.current_offset)
        };
        self.current_offset += size;
        Some(ptr)
    }

    /// Reset arena for reuse without freeing memory
    pub fn reset(&mut self) {
        self.current_block = 0;
        self.current_offset = 0;
    }

    pub fn total_capacity(&self) -> usize {
        self.blocks.len() * self.block_size
    }
}

impl Drop for MemoryArena {
    fn drop(&mut self) {
        for &block in &self.blocks {
            unsafe {
                let _ = windows::Win32::System::Memory::VirtualFree(
                    block as *mut c_void,
                    0,
                    windows::Win32::System::Memory::MEM_RELEASE,
                );
            }
        }
    }
}

/// TickDataAllocator - Specialized allocator for market tick structures
pub struct TickDataAllocator<T> {
    arena: MemoryArena,
    _marker: std::marker::PhantomData<T>,
}

unsafe impl<T> Send for TickDataAllocator<T> {}
unsafe impl<T> Sync for TickDataAllocator<T> {}

impl<T> TickDataAllocator<T> {
    pub fn new(capacity: usize) -> Result<Self, String> 
    where
        T: Sized,
    {
        let block_size = ARENA_BLOCK_SIZE;
        let num_blocks = (capacity * std::mem::size_of::<T>() / block_size) + 1;
        let arena = MemoryArena::new(num_blocks.min(MAX_ARENA_COUNT), block_size)?;
        
        Ok(TickDataAllocator {
            arena,
            _marker: std::marker::PhantomData,
        })
    }

    pub fn alloc(&mut self, value: T) -> Option<*mut T> {
        self.arena
            .alloc(std::mem::size_of::<T>())
            .map(|ptr| ptr as *mut T)
            .map(|ptr| {
                unsafe { ptr.write(value) };
                ptr
            })
    }

    pub fn reset(&mut self) {
        self.arena.reset();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_custom_heap() {
        let mut heap = HFTheap::create().expect("Failed to create heap");
        let ptr = heap.alloc(1024).expect("Failed to allocate");
        assert!(!ptr.is_null());
        assert_eq!(heap.allocated_bytes(), 1024);
        heap.free(ptr, 1024).expect("Failed to free");
        assert_eq!(heap.allocated_bytes(), 0);
    }

    #[test]
    fn test_memory_arena() {
        let mut arena = MemoryArena::new(4, 64 * 1024).expect("Failed to create arena");
        let ptr1 = arena.alloc(256).expect("Failed to allocate");
        let ptr2 = arena.alloc(256).expect("Failed to allocate");
        assert!(!ptr1.is_null());
        assert!(!ptr2.is_null());
        assert_ne!(ptr1, ptr2);
        arena.reset();
    }
}
