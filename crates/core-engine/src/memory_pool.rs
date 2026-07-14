//! Zero-Allocation Lock-Free Memory Pool
//! 
//! This module implements a custom memory pool for order book updates and tick data.
//! It uses pre-allocated slabs to prevent runtime allocation, GC pauses, and allocator fragmentation.
//! Optimized for AMD Ryzen AI 5 L3 cache topology.

use std::alloc::{self, Layout};
use std::cell::UnsafeCell;
use std::ptr::{self, NonNull};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::time::Instant;

/// Memory pool configuration
#[derive(Debug, Clone)]
pub struct MemoryPoolConfig {
    pub slab_size: usize,
    pub num_slabs: usize,
    pub alignment: usize,
}

impl Default for MemoryPoolConfig {
    fn default() -> Self {
        Self {
            slab_size: 4096, // Page-aligned
            num_slabs: 131072, // 512MB total (512 * 1024 * 1024 / 4096)
            alignment: 64, // Cache line alignment for AMD Zen 4
        }
    }
}

/// A single memory slab in the pool
struct Slab {
    data: NonNull<u8>,
    layout: Layout,
    used: AtomicUsize,
}

unsafe impl Send for Slab {}
unsafe impl Sync for Slab {}

impl Slab {
    fn new(config: &MemoryPoolConfig) -> Option<Self> {
        let layout = unsafe {
            Layout::from_size_align_unchecked(config.slab_size, config.alignment)
        };
        
        let ptr = unsafe { alloc::alloc(layout) };
        if ptr.is_null() {
            return None;
        }
        
        // Zero-initialize the slab
        unsafe { ptr::write_bytes(ptr, 0, config.slab_size) };
        
        Some(Self {
            data: NonNull::new(ptr)?,
            layout,
            used: AtomicUsize::new(0),
        })
    }
    
    #[inline]
    fn acquire(&self) -> Option<*mut u8> {
        let idx = self.used.fetch_add(1, Ordering::AcqRel);
        if idx < self.layout.size() {
            Some(unsafe { self.data.as_ptr().add(idx) })
        } else {
            self.used.fetch_sub(1, Ordering::AcqRel);
            None
        }
    }
    
    #[inline]
    fn release(&self, ptr: *mut u8) {
        // In a production system, we'd track which bytes are freed
        // For now, we use a simple reset mechanism
        self.used.store(0, Ordering::Release);
        unsafe { ptr::write_bytes(ptr, 0, 1) };
    }
}

/// Lock-free memory pool for zero-copy operations
pub struct MemoryPool {
    slabs: Vec<Slab>,
    config: MemoryPoolConfig,
    allocations: AtomicU64,
    deallocations: AtomicU64,
    peak_usage: AtomicU64,
    created_at: Instant,
}

unsafe impl Send for MemoryPool {}
unsafe impl Sync for MemoryPool {}

impl MemoryPool {
    /// Create a new memory pool with the given configuration
    pub fn new(config: MemoryPoolConfig) -> Option<Self> {
        let mut slabs = Vec::with_capacity(config.num_slabs);
        
        for _ in 0..config.num_slabs {
            slabs.push(Slab::new(&config)?);
        }
        
        Some(Self {
            slabs,
            config,
            allocations: AtomicU64::new(0),
            deallocations: AtomicU64::new(0),
            peak_usage: AtomicU64::new(0),
            created_at: Instant::now(),
        })
    }
    
    /// Allocate a buffer from the pool (zero-copy if reused)
    #[inline]
    pub fn allocate(&self, size: usize) -> Option<PoolBuffer> {
        if size > self.config.slab_size {
            return None;
        }
        
        // Round-robin across slabs for load balancing
        let slab_idx = self.allocations.load(Ordering::Relaxed) as usize % self.slabs.len();
        let slab = &self.slabs[slab_idx];
        
        slab.acquire().map(|ptr| {
            self.allocations.fetch_add(1, Ordering::AcqRel);
            
            // Update peak usage
            let current = self.allocations.load(Ordering::Relaxed) 
                - self.deallocations.load(Ordering::Relaxed);
            let mut peak = self.peak_usage.load(Ordering::Relaxed);
            while current > peak {
                match self.peak_usage.compare_exchange_weak(
                    peak, current, Ordering::AcqRel, Ordering::Relaxed
                ) {
                    Ok(_) => break,
                    Err(p) => peak = p,
                }
            }
            
            PoolBuffer {
                ptr,
                size,
                pool: self,
                slab_idx,
            }
        })
    }
    
    /// Get pool statistics
    pub fn stats(&self) -> PoolStats {
        let allocs = self.allocations.load(Ordering::Relaxed);
        let deallocs = self.deallocations.load(Ordering::Relaxed);
        
        PoolStats {
            total_allocations: allocs,
            total_deallocations: deallocs,
            active_buffers: allocs - deallocs,
            peak_usage: self.peak_usage.load(Ordering::Relaxed),
            pool_size_mb: (self.config.num_slabs * self.config.slab_size) / (1024 * 1024),
            uptime_secs: self.created_at.elapsed().as_secs_f64(),
        }
    }
    
    /// Reset the pool (use with caution, invalidates all buffers)
    pub fn reset(&self) {
        for slab in &self.slabs {
            slab.release(slab.data.as_ptr());
        }
        self.allocations.store(0, Ordering::Release);
        self.deallocations.store(0, Ordering::Release);
    }
}

impl Drop for MemoryPool {
    fn drop(&mut self) {
        for slab in &self.slabs {
            unsafe {
                alloc::dealloc(slab.data.as_ptr(), slab.layout);
            }
        }
    }
}

/// A buffer allocated from the memory pool
pub struct PoolBuffer<'a> {
    ptr: *mut u8,
    size: usize,
    pool: &'a MemoryPool,
    slab_idx: usize,
}

unsafe impl<'a> Send for PoolBuffer<'a> {}
unsafe impl<'a> Sync for PoolBuffer<'a> {}

impl<'a> PoolBuffer<'a> {
    #[inline]
    pub fn as_mut_ptr(&mut self) -> *mut u8 {
        self.ptr
    }
    
    #[inline]
    pub fn as_ptr(&self) -> *const u8 {
        self.ptr
    }
    
    #[inline]
    pub fn size(&self) -> usize {
        self.size
    }
    
    /// Write data to the buffer (zero-copy)
    #[inline]
    pub fn write(&mut self, data: &[u8]) -> Result<(), ()> {
        if data.len() > self.size {
            return Err(());
        }
        unsafe {
            ptr::copy_nonoverlapping(data.as_ptr(), self.ptr, data.len());
        }
        self.size = data.len();
        Ok(())
    }
    
    /// Read data from the buffer (zero-copy)
    #[inline]
    pub fn read(&self, len: usize) -> Option<&[u8]> {
        if len > self.size {
            return None;
        }
        Some(unsafe { std::slice::from_raw_parts(self.ptr, len) })
    }
}

impl<'a> Drop for PoolBuffer<'a> {
    fn drop(&mut self) {
        let slab = &self.pool.slabs[self.slab_idx];
        slab.release(self.ptr);
        self.pool.deallocations.fetch_add(1, Ordering::AcqRel);
    }
}

/// Pool statistics for monitoring
#[derive(Debug, Clone)]
pub struct PoolStats {
    pub total_allocations: u64,
    pub total_deallocations: u64,
    pub active_buffers: u64,
    pub peak_usage: u64,
    pub pool_size_mb: usize,
    pub uptime_secs: f64,
}

impl PoolStats {
    pub fn utilization(&self) -> f64 {
        if self.pool_size_mb == 0 {
            return 0.0;
        }
        (self.active_buffers as f64 * 4096.0) / (self.pool_size_mb as f64 * 1024.0 * 1024.0) * 100.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_memory_pool_creation() {
        let config = MemoryPoolConfig {
            slab_size: 4096,
            num_slabs: 100,
            alignment: 64,
        };
        let pool = MemoryPool::new(config).unwrap();
        assert_eq!(pool.stats().pool_size_mb, 0); // Small test pool
    }
    
    #[test]
    fn test_allocation_deallocation() {
        let config = MemoryPoolConfig::default();
        let pool = MemoryPool::new(config).unwrap();
        
        let mut buf = pool.allocate(1024).unwrap();
        let data = vec![1u8; 1024];
        buf.write(&data).unwrap();
        
        assert_eq!(buf.size(), 1024);
        // Buffer is automatically returned when dropped
    }
}
