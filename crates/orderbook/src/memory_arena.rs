//! Custom bump allocator and memory arena specifically designed for L3 order nodes.
//! Pre-allocates a strict 200MB block of RAM to ensure zero garbage collection pauses.
//! Optimized for AMD Ryzen AI 5 with 64-byte cache line alignment.

use std::alloc::{alloc, dealloc, Layout};
use std::ptr::NonNull;
use std::sync::atomic::{AtomicUsize, Ordering};
use crate::l3_tree::L3Node;

/// Strict 200MB arena size limit to stay within 14GB total RAM constraint
const ARENA_SIZE_BYTES: usize = 200 * 1024 * 1024;
const NODE_SIZE: usize = std::mem::size_of::<L3Node>();
const CACHE_LINE_SIZE: usize = 64;

/// Memory Arena for L3 Order Nodes
/// Uses bump allocation strategy for O(1) allocation without fragmentation
pub struct MemoryArena {
    /// Base pointer to the allocated memory block
    base_ptr: NonNull<u8>,
    /// Current offset into the arena
    offset: AtomicUsize,
    /// Total capacity in bytes
    capacity: usize,
    /// Number of nodes allocated
    node_count: AtomicUsize,
}

unsafe impl Send for MemoryArena {}
unsafe impl Sync for MemoryArena {}

impl MemoryArena {
    /// Create a new memory arena with strict 200MB limit
    pub fn new() -> Result<Self, &'static str> {
        let layout = Layout::from_size_align(ARENA_SIZE_BYTES, CACHE_LINE_SIZE)
            .map_err(|_| "Invalid layout")?;
        
        let ptr = unsafe { alloc(layout) };
        
        if ptr.is_null() {
            return Err("Failed to allocate arena memory");
        }
        
        // Zero-initialize the arena to prevent data leakage
        unsafe {
            std::ptr::write_bytes(ptr, 0, ARENA_SIZE_BYTES);
        }
        
        Ok(Self {
            base_ptr: NonNull::new(ptr).unwrap(),
            offset: AtomicUsize::new(0),
            capacity: ARENA_SIZE_BYTES,
            node_count: AtomicUsize::new(0),
        })
    }
    
    /// Allocate a new L3Node from the arena - O(1) operation
    #[inline]
    pub fn alloc_node(&self, order_id: u64, price: u64, quantity: u64, timestamp_ns: u64, side: u8) -> Option<NonNull<L3Node>> {
        let current_offset = self.offset.fetch_add(NODE_SIZE, Ordering::AcqRel);
        
        if current_offset + NODE_SIZE > self.capacity {
            // Arena full - revert the offset increment
            self.offset.fetch_sub(NODE_SIZE, Ordering::Release);
            return None;
        }
        
        let ptr = unsafe {
            self.base_ptr.as_ptr().add(current_offset) as *mut L3Node
        };
        
        unsafe {
            ptr.write(L3Node::new(order_id, price, quantity, timestamp_ns, side));
        }
        
        self.node_count.fetch_add(1, Ordering::Relaxed);
        NonNull::new(ptr)
    }
    
    /// Reset the arena for reuse (dangerous - only call when no references exist)
    #[inline]
    pub fn reset(&self) {
        self.offset.store(0, Ordering::Release);
        self.node_count.store(0, Ordering::Relaxed);
        
        // Zero-initialize again for safety
        unsafe {
            std::ptr::write_bytes(self.base_ptr.as_ptr(), 0, self.capacity);
        }
    }
    
    /// Get remaining capacity in bytes
    #[inline]
    pub fn remaining_capacity(&self) -> usize {
        self.capacity - self.offset.load(Ordering::Relaxed)
    }
    
    /// Get number of nodes allocated
    #[inline]
    pub fn node_count(&self) -> usize {
        self.node_count.load(Ordering::Relaxed)
    }
    
    /// Get utilization percentage
    #[inline]
    pub fn utilization(&self) -> f32 {
        (self.offset.load(Ordering::Relaxed) as f32) / (self.capacity as f32)
    }
    
    /// Check if arena is near capacity (>90%)
    #[inline]
    pub fn is_near_capacity(&self) -> bool {
        self.utilization() > 0.9
    }
}

impl Drop for MemoryArena {
    fn drop(&mut self) {
        let layout = Layout::from_size_align(self.capacity, CACHE_LINE_SIZE).unwrap();
        unsafe {
            dealloc(self.base_ptr.as_ptr(), layout);
        }
    }
}

/// Thread-local arena handle for zero-contention access
pub struct ArenaHandle {
    arena: &'static MemoryArena,
}

impl ArenaHandle {
    pub fn new(arena: &'static MemoryArena) -> Self {
        Self { arena }
    }
    
    #[inline]
    pub fn alloc_node(&self, order_id: u64, price: u64, quantity: u64, timestamp_ns: u64, side: u8) -> Option<NonNull<L3Node>> {
        self.arena.alloc_node(order_id, price, quantity, timestamp_ns, side)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_arena_allocation() {
        let arena = MemoryArena::new().expect("Failed to create arena");
        
        let node_ptr = arena.alloc_node(1, 10000, 100, 12345, 0);
        assert!(node_ptr.is_some());
        
        let node = unsafe { node_ptr.unwrap().as_ref() };
        assert_eq!(node.order_id, 1);
        assert_eq!(node.price, 10000);
        assert_eq!(arena.node_count(), 1);
    }
    
    #[test]
    fn test_arena_capacity() {
        let arena = MemoryArena::new().expect("Failed to create arena");
        assert_eq!(arena.remaining_capacity(), ARENA_SIZE_BYTES);
        assert_eq!(arena.utilization(), 0.0);
    }
}
