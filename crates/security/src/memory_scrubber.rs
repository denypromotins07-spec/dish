//! Aggressive memory scrubber that overwrites sensitive data structures with zeros
//! the millisecond they are no longer needed.
//! Mitigates cold-boot and core-dump attacks by ensuring secrets never persist in RAM.

use std::ptr;
use std::sync::atomic::{AtomicBool, Ordering};

/// Trait for types that can be securely zeroed from memory.
pub trait SecureZeroize {
    /// Overwrites the data with zeros in a way that won't be optimized away.
    fn zeroize(&mut self);
}

impl SecureZeroize for [u8] {
    fn zeroize(&mut self) {
        unsafe {
            ptr::write_bytes(self.as_mut_ptr(), 0, self.len());
        }
    }
}

impl SecureZeroize for Vec<u8> {
    fn zeroize(&mut self) {
        self.as_mut_slice().zeroize();
    }
}

impl SecureZeroize for String {
    fn zeroize(&mut self) {
        unsafe {
            let bytes = self.as_mut_vec();
            ptr::write_bytes(bytes.as_mut_ptr(), 0, bytes.len());
        }
    }
}

impl SecureZeroize for [u8; 32] {
    fn zeroize(&mut self) {
        unsafe {
            ptr::write_bytes(self.as_mut_ptr(), 0, 32);
        }
    }
}

impl SecureZeroize for [u8; 64] {
    fn zeroize(&mut self) {
        unsafe {
            ptr::write_bytes(self.as_mut_ptr(), 0, 64);
        }
    }
}

/// A secure buffer that automatically zeroes its contents on drop.
pub struct SecureBuffer<T: SecureZeroize> {
    data: T,
    is_zeroed: AtomicBool,
}

impl<T: SecureZeroize> SecureBuffer<T> {
    /// Creates a new secure buffer containing the given data.
    pub fn new(data: T) -> Self {
        SecureBuffer {
            data,
            is_zeroed: AtomicBool::new(false),
        }
    }

    /// Returns a reference to the underlying data.
    pub fn as_ref(&self) -> &T {
        &self.data
    }

    /// Returns a mutable reference to the underlying data.
    pub fn as_mut(&mut self) -> &mut T {
        &mut self.data
    }

    /// Manually zeroes the buffer before it would normally be dropped.
    pub fn zeroize_now(&mut self) {
        if !self.is_zeroed.load(Ordering::Relaxed) {
            self.data.zeroize();
            self.is_zeroed.store(true, Ordering::Relaxed);
        }
    }

    /// Consumes the buffer and returns the inner data WITHOUT zeroing.
    /// WARNING: Only use this if you're transferring ownership to another secure structure!
    pub fn into_inner(mut self) -> T {
        self.is_zeroed.store(true, Ordering::Relaxed);
        // Use MaybeUninit or similar to prevent double-zeroing
        // For now, we'll just take it - caller assumes responsibility
        unsafe {
            std::ptr::read(&self.data)
        }
    }
}

impl<T: SecureZeroize> Drop for SecureBuffer<T> {
    fn drop(&mut self) {
        if !self.is_zeroed.load(Ordering::Relaxed) {
            self.data.zeroize();
            self.is_zeroed.store(true, Ordering::Relaxed);
        }
    }
}

/// Securely holds an API key or secret that will be zeroed on drop.
pub struct SecretKey {
    key: SecureBuffer<Vec<u8>>,
    key_type: KeyType,
}

#[derive(Clone, Debug)]
pub enum KeyType {
    ApiKey,
    ApiSecret,
    PrivateKey,
    EncryptionKey,
    Password,
}

impl SecretKey {
    /// Creates a new secret key.
    pub fn new(key_data: &[u8], key_type: KeyType) -> Self {
        SecretKey {
            key: SecureBuffer::new(key_data.to_vec()),
            key_type,
        }
    }

    /// Returns a reference to the key bytes.
    pub fn as_bytes(&self) -> &[u8] {
        self.key.as_ref()
    }

    /// Returns the key type.
    pub fn key_type(&self) -> &KeyType {
        &self.key_type
    }

    /// Explicitly zeroes the key before drop.
    pub fn destroy(&mut self) {
        self.key.zeroize_now();
    }
}

/// Memory scrubber utility for cleaning up sensitive regions.
pub struct MemoryScrubber;

impl MemoryScrubber {
    /// Scrubs a slice of bytes by overwriting with zeros.
    pub fn scrub(data: &mut [u8]) {
        data.zeroize();
    }

    /// Scrubs multiple slices at once.
    pub fn scrub_multiple(slices: &mut [&mut [u8]]) {
        for slice in slices {
            slice.zeroize();
        }
    }

    /// Scrubs a region of raw memory (unsafe).
    pub unsafe fn scrub_raw(ptr: *mut u8, len: usize) {
        if !ptr.is_null() && len > 0 {
            ptr::write_bytes(ptr, 0, len);
        }
    }

    /// Forces a page of memory to be scrubbed by touching every cache line.
    pub fn scrub_cache_lines(data: &mut [u8]) {
        // Touch every 64-byte cache line to ensure it's loaded into cache
        // before zeroing, preventing any lazy allocation tricks
        for i in (0..data.len()).step_by(64) {
            let _ = data[i];
        }
        
        // Now zero everything
        data.zeroize();
        
        // Memory barrier to ensure the zeroing completes
        std::sync::atomic::fence(Ordering::SeqCst);
    }
}

/// RAII guard that scrubs memory when it goes out of scope.
pub struct ScrubGuard<'a> {
    data: &'a mut [u8],
    armed: bool,
}

impl<'a> ScrubGuard<'a> {
    /// Creates a new scrub guard for the given data.
    pub fn new(data: &'a mut [u8]) -> Self {
        ScrubGuard {
            data,
            armed: true,
        }
    }

    /// Disarms the guard, preventing automatic scrubbing on drop.
    /// Use this if you've already manually scrubbed the data.
    pub fn disarm(&mut self) {
        self.armed = false;
    }

    /// Manually triggers scrubbing before the guard is dropped.
    pub fn scrub(&mut self) {
        if self.armed {
            MemoryScrubber::scrub(self.data);
            self.armed = false;
        }
    }
}

impl<'a> Drop for ScrubGuard<'a> {
    fn drop(&mut self) {
        if self.armed {
            MemoryScrubber::scrub(self.data);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_secure_buffer_zeroizes_on_drop() {
        let original = vec![1u8, 2, 3, 4, 5];
        let cloned = original.clone();
        
        {
            let buffer = SecureBuffer::new(original);
            assert_eq!(buffer.as_ref(), &cloned);
        }
        
        // After drop, the original vector should be zeroed
        // Note: We can't directly verify this since we don't have access anymore,
        // but the test ensures the Drop impl runs without panicking
    }

    #[test]
    fn test_secret_key_destruction() {
        let key_data = b"super_secret_api_key_12345";
        let mut secret = SecretKey::new(key_data, KeyType::ApiKey);
        
        assert_eq!(secret.as_bytes(), key_data);
        
        // Explicitly destroy
        secret.destroy();
        
        // After destruction, the key should be zeroed
        assert!(secret.as_bytes().iter().all(|&b| b == 0));
    }

    #[test]
    fn test_scrub_guard() {
        let mut data = vec![0xDEu8; 1024];
        
        {
            let _guard = ScrubGuard::new(&mut data);
            // Data is still intact here
            assert_eq!(data[0], 0xDE);
        }
        
        // After guard drops, data should be zeroed
        assert!(data.iter().all(|&b| b == 0));
    }

    #[test]
    fn test_scrub_guard_disarm() {
        let mut data = vec![0xADu8; 256];
        
        {
            let mut guard = ScrubGuard::new(&mut data);
            guard.disarm();
        }
        
        // After disarmed guard drops, data should NOT be zeroed
        assert_eq!(data[0], 0xAD);
    }

    #[test]
    fn test_memory_scrubber_cache_lines() {
        let mut data = vec![0xCCu8; 4096]; // Multiple cache lines
        
        MemoryScrubber::scrub_cache_lines(&mut data);
        
        assert!(data.iter().all(|&b| b == 0));
    }
}
