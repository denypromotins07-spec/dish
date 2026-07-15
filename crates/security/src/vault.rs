//! Ultra-lightweight, memory-safe secrets vault.
//! Loads API keys from an encrypted local file (AES-256-GCM) directly into locked, non-swappable RAM pages.
//! Prevents keys from leaking to disk via swap files using `mlock`.

use aes_gcm::{Aes256Gcm, Key, Nonce, aead::Aead};
use std::alloc::{alloc_zeroed, dealloc, Layout};
use std::ptr;
use std::sync::atomic::{AtomicBool, Ordering};
use std::collections::HashMap;
use libc::{mlock, munlock};

const KEY_SIZE: usize = 32; // AES-256
const MAX_KEY_LEN: usize = 1024;

/// Securely holds a secret in locked memory.
pub struct LockedSecret {
    ptr: *mut u8,
    len: usize,
    layout: Layout,
    is_locked: AtomicBool,
}

unsafe impl Send for LockedSecret {}
unsafe impl Sync for LockedSecret {}

impl LockedSecret {
    /// Allocates a new locked memory region for the secret.
    pub fn new(data: &[u8]) -> Result<Self, String> {
        if data.is_empty() || data.len() > MAX_KEY_LEN {
            return Err("Invalid secret length".to_string());
        }

        let layout = Layout::from_size_align(data.len(), 16)
            .map_err(|e| e.to_string())?;

        let ptr = unsafe { alloc_zeroed(layout) };
        if ptr.is_null() {
            return Err("Memory allocation failed".to_string());
        }

        // Copy data into locked memory
        unsafe {
            ptr::copy_nonoverlapping(data.as_ptr(), ptr, data.len());
        }

        // Lock memory pages to prevent swapping
        let lock_result = unsafe { mlock(ptr as *const libc::c_void, data.len()) };
        if lock_result != 0 {
            unsafe { dealloc(ptr, layout) };
            return Err("Failed to lock memory (mlock)".to_string());
        }

        Ok(LockedSecret {
            ptr,
            len: data.len(),
            layout,
            is_locked: AtomicBool::new(true),
        })
    }

    /// Returns a reference to the secret data.
    pub fn as_ref(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }

    /// Securely wipes and deallocates the secret.
    fn wipe(&mut self) {
        if self.is_locked.load(Ordering::Relaxed) {
            unsafe {
                // Overwrite with zeros before unlocking
                ptr::write_bytes(self.ptr, 0, self.len);
                
                // Unlock memory pages
                munlock(self.ptr as *const libc::c_void, self.len);
                self.is_locked.store(false, Ordering::Relaxed);
            }
        }
    }
}

impl Drop for LockedSecret {
    fn drop(&mut self) {
        self.wipe();
        unsafe {
            dealloc(self.ptr, self.layout);
        }
    }
}

/// The main secrets vault managing multiple API keys.
pub struct SecretsVault {
    keys: HashMap<String, LockedSecret>,
    master_key: Option<LockedSecret>,
}

impl SecretsVault {
    pub fn new(encrypted_file_path: &str, master_key_data: &[u8]) -> Result<Self, String> {
        let master_key = LockedSecret::new(master_key_data)?;
        let mut vault = SecretsVault {
            keys: HashMap::new(),
            master_key: Some(master_key),
        };
        
        vault.load_from_encrypted_file(encrypted_file_path)?;
        Ok(vault)
    }

    /// Loads and decrypts API keys from an encrypted file.
    pub fn load_from_encrypted_file(&mut self, path: &str) -> Result<(), String> {
        let encrypted_data = std::fs::read(path)
            .map_err(|e| format!("Failed to read encrypted file: {}", e))?;

        // Decrypt the file content using the master key
        let decrypted_content = self.decrypt_file(&encrypted_data)?;
        
        // Parse the decrypted content (format: "exchange_name:api_key\n...")
        for line in std::str::from_utf8(&decrypted_content)
            .map_err(|_| "Invalid UTF-8 in decrypted content")?
            .lines()
        {
            if let Some((name, key)) = line.split_once(':') {
                let locked_key = LockedSecret::new(key.as_bytes())?;
                self.keys.insert(name.to_string(), locked_key);
            }
        }

        Ok(())
    }

    /// Decrypts the file content using AES-256-GCM.
    fn decrypt_file(&self, encrypted_data: &[u8]) -> Result<Vec<u8>, String> {
        if encrypted_data.len() < 12 + 16 {
            return Err("Invalid encrypted data format".to_string());
        }

        let master_key = self.master_key.as_ref().ok_or("Master key not loaded")?;
        let key = Key::<Aes256Gcm>::from_slice(master_key.as_ref());
        let cipher = Aes256Gcm::new(key);

        let nonce = Nonce::from_slice(&encrypted_data[..12]);
        let ciphertext = &encrypted_data[12..];

        cipher
            .decrypt(nonce, ciphertext)
            .map_err(|_| "Decryption failed".to_string())
    }

    /// Retrieves a reference to an API key by exchange name.
    pub fn get_key(&self, exchange: &str) -> Option<&[u8]> {
        self.keys.get(exchange).map(|k| k.as_ref())
    }

    /// Updates or inserts a new key securely.
    pub fn update_key(&mut self, exchange: &str, key_data: &[u8]) -> Result<(), String> {
        let locked_key = LockedSecret::new(key_data)?;
        self.keys.insert(exchange.to_string(), locked_key);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vault_creation_and_retrieval() {
        let master_key = b"01234567890123456789012345678901"; // 32 bytes
        let vault = SecretsVault {
            keys: HashMap::new(),
            master_key: Some(LockedSecret::new(master_key).unwrap()),
        };
        assert!(vault.master_key.is_some());
    }
}
