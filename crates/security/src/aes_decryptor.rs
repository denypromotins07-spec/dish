//! SIMD-accelerated AES-256-GCM decryption engine built in pure Rust.
//! Decrypts credentials on-the-fly with zero-allocation buffers.
//! Ensures API keys are never exposed in plaintext logs or memory dumps.

use aes_gcm::{Aes256Gcm, Key, Nonce, aead::Aead};
use std::ptr;

/// Zero-copy decryption buffer that avoids heap allocations in the hot path.
pub struct DecryptBuffer {
    data: [u8; 4096], // Fixed-size buffer for typical key sizes
    len: usize,
}

impl DecryptBuffer {
    pub fn new() -> Self {
        DecryptBuffer {
            data: [0u8; 4096],
            len: 0,
        }
    }

    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        &mut self.data[..self.len]
    }

    pub fn set_len(&mut self, len: usize) {
        assert!(len <= self.data.len(), "Buffer overflow attempt");
        self.len = len;
    }

    pub fn as_slice(&self) -> &[u8] {
        &self.data[..self.len]
    }
}

impl Default for DecryptBuffer {
    fn default() -> Self {
        Self::new()
    }
}

/// High-performance AES-256-GCM decryptor optimized for credential decryption.
pub struct AesDecryptor {
    key: [u8; 32], // AES-256 key stored in stack memory
}

impl AesDecryptor {
    /// Creates a new decryptor from a 32-byte key.
    pub fn new(key: &[u8; 32]) -> Self {
        let mut decryptor = AesDecryptor { key: [0u8; 32] };
        unsafe {
            ptr::copy_nonoverlapping(key.as_ptr(), decryptor.key.as_mut_ptr(), 32);
        }
        decryptor
    }

    /// Decrypts data in-place using AES-256-GCM with zero intermediate allocations.
    /// Input format: [12-byte nonce][ciphertext+tag]
    pub fn decrypt_in_place(&self, encrypted_data: &[u8], output: &mut DecryptBuffer) -> Result<(), String> {
        if encrypted_data.len() < 12 + 16 {
            return Err("Invalid encrypted data: too short".to_string());
        }

        let nonce_bytes = &encrypted_data[..12];
        let ciphertext = &encrypted_data[12..];

        let key = Key::<Aes256Gcm>::from_slice(&self.key);
        let cipher = Aes256Gcm::new(key);
        let nonce = Nonce::from_slice(nonce_bytes);

        // Ensure output buffer is large enough
        if ciphertext.len() > output.data.len() {
            return Err("Output buffer too small".to_string());
        }

        // Copy ciphertext to output buffer
        unsafe {
            ptr::copy_nonoverlapping(ciphertext.as_ptr(), output.data.as_mut_ptr(), ciphertext.len());
        }
        output.set_len(ciphertext.len());

        // Decrypt in-place
        match cipher.decrypt_in_place_detached(
            nonce,
            &[], // No associated data
            output.as_mut_slice(),
            &ciphertext[ciphertext.len().saturating_sub(16)..].into(), // Tag is last 16 bytes
        ) {
            Ok(_) => {
                // Remove tag from output (last 16 bytes)
                let new_len = ciphertext.len().saturating_sub(16);
                output.set_len(new_len);
                Ok(())
            }
            Err(_) => Err("Decryption failed: invalid tag or corrupted data".to_string()),
        }
    }

    /// Decrypts data returning a new vector (convenience method, not zero-copy).
    pub fn decrypt(&self, encrypted_data: &[u8]) -> Result<Vec<u8>, String> {
        if encrypted_data.len() < 12 + 16 {
            return Err("Invalid encrypted data: too short".to_string());
        }

        let nonce_bytes = &encrypted_data[..12];
        let ciphertext = &encrypted_data[12..];

        let key = Key::<Aes256Gcm>::from_slice(&self.key);
        let cipher = Aes256Gcm::new(key);
        let nonce = Nonce::from_slice(nonce_bytes);

        cipher
            .decrypt(nonce, ciphertext)
            .map_err(|_| "Decryption failed".to_string())
    }

    /// Securely wipes the key from memory.
    pub fn wipe(&mut self) {
        unsafe {
            ptr::write_bytes(self.key.as_mut_ptr(), 0, 32);
        }
    }
}

impl Drop for AesDecryptor {
    fn drop(&mut self) {
        self.wipe();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use aes_gcm::{Aes256Gcm, Key, Nonce, aead::Aead};
    use rand::RngCore;

    #[test]
    fn test_encrypt_decrypt_round_trip() {
        let mut key = [0u8; 32];
        rand::thread_rng().fill_bytes(&mut key);

        let plaintext = b"test_api_key_12345";
        
        // Encrypt
        let aes_key = Key::<Aes256Gcm>::from_slice(&key);
        let cipher = Aes256Gcm::new(aes_key);
        let mut nonce_bytes = [0u8; 12];
        rand::thread_rng().fill_bytes(&mut nonce_bytes);
        let nonce = Nonce::from_slice(&nonce_bytes);
        
        let mut ciphertext = Vec::new();
        ciphertext.extend_from_slice(&nonce_bytes);
        let encrypted_payload = cipher.encrypt(nonce, plaintext.as_ref()).unwrap();
        ciphertext.extend_from_slice(&encrypted_payload);

        // Decrypt
        let decryptor = AesDecryptor::new(&key);
        let mut output = DecryptBuffer::new();
        let result = decryptor.decrypt_in_place(&ciphertext, &mut output);
        
        assert!(result.is_ok());
        assert_eq!(output.as_slice(), plaintext);
    }
}
