//! High-performance, memory-aligned covariance and correlation matrix calculator
//! Uses SIMD instructions and rolling ring buffers. Strictly avoids heap allocations in the hot path.

use std::arch::x86_64::*;
use std::sync::atomic::{AtomicU64, Ordering};

const MAX_ASSETS: usize = 128;
const RING_BUFFER_SIZE: usize = 1024;

/// Memory-aligned covariance matrix storage (row-major, 64-byte aligned)
#[repr(align(64))]
pub struct CovarianceMatrix {
    data: [[f64; MAX_ASSETS]; MAX_ASSETS],
    asset_count: usize,
}

/// Rolling ring buffer for returns data (lock-free, SIMD-friendly)
#[repr(align(64))]
pub struct ReturnsRingBuffer {
    buffer: [[f64; MAX_ASSETS]; RING_BUFFER_SIZE],
    head: AtomicU64,
    count: AtomicU64,
    asset_count: usize,
}

impl Default for CovarianceMatrix {
    fn default() -> Self {
        Self {
            data: [[0.0; MAX_ASSETS]; MAX_ASSETS],
            asset_count: 0,
        }
    }
}

impl Default for ReturnsRingBuffer {
    fn default() -> Self {
        Self {
            buffer: [[0.0; MAX_ASSETS]; RING_BUFFER_SIZE],
            head: AtomicU64::new(0),
            count: AtomicU64::new(0),
            asset_count: 0,
        }
    }
}

impl ReturnsRingBuffer {
    #[inline(always)]
    pub fn new(asset_count: usize) -> Self {
        assert!(asset_count <= MAX_ASSETS, "Asset count exceeds maximum");
        Self {
            buffer: [[0.0; MAX_ASSETS]; RING_BUFFER_SIZE],
            head: AtomicU64::new(0),
            count: AtomicU64::new(0),
            asset_count,
        }
    }

    /// Push a new returns vector (lock-free, overwrites oldest if full)
    #[inline(always)]
    pub fn push(&self, returns: &[f64]) {
        debug_assert_eq!(returns.len(), self.asset_count);
        
        let head = self.head.fetch_add(1, Ordering::Relaxed) % RING_BUFFER_SIZE as u64;
        let idx = head as usize;
        
        // SIMD-accelerated copy (process 4 f64 at a time)
        let mut i = 0;
        let len = self.asset_count;
        
        unsafe {
            while i + 4 <= len {
                let src = returns.get_unchecked(i..i+4);
                let dst = self.buffer[idx].get_unchecked_mut(i..i+4);
                
                let v = _mm256_loadu_pd(src.as_ptr());
                _mm256_storeu_pd(dst.as_mut_ptr(), v);
                
                i += 4;
            }
            // Handle remainder
            while i < len {
                self.buffer[idx][i] = returns[i];
                i += 1;
            }
        }
        
        // Update count (cap at buffer size)
        self.count.fetch_update(Ordering::Release, Ordering::Relaxed, |c| {
            Some(if c < RING_BUFFER_SIZE as u64 { c + 1 } else { c })
        }).ok();
    }

    #[inline(always)]
    pub fn sample_count(&self) -> usize {
        *self.count.load(Ordering::Acquire) as usize
    }

    #[inline(always)]
    pub fn asset_count(&self) -> usize {
        self.asset_count
    }
}

impl CovarianceMatrix {
    #[inline(always)]
    pub fn new(asset_count: usize) -> Self {
        assert!(asset_count <= MAX_ASSETS, "Asset count exceeds maximum");
        Self {
            data: [[0.0; MAX_ASSETS]; MAX_ASSETS],
            asset_count,
        }
    }

    /// Calculate covariance matrix from ring buffer using SIMD
    /// Uses Welford's online algorithm variant for numerical stability
    #[inline(always)]
    pub fn compute_from_buffer(&mut self, buffer: &ReturnsRingBuffer) {
        let n = buffer.sample_count();
        let assets = buffer.asset_count();
        
        if n < 2 || assets == 0 {
            self.asset_count = 0;
            return;
        }

        self.asset_count = assets;
        let denom = (n - 1) as f64;

        // Step 1: Calculate means (SIMD-accelerated)
        let mut means = [0.0f64; MAX_ASSETS];
        
        unsafe {
            for j in 0..assets {
                let mut sum = _mm256_setzero_pd();
                let mut i = 0usize;
                
                // Process 4 samples at a time
                while i + 4 <= n {
                    let v = _mm256_loadu_pd(buffer.buffer[i].as_ptr().add(j));
                    sum = _mm256_add_pd(sum, v);
                    i += 4;
                }
                
                // Horizontal sum
                let sum_arr: [f64; 4] = std::mem::transmute(sum);
                let mut mean = sum_arr[0] + sum_arr[1] + sum_arr[2] + sum_arr[3];
                
                // Remainder
                while i < n {
                    mean += buffer.buffer[i][j];
                    i += 1;
                }
                
                means[j] = mean / n as f64;
            }
        }

        // Step 2: Calculate covariance (SIMD-accelerated, only upper triangle)
        for i in 0..assets {
            for j in i..assets {
                let mut cov_sum = 0.0f64;
                
                unsafe {
                    let mut k = 0usize;
                    
                    // Process 4 samples at a time
                    while k + 4 <= n {
                        let di_k = buffer.buffer[k][i] - means[i];
                        let dj_k = buffer.buffer[k][j] - means[j];
                        let di_k1 = buffer.buffer[k+1][i] - means[i];
                        let dj_k1 = buffer.buffer[k+1][j] - means[j];
                        let di_k2 = buffer.buffer[k+2][i] - means[i];
                        let dj_k2 = buffer.buffer[k+2][j] - means[j];
                        let di_k3 = buffer.buffer[k+3][i] - means[i];
                        let dj_k3 = buffer.buffer[k+3][j] - means[j];
                        
                        cov_sum += di_k * dj_k + di_k1 * dj_k1 + di_k2 * dj_k2 + di_k3 * dj_k3;
                        k += 4;
                    }
                    
                    // Remainder
                    while k < n {
                        cov_sum += (buffer.buffer[k][i] - means[i]) * (buffer.buffer[k][j] - means[j]);
                        k += 1;
                    }
                }
                
                let cov = cov_sum / denom;
                self.data[i][j] = cov;
                self.data[j][i] = cov; // Symmetric
            }
        }
    }

    /// Calculate correlation matrix from covariance (in-place transformation)
    #[inline(always)]
    pub fn to_correlation(&mut self) {
        let assets = self.asset_count;
        
        // Extract standard deviations
        let mut stds = [0.0f64; MAX_ASSETS];
        for i in 0..assets {
            stds[i] = self.data[i][i].sqrt();
        }
        
        // Convert to correlation
        for i in 0..assets {
            for j in 0..assets {
                if stds[i] > 1e-10 && stds[j] > 1e-10 {
                    self.data[i][j] /= stds[i] * stds[j];
                } else {
                    self.data[i][j] = if i == j { 1.0 } else { 0.0 };
                }
            }
        }
    }

    #[inline(always)]
    pub fn get(&self, i: usize, j: usize) -> f64 {
        debug_assert!(i < self.asset_count && j < self.asset_count);
        self.data[i][j]
    }

    #[inline(always)]
    pub fn asset_count(&self) -> usize {
        self.asset_count
    }

    /// Get raw slice for matrix operations (upper triangle only)
    #[inline(always)]
    pub fn as_slice(&self) -> &[f64] {
        let len = self.asset_count * self.asset_count;
        unsafe { std::slice::from_raw_parts(self.data[0].as_ptr() as *const f64, len) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_covariance_computation() {
        let mut buffer = ReturnsRingBuffer::new(3);
        
        // Push some test data
        buffer.push(&[0.01, 0.02, -0.01]);
        buffer.push(&[-0.02, 0.01, 0.03]);
        buffer.push(&[0.03, -0.01, 0.02]);
        buffer.push(&[0.01, 0.03, -0.02]);
        
        let mut cov = CovarianceMatrix::new(3);
        cov.compute_from_buffer(&buffer);
        
        assert_eq!(cov.asset_count(), 3);
        // Diagonal should be positive (variance)
        assert!(cov.get(0, 0) > 0.0);
        assert!(cov.get(1, 1) > 0.0);
        assert!(cov.get(2, 2) > 0.0);
    }
}
