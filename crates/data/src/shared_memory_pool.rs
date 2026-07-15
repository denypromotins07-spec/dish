//! POSIX Shared Memory Pool for Zero-Copy Data Sharing
//! Enables zero-copy data sharing between Rust core engine and Python Ray workers.
//! Eliminates serialization overhead and RAM duplication.

use anyhow::{Result, anyhow};
use std::sync::Arc;
use std::ptr;
use std::ffi::CString;
use std::os::unix::io::RawFd;
use libc::{mmap, munmap, shm_open, shm_unlink, ftruncate, PROT_READ, PROT_WRITE, MAP_SHARED, O_CREAT, O_RDWR};
use std::slice;

/// Shared memory segment configuration
#[derive(Debug, Clone)]
pub struct SharedMemConfig {
    pub name: String,
    pub size_bytes: usize,
}

/// POSIX shared memory pool
pub struct SharedMemoryPool {
    fd: RawFd,
    ptr: *mut u8,
    size: usize,
    name: CString,
    is_owner: bool,
}

unsafe impl Send for SharedMemoryPool {}
unsafe impl Sync for SharedMemoryPool {}

impl SharedMemoryPool {
    /// Create or open a shared memory segment
    pub fn new(config: &SharedMemConfig, create: bool) -> Result<Self> {
        let name = CString::new(config.name.as_str())
            .map_err(|e| anyhow!("Invalid shared mem name: {}", e))?;
        
        // Open or create shared memory
        let flags = if create { O_CREAT | O_RDWR } else { O_RDWR };
        let fd = unsafe {
            shm_open(name.as_ptr(), flags, 0o666)
        };
        
        if fd == -1 {
            return Err(anyhow!("Failed to open shared memory: {}", std::io::Error::last_os_error()));
        }
        
        // Set size if creating
        if create {
            let ret = unsafe { ftruncate(fd, config.size_bytes as i64) };
            if ret == -1 {
                unsafe { libc::close(fd) };
                return Err(anyhow!("Failed to set shared mem size: {}", std::io::Error::last_os_error()));
            }
        }
        
        // Memory map the segment
        let ptr = unsafe {
            mmap(
                ptr::null_mut(),
                config.size_bytes,
                PROT_READ | PROT_WRITE,
                MAP_SHARED,
                fd,
                0,
            )
        };
        
        if ptr == libc::MAP_FAILED {
            unsafe { libc::close(fd) };
            return Err(anyhow!("Failed to mmap shared memory: {}", std::io::Error::last_os_error()));
        }
        
        Ok(Self {
            fd,
            ptr: ptr as *mut u8,
            size: config.size_bytes,
            name,
            is_owner: create,
        })
    }
    
    /// Get a slice of the shared memory
    pub fn as_slice(&self) -> &[u8] {
        unsafe { slice::from_raw_parts(self.ptr, self.size) }
    }
    
    /// Get a mutable slice of the shared memory
    pub fn as_slice_mut(&mut self) -> &mut [u8] {
        unsafe { slice::from_raw_parts_mut(self.ptr, self.size) }
    }
    
    /// Write data to shared memory at offset
    pub fn write_at(&mut self, offset: usize, data: &[u8]) -> Result<()> {
        if offset + data.len() > self.size {
            return Err(anyhow!("Write exceeds shared memory bounds"));
        }
        
        let slice = self.as_slice_mut();
        slice[offset..offset + data.len()].copy_from_slice(data);
        
        Ok(())
    }
    
    /// Read data from shared memory at offset
    pub fn read_at(&self, offset: usize, len: usize) -> Result<Vec<u8>> {
        if offset + len > self.size {
            return Err(anyhow!("Read exceeds shared memory bounds"));
        }
        
        let slice = self.as_slice();
        Ok(slice[offset..offset + len].to_vec())
    }
    
    /// Get the file descriptor
    pub fn fd(&self) -> RawFd {
        self.fd
    }
    
    /// Get the size in bytes
    pub fn size(&self) -> usize {
        self.size
    }
    
    /// Get the shared memory name (for passing to Python)
    pub fn name(&self) -> &str {
        self.name.to_str().unwrap_or("")
    }
}

impl Drop for SharedMemoryPool {
    fn drop(&mut self) {
        // Unmap
        unsafe {
            munmap(self.ptr as *mut _, self.size);
        }
        
        // Close FD
        unsafe {
            libc::close(self.fd);
        }
        
        // Unlink if owner
        if self.is_owner {
            unsafe {
                shm_unlink(self.name.as_ptr());
            }
        }
    }
}

/// Header for messages in shared memory
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct ShmHeader {
    pub magic: u32,
    pub data_type: u32,
    pub length: u64,
    pub timestamp_ns: u64,
    pub checksum: u32,
}

impl ShmHeader {
    pub const MAGIC: u32 = 0x53484D45; // "SHME"
    pub const SIZE: usize = std::mem::size_of::<ShmHeader>();
    
    pub fn new(data_type: u32, length: u64) -> Self {
        use std::time::{SystemTime, UNIX_EPOCH};
        
        Self {
            magic: Self::MAGIC,
            data_type,
            length,
            timestamp_ns: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos() as u64,
            checksum: 0,
        }
    }
}

/// Typed shared memory buffer for specific data types
pub struct ShmBuffer<T> {
    pool: Arc<std::sync::Mutex<SharedMemoryPool>>,
    _phantom: std::marker::PhantomData<T>,
}

impl<T> ShmBuffer<T> 
where
    T: Copy + Sized,
{
    pub fn new(pool: Arc<std::sync::Mutex<SharedMemoryPool>>) -> Self {
        Self {
            pool,
            _phantom: std::marker::PhantomData,
        }
    }
    
    /// Write a slice of items to shared memory
    pub fn write(&self, items: &[T], offset_items: usize) -> Result<()> {
        let mut pool = self.pool.lock().unwrap();
        let byte_offset = offset_items * std::mem::size_of::<T>();
        let byte_len = items.len() * std::mem::size_of::<T>();
        
        // Write header
        let header = ShmHeader::new(1, items.len() as u64);
        pool.write_at(0, bytemuck::cast_slice(&[header]))?;
        
        // Write data
        pool.write_at(ShmHeader::SIZE + byte_offset, bytemuck::cast_slice(items))?;
        
        Ok(())
    }
    
    /// Read a slice of items from shared memory
    pub fn read(&self, offset_items: usize, len: usize) -> Result<Vec<T>> {
        let pool = self.pool.lock().unwrap();
        let byte_offset = ShmHeader::SIZE + offset_items * std::mem::size_of::<T>();
        let byte_len = len * std::mem::size_of::<T>();
        
        let bytes = pool.read_at(byte_offset, byte_len)?;
        let items = bytemuck::cast_slice::<u8, T>(&bytes).to_vec();
        
        Ok(items)
    }
}

/// Create a shared memory pool accessible from Python
pub fn create_python_accessible_shm(
    name: &str,
    size_bytes: usize,
) -> Result<(SharedMemoryPool, String)> {
    let config = SharedMemConfig {
        name: name.to_string(),
        size_bytes,
    };
    
    let pool = SharedMemoryPool::new(&config, true)?;
    let shm_name = format!("/{}", name);
    
    Ok((pool, shm_name))
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_shared_memory_write_read() {
        let config = SharedMemConfig {
            name: "/test_shm_123".to_string(),
            size_bytes: 1024 * 1024, // 1MB
        };
        
        // Create
        let mut pool = SharedMemoryPool::new(&config, true).unwrap();
        
        // Write
        let data = b"Hello, shared memory!";
        pool.write_at(0, data).unwrap();
        
        // Read
        let read_data = pool.read_at(0, data.len()).unwrap();
        assert_eq!(read_data, data);
        
        // Cleanup explicitly (drop will also unlink)
        drop(pool);
        
        // Verify unlinked
        let result = SharedMemoryPool::new(&config, false);
        assert!(result.is_err());
    }
    
    #[test]
    fn test_typed_buffer() {
        let config = SharedMemConfig {
            name: "/test_typed_buf".to_string(),
            size_bytes: 1024 * 1024,
        };
        
        let pool = SharedMemoryPool::new(&config, true).unwrap();
        let arc_pool = Arc::new(std::sync::Mutex::new(pool));
        
        let buffer: ShmBuffer<f64> = ShmBuffer::new(arc_pool.clone());
        
        // Write some doubles
        let data: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        buffer.write(&data, 0).unwrap();
        
        // Read back
        let read_data = buffer.read(0, data.len()).unwrap();
        assert_eq!(read_data, data);
    }
}
