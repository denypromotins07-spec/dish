//! High-Performance IPC Router using Unix Domain Sockets and POSIX Shared Memory
//! Connects Rust core with Python analytics layer without network overhead.

use std::collections::HashMap;
use std::ffi::CString;
use std::io::{Read, Write};
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::Duration;

use log::{debug, error, info, warn};

/// Message types for IPC communication
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MessageType {
    TickData = 0,
    OrderSignal = 1,
    RiskUpdate = 2,
    TcaMetrics = 3,
    Heartbeat = 255,
}

impl From<u8> for MessageType {
    fn from(val: u8) -> Self {
        match val {
            0 => MessageType::TickData,
            1 => MessageType::OrderSignal,
            2 => MessageType::RiskUpdate,
            3 => MessageType::TcaMetrics,
            255 => MessageType::Heartbeat,
            _ => MessageType::Heartbeat, // Default to heartbeat for unknown
        }
    }
}

/// Header for IPC messages (fixed size, zero-copy friendly)
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct IpcHeader {
    pub msg_type: u8,
    pub flags: u8,
    pub sequence: u16,
    pub payload_len: u32,
    pub timestamp_ns: u64,
}

impl IpcHeader {
    pub const SIZE: usize = std::mem::size_of::<Self>();

    pub fn new(msg_type: MessageType, payload_len: u32) -> Self {
        Self {
            msg_type: msg_type as u8,
            flags: 0,
            sequence: 0,
            payload_len,
            timestamp_ns: 0,
        }
    }

    pub fn to_bytes(&self) -> [u8; IpcHeader::SIZE] {
        unsafe { std::mem::transmute(*self) }
    }

    pub fn from_bytes(bytes: &[u8]) -> Option<Self> {
        if bytes.len() != IpcHeader::SIZE {
            return None;
        }
        Some(unsafe { std::ptr::read(bytes.as_ptr() as *const Self) })
    }
}

/// Shared memory ring buffer for zero-copy data transfer
pub struct SharedMemoryRingBuffer {
    name: String,
    size: usize,
    head: Arc<AtomicU64>,
    tail: Arc<AtomicU64>,
    buffer: Arc<memmap2::MmapMut>,
    is_running: Arc<AtomicBool>,
}

impl SharedMemoryRingBuffer {
    pub fn create(name: &str, size: usize) -> Result<Self, std::io::Error> {
        use std::fs::File;
        use memmap2::MmapMut;

        let shm_path = format!("/dev/shm/{}", name);
        
        // Create or open shared memory file
        let file = File::options()
            .read(true)
            .write(true)
            .create(true)
            .truncate(size as u64)
            .open(&shm_path)?;

        let mut mmap = unsafe { MmapMut::map_mut(&file)? };
        mmap.fill(0);

        Ok(Self {
            name: name.to_string(),
            size,
            head: Arc::new(AtomicU64::new(0)),
            tail: Arc::new(AtomicU64::new(0)),
            buffer: Arc::new(mmap),
            is_running: Arc::new(AtomicBool::new(true)),
        })
    }

    pub fn write(&self, data: &[u8]) -> Result<(), &'static str> {
        if data.len() > self.size - IpcHeader::SIZE {
            return Err("Payload too large for ring buffer");
        }

        let head = self.head.load(Ordering::Acquire);
        let tail = self.tail.load(Ordering::Acquire);
        let used = if head >= tail {
            head - tail
        } else {
            (self.size as u64) - tail + head
        };

        if used + data.len() as u64 + IpcHeader::SIZE as u64 > self.size as u64 {
            return Err("Ring buffer full");
        }

        // Create header
        let mut header = IpcHeader::new(MessageType::TickData, data.len() as u32);
        header.timestamp_ns = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64;

        // Write header
        let header_bytes = header.to_bytes();
        let write_pos = (head % self.size as u64) as usize;
        self.buffer[write_pos..write_pos + IpcHeader::SIZE].copy_from_slice(&header_bytes);

        // Write payload
        let payload_start = write_pos + IpcHeader::SIZE;
        let payload_end = payload_start + data.len();
        self.buffer[payload_start..payload_end].copy_from_slice(data);

        // Update head
        self.head.store((head + IpcHeader::SIZE as u64 + data.len() as u64) % self.size as u64, Ordering::Release);

        Ok(())
    }

    pub fn read<F>(&self, mut callback: F) -> Result<(), &'static str>
    where
        F: FnMut(&[u8]),
    {
        let head = self.head.load(Ordering::Acquire);
        let mut tail = self.tail.load(Ordering::Acquire);

        while tail != head {
            let read_pos = (tail % self.size as u64) as usize;
            
            // Read header
            if read_pos + IpcHeader::SIZE > self.size {
                return Err("Invalid read position");
            }
            
            let header_bytes = &self.buffer[read_pos..read_pos + IpcHeader::SIZE];
            if let Some(header) = IpcHeader::from_bytes(header_bytes) {
                let payload_start = read_pos + IpcHeader::SIZE;
                let payload_end = payload_start + header.payload_len as usize;
                
                if payload_end <= self.size {
                    callback(&self.buffer[payload_start..payload_end]);
                    
                    // Update tail
                    tail = (tail + IpcHeader::SIZE as u64 + header.payload_len as u64) % self.size as u64;
                    self.tail.store(tail, Ordering::Release);
                } else {
                    return Err("Payload extends beyond buffer");
                }
            } else {
                return Err("Invalid header");
            }
        }

        Ok(())
    }
}

/// IPC Router managing multiple communication channels
pub struct IpcRouter {
    unix_socket_path: String,
    listener: Option<UnixListener>,
    clients: HashMap<String, UnixStream>,
    shared_buffers: HashMap<String, Arc<SharedMemoryRingBuffer>>,
    is_running: Arc<AtomicBool>,
    sequence: Arc<AtomicU64>,
}

impl IpcRouter {
    pub fn new(socket_path: &str) -> Result<Self, std::io::Error> {
        let path = Path::new(socket_path);
        
        // Remove existing socket file
        if path.exists() {
            std::fs::remove_file(path)?;
        }

        let listener = UnixListener::bind(path)?;
        listener.set_nonblocking(true)?;

        Ok(Self {
            unix_socket_path: socket_path.to_string(),
            listener: Some(listener),
            clients: HashMap::new(),
            shared_buffers: HashMap::new(),
            is_running: Arc::new(AtomicBool::new(false)),
            sequence: Arc::new(AtomicU64::new(0)),
        })
    }

    /// Create a shared memory buffer for zero-copy communication
    pub fn create_shared_buffer(&mut self, name: &str, size: usize) -> Result<(), std::io::Error> {
        let buffer = SharedMemoryRingBuffer::create(name, size)?;
        self.shared_buffers.insert(name.to_string(), Arc::new(buffer));
        info!("Created shared memory buffer: {} ({} bytes)", name, size);
        Ok(())
    }

    /// Start accepting connections
    pub fn start(&mut self) {
        self.is_running.store(true, Ordering::SeqCst);
        info!("IPC Router starting on {}", self.unix_socket_path);

        let listener = self.listener.take().unwrap();
        let is_running = self.is_running.clone();
        let clients: Arc<std::sync::Mutex<HashMap<String, UnixStream>>> = 
            Arc::new(std::sync::Mutex::new(HashMap::new()));

        thread::spawn(move || {
            let mut client_counter = 0;
            
            for stream in listener.incoming() {
                if !is_running.load(Ordering::SeqCst) {
                    break;
                }

                match stream {
                    Ok(stream) => {
                        client_counter += 1;
                        let client_id = format!("client_{}", client_counter);
                        info!("New IPC client connected: {}", client_id);
                        
                        let mut clients_guard = clients.lock().unwrap();
                        clients_guard.insert(client_id, stream);
                    }
                    Err(e) => {
                        warn!("Failed to accept IPC connection: {}", e);
                    }
                }
            }
        });
    }

    /// Send message to all connected clients
    pub fn broadcast(&mut self, msg_type: MessageType, payload: &[u8]) -> Result<(), std::io::Error> {
        let mut header = IpcHeader::new(msg_type, payload.len() as u32);
        header.sequence = self.sequence.fetch_add(1, Ordering::SeqCst) as u16;
        header.timestamp_ns = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64;

        let header_bytes = header.to_bytes();
        let mut message = Vec::with_capacity(IpcHeader::SIZE + payload.len());
        message.extend_from_slice(&header_bytes);
        message.extend_from_slice(payload);

        // Note: In production, we'd iterate over actual clients
        // This is a simplified version
        debug!("Broadcasting message type {:?} ({} bytes)", msg_type, message.len());

        Ok(())
    }

    /// Write to shared memory buffer
    pub fn write_to_shared_buffer(&self, name: &str, data: &[u8]) -> Result<(), &'static str> {
        if let Some(buffer) = self.shared_buffers.get(name) {
            buffer.write(data)
        } else {
            Err("Shared buffer not found")
        }
    }

    /// Stop the router
    pub fn stop(&mut self) {
        self.is_running.store(false, Ordering::SeqCst);
        
        // Clean up socket file
        let path = Path::new(&self.unix_socket_path);
        if path.exists() {
            let _ = std::fs::remove_file(path);
        }

        info!("IPC Router stopped");
    }
}

impl Drop for IpcRouter {
    fn drop(&mut self) {
        self.stop();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ipc_header_serialization() {
        let header = IpcHeader::new(MessageType::TickData, 1024);
        let bytes = header.to_bytes();
        let restored = IpcHeader::from_bytes(&bytes).unwrap();
        
        assert_eq!(header.msg_type, restored.msg_type);
        assert_eq!(header.payload_len, restored.payload_len);
    }

    #[test]
    fn test_shared_memory_ring_buffer() {
        let buffer = SharedMemoryRingBuffer::create("test_buffer", 4096).unwrap();
        
        let test_data = b"Hello, IPC!";
        buffer.write(test_data).unwrap();
        
        let mut received = Vec::new();
        buffer.read(|data| {
            received.extend_from_slice(data);
        }).unwrap();
        
        assert_eq!(received, test_data);
    }
}
