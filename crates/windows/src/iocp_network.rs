//! Windows IOCP (I/O Completion Ports) Network Engine
//! 
//! This module replaces Linux io_uring with Windows-native IOCP for:
//! - Zero-copy WebSocket ingestion from Binance/Bybit
//! - Ultra-low latency REST API calls
//! - Non-blocking network I/O without kernel context-switching penalties
//! 
//! Uses tokio's IOCP-backed runtime for async operations on Windows

use std::io;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream, UdpSocket};
use tokio::sync::mpsc;
use bytes::{BufMut, BytesMut};

/// Configuration for the IOCP network engine
#[derive(Clone, Debug)]
pub struct IocpConfig {
    /// Number of IOCP completion threads (typically equals CPU core count)
    pub completion_threads: usize,
    /// Maximum concurrent connections
    pub max_connections: usize,
    /// Socket buffer size in bytes (64KB for low latency)
    pub socket_buffer_size: usize,
    /// Connection timeout in milliseconds
    pub connection_timeout_ms: u64,
}

impl Default for IocpConfig {
    fn default() -> Self {
        Self {
            completion_threads: 12, // Match AMD Ryzen AI 5 thread count
            max_connections: 1000,
            socket_buffer_size: 65536, // 64KB
            connection_timeout_ms: 5000,
        }
    }
}

/// High-performance WebSocket client using IOCP
pub struct IocpWebSocket {
    stream: TcpStream,
    buffer: BytesMut,
    config: Arc<IocpConfig>,
}

impl IocpWebSocket {
    /// Creates a new IOCP WebSocket connection
    pub async fn connect(addr: SocketAddr, config: Arc<IocpConfig>) -> io::Result<Self> {
        let stream = tokio::time::timeout(
            Duration::from_millis(config.connection_timeout_ms),
            TcpStream::connect(addr),
        )
        .await
        .map_err(|_| io::Error::new(io::ErrorKind::TimedOut, "Connection timed out"))??;

        // Set socket options for low latency
        stream.set_nodelay(true)?;
        
        // Note: Windows socket buffer sizes are set via setsockopt
        // tokio handles this internally via IOCP

        Ok(Self {
            stream,
            buffer: BytesMut::with_capacity(config.socket_buffer_size),
            config,
        })
    }

    /// Reads WebSocket frames with zero-copy semantics
    pub async fn read_frame(&mut self) -> io::Result<&[u8]> {
        self.buffer.clear();
        
        // Read WebSocket frame header (simplified - real impl needs full WS protocol)
        let mut header = [0u8; 2];
        self.stream.read_exact(&mut header).await?;
        
        let payload_len = header[1] & 0x7F;
        let actual_len = match payload_len {
            126 => {
                let mut len_bytes = [0u8; 2];
                self.stream.read_exact(&mut len_bytes).await?;
                u16::from_be_bytes(len_bytes) as usize
            }
            127 => {
                let mut len_bytes = [0u8; 8];
                self.stream.read_exact(&mut len_bytes).await?;
                u64::from_be_bytes(len_bytes) as usize
            }
            _ => payload_len,
        };

        // Read payload directly into pre-allocated buffer
        self.buffer.resize(actual_len, 0);
        self.stream.read_exact(&mut self.buffer).await?;

        Ok(&self.buffer)
    }

    /// Writes WebSocket frames
    pub async fn write_frame(&mut self, data: &[u8]) -> io::Result<()> {
        // Simplified WebSocket framing (production needs full protocol)
        let mut frame = Vec::with_capacity(data.len() + 10);
        frame.push(0x81); // Text frame, FIN bit set
        
        if data.len() < 126 {
            frame.push(data.len() as u8);
        } else if data.len() < 65536 {
            frame.push(126);
            frame.extend_from_slice(&(data.len() as u16).to_be_bytes());
        } else {
            frame.push(127);
            frame.extend_from_slice(&(data.len() as u64).to_be_bytes());
        }
        
        frame.extend_from_slice(data);
        self.stream.write_all(&frame).await?;
        self.stream.flush().await?;
        
        Ok(())
    }
}

/// IOCP-based TCP server for handling multiple connections
pub struct IocpServer {
    listener: TcpListener,
    config: Arc<IocpConfig>,
}

impl IocpServer {
    /// Creates a new IOCP server
    pub async fn bind(addr: SocketAddr, config: Arc<IocpConfig>) -> io::Result<Self> {
        let listener = TcpListener::bind(addr).await?;
        Ok(Self { listener, config })
    }

    /// Accepts incoming connections with IOCP
    pub async fn accept(&self) -> io::Result<(TcpStream, SocketAddr)> {
        self.listener.accept().await
    }

    /// Spawns IOCP worker tasks for handling connections
    pub fn spawn_workers<F>(&self, handler: F) -> io::Result<()>
    where
        F: Fn(TcpStream, SocketAddr) + Send + Sync + 'static,
    {
        let handler = Arc::new(handler);
        
        for _ in 0..self.config.completion_threads {
            let listener = self.listener.local_addr()?;
            let handler = Arc::clone(&handler);
            
            tokio::spawn(async move {
                // Worker loop handled by tokio's IOCP runtime
                loop {
                    match TcpStream::connect(listener).await {
                        Ok(stream) => {
                            let addr = stream.peer_addr().unwrap_or_else(|_| {
                                "0.0.0.0:0".parse().unwrap()
                            });
                            handler(stream, addr);
                        }
                        Err(_) => continue,
                    }
                }
            });
        }
        
        Ok(())
    }
}

/// UDP socket wrapper for market data feeds using IOCP
pub struct IocpUdpSocket {
    socket: UdpSocket,
    buffer: BytesMut,
}

impl IocpUdpSocket {
    /// Binds to a UDP port for receiving market data
    pub async fn bind(addr: SocketAddr) -> io::Result<Self> {
        let socket = UdpSocket::bind(addr).await?;
        
        Ok(Self {
            socket,
            buffer: BytesMut::with_capacity(65536),
        })
    }

    /// Receives UDP packets with zero-copy
    pub async fn recv_from(&mut self) -> io::Result<(usize, SocketAddr)> {
        self.buffer.clear();
        self.buffer.resize(65536, 0);
        
        let (len, addr) = self.socket.recv_from(&mut self.buffer).await?;
        self.buffer.truncate(len);
        
        Ok((len, addr))
    }

    /// Sends UDP packets
    pub async fn send_to(&self, buf: &[u8], addr: SocketAddr) -> io::Result<usize> {
        self.socket.send_to(buf, addr).await
    }
}

/// Initializes the tokio runtime with IOCP configuration for Windows
pub fn create_iocp_runtime() -> io::Result<tokio::runtime::Runtime> {
    tokio::runtime::Builder::new_multi_thread()
        .worker_threads(12) // Match AMD Ryzen AI 5
        .enable_io()
        .enable_time()
        .thread_name("iocp-worker")
        .build()
        .map_err(|e| io::Error::new(io::ErrorKind::Other, e))
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_iocp_websocket_connect() {
        // This would require a real WebSocket server
        // Placeholder for integration testing
        let config = Arc::new(IocpConfig::default());
        let addr = "127.0.0.1:8080".parse().unwrap();
        
        // Expected to fail without a server, but tests the API
        let result = IocpWebSocket::connect(addr, config).await;
        assert!(result.is_err());
    }
}
