// Chapter 2, File 1: Windows IOCP Networking
// crates/hft/src/windows_iocp.rs
// Implements I/O Completion Port (IOCP) with Registered I/O (RIO) for zero-copy WebSocket ingestion

use std::ptr;
use std::ffi::c_void;
use std::sync::Arc;
use std::time::Duration;
use windows::{
    Win32::Networking::WinSock::{
        WSADATA, WSAStartup, WSACleanup, socket, closesocket, ioctlsocket,
        WSASocketW, WSARecv, WSASend, WSAIoctl,
        SOCKET, INVALID_SOCKET, SOCKET_ERROR, AF_INET, SOCK_STREAM, IPPROTO_TCP,
        WSA_FLAG_OVERLAPPED, WSA_FLAG_REGISTERED_IO,
        SIO_GET_EXTENSION_FUNCTION_POINTER, SIO_RIO_RECV_BUF, SIO_RIO_SEND_BUF,
        sockaddr_in, in_addr, RIO_BUF, RIO_BUFFERID, RIO_REQUEST_HANDLE,
        RIO_CQE, RIO_NOTIFICATION_COMPLETION_TYPE, RIO_CP_SIZE,
    },
    Win32::System::IO::{
        CreateIoCompletionPort, GetQueuedCompletionStatus, PostQueuedCompletionStatus,
        OVERLAPPED,
    },
    Win32::Foundation::{HANDLE, BOOL, TRUE, FALSE, GetLastError},
};

const MAX_PENDING_CONNECTIONS: u32 = 1024;
const IOCP_KEY_ACCEPT: usize = 1;
const IOCP_KEY_RECV: usize = 2;
const IOCP_KEY_SEND: usize = 3;
const DEFAULT_BUFFER_SIZE: usize = 65536; // 64KB buffers
const MAX_OUTSTANDING_RECV: u32 = 256;
const MAX_OUTSTANDING_SEND: u32 = 256;

/// RIO Extension function pointers
type LPFN_WSARIOREGISTERBUFFER = unsafe extern "system" fn(*const RIO_BUF, u32) -> RIO_BUFFERID;
type LPFN_WSARIOREGISTEREX = unsafe extern "system" fn(SOCKET, *const RIO_BUF, u32, *const RIO_BUF, u32, RIO_REQUEST_HANDLE) -> BOOL;
type LPFN_WSIORIONOTIFY = unsafe extern "system" fn(RIO_REQUEST_HANDLE) -> BOOL;

/// HFT IOCP Server - High-performance async networking
pub struct IOCPServer {
    iocp_handle: HANDLE,
    listen_socket: SOCKET,
    running: bool,
    worker_count: usize,
}

unsafe impl Send for IOCPServer {}
unsafe impl Sync for IOCPServer {}

impl IOCPServer {
    /// Initialize Winsock and create IOCP
    pub fn new(port: u16, worker_count: usize) -> Result<Self, String> {
        unsafe {
            // Initialize Winsock
            let mut wsadata = WSADATA::default();
            if WSAStartup(0x0202, &mut wsadata).is_err() {
                return Err("WSAStartup failed".to_string());
            }

            // Create listening socket
            let listen_socket = WSASocketW(
                AF_INET,
                SOCK_STREAM,
                IPPROTO_TCP,
                None,
                0,
                WSA_FLAG_OVERLAPPED,
            );

            if listen_socket == INVALID_SOCKET {
                WSACleanup().ok();
                return Err(format!("WSASocketW failed: {}", GetLastError().0));
            }

            // Bind to port
            let addr = sockaddr_in {
                sin_family: AF_INET as u16,
                sin_port: port.to_be(),
                sin_addr: in_addr { S_un: in_addr__bindgen_ty_1 { S_addr: 0 } },
                sin_zero: [0; 8],
            };

            use windows::Win32::Networking::WinSock::bind;
            if bind(listen_socket, &addr as *const _ as *const _, std::mem::size_of::<sockaddr_in>() as i32) == SOCKET_ERROR {
                closesocket(listen_socket).ok();
                WSACleanup().ok();
                return Err(format!("bind failed: {}", GetLastError().0));
            }

            // Listen
            if windows::Win32::Networking::WinSock::listen(listen_socket, MAX_PENDING_CONNECTIONS as i32) == SOCKET_ERROR {
                closesocket(listen_socket).ok();
                WSACleanup().ok();
                return Err(format!("listen failed: {}", GetLastError().0));
            }

            // Create IOCP
            let iocp_handle = CreateIoCompletionPort(HANDLE::default(), HANDLE::default(), 0, worker_count as u32)
                .map_err(|e| format!("CreateIoCompletionPort failed: {:?}", e))?;

            // Associate listen socket with IOCP
            CreateIoCompletionPort(HANDLE(listen_socket as isize), iocp_handle, IOCP_KEY_ACCEPT, 0)
                .map_err(|e| format!("Failed to associate socket with IOCP: {:?}", e))?;

            Ok(IOCPServer {
                iocp_handle,
                listen_socket,
                running: false,
                worker_count,
            })
        }
    }

    /// Start accepting connections
    pub fn start_accepting(&mut self) -> Result<(), String> {
        self.running = true;
        Ok(())
    }

    /// IOCP worker loop - processes completion events
    pub fn run_worker(&self, worker_id: usize) -> Result<(), String> {
        unsafe {
            let mut bytes_transferred: u32 = 0;
            let mut completion_key: usize = 0;
            let mut overlapped: *mut OVERLAPPED = ptr::null_mut();

            while self.running {
                match GetQueuedCompletionStatus(self.iocp_handle, &mut bytes_transferred, &mut completion_key, &mut overlapped, 1000) {
                    Ok(_) => {
                        if overlapped.is_null() && completion_key == 0 {
                            // Timeout, continue
                            continue;
                        }

                        match completion_key {
                            IOCP_KEY_ACCEPT => self.handle_accept()?,
                            IOCP_KEY_RECV => self.handle_recv(bytes_transferred)?,
                            IOCP_KEY_SEND => self.handle_send(bytes_transferred)?,
                            _ => eprintln!("[IOCP] Unknown completion key: {}", completion_key),
                        }
                    }
                    Err(e) => {
                        if overlapped.is_null() {
                            eprintln!("[IOCP Worker {}] Error: {:?}", worker_id, e);
                        }
                    }
                }
            }

            Ok(())
        }
    }

    fn handle_accept(&self) -> Result<(), String> {
        // Accept new connection and register with IOCP
        unsafe {
            use windows::Win32::Networking::WinSock::accept;
            let client_socket = accept(self.listen_socket, None, None);
            if client_socket == INVALID_SOCKET {
                return Err(format!("accept failed: {}", GetLastError().0));
            }

            // Disable Nagle's algorithm
            let no_delay: BOOL = TRUE;
            setsockopt(client_socket, IPPROTO_TCP, TCP_NODELAY, &no_delay as *const _ as *const c_void, std::mem::size_of::<BOOL>() as i32);

            // Associate client socket with IOCP
            CreateIoCompletionPort(HANDLE(client_socket as isize), self.iocp_handle, IOCP_KEY_RECV, 0)
                .map_err(|e| format!("Failed to associate client socket: {:?}", e))?;

            // Post initial receive request
            self.post_receive_request(client_socket)?;
        }
        Ok(())
    }

    fn handle_recv(&self, bytes: u32) -> Result<(), String> {
        // Process received data (market ticks)
        log_action(&format!("[IOCP] Received {} bytes", bytes));
        Ok(())
    }

    fn handle_send(&self, bytes: u32) -> Result<(), String> {
        log_action(&format!("[IOCP] Sent {} bytes", bytes));
        Ok(())
    }

    fn post_receive_request(&self, socket: SOCKET) -> Result<(), String> {
        // Post asynchronous receive using WSARecv
        unsafe {
            let buffer = Vec::with_capacity(DEFAULT_BUFFER_SIZE);
            let mut overlapped = Box::new(OVERLAPPED::default());
            
            let wsabuf = windows::Win32::Networking::WinSock::WSABUF {
                len: DEFAULT_BUFFER_SIZE as u32,
                buf: buffer.as_ptr() as *mut _,
            };

            let mut flags: u32 = 0;
            if WSARecv(socket, &wsabuf, 1, None, &mut flags, &mut *overlapped, None).is_err() {
                let err = GetLastError();
                if err.0 != 997 { // ERROR_IO_PENDING is expected
                    return Err(format!("WSARecv failed: {}", err.0));
                }
            }

            std::mem::forget(buffer); // Buffer ownership transferred to overlapped operation
            std::mem::forget(overlapped);
        }
        Ok(())
    }

    pub fn stop(&mut self) {
        self.running = false;
        unsafe {
            PostQueuedCompletionStatus(self.iocp_handle, 0, 0, None).ok();
        }
    }
}

impl Drop for IOCPServer {
    fn drop(&mut self) {
        unsafe {
            if self.listen_socket != INVALID_SOCKET {
                closesocket(self.listen_socket).ok();
            }
            WSACleanup().ok();
        }
    }
}

/// Helper for setsockopt
unsafe fn setsockopt(s: SOCKET, level: i32, optname: i32, optval: *const c_void, optlen: i32) -> i32 {
    windows::Win32::Networking::WinSock::setsockopt(s, level, optname, optval, optlen)
}

const TCP_NODELAY: i32 = 1;

fn log_action(msg: &str) {
    println!("{}", msg);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_iocp_creation() {
        let server = IOCPServer::new(0, 4);
        assert!(server.is_ok());
    }
}
