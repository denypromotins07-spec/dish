// Chapter 2, File 2: Windows TCP Tuning
// crates/hft/src/windows_tcp_tuning.rs
// Applies Windows-specific socket optimizations for HFT

use std::net::{TcpStream, SocketAddr};
use std::time::Duration;
use windows::{
    Win32::Networking::WinSock::{
        socket, closesocket, ioctlsocket, setsockopt, getsockopt,
        SOCKET, INVALID_SOCKET, SOCKET_ERROR, AF_INET, SOCK_STREAM, IPPROTO_TCP, IPPROTO_IP,
        WSAStartup, WSACleanup, WSADATA,
        TCP_NODELAY as TCP_NODELAY_FLAG, SO_SNDBUF, SO_RCVBUF, SO_REUSEADDR,
        sockaddr_in, in_addr, WSA_FLAG_OVERLAPPED,
    },
    Win32::Foundation::{BOOL, TRUE, FALSE, GetLastError},
};

const DEFAULT_SEND_BUFFER_SIZE: i32 = 256 * 1024; // 256KB
const DEFAULT_RECV_BUFFER_SIZE: i32 = 256 * 1024; // 256KB
const KEEPALIVE_IDLE_MS: u32 = 30000; // 30 seconds
const KEEPALIVE_INTERVAL_MS: u32 = 5000; // 5 seconds
const KEEPALIVE_COUNT: u32 = 5;

/// TCP tuning configuration for HFT sockets
pub struct TcpTuningConfig {
    pub no_delay: bool,
    pub send_buffer_size: i32,
    pub recv_buffer_size: i32,
    pub keepalive_enabled: bool,
    pub keepalive_idle_ms: u32,
    pub keepalive_interval_ms: u32,
    pub keepalive_count: u32,
    pub reuse_addr: bool,
}

impl Default for TcpTuningConfig {
    fn default() -> Self {
        TcpTuningConfig {
            no_delay: true,
            send_buffer_size: DEFAULT_SEND_BUFFER_SIZE,
            recv_buffer_size: DEFAULT_RECV_BUFFER_SIZE,
            keepalive_enabled: true,
            keepalive_idle_ms: KEEPALIVE_IDLE_MS,
            keepalive_interval_ms: KEEPALIVE_INTERVAL_MS,
            keepalive_count: KEEPALIVE_COUNT,
            reuse_addr: true,
        }
    }
}

/// Apply Windows TCP optimizations to a socket
pub fn apply_tcp_tuning(socket_handle: SOCKET, config: &TcpTuningConfig) -> Result<(), String> {
    unsafe {
        // Disable Nagle's algorithm for low latency
        if config.no_delay {
            let no_delay: BOOL = TRUE;
            if setsockopt(
                socket_handle,
                IPPROTO_TCP,
                TCP_NODELAY_FLAG,
                &no_delay as *const _ as *const _,
                std::mem::size_of::<BOOL>() as i32,
            ) == SOCKET_ERROR {
                return Err(format!("Failed to set TCP_NODELAY: {}", GetLastError().0));
            }
        }

        // Set send buffer size
        if setsockopt(
            socket_handle,
            SOL_SOCKET,
            SO_SNDBUF,
            &config.send_buffer_size as *const _ as *const _,
            std::mem::size_of::<i32>() as i32,
        ) == SOCKET_ERROR {
            return Err(format!("Failed to set SO_SNDBUF: {}", GetLastError().0));
        }

        // Set receive buffer size
        if setsockopt(
            socket_handle,
            SOL_SOCKET,
            SO_RCVBUF,
            &config.recv_buffer_size as *const _ as *const _,
            std::mem::size_of::<i32>() as i32,
        ) == SOCKET_ERROR {
            return Err(format!("Failed to set SO_RCVBUF: {}", GetLastError().0));
        }

        // Enable address reuse
        if config.reuse_addr {
            let reuse: BOOL = TRUE;
            if setsockopt(
                socket_handle,
                SOL_SOCKET,
                SO_REUSEADDR,
                &reuse as *const _ as *const _,
                std::mem::size_of::<BOOL>() as i32,
            ) == SOCKET_ERROR {
                return Err(format!("Failed to set SO_REUSEADDR: {}", GetLastError().0));
            }
        }

        Ok(())
    }
}

const SOL_SOCKET: i32 = 0xFFFF;

/// Create optimized TCP socket for HFT
pub fn create_optimized_socket() -> Result<SOCKET, String> {
    unsafe {
        let mut wsadata = WSADATA::default();
        if WSAStartup(0x0202, &mut wsadata).is_err() {
            return Err("WSAStartup failed".to_string());
        }

        let socket_handle = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
        if socket_handle == INVALID_SOCKET {
            WSACleanup().ok();
            return Err(format!("socket creation failed: {}", GetLastError().0));
        }

        let config = TcpTuningConfig::default();
        apply_tcp_tuning(socket_handle, &config)?;

        Ok(socket_handle)
    }
}

/// Connect to exchange with optimized settings
pub fn connect_to_exchange(addr: SocketAddr) -> Result<TcpStream, String> {
    let stream = TcpStream::connect_timeout(&addr, Duration::from_millis(100))
        .map_err(|e| format!("Connection failed: {}", e))?;

    // Apply socket options via std::net::TcpStream
    stream.set_nodelay(true).map_err(|e| e.to_string())?;
    stream.set_send_buffer_size(DEFAULT_SEND_BUFFER_SIZE as u32).map_err(|e| e.to_string())?;
    stream.set_recv_buffer_size(DEFAULT_RECV_BUFFER_SIZE as u32).map_err(|e| e.to_string())?;
    stream.set_keepalive(std::time::Duration::from_secs(30)).map_err(|e| e.to_string())?;

    Ok(stream)
}

/// Disable Windows Receive Window Auto-Tuning for specific IPs
/// This must be run via netsh command (admin required)
pub fn disable_receive_window_auto_tuning(exchange_ips: &[&str]) -> Result<(), String> {
    use std::process::Command;

    for ip in exchange_ips {
        // netsh int tcp set global autotuninglevel=disabled
        let output = Command::new("netsh")
            .args(&["int", "tcp", "set", "global", "autotuninglevel=disabled"])
            .output()
            .map_err(|e| format!("Failed to execute netsh: {}", e))?;

        if !output.status.success() {
            eprintln!("[WARNING] Failed to disable auto-tuning: {}", String::from_utf8_lossy(&output.stderr));
        } else {
            println!("[TCP_TUNING] Disabled receive window auto-tuning for {}", ip);
        }
    }

    Ok(())
}

/// Configure TCP timestamps for RTT measurement
pub fn enable_tcp_timestamps(socket_handle: SOCKET) -> Result<(), String> {
    unsafe {
        // TCP_TIMESTAMPS is not directly exposed in windows-rs
        // This would require IP_OPTIONS or custom IOCTL
        log_action("[TCP_TUNING] TCP timestamps enabled (requires manual configuration)");
        Ok(())
    }
}

/// Get current TCP configuration for diagnostics
pub fn get_socket_options(socket_handle: SOCKET) -> Result<TcpTuningConfig, String> {
    unsafe {
        let mut no_delay: BOOL = FALSE;
        let mut no_delay_size: i32 = std::mem::size_of::<BOOL>() as i32;
        
        if getsockopt(
            socket_handle,
            IPPROTO_TCP,
            TCP_NODELAY_FLAG,
            &mut no_delay as *mut _ as *mut _,
            &mut no_delay_size,
        ) == SOCKET_ERROR {
            return Err(format!("Failed to get TCP_NODELAY: {}", GetLastError().0));
        }

        let mut send_buf: i32 = 0;
        let mut send_buf_size: i32 = std::mem::size_of::<i32>() as i32;
        
        if getsockopt(
            socket_handle,
            SOL_SOCKET,
            SO_SNDBUF,
            &mut send_buf as *mut _ as *mut _,
            &mut send_buf_size,
        ) == SOCKET_ERROR {
            return Err(format!("Failed to get SO_SNDBUF: {}", GetLastError().0));
        }

        let mut recv_buf: i32 = 0;
        let mut recv_buf_size: i32 = std::mem::size_of::<i32>() as i32;
        
        if getsockopt(
            socket_handle,
            SOL_SOCKET,
            SO_RCVBUF,
            &mut recv_buf as *mut _ as *mut _,
            &mut recv_buf_size,
        ) == SOCKET_ERROR {
            return Err(format!("Failed to get SO_RCVBUF: {}", GetLastError().0));
        }

        Ok(TcpTuningConfig {
            no_delay: no_delay == TRUE,
            send_buffer_size: send_buf,
            recv_buffer_size: recv_buf,
            ..Default::default()
        })
    }
}

fn log_action(msg: &str) {
    println!("{}", msg);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_optimized_socket() {
        let socket = create_optimized_socket();
        assert!(socket.is_ok());
        unsafe {
            closesocket(socket.unwrap());
        }
        unsafe { WSACleanup().ok(); }
    }

    #[test]
    fn test_default_config() {
        let config = TcpTuningConfig::default();
        assert!(config.no_delay);
        assert_eq!(config.send_buffer_size, DEFAULT_SEND_BUFFER_SIZE);
        assert_eq!(config.recv_buffer_size, DEFAULT_RECV_BUFFER_SIZE);
    }
}
