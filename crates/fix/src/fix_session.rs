//! FIX Session Management
//! Logon, Heartbeat, ResendRequest, SequenceReset handling
//! TCP_NODELAY, Nagle's algorithm disabled, io_uring/epoll configurations

use std::io;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;
use tokio::sync::{mpsc, Mutex};
use tokio::time;

use super::fix_parser::{FixParser, FixMessageType, FixHeaderInfo, tags};

/// FIX session state
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SessionState {
    Disconnected,
    Connecting,
    SentLogon,
    LoggedOn,
    LogoutSent,
}

/// FIX session configuration
#[derive(Clone)]
pub struct SessionConfig {
    pub sender_comp_id: String,
    pub target_comp_id: String,
    pub host: String,
    pub port: u16,
    pub heart_bt_sec: i32,
    pub reconnect_delay_ms: u64,
    pub logon_timeout_sec: u64,
    pub max_seq_gap: u64,
}

/// FIX session manager
pub struct FixSession {
    config: SessionConfig,
    state: SessionState,
    parser: FixParser,
    outgoing_seq: u64,
    incoming_seq: u64,
    last_msg_time: Instant,
    test_request_counter: u64,
    tx: mpsc::Sender<Vec<u8>>,
    rx: Option<mpsc::Receiver<Vec<u8>>>,
    shutdown: bool,
}

impl FixSession {
    pub fn new(config: SessionConfig, channel_capacity: usize) -> Self {
        let (tx, rx) = mpsc::channel(channel_capacity);
        
        Self {
            parser: FixParser::new(8192),
            config,
            state: SessionState::Disconnected,
            outgoing_seq: 1,
            incoming_seq: 1,
            last_msg_time: Instant::now(),
            test_request_counter: 0,
            tx,
            rx: Some(rx),
            shutdown: false,
        }
    }

    /// Connect to FIX counterparty with TCP_NODELAY
    pub async fn connect(&mut self) -> Result<TcpStream, io::Error> {
        self.state = SessionState::Connecting;
        
        let addr = format!("{}:{}", self.config.host, self.config.port);
        let stream = TcpStream::connect(&addr).await?;
        
        // Disable Nagle's algorithm for low latency
        stream.set_nodelay(true)?;
        
        // Set socket options for optimal performance
        #[cfg(target_os = "linux")]
        {
            use std::os::unix::io::AsRawFd;
            let fd = stream.as_raw_fd();
            unsafe {
                libc::setsockopt(fd, libc::SOL_SOCKET, libc::SO_KEEPALIVE, &1i32 as *const _ as _, 4);
            }
        }
        
        Ok(stream)
    }

    /// Send Logon message
    pub async fn send_logon(&mut self, stream: &mut TcpStream) -> Result<(), io::Error> {
        let mut builder = super::fix_parser::FixMessageBuilder::new(
            &self.config.sender_comp_id,
            &self.config.target_comp_id,
            self.outgoing_seq,
        );
        
        let logon_msg = builder.build_logon(self.config.heart_bt_sec, "NONE");
        self.outgoing_seq += 1;
        
        stream.write_all(&logon_msg).await?;
        stream.flush().await?;
        
        self.state = SessionState::SentLogon;
        self.last_msg_time = Instant::now();
        
        log::info!("FIX Logon sent to {}", self.config.target_comp_id);
        Ok(())
    }

    /// Process incoming FIX messages
    pub async fn process_message(
        &mut self,
        data: &[u8],
    ) -> Result<Option<FixMessageType>, crate::fix_parser::FixParseError> {
        if !self.parser.parse(data)? {
            return Ok(None);
        }

        // Validate sequence number
        if let Some(header) = self.parser.get_header_info() {
            if header.msg_seq_num != self.incoming_seq {
                log::warn!(
                    "Sequence gap: expected {}, got {}",
                    self.incoming_seq,
                    header.msg_seq_num
                );
                
                if header.msg_seq_num > self.incoming_seq {
                    // Request resend
                    self.send_resend_request(self.incoming_seq, header.msg_seq_num - 1)
                        .await?;
                }
                return Ok(None);
            }
            self.incoming_seq += 1;
        }

        self.last_msg_time = Instant::now();

        // Handle message based on type
        match self.parser.get_msg_type() {
            Some(FixMessageType::Logon) => {
                self.state = SessionState::LoggedOn;
                log::info!("FIX session logged on");
            }
            Some(FixMessageType::Heartbeat) => {
                log::debug!("Received heartbeat");
            }
            Some(FixMessageType::TestRequest) => {
                // Auto-respond with heartbeat
                if let Some(test_req_id) = self.parser.get_field(tags::TEST_REQ_ID) {
                    if let Ok(req_id) = test_req_id.value_str() {
                        self.send_heartbeat(req_id).await?;
                    }
                }
            }
            Some(FixMessageType::Logout) => {
                self.state = SessionState::LogoutSent;
                log::info!("Received logout request");
            }
            Some(FixMessageType::ResendRequest) => {
                // Handle resend request
                self.handle_resend_request().await?;
            }
            Some(FixMessageType::SequenceReset) => {
                // Handle sequence reset
                if let Some(new_seq) = self.parser.get_field(tags::NEW_SEQ_NUM) {
                    if let Some(seq) = new_seq.value_as_int() {
                        self.incoming_seq = seq as u64;
                        log::info!("Sequence reset to {}", self.incoming_seq);
                    }
                }
            }
            _ => {}
        }

        Ok(self.parser.get_msg_type())
    }

    /// Send Heartbeat message
    async fn send_heartbeat(&mut self, test_req_id: &str) -> Result<(), io::Error> {
        let mut builder = super::fix_parser::FixMessageBuilder::new(
            &self.config.sender_comp_id,
            &self.config.target_comp_id,
            self.outgoing_seq,
        );
        
        let hb_msg = builder.build_heartbeat(test_req_id);
        self.outgoing_seq += 1;
        
        // Queue for sending
        let _ = self.tx.send(hb_msg).await;
        Ok(())
    }

    /// Send ResendRequest
    async fn send_resend_request(&mut self, begin: u64, end: u64) -> Result<(), io::Error> {
        let mut builder = super::fix_parser::FixMessageBuilder::new(
            &self.config.sender_comp_id,
            &self.config.target_comp_id,
            self.outgoing_seq,
        );
        
        let now = chrono::Utc::now().format("%Y%m%d-%H:%M:%S.%f").to_string();
        let msg = format!(
            "8=FIX.4.4\x019=64\x0135=2\x0149={}\x0156={}\x0134={}\x0152={}\x017={}\x0116={}\x0110={:03}\x01",
            self.config.sender_comp_id,
            self.config.target_comp_id,
            self.outgoing_seq,
            now,
            begin,
            end,
            0u8, // Placeholder checksum
        );
        
        self.outgoing_seq += 1;
        let _ = self.tx.send(msg.into_bytes()).await;
        
        log::info!("ResendRequest sent: {}-{}", begin, end);
        Ok(())
    }

    /// Handle incoming ResendRequest
    async fn handle_resend_request(&mut self) -> Result<(), io::Error> {
        let begin_seq = self.parser.get_field(tags::BEGIN_SEQ_NO)
            .and_then(|f| f.value_as_int())
            .unwrap_or(1) as u64;
        
        let end_seq = self.parser.get_field(tags::END_SEQ_NO)
            .and_then(|f| f.value_as_int())
            .unwrap_or(i64::MAX) as u64;
        
        log::info!("ResendRequest received: {}-{}", begin_seq, end_seq);
        
        // In production, retrieve and resend messages from persistent store
        // For now, send sequence reset
        self.send_sequence_reset(begin_seq, end_seq + 1).await?;
        
        Ok(())
    }

    /// Send SequenceReset-GAP
    async fn send_sequence_reset(&mut self, new_seq: u64, _end_seq: u64) -> Result<(), io::Error> {
        let mut builder = super::fix_parser::FixMessageBuilder::new(
            &self.config.sender_comp_id,
            &self.config.target_comp_id,
            self.outgoing_seq,
        );
        
        let now = chrono::Utc::now().format("%Y%m%d-%H:%M:%S.%f").to_string();
        let msg = format!(
            "8=FIX.4.4\x019=52\x0135=4\x0149={}\x0156={}\x0134={}\x0152={}\x01123=Y\x0136={}\x0110={:03}\x01",
            self.config.sender_comp_id,
            self.config.target_comp_id,
            self.outgoing_seq,
            now,
            new_seq,
            0u8,
        );
        
        self.outgoing_seq += 1;
        let _ = self.tx.send(msg.into_bytes()).await;
        
        self.incoming_seq = new_seq;
        Ok(())
    }

    /// Check if heartbeat timeout occurred
    pub fn check_heartbeat_timeout(&self) -> bool {
        let elapsed = self.last_msg_time.elapsed();
        let timeout = Duration::from_secs((self.config.heart_bt_sec * 2) as u64);
        elapsed > timeout
    }

    /// Send test request if no messages received
    pub async fn maybe_send_test_request(&mut self) -> Result<(), io::Error> {
        if self.check_heartbeat_timeout() && self.state == SessionState::LoggedOn {
            self.test_request_counter += 1;
            let test_req_id = format!("TEST{}", self.test_request_counter);
            
            let mut builder = super::fix_parser::FixMessageBuilder::new(
                &self.config.sender_comp_id,
                &self.config.target_comp_id,
                self.outgoing_seq,
            );
            
            let now = chrono::Utc::now().format("%Y%m%d-%H:%M:%S.%f").to_string();
            let msg = format!(
                "8=FIX.4.4\x019=58\x0135=1\x0149={}\x0156={}\x0134={}\x0152={}\x01112={}\x0110={:03}\x01",
                self.config.sender_comp_id,
                self.config.target_comp_id,
                self.outgoing_seq,
                now,
                test_req_id,
                0u8,
            );
            
            self.outgoing_seq += 1;
            let _ = self.tx.send(msg.into_bytes()).await;
            
            log::warn!("TestRequest sent due to heartbeat timeout");
        }
        Ok(())
    }

    pub fn get_state(&self) -> SessionState {
        self.state
    }

    pub fn is_logged_on(&self) -> bool {
        self.state == SessionState::LoggedOn
    }

    pub fn shutdown(&mut self) {
        self.shutdown = true;
        self.state = SessionState::LogoutSent;
    }
}

// Additional tags needed
impl tags {
    pub const BEGIN_SEQ_NO: u32 = 7;
    pub const END_SEQ_NO: u32 = 16;
    pub const NEW_SEQ_NUM: u32 = 36;
    pub const GAP_FILL_FLAG: u32 = 123;
    pub const TEST_REQ_ID: u32 = 112;
}

#[cfg(target_os = "linux")]
extern crate libc;
