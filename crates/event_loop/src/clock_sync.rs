//! NTP and PTP (Precision Time Protocol) Client for Microsecond Clock Synchronization
//! Synchronizes local clock with exchange server time for timestamp-dependent strategies.

use std::sync::atomic::{AtomicBool, AtomicI64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use std::thread;

use log::{debug, error, info, warn};

/// Clock synchronization result
#[derive(Debug, Clone, Copy)]
pub struct ClockSyncResult {
    /// Offset from true time in nanoseconds (positive = local is ahead)
    pub offset_ns: i64,
    /// Round-trip time in nanoseconds
    pub rtt_ns: u64,
    /// Estimated jitter in nanoseconds
    pub jitter_ns: u64,
    /// Number of samples used
    pub sample_count: u32,
    /// Timestamp of sync
    pub sync_timestamp_ns: u64,
}

impl ClockSyncResult {
    pub fn is_accurate(&self, max_offset_ns: i64) -> bool {
        self.offset_ns.abs() <= max_offset_ns && self.jitter_ns < 1_000_000 // < 1ms jitter
    }
}

/// NTP/PTP client configuration
#[derive(Debug, Clone)]
pub struct ClockSyncConfig {
    /// NTP servers to query
    pub ntp_servers: Vec<String>,
    /// Exchange timestamp endpoint (for direct exchange time)
    pub exchange_time_endpoint: Option<String>,
    /// Sync interval in seconds
    pub sync_interval_sec: u64,
    /// Maximum acceptable offset in nanoseconds
    pub max_offset_ns: i64,
    /// Number of samples per sync
    pub samples_per_sync: u32,
    /// Enable PTP (requires hardware support)
    pub enable_ptp: bool,
}

impl Default for ClockSyncConfig {
    fn default() -> Self {
        Self {
            ntp_servers: vec![
                "pool.ntp.org".to_string(),
                "time.google.com".to_string(),
                "time.cloudflare.com".to_string(),
            ],
            exchange_time_endpoint: None,
            sync_interval_sec: 60,
            max_offset_ns: 100_000, // 100 microseconds
            samples_per_sync: 8,
            enable_ptp: false,
        }
    }
}

/// Clock synchronizer maintaining offset state
pub struct ClockSynchronizer {
    config: ClockSyncConfig,
    current_offset_ns: Arc<AtomicI64>,
    current_jitter_ns: Arc<AtomicI64>,
    last_sync_timestamp_ns: Arc<AtomicU64>,
    is_running: Arc<AtomicBool>,
    sync_count: Arc<AtomicU64>,
}

impl ClockSynchronizer {
    pub fn new(config: ClockSyncConfig) -> Self {
        Self {
            config,
            current_offset_ns: Arc::new(AtomicI64::new(0)),
            current_jitter_ns: Arc::new(AtomicI64::new(0)),
            last_sync_timestamp_ns: Arc::new(AtomicU64::new(0)),
            is_running: Arc::new(AtomicBool::new(false)),
            sync_count: Arc::new(AtomicU64::new(0)),
        }
    }

    /// Get current time offset in nanoseconds
    pub fn get_offset_ns(&self) -> i64 {
        self.current_offset_ns.load(Ordering::Acquire)
    }

    /// Get current jitter estimate in nanoseconds
    pub fn get_jitter_ns(&self) -> i64 {
        self.current_jitter_ns.load(Ordering::Acquire)
    }

    /// Get synchronized timestamp in nanoseconds since epoch
    pub fn get_synced_timestamp_ns(&self) -> u64 {
        let local_ns = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64;
        
        let offset = self.current_offset_ns.load(Ordering::Acquire);
        if offset > 0 {
            local_ns.saturating_sub(offset as u64)
        } else {
            local_ns.saturating_add((-offset) as u64)
        }
    }

    /// Perform a single sync operation
    pub fn sync_once(&self) -> Result<ClockSyncResult, String> {
        let mut offsets: Vec<i64> = Vec::with_capacity(self.config.samples_per_sync as usize);
        let mut rtts: Vec<u64> = Vec::with_capacity(self.config.samples_per_sync as usize);

        // Query NTP servers
        for server in &self.config.ntp_servers {
            match self.query_ntp(server) {
                Ok((offset, rtt)) => {
                    offsets.push(offset);
                    rtts.push(rtt);
                    debug!("NTP {} - Offset: {}ns, RTT: {}ns", server, offset, rtt);
                }
                Err(e) => {
                    warn!("Failed to query NTP {}: {}", server, e);
                }
            }
        }

        // Query exchange time if configured
        if let Some(ref endpoint) = self.config.exchange_time_endpoint {
            match self.query_exchange_time(endpoint) {
                Ok((offset, rtt)) => {
                    offsets.push(offset);
                    rtts.push(rtt);
                    debug!("Exchange {} - Offset: {}ns, RTT: {}ns", endpoint, offset, rtt);
                }
                Err(e) => {
                    warn!("Failed to query exchange time {}: {}", endpoint, e);
                }
            }
        }

        if offsets.is_empty() {
            return Err("No successful time samples".to_string());
        }

        // Calculate median offset (more robust than mean)
        offsets.sort();
        let median_offset = if offsets.len() % 2 == 0 {
            (offsets[offsets.len() / 2 - 1] + offsets[offsets.len() / 2]) / 2
        } else {
            offsets[offsets.len() / 2]
        };

        // Calculate jitter as standard deviation
        let mean_offset = offsets.iter().sum::<i64>() / offsets.len() as i64;
        let variance = offsets.iter()
            .map(|&x| (x - mean_offset).pow(2) as u64)
            .sum::<u64>() / offsets.len() as u64;
        let jitter = (variance as f64).sqrt() as u64;

        let avg_rtt = rtts.iter().sum::<u64>() / rtts.len() as u64;
        let sync_timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64;

        let result = ClockSyncResult {
            offset_ns: median_offset,
            rtt_ns: avg_rtt,
            jitter_ns: jitter,
            sample_count: offsets.len() as u32,
            sync_timestamp_ns: sync_timestamp,
        };

        // Update state
        self.current_offset_ns.store(median_offset, Ordering::Release);
        self.current_jitter_ns.store(jitter as i64, Ordering::Release);
        self.last_sync_timestamp_ns.store(sync_timestamp, Ordering::Release);
        self.sync_count.fetch_add(1, Ordering::Relaxed);

        Ok(result)
    }

    /// Query NTP server (simplified implementation)
    fn query_ntp(&self, _server: &str) -> Result<(i64, u64), String> {
        // In production, use `sntpc` or `chrono` crate for actual NTP
        // This is a stub that simulates NTP behavior
        
        // Simulate network latency
        thread::sleep(Duration::from_millis(10));
        
        let rtt_ns = 20_000_000u64; // 20ms simulated RTT
        let offset_ns = 500_000i64; // 500μs simulated offset
        
        Ok((offset_ns, rtt_ns))
    }

    /// Query exchange time endpoint
    fn query_exchange_time(&self, _endpoint: &str) -> Result<(i64, u64), String> {
        // In production, make HTTP request to exchange's time API
        // e.g., Binance: GET /api/v3/time
        
        thread::sleep(Duration::from_millis(5));
        
        let rtt_ns = 10_000_000u64; // 10ms simulated RTT
        let offset_ns = 100_000i64; // 100μs simulated offset
        
        Ok((offset_ns, rtt_ns))
    }

    /// Start continuous synchronization loop
    pub fn start_sync_loop(&self) {
        self.is_running.store(true, Ordering::SeqCst);
        info!("Clock synchronizer started");

        let is_running = self.is_running.clone();
        let sync_interval = Duration::from_secs(self.config.sync_interval_sec);

        thread::spawn(move || {
            while is_running.load(Ordering::SeqCst) {
                match self.sync_once() {
                    Ok(result) => {
                        info!(
                            "Clock sync complete - Offset: {}ns, Jitter: {}ns, Samples: {}",
                            result.offset_ns, result.jitter_ns, result.sample_count
                        );

                        if !result.is_accurate(self.config.max_offset_ns) {
                            warn!("Clock sync accuracy degraded!");
                        }
                    }
                    Err(e) => {
                        error!("Clock sync failed: {}", e);
                    }
                }

                // Sleep with early exit capability
                let start = Instant::now();
                while is_running.load(Ordering::SeqCst) && start.elapsed() < sync_interval {
                    thread::sleep(Duration::from_millis(100));
                }
            }

            info!("Clock synchronizer stopped");
        });
    }

    /// Stop the synchronization loop
    pub fn stop(&self) {
        self.is_running.store(false, Ordering::SeqCst);
        info!("Stopping clock synchronizer...");
    }

    /// Get synchronization statistics
    pub fn get_stats(&self) -> ClockSyncStats {
        ClockSyncStats {
            offset_ns: self.current_offset_ns.load(Ordering::Acquire),
            jitter_ns: self.current_jitter_ns.load(Ordering::Acquire) as u64,
            sync_count: self.sync_count.load(Ordering::Relaxed),
            last_sync_ns: self.last_sync_timestamp_ns.load(Ordering::Acquire),
            is_running: self.is_running.load(Ordering::Acquire),
        }
    }
}

/// Clock synchronization statistics
#[derive(Debug, Clone)]
pub struct ClockSyncStats {
    pub offset_ns: i64,
    pub jitter_ns: u64,
    pub sync_count: u64,
    pub last_sync_ns: u64,
    pub is_running: bool,
}

impl Default for ClockSynchronizer {
    fn default() -> Self {
        Self::new(ClockSyncConfig::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_clock_sync_basic() {
        let config = ClockSyncConfig {
            ntp_servers: vec!["pool.ntp.org".to_string()],
            exchange_time_endpoint: None,
            sync_interval_sec: 60,
            max_offset_ns: 100_000,
            samples_per_sync: 4,
            enable_ptp: false,
        };

        let sync = ClockSynchronizer::new(config);
        
        // Initial offset should be 0
        assert_eq!(sync.get_offset_ns(), 0);

        // Perform sync
        let result = sync.sync_once();
        assert!(result.is_ok());

        let result = result.unwrap();
        assert!(result.sample_count > 0);
        assert!(result.jitter_ns < 100_000_000); // < 100ms jitter
    }

    #[test]
    fn test_synced_timestamp() {
        let sync = ClockSynchronizer::default();
        
        let local_ts = sync.get_synced_timestamp_ns();
        assert!(local_ts > 0);
        
        // Should be close to actual time (within 1 second)
        let expected_ts = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64;
        
        let diff = local_ts.abs_diff(expected_ts);
        assert!(diff < 1_000_000_000); // < 1 second
    }
}
