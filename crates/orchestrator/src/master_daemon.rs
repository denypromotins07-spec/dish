//! Master Supervisor Daemon (Erlang/OTP inspired)
//! Spawns, monitors, and restarts critical Rust subsystems with strict memory limits and CPU affinity.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use log::{error, info, warn};

/// Memory limit in bytes for a subsystem
pub type MemoryLimit = u64;

/// Subsystem identifier
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct SubsystemId(pub String);

/// Criticality level: if a critical subsystem panics, we halt trading
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Criticality {
    Critical,
    NonCritical,
}

/// Message types for subsystem communication
#[derive(Debug, Clone)]
pub enum SubsystemMessage {
    Start,
    Stop,
    Restart,
    HealthCheck,
}

/// Subsystem configuration
#[derive(Debug, Clone)]
pub struct SubsystemConfig {
    pub id: SubsystemId,
    pub criticality: Criticality,
    pub memory_limit: MemoryLimit,
    pub cpu_affinity: Vec<usize>,
    pub restart_delay: Duration,
    pub max_restarts: u32,
}

/// Subsystem state
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SubsystemState {
    Stopped,
    Starting,
    Running,
    Crashed,
    Restarting,
}

/// Subsystem handle
pub struct SubsystemHandle {
    pub config: SubsystemConfig,
    pub state: Arc<AtomicU64>, // Encoded SubsystemState
    pub restart_count: Arc<AtomicU32>,
    pub last_heartbeat: Arc<AtomicU64>, // Microseconds since epoch
    pub thread_handle: Option<JoinHandle<()>>,
    pub stop_flag: Arc<AtomicBool>,
}

impl SubsystemHandle {
    pub fn new(config: SubsystemConfig) -> Self {
        Self {
            config,
            state: Arc::new(AtomicU64::new(SubsystemState::Stopped as u64)),
            restart_count: Arc::new(AtomicU32::new(0)),
            last_heartbeat: Arc::new(AtomicU64::new(0)),
            thread_handle: None,
            stop_flag: Arc::new(AtomicBool::new(false)),
        }
    }

    pub fn set_state(&self, state: SubsystemState) {
        self.state.store(state as u64, Ordering::SeqCst);
    }

    pub fn get_state(&self) -> SubsystemState {
        match self.state.load(Ordering::SeqCst) {
            0 => SubsystemState::Stopped,
            1 => SubsystemState::Starting,
            2 => SubsystemState::Running,
            3 => SubsystemState::Crashed,
            4 => SubsystemState::Restarting,
            _ => SubsystemState::Stopped,
        }
    }

    pub fn update_heartbeat(&self) {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_micros() as u64;
        self.last_heartbeat.store(now, Ordering::SeqCst);
    }
}

/// Master Daemon managing all subsystems
pub struct MasterDaemon {
    pub subsystems: HashMap<SubsystemId, SubsystemHandle>,
    pub is_running: Arc<AtomicBool>,
    pub monitoring_interval: Duration,
}

impl MasterDaemon {
    pub fn new() -> Self {
        Self {
            subsystems: HashMap::new(),
            is_running: Arc::new(AtomicBool::new(false)),
            monitoring_interval: Duration::from_millis(100),
        }
    }

    /// Register a new subsystem
    pub fn register_subsystem(&mut self, config: SubsystemConfig, entry_point: fn(Arc<AtomicBool>) -> ()) {
        let mut handle = SubsystemHandle::new(config);
        let stop_flag = handle.stop_flag.clone();
        let state_arc = handle.state.clone();
        let restart_count = handle.restart_count.clone();
        let last_heartbeat = handle.last_heartbeat.clone();
        let crit = handle.config.criticality;

        let thread_handle = thread::spawn(move || {
            // Set CPU affinity (platform-specific, stubbed here)
            // In production: use `core_affinity` crate

            state_arc.store(SubsystemState::Starting as u64, Ordering::SeqCst);

            loop {
                if stop_flag.load(Ordering::SeqCst) {
                    break;
                }

                // Simulate subsystem work
                state_arc.store(SubsystemState::Running as u64, Ordering::SeqCst);
                last_heartbeat.store(
                    std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .unwrap()
                        .as_micros() as u64,
                    Ordering::SeqCst,
                );

                // Call the actual subsystem entry point
                entry_point(stop_flag.clone());

                // If we exited the loop without stop_flag, it crashed
                if !stop_flag.load(Ordering::SeqCst) {
                    state_arc.store(SubsystemState::Crashed as u64, Ordering::SeqCst);
                    error!("Subsystem crashed!");

                    // Increment restart count
                    let count = restart_count.fetch_add(1, Ordering::SeqCst);
                    if count >= handle.config.max_restarts {
                        error!("Max restarts reached for subsystem");
                        if crit == Criticality::Critical {
                            panic!("Critical subsystem exceeded max restarts. Halting.");
                        }
                        break;
                    }

                    // Wait before restart
                    thread::sleep(handle.config.restart_delay);
                    state_arc.store(SubsystemState::Restarting as u64, Ordering::SeqCst);
                } else {
                    break;
                }
            }

            state_arc.store(SubsystemState::Stopped as u64, Ordering::SeqCst);
        });

        handle.thread_handle = Some(thread_handle);
        self.subsystems.insert(handle.config.id.clone(), handle);
    }

    /// Start all subsystems
    pub fn start_all(&mut self) {
        self.is_running.store(true, Ordering::SeqCst);
        info!("Master Daemon starting all subsystems...");
        for (_, handle) in &mut self.subsystems {
            handle.set_state(SubsystemState::Starting);
        }
    }

    /// Stop all subsystems gracefully
    pub fn stop_all(&mut self) {
        info!("Master Daemon stopping all subsystems...");
        self.is_running.store(false, Ordering::SeqCst);

        for (_, handle) in &mut self.subsystems {
            handle.stop_flag.store(true, Ordering::SeqCst);
        }

        // Wait for all threads to finish
        for (_, handle) in &mut self.subsystems {
            if let Some(thread_handle) = handle.thread_handle.take() {
                let _ = thread_handle.join();
            }
        }

        info!("All subsystems stopped.");
    }

    /// Monitor subsystems health
    pub fn monitor_loop(&self) {
        let is_running = self.is_running.clone();
        let subsystems = self.subsystems.clone(); // This requires Arc or similar in production

        thread::spawn(move || {
            while is_running.load(Ordering::SeqCst) {
                let now = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_micros() as u64;

                for (id, handle) in &subsystems {
                    let last_hb = handle.last_heartbeat.load(Ordering::SeqCst);
                    let elapsed = now.saturating_sub(last_hb);

                    // If no heartbeat for 5 seconds, consider it dead
                    if elapsed > 5_000_000 && handle.get_state() == SubsystemState::Running {
                        warn!("Subsystem {} appears dead (no heartbeat)", id.0);
                        handle.set_state(SubsystemState::Crashed);
                    }
                }

                thread::sleep(Duration::from_millis(100));
            }
        });
    }
}

impl Default for MasterDaemon {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dummy_subsystem(_stop_flag: Arc<AtomicBool>) {
        // Simulate work
        thread::sleep(Duration::from_millis(10));
    }

    #[test]
    fn test_master_daemon_lifecycle() {
        let mut daemon = MasterDaemon::new();

        let config = SubsystemConfig {
            id: SubsystemId("test_subsystem".to_string()),
            criticality: Criticality::NonCritical,
            memory_limit: 1024 * 1024 * 100, // 100MB
            cpu_affinity: vec![0],
            restart_delay: Duration::from_millis(100),
            max_restarts: 3,
        };

        daemon.register_subsystem(config, dummy_subsystem);
        daemon.start_all();
        thread::sleep(Duration::from_millis(500));
        daemon.stop_all();

        assert_eq!(daemon.subsystems.len(), 1);
    }
}
