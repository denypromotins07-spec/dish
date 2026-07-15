//! Deep health check daemon for continuous system verification.
//! Verifies WebSocket heartbeats, LMDB database locks, and thread liveness.
//! Instantly flags zombie threads or deadlocks without blocking trading.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use std::thread;
use std::collections::HashMap;

/// Health status enumeration
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum HealthStatus {
    Healthy,
    Degraded,
    Critical,
    Dead,
}

/// Component health tracker
pub struct ComponentHealth {
    pub last_heartbeat: AtomicU64, // Nanoseconds since epoch
    pub is_alive: AtomicBool,
    pub failure_count: AtomicU64,
    pub component_id: u32,
}

impl ComponentHealth {
    pub fn new(component_id: u32) -> Self {
        Self {
            last_heartbeat: AtomicU64::new(0),
            is_alive: AtomicBool::new(false),
            failure_count: AtomicU64::new(0),
            component_id,
        }
    }

    #[inline]
    pub fn heartbeat(&self) {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64;
        self.last_heartbeat.store(now, Ordering::Relaxed);
        self.is_alive.store(true, Ordering::Relaxed);
    }

    #[inline]
    pub fn mark_dead(&self) {
        self.is_alive.store(false, Ordering::Relaxed);
        self.failure_count.fetch_add(1, Ordering::Relaxed);
    }

    pub fn time_since_heartbeat_ns(&self) -> u64 {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64;
        let last = self.last_heartbeat.load(Ordering::Relaxed);
        if last == 0 { return u64::MAX; }
        now.saturating_sub(last)
    }
}

/// Main health check daemon
pub struct HealthCheckDaemon {
    components: Vec<Arc<ComponentHealth>>,
    ws_heartbeat_timeout_ns: u64,
    db_lock_check_interval_ns: u64,
    thread_check_interval_ns: u64,
    is_running: AtomicBool,
    critical_alert_triggered: AtomicBool,
}

impl HealthCheckDaemon {
    pub fn new() -> Self {
        Self {
            components: Vec::with_capacity(16), // Pre-allocated, bounded
            ws_heartbeat_timeout_ns: 5_000_000_000, // 5 seconds
            db_lock_check_interval_ns: 1_000_000_000, // 1 second
            thread_check_interval_ns: 2_000_000_000, // 2 seconds
            is_running: AtomicBool::new(false),
            critical_alert_triggered: AtomicBool::new(false),
        }
    }

    pub fn register_component(&mut self, component_id: u32) -> Arc<ComponentHealth> {
        let health = Arc::new(ComponentHealth::new(component_id));
        self.components.push(Arc::clone(&health));
        health
    }

    /// Check all components and return overall system health
    pub fn check_system_health(&self) -> HealthStatus {
        let mut worst_status = HealthStatus::Healthy;
        let now_ns = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64;

        for component in &self.components {
            let status = self.evaluate_component(component, now_ns);
            if status as u8 > worst_status as u8 {
                worst_status = status;
                if worst_status == HealthStatus::Dead {
                    break; // Early exit on critical failure
                }
            }
        }

        worst_status
    }

    fn evaluate_component(&self, comp: &ComponentHealth, now_ns: u64) -> HealthStatus {
        if !comp.is_alive.load(Ordering::Relaxed) {
            return HealthStatus::Dead;
        }

        let elapsed = now_ns.saturating_sub(comp.last_heartbeat.load(Ordering::Relaxed));
        
        if elapsed > self.ws_heartbeat_timeout_ns * 3 {
            return HealthStatus::Dead;
        } else if elapsed > self.ws_heartbeat_timeout_ns * 2 {
            return HealthStatus::Critical;
        } else if elapsed > self.ws_heartbeat_timeout_ns {
            return HealthStatus::Degraded;
        }

        HealthStatus::Healthy
    }

    /// Check for LMDB lock file staleness (simulated)
    pub fn check_database_locks(&self) -> Result<(), &'static str> {
        // In production: check actual LMDB lock files
        // Here we simulate by checking if any DB component is dead
        for comp in &self.components {
            if comp.component_id == 999 && !comp.is_alive.load(Ordering::Relaxed) {
                return Err("LMDB lock stale - database may be corrupted");
            }
        }
        Ok(())
    }

    /// Detect zombie threads by checking heartbeat freshness
    pub fn detect_zombie_threads(&self) -> Vec<u32> {
        let mut zombies = Vec::with_capacity(8);
        let threshold = self.ws_heartbeat_timeout_ns * 2;

        for comp in &self.components {
            if comp.time_since_heartbeat_ns() > threshold {
                zombies.push(comp.component_id);
            }
        }
        zombies
    }

    /// Start the background health monitoring loop
    pub fn start_monitoring(&self) -> thread::JoinHandle<()> {
        let components = self.components.clone();
        let timeout = self.ws_heartbeat_timeout_ns;
        
        thread::spawn(move || {
            loop {
                thread::sleep(Duration::from_millis(500));
                
                let now = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_nanos() as u64;

                for comp in &components {
                    let elapsed = now.saturating_sub(comp.last_heartbeat.load(Ordering::Relaxed));
                    if elapsed > timeout && comp.is_alive.load(Ordering::Relaxed) {
                        eprintln!("WARNING: Component {} heartbeat timeout!", comp.component_id);
                    }
                }
            }
        })
    }

    /// Get count of healthy components
    pub fn healthy_component_count(&self) -> usize {
        self.components.iter()
            .filter(|c| c.is_alive.load(Ordering::Relaxed))
            .count()
    }

    /// Get total registered components
    pub fn total_components(&self) -> usize {
        self.components.len()
    }
}

impl Default for HealthCheckDaemon {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_component_heartbeat() {
        let comp = ComponentHealth::new(1);
        assert!(!comp.is_alive.load(Ordering::Relaxed));
        
        comp.heartbeat();
        assert!(comp.is_alive.load(Ordering::Relaxed));
        assert!(comp.time_since_heartbeat_ns() < 1_000_000_000);
    }

    #[test]
    fn test_daemon_health_check() {
        let mut daemon = HealthCheckDaemon::new();
        let comp1 = daemon.register_component(1);
        let comp2 = daemon.register_component(2);
        
        comp1.heartbeat();
        comp2.heartbeat();
        
        assert_eq!(daemon.check_system_health(), HealthStatus::Healthy);
        assert_eq!(daemon.healthy_component_count(), 2);
    }
}
