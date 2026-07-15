//! Custom Panic Hook and Graceful Degradation Handler
//! Logs stack traces, isolates faults, and safely flattens portfolio on critical panics.

use std::panic::{self, PanicInfo};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

use log::{error, info, warn};

/// Severity level of a panic
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PanicSeverity {
    /// Non-critical: log and isolate
    NonCritical,
    /// Critical: flatten portfolio and halt
    Critical,
}

/// Panic context information
#[derive(Debug, Clone)]
pub struct PanicContext {
    pub thread_name: String,
    pub location: Option<String>,
    pub message: Option<String>,
    pub severity: PanicSeverity,
    pub timestamp_ns: u64,
    pub subsystem: String,
}

impl PanicContext {
    pub fn from_panic_info(info: &PanicInfo, subsystem: &str, severity: PanicSeverity) -> Self {
        let location = info.location().map(|loc| {
            format!("{}:{}:{}", loc.file(), loc.line(), loc.column())
        });

        let message = if let Some(msg) = info.payload().downcast_ref::<&str>() {
            Some(msg.to_string())
        } else if let Some(msg) = info.payload().downcast_ref::<String>() {
            Some(msg.clone())
        } else {
            None
        };

        Self {
            thread_name: thread::current().name().unwrap_or("unknown").to_string(),
            location,
            message,
            severity,
            timestamp_ns: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos() as u64,
            subsystem: subsystem.to_string(),
        }
    }
}

/// Panic handler statistics
#[derive(Debug, Clone, Default)]
pub struct PanicStats {
    pub total_panics: usize,
    pub critical_panics: usize,
    pub non_critical_panics: usize,
    pub last_panic_timestamp_ns: u64,
    pub subsystems_affected: Vec<String>,
}

/// Graceful panic handler
pub struct GracefulPanicHandler {
    is_initialized: Arc<AtomicBool>,
    panic_count: Arc<AtomicUsize>,
    critical_panic_count: Arc<AtomicUsize>,
    stats: Arc<parking_lot::RwLock<PanicStats>>,
    shutdown_callback: Option<Arc<dyn Fn() + Send + Sync>>,
    portfolio_flatten_callback: Option<Arc<dyn Fn() + Send + Sync>>,
}

impl GracefulPanicHandler {
    pub fn new() -> Self {
        Self {
            is_initialized: Arc::new(AtomicBool::new(false)),
            panic_count: Arc::new(AtomicUsize::new(0)),
            critical_panic_count: Arc::new(AtomicUsize::new(0)),
            stats: Arc::new(parking_lot::RwLock::new(PanicStats::default())),
            shutdown_callback: None,
            portfolio_flatten_callback: None,
        }
    }

    /// Set callback for critical panic shutdown
    pub fn set_shutdown_callback<F>(&mut self, callback: F)
    where
        F: Fn() + Send + Sync + 'static,
    {
        self.shutdown_callback = Some(Arc::new(callback));
    }

    /// Set callback to flatten portfolio on critical panic
    pub fn set_portfolio_flatten_callback<F>(&mut self, callback: F)
    where
        F: Fn() + Send + Sync + 'static,
    {
        self.portfolio_flatten_callback = Some(Arc::new(callback));
    }

    /// Install the custom panic hook
    pub fn install(&self, default_subsystem: &str) {
        if self.is_initialized.swap(true, Ordering::SeqCst) {
            warn!("Panic handler already initialized");
            return;
        }

        let panic_count = self.panic_count.clone();
        let critical_count = self.critical_panic_count.clone();
        let stats = self.stats.clone();
        let shutdown_cb = self.shutdown_callback.clone();
        let flatten_cb = self.portfolio_flatten_callback.clone();
        let subsystem = default_subsystem.to_string();

        panic::set_hook(Box::new(move |info: &PanicInfo| {
            let count = panic_count.fetch_add(1, Ordering::SeqCst);
            let timestamp = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos() as u64;

            // Parse panic context
            let ctx = PanicContext::from_panic_info(info, &subsystem, PanicSeverity::NonCritical);

            // Build error message
            let mut error_msg = String::new();
            error_msg.push_str(&format!("PANIC in thread '{}'", ctx.thread_name));
            if let Some(ref loc) = ctx.location {
                error_msg.push_str(&format!(" at {}", loc));
            }
            if let Some(ref msg) = ctx.message {
                error_msg.push_str(&format!(" - {}", msg));
            }

            // Get backtrace (in production, use backtrace crate)
            let backtrace = std::backtrace::Backtrace::capture();

            error!("{}\nBacktrace:\n{}", error_msg, backtrace);

            // Update stats
            {
                let mut stats_guard = stats.write();
                stats_guard.total_panics = count + 1;
                stats_guard.last_panic_timestamp_ns = timestamp;
                if !stats_guard.subsystems_affected.contains(&ctx.subsystem) {
                    stats_guard.subsystems_affected.push(ctx.subsystem.clone());
                }
            }

            // Handle based on severity
            // In production, you'd analyze the panic to determine severity
            // For now, we treat all panics as potentially critical
            let is_critical = ctx.severity == PanicSeverity::Critical 
                || ctx.subsystem.contains("execution")
                || ctx.subsystem.contains("risk");

            if is_critical {
                critical_count.fetch_add(1, Ordering::SeqCst);

                {
                    let mut stats_guard = stats.write();
                    stats_guard.critical_panics += 1;
                }

                error!("CRITICAL PANIC DETECTED - Initiating emergency shutdown...");

                // Flatten portfolio first
                if let Some(ref cb) = flatten_cb {
                    warn!("Flattening portfolio...");
                    cb();
                }

                // Then shutdown
                if let Some(ref cb) = shutdown_cb {
                    warn!("Executing shutdown callback...");
                    cb();
                }

                // Force exit after brief delay to ensure logs are flushed
                thread::sleep(Duration::from_millis(100));
                std::process::exit(1);
            } else {
                // Non-critical: just log and continue
                warn!("Non-critical panic in subsystem '{}' - continuing operation", ctx.subsystem);
            }
        }));

        info!("Graceful panic handler installed");
    }

    /// Get current panic statistics
    pub fn get_stats(&self) -> PanicStats {
        self.stats.read().clone()
    }

    /// Check if any critical panics have occurred
    pub fn has_critical_panics(&self) -> bool {
        self.critical_panic_count.load(Ordering::Acquire) > 0
    }

    /// Reset statistics (for testing)
    pub fn reset_stats(&self) {
        *self.stats.write() = PanicStats::default();
        self.panic_count.store(0, Ordering::SeqCst);
        self.critical_panic_count.store(0, Ordering::SeqCst);
    }
}

impl Default for GracefulPanicHandler {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for GracefulPanicHandler {
    fn drop(&mut self) {
        // Restore default panic hook
        panic::set_hook(Box::new(|_| {}));
    }
}

/// Test helper to trigger a panic in a controlled way
#[cfg(test)]
pub fn trigger_test_panic() {
    panic!("Test panic triggered");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_panic_handler_initialization() {
        let handler = GracefulPanicHandler::new();
        handler.install("test_subsystem");
        
        assert!(handler.is_initialized.load(Ordering::SeqCst));
        
        let stats = handler.get_stats();
        assert_eq!(stats.total_panics, 0);
    }

    #[test]
    fn test_panic_stats_tracking() {
        let handler = GracefulPanicHandler::new();
        
        // Manually increment counters for testing
        handler.panic_count.fetch_add(3, Ordering::SeqCst);
        handler.critical_panic_count.fetch_add(1, Ordering::SeqCst);
        
        let stats = handler.get_stats();
        assert_eq!(stats.total_panics, 3);
        assert_eq!(stats.critical_panics, 1);
        assert!(handler.has_critical_panics());
    }
}
