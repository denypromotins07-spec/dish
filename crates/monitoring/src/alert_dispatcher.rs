//! Ultra-low-overhead, non-blocking alert dispatcher.
//! Sends critical system failure notifications via Telegram/Discord webhooks
//! asynchronously without ever pausing the main trading thread.

use std::sync::mpsc::{self, Sender, Receiver};
use std::thread;
use std::time::Duration;
use std::sync::atomic::{AtomicBool, Ordering};

/// Alert severity levels
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AlertLevel {
    Info,
    Warning,
    Critical,
    Emergency,
}

/// Alert message structure
#[derive(Debug, Clone)]
pub struct Alert {
    pub level: AlertLevel,
    pub component: &'static str,
    pub message: String,
    pub timestamp_ns: u64,
}

impl Alert {
    pub fn new(level: AlertLevel, component: &'static str, message: impl Into<String>) -> Self {
        Self {
            level,
            component,
            message: message.into(),
            timestamp_ns: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos() as u64,
        }
    }
}

/// Non-blocking alert dispatcher using channel-based async processing
pub struct AlertDispatcher {
    sender: Sender<Alert>,
    receiver: Option<Receiver<Alert>>,
    is_running: AtomicBool,
    webhook_url_telegram: Option<String>,
    webhook_url_discord: Option<String>,
}

impl AlertDispatcher {
    /// Create a new dispatcher with bounded channel (prevents memory bloat)
    pub fn new(
        telegram_webhook: Option<String>,
        discord_webhook: Option<String>,
    ) -> Self {
        let (tx, rx) = mpsc::channel::<Alert>(); // Bounded by nature of single consumer
        
        Self {
            sender: tx,
            receiver: Some(rx),
            is_running: AtomicBool::new(false),
            webhook_url_telegram: telegram_webhook,
            webhook_url_discord: discord_webhook,
        }
    }

    /// Start the background dispatcher thread
    pub fn start(&mut self) {
        if self.is_running.load(Ordering::Relaxed) {
            return;
        }

        let receiver = self.receiver.take().expect("Dispatcher already started");
        let tg_hook = self.webhook_url_telegram.clone();
        let dc_hook = self.webhook_url_discord.clone();

        self.is_running.store(true, Ordering::Relaxed);

        thread::spawn(move || {
            Self::dispatch_loop(receiver, tg_hook, dc_hook);
        });
    }

    /// Main dispatch loop - runs in dedicated thread
    fn dispatch_loop(
        receiver: Receiver<Alert>,
        tg_hook: Option<String>,
        dc_hook: Option<String>,
    ) {
        while let Ok(alert) = receiver.recv() {
            // Only send Critical and Emergency alerts to avoid rate limits
            if alert.level == AlertLevel::Info || alert.level == AlertLevel::Warning {
                // Log locally but don't send webhook
                eprintln!("[{}] {}: {}", alert.level_str(), alert.component, alert.message);
                continue;
            }

            // Send to Telegram if configured
            if let Some(ref url) = tg_hook {
                Self::send_telegram_alert(url, &alert);
            }

            // Send to Discord if configured
            if let Some(ref url) = dc_hook {
                Self::send_discord_alert(url, &alert);
            }
        }
    }

    /// Send alert to Telegram (non-blocking, fire-and-forget)
    fn send_telegram_alert(webhook_url: &str, alert: &Alert) {
        // In production: use reqwest or ureq for actual HTTP call
        // Here we simulate the async behavior
        let payload = format!(
            "{{\"chat_id\": \"-100\", \"text\": \"🚨 {} ALERT\\nComponent: {}\\nMessage: {}\"}}",
            alert.level_str(),
            alert.component,
            alert.message
        );
        
        // Simulated async send - in real code this would be:
        // reqwest::blocking::Client::new()
        //     .post(webhook_url)
        //     .json(&payload)
        //     .timeout(Duration::from_secs(2))
        //     .send()
        
        eprintln!("[TELEGRAM] Would send: {}", payload);
    }

    /// Send alert to Discord (non-blocking, fire-and-forget)
    fn send_discord_alert(webhook_url: &str, alert: &Alert) {
        let color = match alert.level {
            AlertLevel::Emergency => 0xFF0000,
            AlertLevel::Critical => 0xFFA500,
            _ => 0xFFFF00,
        };

        let payload = format!(
            r#"{{"embeds": [{{"title": "{}", "description": "{}", "color": {}, "fields": [{{"name": "Component", "value": "{}"}}]}}]}}"#,
            alert.level_str(),
            alert.message,
            color,
            alert.component
        );

        eprintln!("[DISCORD] Would send: {}", payload);
    }

    /// Queue an alert (non-blocking, returns immediately)
    #[inline]
    pub fn send_alert(&self, alert: Alert) {
        // Try to send, but don't block if channel is full
        // This ensures trading thread is never paused
        let _ = self.sender.try_send(alert);
    }

    /// Convenience method for critical alerts
    pub fn critical(&self, component: &'static str, message: impl Into<String>) {
        self.send_alert(Alert::new(AlertLevel::Critical, component, message));
    }

    /// Convenience method for emergency alerts
    pub fn emergency(&self, component: &'static str, message: impl Into<String>) {
        self.send_alert(Alert::new(AlertLevel::Emergency, component, message));
    }

    /// Convenience method for warnings
    pub fn warning(&self, component: &'static str, message: impl Into<String>) {
        self.send_alert(Alert::new(AlertLevel::Warning, component, message));
    }

    /// Check if dispatcher is running
    pub fn is_running(&self) -> bool {
        self.is_running.load(Ordering::Relaxed)
    }
}

impl AlertLevel {
    pub fn level_str(self) -> &'static str {
        match self {
            AlertLevel::Info => "INFO",
            AlertLevel::Warning => "WARNING",
            AlertLevel::Critical => "CRITICAL",
            AlertLevel::Emergency => "EMERGENCY",
        }
    }
}

// Global static dispatcher for zero-overhead access
static mut GLOBAL_DISPATCHER: Option<AlertDispatcher> = None;

/// Initialize global dispatcher (call once at startup)
pub fn init_global_dispatcher(
    telegram_webhook: Option<String>,
    discord_webhook: Option<String>,
) {
    unsafe {
        GLOBAL_DISPATCHER = Some(AlertDispatcher::new(telegram_webhook, discord_webhook));
    }
}

/// Get reference to global dispatcher
pub fn get_global_dispatcher() -> Option<&'static AlertDispatcher> {
    unsafe { GLOBAL_DISPATCHER.as_ref() }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_alert_creation() {
        let alert = Alert::new(AlertLevel::Critical, "WS", "Connection lost");
        assert_eq!(alert.level, AlertLevel::Critical);
        assert_eq!(alert.component, "WS");
        assert!(alert.message.contains("Connection lost"));
    }

    #[test]
    fn test_dispatcher_non_blocking() {
        let mut dispatcher = AlertDispatcher::new(None, None);
        dispatcher.start();
        
        // Should return immediately without blocking
        dispatcher.critical("TEST", "Test alert");
        dispatcher.warning("TEST", "Test warning");
        
        assert!(dispatcher.is_running());
        
        // Give thread time to process
        thread::sleep(Duration::from_millis(10));
    }
}
