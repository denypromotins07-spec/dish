//! Backtest Event Streamer: High-speed replay engine for historical backtest events.
//! Streams to UI at variable speeds (1x, 10x, 100x, 1000x) without blocking live trading.
//! Uses non-blocking channels and fixed-size buffers.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use tokio::sync::{broadcast, mpsc};

/// Replay speed multiplier
#[derive(Debug, Clone, Copy)]
pub enum ReplaySpeed {
    X1,
    X10,
    X100,
    X1000,
    Custom(u32),
}

impl ReplaySpeed {
    pub fn delay_ms(&self) -> u64 {
        match self {
            ReplaySpeed::X1 => 100,
            ReplaySpeed::X10 => 10,
            ReplaySpeed::X100 => 1,
            ReplaySpeed::X1000 => 0,
            ReplaySpeed::Custom(multiplier) => {
                if *multiplier >= 1000 {
                    0
                } else {
                    100 / multiplier
                }
            }
        }
    }
}

/// Backtest event types for UI visualization
#[derive(Debug, Clone)]
pub enum BacktestEvent {
    Tick {
        timestamp_us: u64,
        symbol_id: u32,
        bid: f64,
        ask: f64,
        last: f64,
    },
    OrderFilled {
        order_id: u64,
        symbol_id: u32,
        side: u8, // 0=buy, 1=sell
        price: f64,
        quantity: f64,
        pnl: f64,
    },
    SignalGenerated {
        signal_id: u32,
        symbol_id: u32,
        signal_type: u8,
        strength: f64,
    },
    EquityUpdate {
        timestamp_us: u64,
        total_equity: f64,
        unrealized_pnl: f64,
        realized_pnl: f64,
    },
    DrawdownAlert {
        current_dd: f64,
        max_dd: f64,
        threshold: f64,
    },
}

/// Lock-free backtest event streamer
pub struct BacktestEventStreamer {
    /// Broadcast channel for UI clients
    tx: broadcast::Sender<BacktestEvent>,
    /// Control channel for speed changes
    speed_tx: mpsc::Sender<ReplaySpeed>,
    speed_rx: parking_lot::Mutex<Option<mpsc::Receiver<ReplaySpeed>>>,
    /// State flags
    is_playing: AtomicBool,
    is_paused: AtomicBool,
    current_position: AtomicU64,
    total_events: AtomicU64,
    /// Current speed
    current_speed: parking_lot::RwLock<ReplaySpeed>,
    /// Pre-allocated event buffer (circular)
    event_buffer: parking_lot::RwLock<Vec<BacktestEvent>>,
    buffer_capacity: usize,
}

impl BacktestEventStreamer {
    pub fn new(buffer_capacity: usize) -> Self {
        let (tx, _) = broadcast::channel(4096);
        let (speed_tx, speed_rx) = mpsc::channel(16);

        Self {
            tx,
            speed_tx,
            speed_rx: parking_lot::Mutex::new(Some(speed_rx)),
            is_playing: AtomicBool::new(false),
            is_paused: AtomicBool::new(false),
            current_position: AtomicU64::new(0),
            total_events: AtomicU64::new(0),
            current_speed: parking_lot::RwLock::new(ReplaySpeed::X100),
            event_buffer: parking_lot::RwLock::new(Vec::with_capacity(buffer_capacity)),
            buffer_capacity,
        }
    }

    /// Load events into the buffer (pre-sorted by timestamp)
    pub fn load_events(&self, events: Vec<BacktestEvent>) {
        let mut buffer = self.event_buffer.write();
        buffer.clear();
        buffer.extend(events.iter().take(self.buffer_capacity));
        self.total_events.store(buffer.len() as u64, Ordering::Release);
        self.current_position.store(0, Ordering::Release);
    }

    /// Start replay
    pub fn play(&self) {
        self.is_playing.store(true, Ordering::Release);
        self.is_paused.store(false, Ordering::Release);
    }

    /// Pause replay
    pub fn pause(&self) {
        self.is_paused.store(true, Ordering::Release);
    }

    /// Stop replay
    pub fn stop(&self) {
        self.is_playing.store(false, Ordering::Release);
        self.is_paused.store(false, Ordering::Release);
        self.current_position.store(0, Ordering::Release);
    }

    /// Change replay speed
    pub fn set_speed(&self, speed: ReplaySpeed) {
        *self.current_speed.write() = speed;
        let _ = self.speed_tx.try_send(speed);
    }

    /// Seek to a specific position
    pub fn seek(&self, position: u64) {
        let pos = position.min(self.total_events.load(Ordering::Acquire));
        self.current_position.store(pos, Ordering::Release);
    }

    /// Run the replay loop (call from async task)
    pub async fn run_replay_loop(&self) {
        // Take the receiver from the mutex
        let mut speed_rx = self.speed_rx.lock().take().unwrap();

        while self.is_playing.load(Ordering::Acquire) {
            if self.is_paused.load(Ordering::Acquire) {
                tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;
                continue;
            }

            let pos = self.current_position.load(Ordering::Acquire);
            let total = self.total_events.load(Ordering::Acquire);

            if pos >= total {
                // End of replay
                self.stop();
                break;
            }

            // Get current speed
            let speed = *self.current_speed.read();

            // Emit next event
            let buffer = self.event_buffer.read();
            if let Some(event) = buffer.get(pos as usize) {
                let _ = self.tx.send(event.clone());
                drop(buffer);

                self.current_position.fetch_add(1, Ordering::AcqRel);
            } else {
                drop(buffer);
            }

            // Delay based on speed
            let delay = speed.delay_ms();
            if delay > 0 {
                tokio::time::sleep(tokio::time::Duration::from_millis(delay)).await;
            } else {
                // At max speed, yield to runtime
                tokio::task::yield_now().await;
            }
        }
    }

    /// Subscribe to replay events
    pub fn subscribe(&self) -> broadcast::Receiver<BacktestEvent> {
        self.tx.subscribe()
    }

    /// Get current state
    pub fn get_state(&self) -> (u64, u64, bool, bool) {
        (
            self.current_position.load(Ordering::Acquire),
            self.total_events.load(Ordering::Acquire),
            self.is_playing.load(Ordering::Acquire),
            self.is_paused.load(Ordering::Acquire),
        )
    }

    /// Check if replay is complete
    pub fn is_complete(&self) -> bool {
        self.current_position.load(Ordering::Acquire) >= self.total_events.load(Ordering::Acquire)
    }
}

impl Default for BacktestEventStreamer {
    fn default() -> Self {
        Self::new(100000)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_replay_streamer() {
        let streamer = Arc::new(BacktestEventStreamer::new(1000));

        // Load sample events
        let events: Vec<BacktestEvent> = (0..100)
            .map(|i| BacktestEvent::Tick {
                timestamp_us: i * 1000,
                symbol_id: 1,
                bid: 100.0 + i as f64 * 0.01,
                ask: 100.05 + i as f64 * 0.01,
                last: 100.02 + i as f64 * 0.01,
            })
            .collect();

        streamer.load_events(events);
        streamer.play();

        // Subscribe
        let mut rx = streamer.subscribe();

        // Run for a bit
        let s = Arc::clone(&streamer);
        tokio::spawn(async move {
            s.run_replay_loop().await;
        });

        // Receive some events
        let mut count = 0;
        while count < 10 {
            if let Ok(_event) = rx.recv().await {
                count += 1;
            }
        }

        let (pos, total, playing, _paused) = streamer.get_state();
        assert!(playing);
        assert!(pos >= 10);
        assert_eq!(total, 100);

        streamer.pause();
        tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;
        
        let (_, _, playing, paused) = streamer.get_state();
        assert!(!playing || paused);
    }
}
