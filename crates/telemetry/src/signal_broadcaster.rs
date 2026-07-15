//! Signal broadcaster for internal "Brain" state telemetry.
//! Broadcasts ML model confidence, market regime, and active strategy triggers to UI.

use std::sync::atomic::{AtomicU64, AtomicBool, Ordering};
use std::sync::Arc;
use tokio::sync::broadcast;

/// Market regime types
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
pub enum MarketRegime {
    HighVolatilityTrending,
    HighVolatilityMeanReversion,
    LowVolatilityTrending,
    LowVolatilityMeanReversion,
    Choppy,
    Unknown,
}

impl MarketRegime {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::HighVolatilityTrending => "High Volatility Trending",
            Self::HighVolatilityMeanReversion => "High Volatility Mean Reversion",
            Self::LowVolatilityTrending => "Low Volatility Trending",
            Self::LowVolatilityMeanReversion => "Low Volatility Mean Reversion",
            Self::Choppy => "Choppy",
            Self::Unknown => "Unknown",
        }
    }
}

/// Active strategy trigger
#[derive(Debug, Clone, serde::Serialize)]
pub struct StrategyTrigger {
    pub strategy_name: String,
    pub signal_type: String,
    pub confidence: f64,
    pub timestamp_ns: u64,
    pub metadata: serde_json::Value,
}

/// Brain state snapshot for UI streaming
#[derive(Debug, Clone, serde::Serialize)]
pub struct BrainStateSnapshot {
    pub timestamp_ns: u64,
    pub market_regime: &'static str,
    pub regime_confidence: f64,
    pub ml_model_confidence: f64,
    pub active_strategies: Vec<String>,
    pub active_triggers: Vec<StrategyTrigger>,
    pub risk_level: f64, // 0.0 to 1.0
    pub inference_latency_us: u64,
}

/// Atomic brain state for lock-free reads
pub struct AtomicBrainState {
    regime: AtomicU64, // Encoded as usize
    regime_confidence: AtomicU64, // Fixed-point * 1e6
    ml_confidence: AtomicU64,
    risk_level: AtomicU64,
    inference_latency_us: AtomicU64,
    last_update_ns: AtomicU64,
}

impl AtomicBrainState {
    /// Create new atomic brain state
    pub fn new() -> Self {
        Self {
            regime: AtomicU64::new(MarketRegime::Unknown as u64),
            regime_confidence: AtomicU64::new(0),
            ml_confidence: AtomicU64::new(0),
            risk_level: AtomicU64::new(0),
            inference_latency_us: AtomicU64::new(0),
            last_update_ns: AtomicU64::new(0),
        }
    }

    /// Update market regime
    pub fn set_regime(&self, regime: MarketRegime) {
        self.regime.store(regime as u64, Ordering::Relaxed);
    }

    /// Get current market regime
    pub fn get_regime(&self) -> MarketRegime {
        unsafe { std::mem::transmute::<u64, MarketRegime>(
            self.regime.load(Ordering::Relaxed)
        )}
    }

    /// Update regime confidence
    pub fn set_regime_confidence(&self, value: f64) {
        self.regime_confidence.store((value * 1e6) as u64, Ordering::Relaxed);
    }

    /// Get regime confidence
    pub fn get_regime_confidence(&self) -> f64 {
        self.regime_confidence.load(Ordering::Relaxed) as f64 / 1e6
    }

    /// Update ML model confidence
    pub fn set_ml_confidence(&self, value: f64) {
        self.ml_confidence.store((value * 1e6) as u64, Ordering::Relaxed);
    }

    /// Get ML model confidence
    pub fn get_ml_confidence(&self) -> f64 {
        self.ml_confidence.load(Ordering::Relaxed) as f64 / 1e6
    }

    /// Update risk level
    pub fn set_risk_level(&self, value: f64) {
        self.risk_level.store((value * 1e6) as u64, Ordering::Relaxed);
    }

    /// Get risk level
    pub fn get_risk_level(&self) -> f64 {
        self.risk_level.load(Ordering::Relaxed) as f64 / 1e6
    }

    /// Update inference latency
    pub fn set_inference_latency_us(&self, value: u64) {
        self.inference_latency_us.store(value, Ordering::Relaxed);
    }

    /// Get inference latency
    pub fn get_inference_latency_us(&self) -> u64 {
        self.inference_latency_us.load(Ordering::Relaxed)
    }

    /// Mark update time
    pub fn mark_update(&self) {
        self.last_update_ns.store(
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos() as u64,
            Ordering::Relaxed,
        );
    }
}

impl Default for AtomicBrainState {
    fn default() -> Self {
        Self::new()
    }
}

/// Signal broadcaster for brain state telemetry
pub struct SignalBroadcaster {
    state: Arc<AtomicBrainState>,
    tx: broadcast::Sender<BrainStateSnapshot>,
    active_strategies: Arc<tokio::sync::RwLock<Vec<String>>>,
    active_triggers: Arc<tokio::sync::RwLock<Vec<StrategyTrigger>>>,
    max_triggers: usize,
}

impl SignalBroadcaster {
    /// Create new signal broadcaster
    pub fn new(state: Arc<AtomicBrainState>, buffer_size: usize, max_triggers: usize) -> Self {
        let (tx, _rx) = broadcast::channel(buffer_size);
        
        Self {
            state,
            tx,
            active_strategies: Arc::new(tokio::sync::RwLock::new(Vec::new())),
            active_triggers: Arc::new(tokio::sync::RwLock::new(Vec::new())),
            max_triggers,
        }
    }

    /// Generate and broadcast current brain state
    pub async fn broadcast_snapshot(&self) -> Result<(), broadcast::error::SendError<BrainStateSnapshot>> {
        let now_ns = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64;

        let regime = self.state.get_regime();
        let strategies = self.active_strategies.read().await.clone();
        let triggers = self.active_triggers.read().await.clone();

        let snapshot = BrainStateSnapshot {
            timestamp_ns: now_ns,
            market_regime: regime.as_str(),
            regime_confidence: self.state.get_regime_confidence(),
            ml_model_confidence: self.state.get_ml_confidence(),
            active_strategies: strategies,
            active_triggers: triggers,
            risk_level: self.state.get_risk_level(),
            inference_latency_us: self.state.get_inference_latency_us(),
        };

        self.tx.send(snapshot)?;
        Ok(())
    }

    /// Register an active strategy
    pub async fn register_strategy(&self, name: String) {
        let mut strategies = self.active_strategies.write().await;
        if !strategies.contains(&name) {
            strategies.push(name);
        }
    }

    /// Unregister a strategy
    pub async fn unregister_strategy(&self, name: &str) {
        let mut strategies = self.active_strategies.write().await;
        if let Some(pos) = strategies.iter().position(|s| s == name) {
            strategies.remove(pos);
        }
    }

    /// Add a strategy trigger
    pub async fn add_trigger(&self, trigger: StrategyTrigger) {
        let mut triggers = self.active_triggers.write().await;
        
        // Enforce memory bound
        if triggers.len() >= self.max_triggers {
            triggers.remove(0);
        }
        
        triggers.push(trigger);
    }

    /// Clear old triggers
    pub async fn clear_old_triggers(&self, max_age_ms: u64) {
        let now_ns = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64;
        let max_age_ns = max_age_ms * 1_000_000;

        let mut triggers = self.active_triggers.write().await;
        triggers.retain(|t| now_ns - t.timestamp_ns < max_age_ns);
    }

    /// Get subscription handle
    pub fn subscribe(&self) -> broadcast::Receiver<BrainStateSnapshot> {
        self.tx.subscribe()
    }

    /// Get reference to atomic state
    pub fn state(&self) -> &Arc<AtomicBrainState> {
        &self.state
    }

    /// Update all brain state values atomically
    pub fn update_full_state(
        &self,
        regime: MarketRegime,
        regime_confidence: f64,
        ml_confidence: f64,
        risk_level: f64,
        inference_latency_us: u64,
    ) {
        self.state.set_regime(regime);
        self.state.set_regime_confidence(regime_confidence);
        self.state.set_ml_confidence(ml_confidence);
        self.state.set_risk_level(risk_level);
        self.state.set_inference_latency_us(inference_latency_us);
        self.state.mark_update();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_atomic_brain_state() {
        let state = AtomicBrainState::new();
        
        state.set_regime(MarketRegime::HighVolatilityMeanReversion);
        state.set_regime_confidence(0.85);
        state.set_ml_confidence(0.72);
        state.set_risk_level(0.45);
        state.set_inference_latency_us(150);
        
        assert_eq!(state.get_regime(), MarketRegime::HighVolatilityMeanReversion);
        assert!((state.get_regime_confidence() - 0.85).abs() < 0.00001);
        assert!((state.get_ml_confidence() - 0.72).abs() < 0.00001);
        assert_eq!(state.get_inference_latency_us(), 150);
    }

    #[tokio::test]
    async fn test_signal_broadcaster() {
        let state = Arc::new(AtomicBrainState::new());
        let broadcaster = SignalBroadcaster::new(state.clone(), 100, 50);
        
        // Register strategies
        broadcaster.register_strategy("momentum".to_string()).await;
        broadcaster.register_strategy("mean_reversion".to_string()).await;
        
        // Update state
        broadcaster.update_full_state(
            MarketRegime::LowVolatilityTrending,
            0.90,
            0.78,
            0.35,
            120,
        );
        
        // Broadcast
        let result = broadcaster.broadcast_snapshot().await;
        assert!(result.is_ok());
        
        // Verify subscription works
        let mut rx = broadcaster.subscribe();
        let snapshot = rx.recv().await.unwrap();
        assert_eq!(snapshot.active_strategies.len(), 2);
        assert_eq!(snapshot.ml_model_confidence, 0.78);
    }
}
