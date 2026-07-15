//! Portfolio streamer for real-time PnL, margin, and position telemetry.
//! Uses atomic reads to prevent locking the portfolio state while streaming to UI.

use std::sync::atomic::{AtomicU64, AtomicI64, AtomicBool, Ordering};
use std::sync::Arc;

/// Real-time portfolio snapshot for streaming
#[derive(Debug, Clone, serde::Serialize)]
pub struct PortfolioSnapshot {
    pub timestamp_ns: u64,
    pub total_equity: f64,
    pub unrealized_pnl: f64,
    pub realized_pnl: f64,
    pub total_margin_used: f64,
    pub available_margin: f64,
    pub margin_ratio: f64,
    pub liquidation_distance_pct: f64,
    pub active_positions: u32,
    pub total_exposure: f64,
    pub net_delta: f64,
}

/// Position data for individual symbol tracking
#[derive(Debug, Clone, serde::Serialize)]
pub struct PositionSnapshot {
    pub symbol: String,
    pub side: String, // "long", "short", or "flat"
    pub size: f64,
    pub entry_price: f64,
    pub mark_price: f64,
    pub unrealized_pnl: f64,
    pub leverage: f64,
    pub margin_used: f64,
    pub liquidation_price: f64,
    pub timestamp_ns: u64,
}

/// Atomic portfolio state for lock-free reads
pub struct AtomicPortfolioState {
    total_equity: AtomicU64, // Stored as fixed-point * 1e6
    unrealized_pnl: AtomicI64,
    realized_pnl: AtomicI64,
    total_margin_used: AtomicU64,
    active_positions: AtomicU32,
    net_delta: AtomicI64,
    last_update_ns: AtomicU64,
}

impl AtomicPortfolioState {
    /// Create new atomic portfolio state
    pub fn new() -> Self {
        Self {
            total_equity: AtomicU64::new(0),
            unrealized_pnl: AtomicI64::new(0),
            realized_pnl: AtomicI64::new(0),
            total_margin_used: AtomicU64::new(0),
            active_positions: AtomicU32::new(0),
            net_delta: AtomicI64::new(0),
            last_update_ns: AtomicU64::new(0),
        }
    }

    /// Update total equity (thread-safe)
    pub fn set_total_equity(&self, value: f64) {
        self.total_equity.store((value * 1e6) as u64, Ordering::Relaxed);
    }

    /// Get total equity
    pub fn get_total_equity(&self) -> f64 {
        self.total_equity.load(Ordering::Relaxed) as f64 / 1e6
    }

    /// Update unrealized PnL (thread-safe)
    pub fn set_unrealized_pnl(&self, value: f64) {
        self.unrealized_pnl.store((value * 1e6) as i64, Ordering::Relaxed);
    }

    /// Get unrealized PnL
    pub fn get_unrealized_pnl(&self) -> f64 {
        self.unrealized_pnl.load(Ordering::Relaxed) as f64 / 1e6
    }

    /// Update realized PnL (thread-safe)
    pub fn set_realized_pnl(&self, value: f64) {
        self.realized_pnl.store((value * 1e6) as i64, Ordering::Relaxed);
    }

    /// Get realized PnL
    pub fn get_realized_pnl(&self) -> f64 {
        self.realized_pnl.load(Ordering::Relaxed) as f64 / 1e6
    }

    /// Update margin used (thread-safe)
    pub fn set_margin_used(&self, value: f64) {
        self.total_margin_used.store((value * 1e6) as u64, Ordering::Relaxed);
    }

    /// Get margin used
    pub fn get_margin_used(&self) -> f64 {
        self.total_margin_used.load(Ordering::Relaxed) as f64 / 1e6
    }

    /// Update active position count (thread-safe)
    pub fn set_active_positions(&self, count: u32) {
        self.active_positions.store(count, Ordering::Relaxed);
    }

    /// Get active position count
    pub fn get_active_positions(&self) -> u32 {
        self.active_positions.load(Ordering::Relaxed)
    }

    /// Update net delta (thread-safe)
    pub fn set_net_delta(&self, value: f64) {
        self.net_delta.store((value * 1e6) as i64, Ordering::Relaxed);
    }

    /// Get net delta
    pub fn get_net_delta(&self) -> f64 {
        self.net_delta.load(Ordering::Relaxed) as f64 / 1e6
    }

    /// Mark last update time
    pub fn mark_update(&self) {
        self.last_update_ns.store(
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos() as u64,
            Ordering::Relaxed,
        );
    }

    /// Get last update timestamp
    pub fn get_last_update_ns(&self) -> u64 {
        self.last_update_ns.load(Ordering::Relaxed)
    }
}

impl Default for AtomicPortfolioState {
    fn default() -> Self {
        Self::new()
    }
}

/// Portfolio streamer that broadcasts snapshots to UI
pub struct PortfolioStreamer {
    state: Arc<AtomicPortfolioState>,
    margin_warning_threshold: f64, // Margin ratio warning level
    liquidation_warning_distance: f64, // Minimum distance to liquidation
}

impl PortfolioStreamer {
    /// Create new portfolio streamer
    pub fn new(
        state: Arc<AtomicPortfolioState>,
        margin_warning_threshold: f64,
        liquidation_warning_distance: f64,
    ) -> Self {
        Self {
            state,
            margin_warning_threshold,
            liquidation_warning_distance,
        }
    }

    /// Generate current portfolio snapshot (lock-free read)
    pub fn generate_snapshot(&self, total_exposure: f64) -> PortfolioSnapshot {
        let now_ns = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64;

        let total_equity = self.state.get_total_equity();
        let margin_used = self.state.get_margin_used();
        
        let available_margin = (total_equity - margin_used).max(0.0);
        let margin_ratio = if total_equity > 0.0 {
            margin_used / total_equity
        } else {
            0.0
        };

        let unrealized_pnl = self.state.get_unrealized_pnl();
        let realized_pnl = self.state.get_realized_pnl();
        
        // Calculate liquidation distance (simplified)
        let liquidation_distance_pct = if total_exposure > 0.0 && margin_used > 0.0 {
            ((available_margin / total_exposure) * 100.0).min(100.0)
        } else {
            100.0
        };

        PortfolioSnapshot {
            timestamp_ns: now_ns,
            total_equity,
            unrealized_pnl,
            realized_pnl,
            total_margin_used: margin_used,
            available_margin,
            margin_ratio,
            liquidation_distance_pct,
            active_positions: self.state.get_active_positions(),
            total_exposure,
            net_delta: self.state.get_net_delta(),
        }
    }

    /// Check if margin is approaching dangerous levels
    pub fn check_margin_warning(&self, snapshot: &PortfolioSnapshot) -> Option<MarginWarning> {
        if snapshot.margin_ratio >= self.margin_warning_threshold {
            return Some(MarginWarning {
                level: "high_margin",
                message: format!("Margin ratio at {:.1}%", snapshot.margin_ratio * 100.0),
                margin_ratio: snapshot.margin_ratio,
            });
        }

        if snapshot.liquidation_distance_pct < self.liquidation_warning_distance {
            return Some(MarginWarning {
                level: "liquidation_risk",
                message: format!("Liquidation distance only {:.1}%", snapshot.liquidation_distance_pct),
                margin_ratio: snapshot.margin_ratio,
            });
        }

        None
    }

    /// Get reference to atomic state for updates
    pub fn state(&self) -> &Arc<AtomicPortfolioState> {
        &self.state
    }
}

/// Margin warning alert
#[derive(Debug, Clone, serde::Serialize)]
pub struct MarginWarning {
    pub level: &'static str,
    pub message: String,
    pub margin_ratio: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_atomic_state_updates() {
        let state = AtomicPortfolioState::new();
        
        state.set_total_equity(100000.0);
        state.set_unrealized_pnl(5000.0);
        state.set_realized_pnl(2500.0);
        state.set_margin_used(25000.0);
        state.set_active_positions(5);
        state.set_net_delta(-15000.0);
        
        assert_eq!(state.get_total_equity(), 100000.0);
        assert_eq!(state.get_unrealized_pnl(), 5000.0);
        assert_eq!(state.get_realized_pnl(), 2500.0);
        assert_eq!(state.get_margin_used(), 25000.0);
        assert_eq!(state.get_active_positions(), 5);
        assert_eq!(state.get_net_delta(), -15000.0);
    }

    #[test]
    fn test_portfolio_snapshot_generation() {
        let state = Arc::new(AtomicPortfolioState::new());
        state.set_total_equity(100000.0);
        state.set_margin_used(30000.0);
        state.set_unrealized_pnl(2000.0);
        
        let streamer = PortfolioStreamer::new(state, 0.5, 20.0);
        let snapshot = streamer.generate_snapshot(50000.0);
        
        assert_eq!(snapshot.total_equity, 100000.0);
        assert_eq!(snapshot.available_margin, 70000.0);
        assert_eq!(snapshot.margin_ratio, 0.3);
        assert!(snapshot.liquidation_distance_pct > 0.0);
    }

    #[test]
    fn test_margin_warning_detection() {
        let state = Arc::new(AtomicPortfolioState::new());
        state.set_total_equity(100000.0);
        state.set_margin_used(60000.0); // 60% margin ratio
        
        let streamer = PortfolioStreamer::new(state.clone(), 0.5, 20.0);
        let snapshot = streamer.generate_snapshot(80000.0);
        
        let warning = streamer.check_margin_warning(&snapshot);
        assert!(warning.is_some());
        assert_eq!(warning.unwrap().level, "high_margin");
    }
}
