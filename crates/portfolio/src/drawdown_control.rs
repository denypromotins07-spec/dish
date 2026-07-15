//! Hardcoded circuit breaker logic for drawdown control.
//! Dynamically scales position sizes, reduces leverage, or halts trading on threshold breaches.

use std::sync::atomic::{AtomicF64, AtomicBool, AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

/// Drawdown state enumeration
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DrawdownState {
    Normal,
    Warning,
    Reduced,
    Critical,
    Halted,
}

/// Circuit breaker configuration
#[derive(Debug, Clone)]
pub struct CircuitBreakerConfig {
    /// Daily drawdown warning threshold (e.g., 1%)
    pub daily_warning_pct: f64,
    /// Daily drawdown reduction threshold (e.g., 1.5%)
    pub daily_reduction_pct: f64,
    /// Daily drawdown critical threshold (e.g., 1.8%)
    pub daily_critical_pct: f64,
    /// Daily drawdown halt threshold (e.g., 2%)
    pub daily_halt_pct: f64,
    /// Weekly drawdown halt threshold (e.g., 5%)
    pub weekly_halt_pct: f64,
    /// Monthly drawdown halt threshold (e.g., 10%)
    pub monthly_halt_pct: f64,
    /// Position size reduction factor when in reduced state
    pub reduction_factor: f64,
    /// Cooldown period after halt (seconds)
    pub cooldown_seconds: u64,
}

impl Default for CircuitBreakerConfig {
    fn default() -> Self {
        Self {
            daily_warning_pct: 0.01,      // 1%
            daily_reduction_pct: 0.015,   // 1.5%
            daily_critical_pct: 0.018,    // 1.8%
            daily_halt_pct: 0.02,         // 2%
            weekly_halt_pct: 0.05,        // 5%
            monthly_halt_pct: 0.10,       // 10%
            reduction_factor: 0.5,        // 50% reduction
            cooldown_seconds: 3600,       // 1 hour
        }
    }
}

/// Drawdown control result
#[derive(Debug, Clone)]
pub struct DrawdownControlResult {
    pub current_state: DrawdownState,
    pub position_size_multiplier: f64,
    pub leverage_multiplier: f64,
    pub trading_allowed: bool,
    pub reason: String,
    pub action_required: DrawdownAction,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DrawdownAction {
    None,
    ReduceSize,
    ReduceLeverage,
    ClosePositions,
    HaltTrading,
    ResumeTrading,
}

/// Main drawdown controller
pub struct DrawdownController {
    /// Starting equity for the day
    daily_start_equity: AtomicF64,
    /// Peak equity today
    daily_peak_equity: AtomicF64,
    /// Minimum equity today
    daily_min_equity: AtomicF64,
    /// Starting equity for the week
    weekly_start_equity: AtomicF64,
    /// Peak equity for the week
    weekly_peak_equity: AtomicF64,
    /// Minimum equity for the week
    weekly_min_equity: AtomicF64,
    /// Starting equity for the month
    monthly_start_equity: AtomicF64,
    /// Peak equity for the month
    monthly_peak_equity: AtomicF64,
    /// Minimum equity for the month
    monthly_min_equity: AtomicF64,
    /// Current state
    current_state: AtomicI8,
    /// Trading halted flag
    trading_halted: AtomicBool,
    /// Halt timestamp
    halt_timestamp: AtomicU64,
    /// Configuration
    config: CircuitBreakerConfig,
    /// Current position size multiplier
    position_size_multiplier: AtomicF64,
    /// Current leverage multiplier
    leverage_multiplier: AtomicF64,
}

impl DrawdownController {
    /// Create new drawdown controller
    pub fn new(initial_equity: f64) -> Self {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();
        
        Self {
            daily_start_equity: AtomicF64::new(initial_equity),
            daily_peak_equity: AtomicF64::new(initial_equity),
            daily_min_equity: AtomicF64::new(initial_equity),
            weekly_start_equity: AtomicF64::new(initial_equity),
            weekly_peak_equity: AtomicF64::new(initial_equity),
            weekly_min_equity: AtomicF64::new(initial_equity),
            monthly_start_equity: AtomicF64::new(initial_equity),
            monthly_peak_equity: AtomicF64::new(initial_equity),
            monthly_min_equity: AtomicF64::new(initial_equity),
            current_state: AtomicI8::new(DrawdownState::Normal as i8),
            trading_halted: AtomicBool::new(false),
            halt_timestamp: AtomicU64::new(0),
            config: CircuitBreakerConfig::default(),
            position_size_multiplier: AtomicF64::new(1.0),
            leverage_multiplier: AtomicF64::new(1.0),
        }
    }

    /// Update equity and check drawdown thresholds
    pub fn update_equity(&self, current_equity: f64) -> DrawdownControlResult {
        // Update peak and min values
        self.update_extremes(current_equity);
        
        // Calculate drawdowns
        let daily_dd = self.get_daily_drawdown();
        let weekly_dd = self.get_weekly_drawdown();
        let monthly_dd = self.get_monthly_drawdown();
        
        // Check for halt conditions first (highest priority)
        if daily_dd >= self.config.daily_halt_pct 
            || weekly_dd >= self.config.weekly_halt_pct 
            || monthly_dd >= self.config.monthly_halt_pct 
        {
            return self.trigger_halt("Drawdown limit breached");
        }
        
        // Check if we can resume from halt
        if self.trading_halted.load(Ordering::Relaxed) {
            return self.check_resume();
        }
        
        // Determine state based on daily drawdown
        let (new_state, size_mult, lev_mult, action, reason) = if daily_dd >= self.config.daily_critical_pct {
            (
                DrawdownState::Critical,
                self.config.reduction_factor * 0.5,
                self.config.reduction_factor * 0.5,
                DrawdownAction::ClosePositions,
                format!("Critical daily drawdown: {:.2}%", daily_dd * 100.0),
            )
        } else if daily_dd >= self.config.daily_reduction_pct {
            (
                DrawdownState::Reduced,
                self.config.reduction_factor,
                self.config.reduction_factor,
                DrawdownAction::ReduceSize,
                format!("Daily drawdown reduction triggered: {:.2}%", daily_dd * 100.0),
            )
        } else if daily_dd >= self.config.daily_warning_pct {
            (
                DrawdownState::Warning,
                0.75,
                0.75,
                DrawdownAction::ReduceLeverage,
                format!("Daily drawdown warning: {:.2}%", daily_dd * 100.0),
            )
        } else {
            (
                DrawdownState::Normal,
                1.0,
                1.0,
                DrawdownAction::None,
                "Normal trading conditions".to_string(),
            )
        };
        
        // Update state
        self.current_state.store(new_state as i8, Ordering::Relaxed);
        self.position_size_multiplier.store(size_mult, Ordering::Relaxed);
        self.leverage_multiplier.store(lev_mult, Ordering::Relaxed);
        
        DrawdownControlResult {
            current_state: new_state,
            position_size_multiplier: size_mult,
            leverage_multiplier: lev_mult,
            trading_allowed: true,
            reason,
            action_required: action,
        }
    }

    /// Update peak and minimum equity values
    fn update_extremes(&self, equity: f64) {
        // Daily
        let daily_peak = self.daily_peak_equity.load(Ordering::Relaxed);
        if equity > daily_peak {
            self.daily_peak_equity.store(equity, Ordering::Relaxed);
        }
        let daily_min = self.daily_min_equity.load(Ordering::Relaxed);
        if equity < daily_min {
            self.daily_min_equity.store(equity, Ordering::Relaxed);
        }
        
        // Weekly
        let weekly_peak = self.weekly_peak_equity.load(Ordering::Relaxed);
        if equity > weekly_peak {
            self.weekly_peak_equity.store(equity, Ordering::Relaxed);
        }
        let weekly_min = self.weekly_min_equity.load(Ordering::Relaxed);
        if equity < weekly_min {
            self.weekly_min_equity.store(equity, Ordering::Relaxed);
        }
        
        // Monthly
        let monthly_peak = self.monthly_peak_equity.load(Ordering::Relaxed);
        if equity > monthly_peak {
            self.monthly_peak_equity.store(equity, Ordering::Relaxed);
        }
        let monthly_min = self.monthly_min_equity.load(Ordering::Relaxed);
        if equity < monthly_min {
            self.monthly_min_equity.store(equity, Ordering::Relaxed);
        }
    }

    /// Get current daily drawdown
    #[inline(always)]
    pub fn get_daily_drawdown(&self) -> f64 {
        let peak = self.daily_peak_equity.load(Ordering::Relaxed);
        let current = self.daily_min_equity.load(Ordering::Relaxed);
        if peak <= 0.0 { return 0.0; }
        (peak - current) / peak
    }

    /// Get current weekly drawdown
    #[inline(always)]
    pub fn get_weekly_drawdown(&self) -> f64 {
        let peak = self.weekly_peak_equity.load(Ordering::Relaxed);
        let current = self.weekly_min_equity.load(Ordering::Relaxed);
        if peak <= 0.0 { return 0.0; }
        (peak - current) / peak
    }

    /// Get current monthly drawdown
    #[inline(always)]
    pub fn get_monthly_drawdown(&self) -> f64 {
        let peak = self.monthly_peak_equity.load(Ordering::Relaxed);
        let current = self.monthly_min_equity.load(Ordering::Relaxed);
        if peak <= 0.0 { return 0.0; }
        (peak - current) / peak
    }

    /// Trigger trading halt
    fn trigger_halt(&self, reason: &str) -> DrawdownControlResult {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();
        
        self.trading_halted.store(true, Ordering::Relaxed);
        self.halt_timestamp.store(now, Ordering::Relaxed);
        self.current_state.store(DrawdownState::Halted as i8, Ordering::Relaxed);
        self.position_size_multiplier.store(0.0, Ordering::Relaxed);
        self.leverage_multiplier.store(0.0, Ordering::Relaxed);
        
        DrawdownControlResult {
            current_state: DrawdownState::Halted,
            position_size_multiplier: 0.0,
            leverage_multiplier: 0.0,
            trading_allowed: false,
            reason: reason.to_string(),
            action_required: DrawdownAction::HaltTrading,
        }
    }

    /// Check if trading can resume after halt
    fn check_resume(&self) -> DrawdownControlResult {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();
        
        let halt_ts = self.halt_timestamp.load(Ordering::Relaxed);
        let elapsed = now - halt_ts;
        
        if elapsed >= self.config.cooldown_seconds {
            // Cooldown complete, allow resume
            self.trading_halted.store(false, Ordering::Relaxed);
            self.current_state.store(DrawdownState::Normal as i8, Ordering::Relaxed);
            self.position_size_multiplier.store(0.25, Ordering::Relaxed); // Start at 25%
            self.leverage_multiplier.store(0.25, Ordering::Relaxed);
            
            DrawdownControlResult {
                current_state: DrawdownState::Normal,
                position_size_multiplier: 0.25,
                leverage_multiplier: 0.25,
                trading_allowed: true,
                reason: "Cooldown period complete, resuming with reduced size".to_string(),
                action_required: DrawdownAction::ResumeTrading,
            }
        } else {
            // Still in cooldown
            DrawdownControlResult {
                current_state: DrawdownState::Halted,
                position_size_multiplier: 0.0,
                leverage_multiplier: 0.0,
                trading_allowed: false,
                reason: format!("In cooldown, {} seconds remaining", self.config.cooldown_seconds - elapsed),
                action_required: DrawdownAction::HaltTrading,
            }
        }
    }

    /// Reset daily counters (call at start of each trading day)
    #[inline(always)]
    pub fn reset_daily(&self, new_start_equity: f64) {
        self.daily_start_equity.store(new_start_equity, Ordering::Relaxed);
        self.daily_peak_equity.store(new_start_equity, Ordering::Relaxed);
        self.daily_min_equity.store(new_start_equity, Ordering::Relaxed);
    }

    /// Reset weekly counters
    #[inline(always)]
    pub fn reset_weekly(&self, new_start_equity: f64) {
        self.weekly_start_equity.store(new_start_equity, Ordering::Relaxed);
        self.weekly_peak_equity.store(new_start_equity, Ordering::Relaxed);
        self.weekly_min_equity.store(new_start_equity, Ordering::Relaxed);
    }

    /// Reset monthly counters
    #[inline(always)]
    pub fn reset_monthly(&self, new_start_equity: f64) {
        self.monthly_start_equity.store(new_start_equity, Ordering::Relaxed);
        self.monthly_peak_equity.store(new_start_equity, Ordering::Relaxed);
        self.monthly_min_equity.store(new_start_equity, Ordering::Relaxed);
    }

    /// Get current position size multiplier
    #[inline(always)]
    pub fn get_position_size_multiplier(&self) -> f64 {
        self.position_size_multiplier.load(Ordering::Relaxed)
    }

    /// Get current leverage multiplier
    #[inline(always)]
    pub fn get_leverage_multiplier(&self) -> f64 {
        self.leverage_multiplier.load(Ordering::Relaxed)
    }

    /// Check if trading is allowed
    #[inline(always)]
    pub fn is_trading_allowed(&self) -> bool {
        !self.trading_halted.load(Ordering::Relaxed)
    }

    /// Manually halt trading
    #[inline(always)]
    pub fn manual_halt(&self) {
        self.trigger_halt("Manual halt requested");
    }

    /// Manually resume trading (override cooldown)
    #[inline(always)]
    pub fn manual_resume(&self) {
        self.trading_halted.store(false, Ordering::Relaxed);
        self.current_state.store(DrawdownState::Normal as i8, Ordering::Relaxed);
        self.position_size_multiplier.store(1.0, Ordering::Relaxed);
        self.leverage_multiplier.store(1.0, Ordering::Relaxed);
    }

    /// Update configuration
    #[inline(always)]
    pub fn update_config(&mut self, config: CircuitBreakerConfig) {
        self.config = config;
    }

    /// Get current state summary
    pub fn get_state_summary(&self) -> DrawdownSummary {
        DrawdownSummary {
            current_state: match self.current_state.load(Ordering::Relaxed) {
                0 => DrawdownState::Normal,
                1 => DrawdownState::Warning,
                2 => DrawdownState::Reduced,
                3 => DrawdownState::Critical,
                _ => DrawdownState::Halted,
            },
            daily_drawdown: self.get_daily_drawdown(),
            weekly_drawdown: self.get_weekly_drawdown(),
            monthly_drawdown: self.get_monthly_drawdown(),
            trading_halted: self.trading_halted.load(Ordering::Relaxed),
            position_size_multiplier: self.position_size_multiplier.load(Ordering::Relaxed),
            leverage_multiplier: self.leverage_multiplier.load(Ordering::Relaxed),
        }
    }
}

/// State summary for monitoring
#[derive(Debug, Clone)]
pub struct DrawdownSummary {
    pub current_state: DrawdownState,
    pub daily_drawdown: f64,
    pub weekly_drawdown: f64,
    pub monthly_drawdown: f64,
    pub trading_halted: bool,
    pub position_size_multiplier: f64,
    pub leverage_multiplier: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_drawdown_warning() {
        let controller = DrawdownController::new(100_000.0);
        
        // Simulate 1.2% drawdown
        controller.update_equity(98_800.0);
        
        let result = controller.update_equity(98_800.0);
        assert_eq!(result.current_state, DrawdownState::Warning);
        assert!((result.position_size_multiplier - 0.75).abs() < 0.01);
    }

    #[test]
    fn test_drawdown_halt() {
        let controller = DrawdownController::new(100_000.0);
        
        // Simulate 2.5% drawdown (exceeds 2% halt threshold)
        controller.update_equity(97_500.0);
        
        let result = controller.update_equity(97_500.0);
        assert_eq!(result.current_state, DrawdownState::Halted);
        assert!(!result.trading_allowed);
    }

    #[test]
    fn test_position_size_reduction() {
        let controller = DrawdownController::new(100_000.0);
        
        // Normal state
        assert!((controller.get_position_size_multiplier() - 1.0).abs() < 0.001);
        
        // Simulate reduction-level drawdown
        controller.update_equity(98_400.0); // 1.6% drawdown
        controller.update_equity(98_400.0);
        
        assert!((controller.get_position_size_multiplier() - 0.5).abs() < 0.01);
    }
}
