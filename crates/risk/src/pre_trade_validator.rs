//! Lock-free pre-trade check engine that validates every outgoing order against strict limits.
//! Validates: max drawdown, daily loss limits, margin utilization, and max position size before execution.

use std::sync::atomic::{AtomicBool, AtomicF64, AtomicU64, Ordering};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

/// Order validation result
#[derive(Debug, Clone, PartialEq)]
pub enum ValidationResult {
    Approved,
    Rejected(RejectionReason),
    Warning(ApprovalWithWarning),
}

#[derive(Debug, Clone, PartialEq)]
pub enum RejectionReason {
    MaxDrawdownBreached,
    DailyLossLimitExceeded,
    MarginUtilizationTooHigh,
    MaxPositionSizeExceeded,
    ConcentrationLimitBreached,
    VelocityLimitExceeded,
    InvalidOrderParameters,
    TradingHalted,
}

#[derive(Debug, Clone)]
pub struct ApprovalWithWarning {
    pub approved: bool,
    pub warning_message: String,
    pub risk_score: f64,
}

/// Risk limits configuration
#[derive(Debug, Clone)]
pub struct RiskLimits {
    pub max_daily_loss_pct: f64,
    pub max_drawdown_pct: f64,
    pub max_margin_utilization: f64,
    pub max_position_size_usd: f64,
    pub max_concentration_pct: f64,
    pub max_orders_per_minute: u32,
    pub trading_halt_flag: bool,
}

impl Default for RiskLimits {
    fn default() -> Self {
        Self {
            max_daily_loss_pct: 0.02,      // 2% daily loss limit
            max_drawdown_pct: 0.05,        // 5% max drawdown
            max_margin_utilization: 0.80,  // 80% margin cap
            max_position_size_usd: 100_000.0,
            max_concentration_pct: 0.25,   // 25% per asset
            max_orders_per_minute: 60,
            trading_halt_flag: false,
        }
    }
}

/// Lock-free pre-trade validator
pub struct PreTradeValidator {
    /// Current account equity
    current_equity: AtomicF64,
    /// Peak equity (for drawdown calculation)
    peak_equity: AtomicF64,
    /// Starting equity for the day
    daily_start_equity: AtomicF64,
    /// Current margin utilized
    margin_used: AtomicF64,
    /// Total margin available
    margin_available: AtomicF64,
    /// Orders placed today (for velocity limiting)
    orders_today_count: AtomicU64,
    /// Last order timestamp (for rate limiting)
    last_order_timestamp: AtomicU64,
    /// Risk limits configuration
    limits: RiskLimits,
    /// Trading halted flag
    trading_halted: AtomicBool,
    /// Last update timestamp
    last_update_ts: AtomicU64,
}

/// Order request to validate
#[derive(Debug, Clone)]
pub struct OrderRequest {
    pub symbol: String,
    pub side: OrderSide,
    pub quantity: f64,
    pub price: f64,
    pub order_type: OrderType,
    pub notional_value: f64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum OrderSide {
    Buy,
    Sell,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum OrderType {
    Market,
    Limit,
    StopMarket,
    StopLimit,
}

impl PreTradeValidator {
    /// Create new pre-trade validator with initial equity
    pub fn new(initial_equity: f64, margin_available: f64) -> Self {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();
        
        Self {
            current_equity: AtomicF64::new(initial_equity),
            peak_equity: AtomicF64::new(initial_equity),
            daily_start_equity: AtomicF64::new(initial_equity),
            margin_used: AtomicF64::new(0.0),
            margin_available: AtomicF64::new(margin_available),
            orders_today_count: AtomicU64::new(0),
            last_order_timestamp: AtomicU64::new(0),
            limits: RiskLimits::default(),
            trading_halted: AtomicBool::new(false),
            last_update_ts: AtomicU64::new(now),
        }
    }

    /// Update current equity (call on every PnL update)
    #[inline(always)]
    pub fn update_equity(&self, new_equity: f64) {
        self.current_equity.store(new_equity, Ordering::Relaxed);
        
        // Update peak equity if new high
        let current_peak = self.peak_equity.load(Ordering::Relaxed);
        if new_equity > current_peak {
            self.peak_equity.store(new_equity, Ordering::Relaxed);
        }
    }

    /// Reset daily counters (call at start of each trading day)
    #[inline(always)]
    pub fn reset_daily_counters(&self) {
        let equity = self.current_equity.load(Ordering::Relaxed);
        self.daily_start_equity.store(equity, Ordering::Relaxed);
        self.orders_today_count.store(0, Ordering::Relaxed);
    }

    /// Calculate current drawdown percentage
    #[inline(always)]
    pub fn get_current_drawdown(&self) -> f64 {
        let peak = self.peak_equity.load(Ordering::Relaxed);
        let current = self.current_equity.load(Ordering::Relaxed);
        
        if peak <= 0.0 {
            return 0.0;
        }
        
        (peak - current) / peak
    }

    /// Calculate daily PnL percentage
    #[inline(always)]
    pub fn get_daily_pnl_pct(&self) -> f64 {
        let start = self.daily_start_equity.load(Ordering::Relaxed);
        let current = self.current_equity.load(Ordering::Relaxed);
        
        if start <= 0.0 {
            return 0.0;
        }
        
        (current - start) / start
    }

    /// Get current margin utilization
    #[inline(always)]
    pub fn get_margin_utilization(&self) -> f64 {
        let used = self.margin_used.load(Ordering::Relaxed);
        let available = self.margin_available.load(Ordering::Relaxed);
        
        if available <= 0.0 {
            return 1.0;
        }
        
        used / available
    }

    /// Validate an order before submission - CORE FUNCTION
    pub fn validate_order(&self, order: &OrderRequest) -> ValidationResult {
        let start = Instant::now();
        
        // Check if trading is halted
        if self.trading_halted.load(Ordering::Relaxed) || self.limits.trading_halt_flag {
            return ValidationResult::Rejected(RejectionReason::TradingHalted);
        }
        
        // Validate order parameters
        if order.quantity <= 0.0 || order.price <= 0.0 || order.notional_value <= 0.0 {
            return ValidationResult::Rejected(RejectionReason::InvalidOrderParameters);
        }
        
        // Check max drawdown
        let drawdown = self.get_current_drawdown();
        if drawdown >= self.limits.max_drawdown_pct {
            return ValidationResult::Rejected(RejectionReason::MaxDrawdownBreached);
        }
        
        // Check daily loss limit
        let daily_pnl = self.get_daily_pnl_pct();
        if daily_pnl <= -self.limits.max_daily_loss_pct {
            return ValidationResult::Rejected(RejectionReason::DailyLossLimitExceeded);
        }
        
        // Check margin utilization after this order
        let current_margin_used = self.margin_used.load(Ordering::Relaxed);
        let new_margin_required = order.notional_value; // Simplified: assume 1x margin
        let projected_margin = current_margin_used + new_margin_required;
        let margin_avail = self.margin_available.load(Ordering::Relaxed);
        
        if margin_avail > 0.0 && (projected_margin / margin_avail) > self.limits.max_margin_utilization {
            return ValidationResult::Rejected(RejectionReason::MarginUtilizationTooHigh);
        }
        
        // Check max position size
        if order.notional_value > self.limits.max_position_size_usd {
            return ValidationResult::Rejected(RejectionReason::MaxPositionSizeExceeded);
        }
        
        // Check order velocity (orders per minute)
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();
        let last_ts = self.last_order_timestamp.load(Ordering::Relaxed);
        
        // Simple velocity check: count orders in last minute
        let orders_count = self.orders_today_count.load(Ordering::Relaxed);
        if orders_count >= self.limits.max_orders_per_minute as u64 {
            // Reset counter if more than a minute has passed
            if now - last_ts < 60 {
                return ValidationResult::Rejected(RejectionReason::VelocityLimitExceeded);
            } else {
                // Reset counter
                self.orders_today_count.store(0, Ordering::Relaxed);
            }
        }
        
        // Increment order counter
        self.orders_today_count.fetch_add(1, Ordering::Relaxed);
        self.last_order_timestamp.store(now, Ordering::Relaxed);
        
        // Calculate risk score for warnings
        let risk_score = self.calculate_risk_score(order);
        
        let latency_ns = start.elapsed().as_nanos() as u64;
        self.last_update_ts.store(latency_ns, Ordering::Relaxed);
        
        // Return approval with warnings if risk is elevated
        if risk_score > 0.7 {
            ValidationResult::Warning(ApprovalWithWarning {
                approved: true,
                warning_message: format!("High risk score: {:.2}", risk_score),
                risk_score,
            })
        } else {
            ValidationResult::Approved
        }
    }

    /// Calculate composite risk score (0.0 to 1.0)
    fn calculate_risk_score(&self, order: &OrderRequest) -> f64 {
        let mut score = 0.0;
        
        // Drawdown component
        let drawdown = self.get_current_drawdown();
        score += (drawdown / self.limits.max_drawdown_pct) * 0.3;
        
        // Daily PnL component
        let daily_pnl = self.get_daily_pnl_pct();
        if daily_pnl < 0.0 {
            score += ((-daily_pnl) / self.limits.max_daily_loss_pct) * 0.3;
        }
        
        // Margin utilization component
        let margin_util = self.get_margin_utilization();
        score += margin_util * 0.2;
        
        // Order size relative to limit
        let size_ratio = order.notional_value / self.limits.max_position_size_usd;
        score += size_ratio * 0.2;
        
        score.min(1.0)
    }

    /// Manually halt trading (circuit breaker)
    #[inline(always)]
    pub fn halt_trading(&self) {
        self.trading_halted.store(true, Ordering::Relaxed);
    }

    /// Resume trading after halt
    #[inline(always)]
    pub fn resume_trading(&self) {
        self.trading_halted.store(false, Ordering::Relaxed);
    }

    /// Update margin used (call when positions change)
    #[inline(always)]
    pub fn update_margin_used(&self, new_margin_used: f64) {
        self.margin_used.store(new_margin_used.max(0.0), Ordering::Relaxed);
    }

    /// Set custom risk limits
    #[inline(always)]
    pub fn set_limits(&mut self, limits: RiskLimits) {
        self.limits = limits;
    }

    /// Get current state summary
    pub fn get_state_summary(&self) -> ValidatorStateSummary {
        ValidatorStateSummary {
            current_equity: self.current_equity.load(Ordering::Relaxed),
            peak_equity: self.peak_equity.load(Ordering::Relaxed),
            drawdown_pct: self.get_current_drawdown(),
            daily_pnl_pct: self.get_daily_pnl_pct(),
            margin_utilization: self.get_margin_utilization(),
            orders_today: self.orders_today_count.load(Ordering::Relaxed),
            trading_halted: self.trading_halted.load(Ordering::Relaxed),
        }
    }
}

/// State summary for monitoring
#[derive(Debug, Clone)]
pub struct ValidatorStateSummary {
    pub current_equity: f64,
    pub peak_equity: f64,
    pub drawdown_pct: f64,
    pub daily_pnl_pct: f64,
    pub margin_utilization: f64,
    pub orders_today: u64,
    pub trading_halted: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_order_validation_approved() {
        let validator = PreTradeValidator::new(100_000.0, 200_000.0);
        
        let order = OrderRequest {
            symbol: "BTCUSDT".to_string(),
            side: OrderSide::Buy,
            quantity: 0.1,
            price: 50_000.0,
            order_type: OrderType::Limit,
            notional_value: 5_000.0,
        };
        
        let result = validator.validate_order(&order);
        assert_eq!(result, ValidationResult::Approved);
    }

    #[test]
    fn test_drawdown_breach() {
        let validator = PreTradeValidator::new(100_000.0, 200_000.0);
        
        // Simulate drawdown by reducing equity
        validator.update_equity(94_000.0); // 6% drawdown
        
        let order = OrderRequest {
            symbol: "BTCUSDT".to_string(),
            side: OrderSide::Buy,
            quantity: 0.1,
            price: 50_000.0,
            order_type: OrderType::Limit,
            notional_value: 5_000.0,
        };
        
        let result = validator.validate_order(&order);
        assert_eq!(result, ValidationResult::Rejected(RejectionReason::MaxDrawdownBreached));
    }

    #[test]
    fn test_position_size_limit() {
        let mut validator = PreTradeValidator::new(100_000.0, 200_000.0);
        validator.limits.max_position_size_usd = 10_000.0;
        
        let order = OrderRequest {
            symbol: "BTCUSDT".to_string(),
            side: OrderSide::Buy,
            quantity: 1.0,
            price: 50_000.0,
            order_type: OrderType::Limit,
            notional_value: 50_000.0,
        };
        
        let result = validator.validate_order(&order);
        assert_eq!(result, ValidationResult::Rejected(RejectionReason::MaxPositionSizeExceeded));
    }
}
