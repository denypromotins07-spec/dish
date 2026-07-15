//! Microsecond portfolio drift monitor
//! Triggers rebalancing only when deviation from target weights exceeds dynamic threshold

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::{Duration, Instant};

const MAX_ASSETS: usize = 128;
const HISTORY_SIZE: usize = 100;

/// Drift measurement result
#[derive(Debug, Clone)]
pub struct DriftMeasurement {
    pub l1_norm: f64,
    pub l2_norm: f64,
    pub linf_norm: f64,
    pub max_drift_asset: usize,
    pub timestamp_ns: u64,
}

/// Dynamic threshold configuration
#[derive(Debug, Clone, Copy)]
pub struct DriftThresholdConfig {
    /// Base threshold (absolute)
    pub base_threshold: f64,
    /// ATR multiplier for dynamic adjustment
    pub atr_multiplier: f64,
    /// Volatility scaling factor
    pub vol_scaling: f64,
    /// Minimum threshold (floor)
    pub min_threshold: f64,
    /// Maximum threshold (cap)
    pub max_threshold: f64,
}

impl Default for DriftThresholdConfig {
    fn default() -> Self {
        Self {
            base_threshold: 0.02,  // 2% base drift
            atr_multiplier: 0.5,
            vol_scaling: 1.0,
            min_threshold: 0.005,  // 0.5% floor
            max_threshold: 0.10,   // 10% cap
        }
    }
}

/// Drift Monitor with atomic state for lock-free reads
#[repr(align(64))]
pub struct DriftMonitor {
    /// Current weights
    current_weights: [f64; MAX_ASSETS],
    /// Target weights
    target_weights: [f64; MAX_ASSETS],
    /// Asset volatilities (for dynamic threshold)
    asset_volatilities: [f64; MAX_ASSETS],
    /// Recent ATR values for each asset
    atr_history: [[f64; HISTORY_SIZE]; MAX_ASSETS],
    /// ATR history index
    atr_index: [usize; MAX_ASSETS],
    /// Asset count
    asset_count: usize,
    /// Current dynamic threshold
    current_threshold: f64,
    /// Threshold configuration
    config: DriftThresholdConfig,
    /// Rebalance needed flag
    rebalance_needed: AtomicBool,
    /// Last check timestamp (nanoseconds)
    last_check_ns: AtomicU64,
    /// Drift alert counter
    alert_counter: AtomicU64,
}

unsafe impl Send for DriftMonitor {}
unsafe impl Sync for DriftMonitor {}

impl Default for DriftMonitor {
    fn default() -> Self {
        Self {
            current_weights: [0.0; MAX_ASSETS],
            target_weights: [0.0; MAX_ASSETS],
            asset_volatilities: [0.2; MAX_ASSETS],  // Default 20% vol
            atr_history: [[0.0; HISTORY_SIZE]; MAX_ASSETS],
            atr_index: [0; MAX_ASSETS],
            asset_count: 0,
            current_threshold: 0.02,
            config: DriftThresholdConfig::default(),
            rebalance_needed: AtomicBool::new(false),
            last_check_ns: AtomicU64::new(0),
            alert_counter: AtomicU64::new(0),
        }
    }
}

impl DriftMonitor {
    #[inline(always)]
    pub fn new(asset_count: usize, config: DriftThresholdConfig) -> Self {
        assert!(asset_count <= MAX_ASSETS);
        
        let mut monitor = Self {
            asset_count,
            config,
            ..Self::default()
        };
        
        // Initialize ATR history with base threshold
        for i in 0..asset_count {
            for j in 0..HISTORY_SIZE {
                monitor.atr_history[i][j] = config.base_threshold;
            }
        }
        
        monitor.update_threshold();
        monitor
    }

    /// Update current weights
    #[inline(always)]
    pub fn set_current_weights(&mut self, weights: &[f64]) {
        assert_eq!(weights.len(), self.asset_count);
        for i in 0..self.asset_count {
            self.current_weights[i] = weights[i];
        }
    }

    /// Set target weights
    #[inline(always)]
    pub fn set_target_weights(&mut self, weights: &[f64]) {
        assert_eq!(weights.len(), self.asset_count);
        for i in 0..self.asset_count {
            self.target_weights[i] = weights[i];
        }
    }

    /// Update asset volatility
    #[inline(always)]
    pub fn set_volatility(&mut self, asset_idx: usize, volatility: f64) {
        if asset_idx < self.asset_count {
            self.asset_volatilities[asset_idx] = volatility.clamp(0.01, 2.0);
        }
    }

    /// Update ATR for an asset
    #[inline(always)]
    pub fn update_atr(&mut self, asset_idx: usize, atr: f64) {
        if asset_idx >= self.asset_count {
            return;
        }
        
        let idx = self.atr_index[asset_idx];
        self.atr_history[asset_idx][idx] = atr;
        self.atr_index[asset_idx] = (idx + 1) % HISTORY_SIZE;
        
        // Update dynamic threshold
        self.update_threshold();
    }

    /// Calculate dynamic threshold based on market conditions
    #[inline(always)]
    fn update_threshold(&mut self) {
        let mut avg_atr = 0.0;
        let mut max_vol = 0.0;
        
        for i in 0..self.asset_count {
            // Average ATR over history
            let sum: f64 = self.atr_history[i].iter().sum();
            avg_atr += sum / HISTORY_SIZE as f64;
            
            max_vol = max_vol.max(self.asset_volatilities[i]);
        }
        
        avg_atr /= self.asset_count as f64;
        
        // Dynamic threshold formula
        let threshold = self.config.base_threshold
            + self.config.atr_multiplier * avg_atr
            + self.config.vol_scaling * max_vol * 0.1;
        
        // Apply bounds
        self.current_threshold = threshold
            .clamp(self.config.min_threshold, self.config.max_threshold);
    }

    /// Check drift and determine if rebalance is needed
    #[inline(always)]
    pub fn check_drift(&self) -> DriftMeasurement {
        let start = Instant::now();
        
        let mut l1_norm = 0.0;
        let mut l2_norm_sq = 0.0;
        let mut linf_norm = 0.0;
        let mut max_drift_asset = 0;
        
        for i in 0..self.asset_count {
            let drift = (self.current_weights[i] - self.target_weights[i]).abs();
            
            l1_norm += drift;
            l2_norm_sq += drift * drift;
            
            if drift > linf_norm {
                linf_norm = drift;
                max_drift_asset = i;
            }
        }
        
        let l2_norm = l2_norm_sq.sqrt();
        let timestamp_ns = start.elapsed().as_nanos() as u64;
        
        DriftMeasurement {
            l1_norm,
            l2_norm,
            linf_norm,
            max_drift_asset,
            timestamp_ns,
        }
    }

    /// Check if rebalance is needed (lock-free)
    #[inline(always)]
    pub fn needs_rebalance(&self) -> bool {
        let measurement = self.check_drift();
        let needed = measurement.linf_norm > self.current_threshold;
        
        // Update atomic flag
        self.rebalance_needed.store(needed, Ordering::Release);
        
        if needed {
            self.alert_counter.fetch_add(1, Ordering::Relaxed);
        }
        
        // Update last check time
        let now_ns = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64;
        self.last_check_ns.store(now_ns, Ordering::Release);
        
        needed
    }

    /// Get current threshold
    #[inline(always)]
    pub fn current_threshold(&self) -> f64 {
        self.current_threshold
    }

    /// Check if rebalance is needed (using cached atomic flag)
    #[inline(always)]
    pub fn is_rebalance_flagged(&self) -> bool {
        self.rebalance_needed.load(Ordering::Acquire)
    }

    /// Get alert count
    #[inline(always)]
    pub fn alert_count(&self) -> u64 {
        self.alert_counter.load(Ordering::Relaxed)
    }

    /// Reset alert counter
    #[inline(always)]
    pub fn reset_alerts(&mut self) {
        self.alert_counter.store(0, Ordering::Release);
    }

    /// Get detailed drift report
    #[inline(always)]
    pub fn get_drift_report(&self) -> DriftReport {
        let mut asset_drifts = Vec::with_capacity(self.asset_count);
        
        for i in 0..self.asset_count {
            let drift = self.current_weights[i] - self.target_weights[i];
            asset_drifts.push(AssetDrift {
                asset_id: i,
                current_weight: self.current_weights[i],
                target_weight: self.target_weights[i],
                drift,
                drift_abs: drift.abs(),
                volatility: self.asset_volatilities[i],
            });
        }
        
        // Sort by absolute drift (largest first)
        asset_drifts.sort_by(|a, b| b.drift_abs.partial_cmp(&a.drift_abs).unwrap());
        
        let measurement = self.check_drift();
        
        DriftReport {
            measurement,
            threshold: self.current_threshold,
            rebalance_needed: self.is_rebalance_flagged(),
            asset_drifts,
        }
    }
}

/// Individual asset drift info
#[derive(Debug, Clone)]
pub struct AssetDrift {
    pub asset_id: usize,
    pub current_weight: f64,
    pub target_weight: f64,
    pub drift: f64,
    pub drift_abs: f64,
    pub volatility: f64,
}

/// Full drift report
#[derive(Debug, Clone)]
pub struct DriftReport {
    pub measurement: DriftMeasurement,
    pub threshold: f64,
    pub rebalance_needed: bool,
    pub asset_drifts: Vec<AssetDrift>,
}

/// Time-based drift checker with cooldown
pub struct CooldownDriftMonitor {
    inner: DriftMonitor,
    /// Minimum time between rebalance signals
    cooldown_duration: Duration,
    /// Last rebalance signal time
    last_signal_ns: AtomicU64,
}

unsafe impl Send for CooldownDriftMonitor {}
unsafe impl Sync for CooldownDriftMonitor {}

impl CooldownDriftMonitor {
    #[inline(always)]
    pub fn new(asset_count: usize, config: DriftThresholdConfig, cooldown_minutes: u32) -> Self {
        Self {
            inner: DriftMonitor::new(asset_count, config),
            cooldown_duration: Duration::from_secs(cooldown_minutes as u64 * 60),
            last_signal_ns: AtomicU64::new(0),
        }
    }

    #[inline(always)]
    pub fn check_with_cooldown(&self) -> bool {
        // First check if drift warrants rebalance
        if !self.inner.needs_rebalance() {
            return false;
        }
        
        // Check cooldown
        let now_ns = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64;
        
        let last_signal = self.last_signal_ns.load(Ordering::Acquire);
        let elapsed_ns = now_ns.saturating_sub(last_signal);
        let cooldown_ns = self.cooldown_duration.as_nanos() as u64;
        
        if elapsed_ns < cooldown_ns {
            return false;  // Still in cooldown
        }
        
        // Update last signal time
        self.last_signal_ns.store(now_ns, Ordering::Release);
        
        true
    }

    #[inline(always)]
    pub fn inner(&self) -> &DriftMonitor {
        &self.inner
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_drift_monitor() {
        let mut monitor = DriftMonitor::new(3, DriftThresholdConfig::default());
        
        // Set target weights
        monitor.set_target_weights(&[0.33, 0.33, 0.34]);
        
        // Set current weights (no drift)
        monitor.set_current_weights(&[0.33, 0.33, 0.34]);
        
        assert!(!monitor.needs_rebalance());
        
        // Introduce drift
        monitor.set_current_weights(&[0.40, 0.30, 0.30]);
        
        let drift = monitor.check_drift();
        assert!(drift.linf_norm > 0.05);
        assert_eq!(drift.max_drift_asset, 0);
        
        // Should trigger rebalance
        assert!(monitor.needs_rebalance());
        
        println!("Drift report:");
        let report = monitor.get_drift_report();
        println!("  L1 norm: {:.4}", report.measurement.l1_norm);
        println!("  L-inf norm: {:.4}", report.measurement.linf_norm);
        println!("  Threshold: {:.4}", report.threshold);
        println!("  Rebalance needed: {}", report.rebalance_needed);
    }
}
