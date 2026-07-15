//! High-throughput Open Interest (OI) and liquidations heatmap tracker
//! Normalizes exchange OI deltas to detect leverage build-ups and liquidation cascades

use std::collections::{HashMap, VecDeque};

/// Open Interest snapshot for a single instrument
#[derive(Debug, Clone, Copy)]
pub struct OISnapshot {
    pub timestamp_ns: u64,
    pub open_interest: f64,
    pub open_interest_usd: f64,
    pub long_oi_ratio: f64, // Ratio of long OI (0.0 to 1.0)
    pub short_oi_ratio: f64,
    pub volume_24h: f64,
}

/// Liquidation event
#[derive(Debug, Clone, Copy)]
pub struct LiquidationEvent {
    pub timestamp_ns: u64,
    pub symbol: [u8; 16],
    pub side: LiquidationSide,
    pub quantity: f64,
    pub price: f64,
    pub value_usd: f64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LiquidationSide {
    Long,
    Short,
}

/// Aggregated liquidation data for heatmap
#[derive(Debug, Clone)]
pub struct LiquidationHeatmapBin {
    pub price_level: f64,
    pub long_liquidations: f64,
    pub short_liquidations: f64,
    pub total_value_usd: f64,
    pub event_count: u32,
}

/// Open Interest tracker with delta detection
pub struct OITracker {
    /// Historical OI snapshots per symbol
    oi_history: HashMap<[u8; 16], VecDeque<OISnapshot>>,
    /// Maximum history length per symbol
    max_history: usize,
    /// Current OI values
    current_oi: HashMap<[u8; 16], OISnapshot>,
    /// OI change thresholds for alerts (in basis points)
    alert_threshold_bps: f64,
}

impl OITracker {
    pub fn new(max_history: usize, alert_threshold_bps: f64) -> Self {
        Self {
            oi_history: HashMap::new(),
            max_history,
            current_oi: HashMap::new(),
            alert_threshold_bps,
        }
    }

    /// Update OI for a symbol
    pub fn update_oi(&mut self, symbol: &[u8; 16], snapshot: OISnapshot) -> Option<OIDelta> {
        let history = self.oi_history.entry(*symbol).or_insert_with(|| {
            VecDeque::with_capacity(self.max_history)
        });

        let delta = if let Some(prev) = self.current_oi.get(symbol) {
            let oi_change = snapshot.open_interest - prev.open_interest;
            let oi_change_pct = if prev.open_interest > 0.0 {
                oi_change / prev.open_interest * 10000.0 // In basis points
            } else {
                0.0
            };

            Some(OIDelta {
                symbol: *symbol,
                oi_change,
                oi_change_bps: oi_change_pct,
                long_ratio_change: snapshot.long_oi_ratio - prev.long_oi_ratio,
                is_significant: oi_change_pct.abs() > self.alert_threshold_bps,
                timestamp_ns: snapshot.timestamp_ns,
            })
        } else {
            None
        };

        // Update history with bounded size
        if history.len() >= self.max_history {
            history.pop_front();
        }
        history.push_back(snapshot);

        self.current_oi.insert(*symbol, snapshot);

        delta
    }

    /// Get current OI for symbol
    pub fn get_current_oi(&self, symbol: &[u8; 16]) -> Option<&OISnapshot> {
        self.current_oi.get(symbol)
    }

    /// Calculate OI trend over last N snapshots
    pub fn get_oi_trend(&self, symbol: &[u8; 16], periods: usize) -> Option<f64> {
        let history = self.oi_history.get(symbol)?;
        if history.len() < periods {
            return None;
        }

        let recent: Vec<OISnapshot> = history.iter().rev().take(periods).copied().collect();
        let start = recent.last()?.open_interest;
        let end = recent.first()?.open_interest;

        if start > 0.0 {
            Some((end - start) / start * 100.0) // Percentage change
        } else {
            Some(0.0)
        }
    }

    /// Detect potential squeeze conditions
    pub fn detect_squeeze(&self, symbol: &[u8; 16]) -> Option<SqueezeSignal> {
        let current = self.current_oi.get(symbol)?;
        
        // Check for extreme long/short ratios
        let long_extreme = current.long_oi_ratio > 0.75 || current.long_oi_ratio < 0.25;
        let short_extreme = current.short_oi_ratio > 0.75 || current.short_oi_ratio < 0.25;

        // Check for rapid OI increase (leverage buildup)
        let oi_trend = self.get_oi_trend(symbol, 5).unwrap_or(0.0);
        let rapid_buildup = oi_trend.abs() > 10.0; // 10% change in recent periods

        if long_extreme || short_extreme || rapid_buildup {
            Some(SqueezeSignal {
                symbol: *symbol,
                is_long_squeeze: current.long_oi_ratio > 0.7,
                is_short_squeeze: current.short_oi_ratio > 0.7,
                leverage_buildup: rapid_buildup,
                oi_ratio: current.long_oi_ratio,
                confidence: (current.long_oi_ratio - 0.5).abs() * 2.0,
            })
        } else {
            None
        }
    }
}

/// Open Interest delta alert
#[derive(Debug, Clone, Copy)]
pub struct OIDelta {
    pub symbol: [u8; 16],
    pub oi_change: f64,
    pub oi_change_bps: f64,
    pub long_ratio_change: f64,
    pub is_significant: bool,
    pub timestamp_ns: u64,
}

/// Squeeze detection signal
#[derive(Debug, Clone, Copy)]
pub struct SqueezeSignal {
    pub symbol: [u8; 16],
    pub is_long_squeeze: bool,
    pub is_short_squeeze: bool,
    pub leverage_buildup: bool,
    pub oi_ratio: f64,
    pub confidence: f64,
}

/// Liquidation heatmap builder
pub struct LiquidationHeatmap {
    /// Price bins for heatmap
    bins: HashMap<u64, LiquidationHeatmapBin>,
    /// Recent liquidation events
    recent_events: VecDeque<LiquidationEvent>,
    /// Bin size in price units
    bin_size: f64,
    /// Maximum events to track
    max_events: usize,
}

impl LiquidationHeatmap {
    pub fn new(bin_size: f64, max_events: usize) -> Self {
        Self {
            bins: HashMap::new(),
            recent_events: VecDeque::with_capacity(max_events),
            bin_size,
            max_events,
        }
    }

    /// Add a liquidation event to the heatmap
    pub fn add_liquidation(&mut self, event: LiquidationEvent) {
        // Determine price bin
        let bin_key = (event.price / self.bin_size) as u64;

        let bin = self.bins.entry(bin_key).or_insert_with(|| LiquidationHeatmapBin {
            price_level: bin_key as f64 * self.bin_size,
            long_liquidations: 0.0,
            short_liquidations: 0.0,
            total_value_usd: 0.0,
            event_count: 0,
        });

        match event.side {
            LiquidationSide::Long => bin.long_liquidations += event.value_usd,
            LiquidationSide::Short => bin.short_liquidations += event.value_usd,
        }

        bin.total_value_usd += event.value_usd;
        bin.event_count += 1;

        // Track recent events
        if self.recent_events.len() >= self.max_events {
            self.recent_events.pop_front();
        }
        self.recent_events.push_back(event);
    }

    /// Get heatmap bins sorted by liquidation value
    pub fn get_significant_bins(&self, min_value_usd: f64) -> Vec<&LiquidationHeatmapBin> {
        let mut bins: Vec<_> = self
            .bins
            .values()
            .filter(|b| b.total_value_usd >= min_value_usd)
            .collect();

        bins.sort_by(|a, b| {
            b.total_value_usd
                .partial_cmp(&a.total_value_usd)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        bins
    }

    /// Get total liquidations in time window
    pub fn get_liquidations_in_window(&self, window_ns: u64) -> LiquidationSummary {
        let now = self.recent_events.back().map(|e| e.timestamp_ns).unwrap_or(0);
        let cutoff = now.saturating_sub(window_ns);

        let mut summary = LiquidationSummary::default();

        for event in &self.recent_events {
            if event.timestamp_ns >= cutoff {
                summary.total_value_usd += event.value_usd;
                summary.event_count += 1;
                match event.side {
                    LiquidationSide::Long => summary.long_liquidations_usd += event.value_usd,
                    LiquidationSide::Short => summary.short_liquidations_usd += event.value_usd,
                }
            }
        }

        summary
    }

    /// Clear old bins to manage memory
    pub fn prune_old_bins(&mut self, min_event_count: u32) {
        self.bins.retain(|_, bin| bin.event_count >= min_event_count);
    }
}

/// Summary of liquidation activity
#[derive(Debug, Default, Clone, Copy)]
pub struct LiquidationSummary {
    pub total_value_usd: f64,
    pub long_liquidations_usd: f64,
    pub short_liquidations_usd: f64,
    pub event_count: u64,
    pub long_short_ratio: f64,
}

impl LiquidationSummary {
    pub fn calculate_ratios(&mut self) {
        if self.short_liquidations_usd > 0.0 {
            self.long_short_ratio = self.long_liquidations_usd / self.short_liquidations_usd;
        } else {
            self.long_short_ratio = f64::MAX;
        }
    }
}

/// Combined OI and liquidation analyzer
pub struct LeverageAnalyzer {
    oi_tracker: OITracker,
    liquidation_heatmap: LiquidationHeatmap,
}

impl LeverageAnalyzer {
    pub fn new(max_oi_history: usize, alert_threshold_bps: f64, bin_size: f64) -> Self {
        Self {
            oi_tracker: OITracker::new(max_oi_history, alert_threshold_bps),
            liquidation_heatmap: LiquidationHeatmap::new(bin_size, 1000),
        }
    }

    pub fn update_oi(&mut self, symbol: &[u8; 16], snapshot: OISnapshot) -> Option<OIDelta> {
        self.oi_tracker.update_oi(symbol, snapshot)
    }

    pub fn add_liquidation(&mut self, event: LiquidationEvent) {
        self.liquidation_heatmap.add_liquidation(event);
    }

    pub fn get_leverage_report(&self, symbol: &[u8; 16]) -> LeverageReport {
        let current_oi = self.oi_tracker.get_current_oi(symbol).copied();
        let squeeze_signal = self.oi_tracker.detect_squeeze(symbol);
        let liq_summary = self.liquidation_heatmap.get_liquidations_in_window(3600 * 1_000_000_000); // 1 hour

        LeverageReport {
            symbol: *symbol,
            current_oi,
            squeeze_signal,
            recent_liquidations: liq_summary,
        }
    }
}

/// Comprehensive leverage report
#[derive(Debug, Clone, Copy)]
pub struct LeverageReport {
    pub symbol: [u8; 16],
    pub current_oi: Option<OISnapshot>,
    pub squeeze_signal: Option<SqueezeSignal>,
    pub recent_liquidations: LiquidationSummary,
}

impl Default for OITracker {
    fn default() -> Self {
        Self::new(100, 500.0) // 5% threshold
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_oi_tracking() {
        let mut tracker = OITracker::new(50, 100.0);
        let symbol = *b"BTC-PERP      ";

        let snapshot1 = OISnapshot {
            timestamp_ns: 1000,
            open_interest: 1000000.0,
            open_interest_usd: 50_000_000_000.0,
            long_oi_ratio: 0.55,
            short_oi_ratio: 0.45,
            volume_24h: 1_000_000_000.0,
        };

        tracker.update_oi(&symbol, snapshot1);

        let snapshot2 = OISnapshot {
            timestamp_ns: 2000,
            open_interest: 1100000.0,
            open_interest_usd: 55_000_000_000.0,
            long_oi_ratio: 0.60,
            short_oi_ratio: 0.40,
            volume_24h: 1_200_000_000.0,
        };

        let delta = tracker.update_oi(&symbol, snapshot2);
        assert!(delta.is_some());
        assert!(delta.unwrap().is_significant);
    }

    #[test]
    fn test_liquidation_heatmap() {
        let mut heatmap = LiquidationHeatmap::new(100.0, 100);
        
        let event = LiquidationEvent {
            timestamp_ns: 1000,
            symbol: *b"BTC-PERP      ",
            side: LiquidationSide::Long,
            quantity: 100.0,
            price: 50000.0,
            value_usd: 5_000_000.0,
        };

        heatmap.add_liquidation(event);
        let bins = heatmap.get_significant_bins(1_000_000.0);
        assert!(!bins.is_empty());
    }
}
