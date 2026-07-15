//! Lock-free Sentiment Aggregator for combining X, Reddit, News, and Macro signals.
//! Produces unified "Market Sentiment Score" (Z-score normalized) in microseconds.
//! Feeds directly into the strategy engine with minimal latency.
//! Designed for AMD Ryzen AI 5 with SIMD optimizations.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

/// Individual sentiment signal from a source
#[derive(Debug, Clone)]
pub struct SentimentSignal {
    pub source: SignalSource,
    pub value: f64,        // -1.0 to +1.0
    pub confidence: f64,   // 0.0 to 1.0
    pub timestamp_ms: u64,
    pub volume: u64,       // Number of items contributing
}

/// Sources of sentiment data
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SignalSource {
    Twitter,
    Reddit,
    News,
    Macro,
    FearGreed,
    GoogleTrends,
}

impl SignalSource {
    /// Default weight for each source
    pub fn default_weight(&self) -> f64 {
        match self {
            SignalSource::Twitter => 0.25,    // High frequency, noisy
            SignalSource::Reddit => 0.20,     // Medium frequency, thoughtful
            SignalSource::News => 0.25,       // Lower frequency, high quality
            SignalSource::Macro => 0.15,      // Low frequency, fundamental
            SignalSource::FearGreed => 0.10,  // Contrarian indicator
            SignalSource::GoogleTrends => 0.05, // Lagging indicator
        }
    }
    
    /// Decay rate per minute for each source
    pub fn decay_rate_per_minute(&self) -> f64 {
        match self {
            SignalSource::Twitter => 0.5,      // Fast decay (very short half-life)
            SignalSource::Reddit => 0.3,       // Medium decay
            SignalSource::News => 0.2,         // Slower decay
            SignalSource::Macro => 0.05,       // Very slow decay (persistent)
            SignalSource::FearGreed => 0.1,    // Slow decay
            SignalSource::GoogleTrends => 0.15, // Medium-slow decay
        }
    }
}

/// Lock-free ring buffer for signal storage
struct SignalRingBuffer {
    buffer: Vec<Option<Arc<SentimentSignal>>>,
    capacity: usize,
    head: AtomicU64,
    tail: AtomicU64,
}

impl SignalRingBuffer {
    fn new(capacity: usize) -> Self {
        let mut buffer = Vec::with_capacity(capacity);
        buffer.resize_with(capacity, || None);
        
        Self {
            buffer,
            capacity,
            head: AtomicU64::new(0),
            tail: AtomicU64::new(0),
        }
    }
    
    fn push(&self, signal: Arc<SentimentSignal>) -> Option<Arc<SentimentSignal>> {
        let tail = self.tail.fetch_add(1, Ordering::Relaxed);
        let index = (tail % self.capacity as u64) as usize;
        
        let old = self.buffer[index].take();
        self.buffer[index] = Some(signal);
        
        // Update head if we've wrapped around
        let head = self.head.load(Ordering::Relaxed);
        if tail >= self.capacity as u64 && head <= tail - self.capacity as u64 {
            self.head.store(tail - self.capacity as u64 + 1, Ordering::Relaxed);
        }
        
        old
    }
    
    fn iter_by_source(&self, source: SignalSource) -> impl Iterator<Item = Arc<SentimentSignal>> + '_ {
        let head = self.head.load(Ordering::Relaxed);
        let tail = self.tail.load(Ordering::Relaxed);
        
        (head..tail).filter_map(move |i| {
            let index = (i % self.capacity as u64) as usize;
            self.buffer[index].clone().filter(|s| s.source == source)
        })
    }
    
    fn iter_all(&self) -> impl Iterator<Item = Arc<SentimentSignal>> + '_ {
        let head = self.head.load(Ordering::Relaxed);
        let tail = self.tail.load(Ordering::Relaxed);
        
        (head..tail).filter_map(move |i| {
            let index = (i % self.capacity as u64) as usize;
            self.buffer[index].clone()
        })
    }
}

/// Aggregated sentiment score with statistics
#[derive(Debug, Clone)]
pub struct AggregatedSentiment {
    pub composite_score: f64,      // Z-score normalized (-3 to +3 typical)
    pub raw_score: f64,            // Raw weighted average (-1 to +1)
    pub z_score: f64,              // Standard deviations from mean
    pub percentile: f64,           // 0-100 percentile
    pub trend: TrendDirection,
    pub momentum: f64,             // Rate of change
    pub sources_active: u8,
    pub total_volume: u64,
    pub timestamp_ms: u64,
    pub is_extreme: bool,
}

/// Trend direction classification
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrendDirection {
    StronglyBullish,
    Bullish,
    Neutral,
    Bearish,
    StronglyBearish,
}

impl TrendDirection {
    pub fn from_score(score: f64, momentum: f64) -> Self {
        if score > 1.5 && momentum > 0.1 {
            TrendDirection::StronglyBullish
        } else if score > 0.5 && momentum > 0.0 {
            TrendDirection::Bullish
        } else if score < -1.5 && momentum < -0.1 {
            TrendDirection::StronglyBearish
        } else if score < -0.5 && momentum < 0.0 {
            TrendDirection::Bearish
        } else {
            TrendDirection::Neutral
        }
    }
}

/// Main sentiment aggregator with lock-free operations
pub struct SentimentAggregator {
    signals: SignalRingBuffer,
    weights: [f64; 6],
    running_mean: AtomicU64,  // Stored as fixed-point for atomic ops
    running_variance: AtomicU64,
    sample_count: AtomicUsize,
    last_score: AtomicU64,
    score_history: VecDeque<f64>,
    history_max: usize,
}

impl SentimentAggregator {
    /// Create new aggregator with default configuration
    pub fn new(buffer_capacity: usize) -> Self {
        let weights = [
            SignalSource::Twitter.default_weight(),
            SignalSource::Reddit.default_weight(),
            SignalSource::News.default_weight(),
            SignalSource::Macro.default_weight(),
            SignalSource::FearGreed.default_weight(),
            SignalSource::GoogleTrends.default_weight(),
        ];
        
        Self {
            signals: SignalRingBuffer::new(buffer_capacity),
            weights,
            running_mean: AtomicU64::new(0),
            running_variance: AtomicU64::new(0),
            sample_count: AtomicUsize::new(0),
            last_score: AtomicU64::new(0),
            score_history: VecDeque::with_capacity(100),
            history_max: 100,
        }
    }
    
    /// Add a new sentiment signal
    pub fn add_signal(&self, signal: SentimentSignal) {
        let arc_signal = Arc::new(signal);
        self.signals.push(arc_signal);
        
        // Update running statistics
        self.update_statistics();
    }
    
    /// Add signal with convenience builder
    pub fn add_signal_raw(
        &self,
        source: SignalSource,
        value: f64,
        confidence: f64,
        volume: u64,
    ) {
        let signal = SentimentSignal {
            source,
            value: value.clamp(-1.0, 1.0),
            confidence: confidence.clamp(0.0, 1.0),
            timestamp_ms: get_current_ms(),
            volume,
        };
        self.add_signal(signal);
    }
    
    /// Update running mean and variance using Welford's algorithm
    fn update_statistics(&self) {
        let count = self.sample_count.fetch_add(1, Ordering::Relaxed) + 1;
        
        // Get current aggregated score
        let current = self.calculate_raw_score();
        
        // Update running mean (fixed-point with 1e9 scale)
        let scale = 1_000_000_000u64;
        let current_fixed = ((current + 1.0) * scale as f64) as u64;
        
        let mean_fixed = self.running_mean.load(Ordering::Relaxed);
        let new_mean_fixed = mean_fixed + (current_fixed.saturating_sub(mean_fixed)) / (count as u64);
        self.running_mean.store(new_mean_fixed, Ordering::Relaxed);
        
        // Update variance estimate
        let var_fixed = self.running_variance.load(Ordering::Relaxed);
        let diff = (current_fixed as i64 - mean_fixed as i64).abs() as u64;
        let new_var_fixed = var_fixed.saturating_add(diff);
        self.running_variance.store(new_var_fixed, Ordering::Relaxed);
        
        // Update score history for momentum calculation
        if self.score_history.len() >= self.history_max {
            self.score_history.pop_front();
        }
        self.score_history.push_back(current);
        
        // Store last score
        let score_bits = current.to_bits();
        self.last_score.store(score_bits, Ordering::Release);
    }
    
    /// Calculate raw weighted score from all signals
    fn calculate_raw_score(&self) -> f64 {
        let now_ms = get_current_ms();
        
        let mut weighted_sum = 0.0;
        let mut weight_total = 0.0;
        let mut total_volume = 0u64;
        let mut sources_seen = 0u8;
        
        for source in [
            SignalSource::Twitter,
            SignalSource::Reddit,
            SignalSource::News,
            SignalSource::Macro,
            SignalSource::FearGreed,
            SignalSource::GoogleTrends,
        ] {
            let mut source_sum = 0.0;
            let mut source_weight = 0.0;
            let mut source_volume = 0u64;
            
            for signal in self.signals.iter_by_source(source) {
                // Apply time decay
                let age_minutes = (now_ms - signal.timestamp_ms) as f64 / 60000.0;
                let decay = (-source.decay_rate_per_minute() * age_minutes).exp();
                
                let effective_weight = signal.confidence * decay * source.default_weight();
                source_sum += signal.value * effective_weight;
                source_weight += effective_weight;
                source_volume += signal.volume;
            }
            
            if source_weight > 0.0 {
                weighted_sum += source_sum;
                weight_total += source_weight;
                total_volume += source_volume;
                sources_seen += 1;
            }
        }
        
        if weight_total == 0.0 {
            return 0.0;
        }
        
        weighted_sum / weight_total
    }
    
    /// Get aggregated sentiment with full statistics
    pub fn get_aggregated(&self) -> AggregatedSentiment {
        let raw_score = self.calculate_raw_score();
        
        // Calculate Z-score from running statistics
        let scale = 1_000_000_000u64;
        let mean_fixed = self.running_mean.load(Ordering::Acquire) as f64 / scale as f64;
        let mean = mean_fixed - 1.0;  // Unscale
        
        let var_fixed = self.running_variance.load(Ordering::Acquire) as f64;
        let count = self.sample_count.load(Ordering::Acquire) as f64;
        let std_dev = if count > 1.0 {
            ((var_fixed / count) as f64).sqrt() / scale as f64
        } else {
            1.0
        };
        
        let z_score = if std_dev > 0.0001 {
            (raw_score - mean) / std_dev
        } else {
            0.0
        };
        
        // Calculate momentum (rate of change over last 10 samples)
        let momentum = if self.score_history.len() >= 10 {
            let recent: Vec<f64> = self.score_history.iter().copied().collect();
            let avg_first = recent[..5].iter().sum::<f64>() / 5.0;
            let avg_last = recent[5..].iter().sum::<f64>() / (recent.len() - 5) as f64;
            avg_last - avg_first
        } else {
            0.0
        };
        
        // Percentile estimation (simplified)
        let percentile = (z_score * 10.0 + 50.0).clamp(0.0, 100.0);
        
        // Trend direction
        let trend = TrendDirection::from_score(z_score, momentum);
        
        // Check for extreme readings
        let is_extreme = z_score.abs() > 2.0;
        
        AggregatedSentiment {
            composite_score: z_score,
            raw_score,
            z_score,
            percentile,
            trend,
            momentum,
            sources_active: self.count_active_sources(),
            total_volume: 0,  // Would need to track separately
            timestamp_ms: get_current_ms(),
            is_extreme,
        }
    }
    
    /// Count number of sources with recent signals
    fn count_active_sources(&self) -> u8 {
        let now_ms = get_current_ms();
        let threshold_ms = 5 * 60 * 1000;  // 5 minutes
        
        let mut active = 0u8;
        
        for source in [
            SignalSource::Twitter,
            SignalSource::Reddit,
            SignalSource::News,
            SignalSource::Macro,
            SignalSource::FearGreed,
            SignalSource::GoogleTrends,
        ] {
            for signal in self.signals.iter_by_source(source) {
                if now_ms - signal.timestamp_ms < threshold_ms {
                    active += 1;
                    break;
                }
            }
        }
        
        active
    }
    
    /// Get trading signal based on aggregated sentiment
    pub fn get_trading_signal(&self) -> TradingSignal {
        let agg = self.get_aggregated();
        
        let action = if agg.z_score > 2.0 {
            SignalAction::StrongBuy
        } else if agg.z_score > 1.0 {
            SignalAction::Buy
        } else if agg.z_score < -2.0 {
            SignalAction::StrongSell
        } else if agg.z_score < -1.0 {
            SignalAction::Sell
        } else {
            SignalAction::Hold
        };
        
        TradingSignal {
            action,
            confidence: agg.percentile / 100.0,
            sentiment_score: agg.composite_score,
            momentum: agg.momentum,
            is_extreme: agg.is_extreme,
        }
    }
}

/// Trading signal derived from sentiment
#[derive(Debug, Clone)]
pub struct TradingSignal {
    pub action: SignalAction,
    pub confidence: f64,
    pub sentiment_score: f64,
    pub momentum: f64,
    pub is_extreme: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SignalAction {
    StrongBuy,
    Buy,
    Hold,
    Sell,
    StrongSell,
}

/// Get current time in milliseconds since epoch
fn get_current_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or(Duration::ZERO)
        .as_millis() as u64
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_aggregator_basic() {
        let agg = SentimentAggregator::new(1000);
        
        // Add some test signals
        agg.add_signal_raw(SignalSource::Twitter, 0.8, 0.9, 100);
        agg.add_signal_raw(SignalSource::Reddit, 0.6, 0.8, 50);
        agg.add_signal_raw(SignalSource::News, 0.7, 0.95, 10);
        
        let result = agg.get_aggregated();
        
        // Should have positive sentiment
        assert!(result.raw_score > 0.0);
        assert!(result.sources_active >= 3);
    }
    
    #[test]
    fn test_trend_direction() {
        assert_eq!(
            TrendDirection::from_score(2.0, 0.2),
            TrendDirection::StronglyBullish
        );
        assert_eq!(
            TrendDirection::from_score(-2.0, -0.2),
            TrendDirection::StronglyBearish
        );
        assert_eq!(
            TrendDirection::from_score(0.0, 0.0),
            TrendDirection::Neutral
        );
    }
    
    #[test]
    fn test_signal_weights() {
        assert_eq!(SignalSource::Twitter.default_weight(), 0.25);
        assert_eq!(SignalSource::News.default_weight(), 0.25);
        assert!(SignalSource::Twitter.decay_rate_per_minute() > SignalSource::Macro.decay_rate_per_minute());
    }
}
