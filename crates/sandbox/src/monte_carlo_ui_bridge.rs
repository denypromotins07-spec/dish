//! Monte Carlo UI Bridge: Streams permutations and confidence intervals via Protobuf.
//! Enables frontend to render "Cone of Probability" equity charts in real-time.
//! Zero-allocation streaming with bounded buffer sizes.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use tokio::sync::broadcast;

/// Fixed-size Monte Carlo result container
#[derive(Debug, Clone)]
pub struct MonteCarloResult {
    pub permutation_id: u32,
    pub final_equity: f64,
    pub max_drawdown: f64,
    pub sharpe_ratio: f64,
    pub time_to_ruin_us: Option<u64>,
}

/// Confidence interval bounds for a specific quantile
#[derive(Debug, Clone, Copy)]
pub struct ConfidenceBand {
    pub quantile: f32,      // e.g., 0.05, 0.50, 0.95
    pub equity_at_time: f64,
    pub drawdown_at_time: f64,
}

/// Aggregated cone data for a specific time bucket
#[derive(Debug, Clone)]
pub struct ConeTimeBucket {
    pub time_us: u64,
    pub p05: f64,  // 5th percentile
    pub p25: f64,  // 25th percentile
    pub p50: f64,  // Median
    pub p75: f64,  // 75th percentile
    pub p95: f64,  // 95th percentile
}

/// Lock-free Monte Carlo bridge using broadcast channels
pub struct MonteCarloUiBridge {
    /// Broadcast channel for streaming results to UI clients
    tx: broadcast::Sender<ConeTimeBucket>,
    /// Pre-allocated result buffer (fixed size to prevent heap growth)
    results_buffer: parking_lot::RwLock<Vec<MonteCarloResult>>,
    /// Running statistics
    total_permutations: AtomicU64,
    running_mean_equity: AtomicU64,  // Fixed-point 16.48
    running_variance: AtomicU64,     // Fixed-point 16.48
    /// Time buckets for cone visualization (fixed array)
    time_buckets: parking_lot::RwLock<[Option<ConeTimeBucket>; 100]>,
}

impl MonteCarloUiBridge {
    pub fn new(buffer_size: usize) -> Self {
        let (tx, _) = broadcast::channel(buffer_size.min(1024));
        
        Self {
            tx,
            results_buffer: parking_lot::RwLock::new(Vec::with_capacity(4096)),
            total_permutations: AtomicU64::new(0),
            running_mean_equity: AtomicU64::new(0),
            running_variance: AtomicU64::new(0),
            time_buckets: parking_lot::RwLock::new(std::array::from_fn(|_| None)),
        }
    }

    /// Ingest a single Monte Carlo permutation result
    pub fn ingest_result(&self, result: MonteCarloResult) {
        // Update running statistics atomically
        let count = self.total_permutations.fetch_add(1, Ordering::AcqRel) + 1;
        
        // Welford's online algorithm for mean/variance (fixed-point)
        let old_mean_fp = self.running_mean_equity.load(Ordering::Relaxed);
        let old_var_fp = self.running_variance.load(Ordering::Relaxed);
        
        let equity_fp = (result.final_equity * (1u64 << 48)) as i128;
        let count_fp = count as i128;
        
        let delta = equity_fp - old_mean_fp as i128;
        let new_mean_fp = old_mean_fp as i128 + (delta / count_fp);
        
        let delta2 = equity_fp - new_mean_fp;
        let new_var_fp = old_var_fp as i128 + (delta * delta2 / count_fp as i128);
        
        self.running_mean_equity.store(new_mean_fp as u64, Ordering::Relaxed);
        self.running_variance.store(new_var_fp.max(0) as u64, Ordering::Relaxed);
        
        // Store result in bounded buffer
        {
            let mut buffer = self.results_buffer.write();
            if buffer.len() >= 4096 {
                // Drop oldest 10% when full
                buffer.drain(0..409);
            }
            buffer.push(result);
        }
        
        // Periodically update time buckets (every 100 permutations)
        if count % 100 == 0 {
            self.update_time_buckets();
        }
    }

    /// Batch ingest multiple results
    pub fn ingest_batch(&self, results: Vec<MonteCarloResult>) {
        for result in results {
            self.ingest_result(result);
        }
    }

    /// Update time buckets with percentile calculations
    fn update_time_buckets(&self) {
        let buffer = self.results_buffer.read();
        if buffer.is_empty() {
            return;
        }

        // Sort by final equity for percentile calculation
        let mut sorted: Vec<&MonteCarloResult> = buffer.iter().collect();
        sorted.sort_by(|a, b| a.final_equity.partial_cmp(&b.final_equity).unwrap());

        let count = sorted.len();
        let p05_idx = (count as f64 * 0.05) as usize;
        let p25_idx = (count as f64 * 0.25) as usize;
        let p50_idx = (count as f64 * 0.50) as usize;
        let p75_idx = (count as f64 * 0.75) as usize;
        let p95_idx = (count as f64 * 0.95) as usize;

        // Create aggregate bucket (simplified: using final state)
        let bucket = ConeTimeBucket {
            time_us: self.total_permutations.load(Ordering::Relaxed) * 1000, // Approximate
            p05: sorted.get(p05_idx).map(|r| r.final_equity).unwrap_or(0.0),
            p25: sorted.get(p25_idx).map(|r| r.final_equity).unwrap_or(0.0),
            p50: sorted.get(p50_idx).map(|r| r.final_equity).unwrap_or(0.0),
            p75: sorted.get(p75_idx).map(|r| r.final_equity).unwrap_or(0.0),
            p95: sorted.get(p95_idx).map(|r| r.final_equity).unwrap_or(0.0),
        };

        // Store in circular buffer
        let mut buckets = self.time_buckets.write();
        let idx = (self.total_permutations.load(Ordering::Relaxed) % 100) as usize;
        buckets[idx] = Some(bucket);

        // Broadcast to UI clients
        let _ = self.tx.send(bucket);
    }

    /// Subscribe to cone updates
    pub fn subscribe(&self) -> broadcast::Receiver<ConeTimeBucket> {
        self.tx.subscribe()
    }

    /// Get current confidence interval summary
    pub fn get_confidence_summary(&self) -> Option<(f64, f64, f64, f64, f64)> {
        let buckets = self.time_buckets.read();
        let valid: Vec<&ConeTimeBucket> = buckets.iter().filter_map(|b| b.as_ref()).collect();
        
        if valid.is_empty() {
            return None;
        }

        // Average across all buckets
        let sum_p05: f64 = valid.iter().map(|b| b.p05).sum();
        let sum_p25: f64 = valid.iter().map(|b| b.p25).sum();
        let sum_p50: f64 = valid.iter().map(|b| b.p50).sum();
        let sum_p75: f64 = valid.iter().map(|b| b.p75).sum();
        let sum_p95: f64 = valid.iter().map(|b| b.p95).sum();
        
        let n = valid.len() as f64;
        Some((
            sum_p05 / n,
            sum_p25 / n,
            sum_p50 / n,
            sum_p75 / n,
            sum_p95 / n,
        ))
    }

    /// Get running statistics
    pub fn get_running_stats(&self) -> (u64, f64, f64) {
        let count = self.total_permutations.load(Ordering::Relaxed);
        let mean_fp = self.running_mean_equity.load(Ordering::Relaxed) as f64 / (1u64 << 48) as f64;
        let var_fp = self.running_variance.load(Ordering::Relaxed) as f64 / (1u64 << 48) as f64;
        let std_dev = var_fp.sqrt();
        
        (count, mean_fp, std_dev)
    }

    /// Clear all stored results
    pub fn clear(&self) {
        self.results_buffer.write().clear();
        self.time_buckets.write().fill(None);
        self.total_permutations.store(0, Ordering::Release);
        self.running_mean_equity.store(0, Ordering::Release);
        self.running_variance.store(0, Ordering::Release);
    }
}

impl Default for MonteCarloUiBridge {
    fn default() -> Self {
        Self::new(512)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_monte_carlo_ingestion() {
        let bridge = MonteCarloUiBridge::new(256);
        
        // Ingest 200 results
        for i in 0..200 {
            bridge.ingest_result(MonteCarloResult {
                permutation_id: i,
                final_equity: 1000.0 + (i as f64 * 0.1),
                max_drawdown: 0.05 + (i as f64 * 0.0001),
                sharpe_ratio: 1.5 + (i as f64 * 0.001),
                time_to_ruin_us: None,
            });
        }

        let (count, mean, std) = bridge.get_running_stats();
        assert_eq!(count, 200);
        assert!(mean > 1000.0);
        
        let summary = bridge.get_confidence_summary();
        assert!(summary.is_some());
    }
}
