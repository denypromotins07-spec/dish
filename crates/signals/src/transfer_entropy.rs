//! Transfer Entropy calculator for measuring non-linear information flow
//! Measures causality between order book microstructure and price movements
//! Real-time computation with memory-efficient implementation

use std::collections::{HashMap, VecDeque};

/// Transfer Entropy calculator using k-nearest neighbor estimation
pub struct TransferEntropy {
    /// History buffer for source variable
    source_history: VecDeque<f64>,
    /// History buffer for target variable  
    target_history: VecDeque<f64>,
    /// Embedding dimension (history length)
    embedding_dim: usize,
    /// Number of nearest neighbors for estimation
    k_neighbors: usize,
    /// Maximum history size
    max_history: usize,
    /// Cache for probability estimates
    prob_cache: HashMap<u64, f64>,
}

impl TransferEntropy {
    pub fn new(embedding_dim: usize, k_neighbors: usize, max_history: usize) -> Self {
        Self {
            source_history: VecDeque::with_capacity(max_history),
            target_history: VecDeque::with_capacity(max_history),
            embedding_dim,
            k_neighbors,
            max_history,
            prob_cache: HashMap::new(),
        }
    }

    /// Update with new observations
    pub fn update(&mut self, source: f64, target: f64) {
        self.source_history.push_back(source);
        self.target_history.push_back(target);

        // Trim to max history
        while self.source_history.len() > self.max_history {
            self.source_history.pop_front();
        }
        while self.target_history.len() > self.max_history {
            self.target_history.pop_front();
        }

        // Clear cache on update
        self.prob_cache.clear();
    }

    /// Calculate transfer entropy from source to target
    /// TE(X->Y) = I(Y_future; X_past | Y_past)
    pub fn calculate(&self) -> Option<f64> {
        let min_samples = self.embedding_dim * 10;
        
        if self.source_history.len() < min_samples || self.target_history.len() < min_samples {
            return None;
        }

        // Build state vectors
        let n = self.source_history.len() - self.embedding_dim;
        if n < self.k_neighbors {
            return None;
        }

        let mut te_sum = 0.0;
        let mut count = 0;

        // For each time point, estimate conditional mutual information
        for t in self.embedding_dim..(self.source_history.len() - 1) {
            // Current state vectors
            let y_past = self._get_target_state(t);
            let x_past = self._get_source_state(t);
            let y_future = self.target_history[t + 1];

            // Find k-nearest neighbors in joint space
            let neighbors = self._find_knn_joint(t, self.k_neighbors);

            if neighbors.is_empty() {
                continue;
            }

            // Estimate probabilities using neighbor counts
            let p_y_future_given_past = self._estimate_conditional_prob(y_future, &y_past, &neighbors);
            let p_y_future_given_both = self._estimate_conditional_prob_full(y_future, &y_past, &x_past, &neighbors);

            if p_y_future_given_past > 0.0 && p_y_future_given_both > 0.0 {
                te_sum += (p_y_future_given_both / p_y_future_given_past).ln();
                count += 1;
            }
        }

        if count == 0 {
            return Some(0.0);
        }

        Some(te_sum / count as f64)
    }

    /// Calculate bidirectional transfer entropy
    pub fn calculate_bidirectional(&self) -> Option<(f64, f64)> {
        let te_source_to_target = self.calculate()?;

        // Swap histories to calculate reverse direction
        let reversed = TransferEntropy {
            source_history: self.target_history.clone(),
            target_history: self.source_history.clone(),
            embedding_dim: self.embedding_dim,
            k_neighbors: self.k_neighbors,
            max_history: self.max_history,
            prob_cache: HashMap::new(),
        };

        let te_target_to_source = reversed.calculate()?;

        Some((te_source_to_target, te_target_to_source))
    }

    /// Get net information flow (positive means source drives target)
    pub fn net_flow(&self) -> Option<f64> {
        let (te_xy, te_yx) = self.calculate_bidirectional()?;
        Some(te_xy - te_yx)
    }

    fn _get_source_state(&self, t: usize) -> Vec<f64> {
        let mut state = Vec::with_capacity(self.embedding_dim);
        for i in 0..self.embedding_dim {
            state.push(self.source_history[t - i]);
        }
        state
    }

    fn _get_target_state(&self, t: usize) -> Vec<f64> {
        let mut state = Vec::with_capacity(self.embedding_dim);
        for i in 0..self.embedding_dim {
            state.push(self.target_history[t - i]);
        }
        state
    }

    fn _find_knn_joint(&self, t: usize, k: usize) -> Vec<usize> {
        let y_past = self._get_target_state(t);
        let x_past = self._get_source_state(t);

        let mut distances: Vec<(usize, f64)> = Vec::new();

        for i in self.embedding_dim..(self.source_history.len().saturating_sub(1)) {
            if i == t {
                continue;
            }

            let y_past_i = self._get_target_state(i);
            let x_past_i = self._get_source_state(i);

            // Euclidean distance in joint space
            let dist = self._euclidean_distance_joint(&y_past, &x_past, &y_past_i, &x_past_i);
            distances.push((i, dist));
        }

        // Sort by distance and take k nearest
        distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        distances.truncate(k);

        distances.into_iter().map(|(idx, _)| idx).collect()
    }

    fn _euclidean_distance_joint(
        &self,
        y1: &[f64],
        x1: &[f64],
        y2: &[f64],
        x2: &[f64],
    ) -> f64 {
        let mut sum = 0.0;
        
        for (a, b) in y1.iter().zip(y2.iter()) {
            sum += (a - b).powi(2);
        }
        
        for (a, b) in x1.iter().zip(x2.iter()) {
            sum += (a - b).powi(2);
        }

        sum.sqrt()
    }

    fn _estimate_conditional_prob(
        &self,
        y_future: f64,
        y_past: &[f64],
        neighbors: &[usize],
    ) -> f64 {
        // Count neighbors with similar y_future
        let mut count = 0;
        let threshold = 0.1; // Bandwidth parameter

        for &idx in neighbors {
            let y_fut_idx = self.target_history[idx + 1];
            if (y_future - y_fut_idx).abs() < threshold {
                count += 1;
            }
        }

        (count as f64) / (neighbors.len() as f64 + 1e-10)
    }

    fn _estimate_conditional_prob_full(
        &self,
        y_future: f64,
        y_past: &[f64],
        x_past: &[f64],
        neighbors: &[usize],
    ) -> f64 {
        // More refined estimate considering x_past
        let mut weighted_count = 0.0;
        let threshold = 0.1;

        for &idx in neighbors {
            let y_fut_idx = self.target_history[idx + 1];
            let x_past_idx = self._get_source_state(idx);

            // Weight by similarity in x_past space
            let x_sim = self._gaussian_kernel(x_past, &x_past_idx, 0.5);
            
            if (y_future - y_fut_idx).abs() < threshold {
                weighted_count += x_sim;
            }
        }

        weighted_count / (neighbors.len() as f64 + 1e-10)
    }

    fn _gaussian_kernel(&self, x1: &[f64], x2: &[f64], bandwidth: f64) -> f64 {
        let dist_sq: f64 = x1.iter()
            .zip(x2.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum();

        (-dist_sq / (2.0 * bandwidth.powi(2))).exp()
    }
}

/// Rolling Transfer Entropy tracker for continuous monitoring
pub struct RollingTransferEntropy {
    te_calculator: TransferEntropy,
    /// Rolling window of TE values
    te_history: VecDeque<f64>,
    /// Significance threshold
    significance_threshold: f64,
    /// Minimum samples before computing
    warmup_samples: usize,
}

impl RollingTransferEntropy {
    pub fn new(
        embedding_dim: usize,
        k_neighbors: usize,
        max_history: usize,
        significance_threshold: f64,
    ) -> Self {
        Self {
            te_calculator: TransferEntropy::new(embedding_dim, k_neighbors, max_history),
            te_history: VecDeque::with_capacity(100),
            significance_threshold,
            warmup_samples: embedding_dim * 20,
        }
    }

    pub fn update(&mut self, source: f64, target: f64) -> Option<f64> {
        self.te_calculator.update(source, target);

        // Only compute after warmup
        if self.te_calculator.source_history.len() < self.warmup_samples {
            return None;
        }

        // Compute TE periodically (every 10 samples to save computation)
        if self.te_calculator.source_history.len() % 10 == 0 {
            if let Some(te) = self.te_calculator.calculate() {
                self.te_history.push_back(te);
                
                // Keep limited history
                if self.te_history.len() > 100 {
                    self.te_history.pop_front();
                }

                return Some(te);
            }
        }

        None
    }

    /// Check if information flow is significant
    pub fn is_significant(&self) -> bool {
        if self.te_history.is_empty() {
            return false;
        }

        let avg_te: f64 = self.te_history.iter().sum::<f64>() / self.te_history.len() as f64;
        avg_te > self.significance_threshold
    }

    /// Get average TE over recent history
    pub fn average_te(&self) -> Option<f64> {
        if self.te_history.is_empty() {
            return None;
        }

        Some(self.te_history.iter().sum::<f64>() / self.te_history.len() as f64)
    }

    /// Detect regime change in information flow
    pub fn detect_regime_change(&self, window: usize) -> Option<bool> {
        if self.te_history.len() < window * 2 {
            return None;
        }

        let recent: f64 = self.te_history.iter().rev().take(window).sum::<f64>() / window as f64;
        let older: f64 = self.te_history.iter().rev().skip(window).take(window).sum::<f64>() / window as f64;

        // Significant change if difference exceeds threshold
        let change = (recent - older).abs();
        Some(change > self.significance_threshold)
    }
}

/// Causality test result
#[derive(Debug, Clone)]
pub struct CausalityResult {
    pub source_to_target: f64,
    pub target_to_source: f64,
    pub net_flow: f64,
    pub is_significant: bool,
    pub dominant_direction: String,
}

impl CausalityResult {
    pub fn interpret(&self) -> &str {
        if !self.is_significant {
            return "No significant causality detected"
        }

        if self.net_flow.abs() < 0.01 {
            return "Bidirectional causality (feedback loop)"
        } else if self.net_flow > 0.0 {
            return "Source drives target (unidirectional)"
        } else {
            return "Target drives source (reverse causality)"
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_transfer_entropy_basic() {
        let mut te = TransferEntropy::new(3, 5, 500);

        // Generate correlated data where source leads target
        for i in 0..200 {
            let source = (i as f64 * 0.1).sin();
            // Target lags source by ~5 steps
            let target = ((i as f64 - 5.0) * 0.1).sin();
            te.update(source, target);
        }

        let result = te.calculate();
        assert!(result.is_some());
    }

    #[test]
    fn test_rolling_te() {
        let mut rolling = RollingTransferEntropy::new(3, 5, 500, 0.01);

        for i in 0..300 {
            let source = (i as f64 * 0.1).sin();
            let target = ((i as f64 - 3.0) * 0.1).sin() + (i as f64 * 0.01).cos() * 0.1;
            rolling.update(source, target);
        }

        let avg = rolling.average_te();
        assert!(avg.is_some());
    }
}
