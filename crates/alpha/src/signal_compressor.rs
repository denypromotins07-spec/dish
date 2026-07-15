//! Rust-based signal compressor using online PCA
//! Reduces dimensionality of thousands of alpha signals into master execution vectors
//! Drastically reduces inference load on the execution engine

use std::collections::VecDeque;

/// Online PCA using incremental covariance estimation
pub struct OnlinePCA {
    /// Number of features (input signals)
    n_features: usize,
    /// Number of principal components to keep
    n_components: usize,
    /// Running mean estimate
    mean: Vec<f64>,
    /// Running covariance estimate (upper triangle stored)
    cov_upper: Vec<f64>,
    /// Current eigenvectors (principal components)
    components: Vec<Vec<f64>>,
    /// Explained variance ratios
    explained_variance: Vec<f64>,
    /// Decay factor for exponential weighting
    decay: f64,
    /// Number of samples seen
    n_samples: usize,
    /// Convergence threshold
    convergence_threshold: f64,
    /// Whether components have converged
    converged: bool,
}

impl OnlinePCA {
    pub fn new(n_features: usize, n_components: usize, decay: f64) -> Self {
        let cov_size = n_features * (n_features + 1) / 2;
        
        Self {
            n_features,
            n_components: n_components.min(n_features),
            mean: vec![0.0; n_features],
            cov_upper: vec![0.0; cov_size],
            components: Vec::new(),
            explained_variance: Vec::new(),
            decay,
            n_samples: 0,
            convergence_threshold: 1e-6,
            converged: false,
        }
    }

    /// Update PCA with new observation
    pub fn update(&mut self, x: &[f64]) {
        assert_eq!(x.len(), self.n_features);
        
        self.n_samples += 1;
        
        // Learning rate based on sample count and decay
        let alpha = if self.n_samples < 100 {
            1.0 / self.n_samples as f64
        } else {
            1.0 - self.decay
        };
        
        // Update running mean
        let diff = Vec::with_capacity(self.n_features);
        for i in 0..self.n_features {
            let delta = x[i] - self.mean[i];
            self.mean[i] += alpha * delta;
        }
        
        // Update running covariance (upper triangle)
        let mut idx = 0;
        for i in 0..self.n_features {
            for j in i..self.n_features {
                let delta_i = x[i] - self.mean[i];
                let delta_j = x[j] - self.mean[j];
                
                let old_cov = self.cov_upper[idx];
                self.cov_upper[idx] += alpha * (delta_i * delta_j - old_cov);
                
                idx += 1;
            }
        }
        
        // Recompute eigendecomposition periodically
        if self.n_samples % 10 == 0 || !self.converged {
            self._update_eigendecomposition();
        }
    }

    /// Transform input to principal component space
    pub fn transform(&self, x: &[f64]) -> Vec<f64> {
        if self.components.is_empty() {
            return x.to_vec();
        }

        // Center the input
        let centered: Vec<f64> = x
            .iter()
            .zip(self.mean.iter())
            .map(|(&xi, &mi)| xi - mi)
            .collect();

        // Project onto principal components
        let mut result = vec![0.0; self.n_components];
        for (k, component) in self.components.iter().enumerate().take(self.n_components) {
            result[k] = centered.iter().zip(component.iter()).map(|(&c, &v)| c * v).sum();
        }

        result
    }

    /// Inverse transform from PC space back to original space
    pub fn inverse_transform(&self, pc_values: &[f64]) -> Vec<f64> {
        if self.components.is_empty() {
            return pc_values.to_vec();
        }

        let mut result = self.mean.clone();
        
        for (k, &pc_val) in pc_values.iter().enumerate().take(self.n_components) {
            if k < self.components.len() {
                for (i, &comp_val) in self.components[k].iter().enumerate() {
                    result[i] += pc_val * comp_val;
                }
            }
        }

        result
    }

    /// Get reconstruction error for an observation
    pub fn reconstruction_error(&self, x: &[f64]) -> f64 {
        let pc_values = self.transform(x);
        let reconstructed = self.inverse_transform(&pc_values);
        
        let error: f64 = x
            .iter()
            .zip(reconstructed.iter())
            .map(|(&xi, &ri)| (xi - ri).powi(2))
            .sum();
        
        error.sqrt()
    }

    fn _update_eigendecomposition(&mut self) {
        // Reconstruct full covariance matrix
        let cov = self._reconstruct_covariance();
        
        // Power iteration to find top eigenvectors
        let (components, variances) = self._power_iteration(&cov, self.n_components);
        
        // Check convergence
        if let Some(old_components) = self.components.first() {
            let mut max_change = 0.0;
            for (new, old) in components.iter().zip(self.components.iter()) {
                let change: f64 = new
                    .iter()
                    .zip(old.iter())
                    .map(|(&a, &b)| (a - b).abs())
                    .sum();
                max_change = max_change.max(change);
            }
            
            if max_change < self.convergence_threshold {
                self.converged = true;
            }
        }
        
        self.components = components;
        self.explained_variance = variances;
    }

    fn _reconstruct_covariance(&self) -> Vec<Vec<f64>> {
        let mut cov = vec![vec![0.0; self.n_features]; self.n_features];
        
        let mut idx = 0;
        for i in 0..self.n_features {
            for j in i..self.n_features {
                cov[i][j] = self.cov_upper[idx];
                if i != j {
                    cov[j][i] = self.cov_upper[idx];
                }
                idx += 1;
            }
        }
        
        cov
    }

    fn _power_iteration(
        &self,
        cov: &[Vec<f64>],
        k: usize,
    ) -> (Vec<Vec<f64>>, Vec<f64>) {
        let n = self.n_features;
        let mut components = Vec::with_capacity(k);
        let mut variances = Vec::with_capacity(k);
        
        let mut residual = cov.to_vec();
        
        for _ in 0..k {
            // Find dominant eigenvector of residual
            let (eigenvec, eigenval) = self._find_dominant_eigenvector(&residual);
            
            components.push(eigenvec);
            variances.push(eigenval);
            
            // Deflate: subtract contribution from residual
            for i in 0..n {
                for j in 0..n {
                    residual[i][j] -= eigenval * components.last().unwrap()[i] * components.last().unwrap()[j];
                }
            }
        }
        
        (components, variances)
    }

    fn _find_dominant_eigenvector(&self, matrix: &[Vec<f64>]) -> (Vec<f64>, f64) {
        let n = matrix.len();
        let mut v = vec![1.0 / (n as f64).sqrt(); n];
        let mut eigenvalue = 0.0;
        
        for _ in 0..100 {
            // Matrix-vector multiplication
            let mut new_v = vec![0.0; n];
            for i in 0..n {
                for j in 0..n {
                    new_v[i] += matrix[i][j] * v[j];
                }
            }
            
            // Normalize
            let norm: f64 = new_v.iter().map(|&x| x.powi(2)).sum::<f64>().sqrt();
            if norm < 1e-10 {
                break;
            }
            
            eigenvalue = norm;
            v = new_v;
            for x in v.iter_mut() {
                *x /= norm;
            }
        }
        
        (v, eigenvalue)
    }

    /// Get total explained variance ratio
    pub fn total_explained_variance_ratio(&self) -> f64 {
        if self.explained_variance.is_empty() {
            return 0.0;
        }

        let total_var: f64 = self.cov_upper.iter().filter(|&&x| x > 0.0).sum();
        if total_var < 1e-10 {
            return 0.0;
        }

        let explained: f64 = self.explained_variance.iter().sum();
        explained / total_var
    }

    /// Get number of components needed to explain target variance
    pub fn components_for_variance(&self, target_ratio: f64) -> usize {
        let total: f64 = self.explained_variance.iter().sum();
        if total < 1e-10 {
            return self.n_components;
        }

        let mut cumulative = 0.0;
        for (i, &var) in self.explained_variance.iter().enumerate() {
            cumulative += var / total;
            if cumulative >= target_ratio {
                return i + 1;
            }
        }

        self.n_components
    }
}

/// Incremental SVD for very high-dimensional signals
pub struct IncrementalSVD {
    n_rows: usize,
    n_cols: usize,
    max_rank: usize,
    /// Left singular vectors
    u: Vec<Vec<f64>>,
    /// Singular values
    singular_values: Vec<f64>,
    /// Right singular vectors
    vt: Vec<Vec<f64>>,
    decay: f64,
    n_updates: usize,
}

impl IncrementalSVD {
    pub fn new(n_rows: usize, n_cols: usize, max_rank: usize, decay: f64) -> Self {
        Self {
            n_rows,
            n_cols,
            max_rank: max_rank.min(n_rows.min(n_cols)),
            u: Vec::new(),
            singular_values: Vec::new(),
            vt: Vec::new(),
            decay,
            n_updates: 0,
        }
    }

    pub fn update_row(&mut self, row_idx: usize, new_row: &[f64]) {
        self.n_updates += 1;
        
        // Initialize if needed
        if self.u.is_empty() {
            self._initialize(new_row);
            return;
        }
        
        // Rank-1 update to SVD (simplified Bunch-Nielsen method)
        self._rank1_update(row_idx, new_row);
    }

    fn _initialize(&mut self, first_row: &[f64]) {
        // Initialize with first observation
        let norm: f64 = first_row.iter().map(|&x| x.powi(2)).sum::<f64>().sqrt();
        
        if norm > 1e-10 {
            self.u = vec![vec![1.0]];
            self.singular_values = vec![norm];
            self.vt = vec![first_row.iter().map(|&x| x / norm).collect()];
        }
    }

    fn _rank1_update(&mut self, _row_idx: usize, new_row: &[f64]) {
        // Simplified rank-1 update
        // Full implementation would use Brand's algorithm
        
        let alpha = 1.0 - self.decay;
        
        // Update singular values with decay
        for sv in &mut self.singular_values {
            *sv = (1.0 - alpha) * (*sv) + alpha * sv.abs();
        }
        
        // Update right singular vectors
        for (vt_row, &new_val) in self.vt.iter_mut().zip(new_row.iter()).take(self.max_rank) {
            for x in vt_row.iter_mut() {
                *x = (1.0 - alpha) * (*x) + alpha * new_val;
            }
        }
    }

    pub fn project(&self, row: &[f64]) -> Vec<f64> {
        if self.vt.is_empty() {
            return row.to_vec();
        }

        let mut result = vec![0.0; self.vt.len()];
        for (k, vt_row) in self.vt.iter().enumerate() {
            result[k] = row.iter().zip(vt_row.iter()).map(|(&r, &v)| r * v).sum();
        }

        result
    }
}

/// Signal compression manager for trading system
pub struct SignalCompressor {
    pca: OnlinePCA,
    /// Raw signal buffer for warmup
    signal_buffer: VecDeque<Vec<f64>>,
    /// Compressed representations
    compressed_cache: VecDeque<Vec<f64>>,
    warmup_size: usize,
    is_warmed_up: bool,
}

impl SignalCompressor {
    pub fn new(n_signals: usize, n_components: usize, warmup_size: usize) -> Self {
        Self {
            pca: OnlinePCA::new(n_signals, n_components, 0.99),
            signal_buffer: VecDeque::with_capacity(warmup_size),
            compressed_cache: VecDeque::with_capacity(warmup_size),
            warmup_size,
            is_warmed_up: false,
        }
    }

    pub fn compress(&mut self, signals: &[f64]) -> Vec<f64> {
        // Store raw signal
        self.signal_buffer.push_back(signals.to_vec());
        
        // Update PCA
        self.pca.update(signals);
        
        // Transform
        let compressed = self.pca.transform(signals);
        self.compressed_cache.push_back(compressed.clone());
        
        // Check warmup
        if !self.is_warmed_up && self.signal_buffer.len() >= self.warmup_size {
            self.is_warmed_up = true;
        }
        
        compressed
    }

    pub fn decompress(&self, compressed: &[f64]) -> Vec<f64> {
        self.pca.inverse_transform(compressed)
    }

    pub fn compression_ratio(&self) -> f64 {
        if !self.is_warmed_up {
            return 1.0;
        }
        
        let n_original = self.pca.n_features;
        let n_compressed = self.pca.n_components;
        
        n_compressed as f64 / n_original as f64
    }

    pub fn explained_variance(&self) -> f64 {
        self.pca.total_explained_variance_ratio()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_online_pca_basic() {
        let mut pca = OnlinePCA::new(5, 3, 0.99);
        
        // Generate correlated data
        for _ in 0..100 {
            let x = vec![
                1.0,
                2.0,
                3.0,
                4.0,
                5.0,
            ];
            pca.update(&x);
        }
        
        // Test transformation
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let transformed = pca.transform(&x);
        
        assert_eq!(transformed.len(), 3);
    }

    #[test]
    fn test_signal_compressor() {
        let mut compressor = SignalCompressor::new(10, 3, 50);
        
        for i in 0..100 {
            let signals: Vec<f64> = (0..10).map(|j| (i + j) as f64 * 0.1).collect();
            let compressed = compressor.compress(&signals);
            
            assert_eq!(compressed.len(), 3);
        }
        
        assert!(compressor.is_warmed_up);
        assert!(compressor.explained_variance() > 0.0);
    }
}
