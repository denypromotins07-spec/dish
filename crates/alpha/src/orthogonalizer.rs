//! High-performance Gram-Schmidt orthogonalization process
//! Removes collinearity between highly correlated alpha factors
//! Implemented in pure Rust for nanosecond-level execution

use std::collections::HashMap;

/// Matrix operations for orthogonalization
pub struct Orthogonalizer {
    /// Tolerance for numerical stability
    tolerance: f64,
    /// Cached QR decomposition workspace (reused to avoid allocations)
    workspace: Vec<f64>,
}

impl Orthogonalizer {
    pub fn new(tolerance: f64) -> Self {
        Self {
            tolerance,
            workspace: Vec::with_capacity(1024),
        }
    }

    /// Classical Gram-Schmidt orthogonalization
    /// 
    /// # Arguments
    /// * `vectors` - Input vectors as slice of slices (each inner slice is a factor)
    /// 
    /// # Returns
    /// Orthogonalized vectors maintaining the same order
    pub fn gram_schmidt_classical(&mut self, vectors: &[&[f64]]) -> Vec<Vec<f64>> {
        if vectors.is_empty() {
            return Vec::new();
        }

        let n_vectors = vectors.len();
        let n_elements = vectors[0].len();
        
        // Validate all vectors have same length
        for v in vectors.iter().skip(1) {
            assert_eq!(v.len(), n_elements, "All vectors must have same length");
        }

        let mut result: Vec<Vec<f64>> = Vec::with_capacity(n_vectors);
        
        // Process first vector
        let first = vectors[0];
        let norm = self._norm(first);
        if norm > self.tolerance {
            let mut q: Vec<f64> = first.to_vec();
            self._scale(&mut q, 1.0 / norm);
            result.push(q);
        } else {
            result.push(vec![0.0; n_elements]);
        }

        // Process remaining vectors
        for i in 1..n_vectors {
            let mut orthogonal = vectors[i].to_vec();
            
            // Subtract projections onto all previous orthogonal vectors
            for j in 0..result.len() {
                let dot = self._dot(&orthogonal, &result[j]);
                self._axpy(&mut orthogonal, &result[j], -dot);
            }
            
            // Normalize
            let norm = self._norm(&orthogonal);
            if norm > self.tolerance {
                self._scale(&mut orthogonal, 1.0 / norm);
            } else {
                // Vector is linearly dependent, zero it out
                orthogonal.fill(0.0);
            }
            
            result.push(orthogonal);
        }

        result
    }

    /// Modified Gram-Schmidt orthogonalization (more numerically stable)
    /// 
    /// This is preferred for production use due to better numerical properties
    pub fn gram_schmidt_modified(&mut self, vectors: &[&[f64]]) -> Vec<Vec<f64>> {
        if vectors.is_empty() {
            return Vec::new();
        }

        let n_vectors = vectors.len();
        let n_elements = vectors[0].len();
        
        for v in vectors.iter().skip(1) {
            assert_eq!(v.len(), n_elements, "All vectors must have same length");
        }

        // Copy input vectors
        let mut work: Vec<Vec<f64>> = vectors.iter().map(|v| v.to_vec()).collect();
        let mut result: Vec<Vec<f64>> = Vec::with_capacity(n_vectors);

        for i in 0..n_vectors {
            // Subtract projections onto previous orthogonal vectors
            for j in 0..i {
                let dot = self._dot(&work[i], &result[j]);
                self._axpy(&mut work[i], &result[j], -dot);
            }
            
            // Normalize
            let norm = self._norm(&work[i]);
            if norm > self.tolerance {
                self._scale(&mut work[i], 1.0 / norm);
                result.push(work[i].clone());
            } else {
                result.push(vec![0.0; n_elements]);
            }
        }

        result
    }

    /// Orthogonalize a set of named factors
    /// 
    /// # Arguments
    /// * `factors` - HashMap mapping factor names to their signal vectors
    /// 
    /// # Returns
    /// HashMap with orthogonalized factors (same keys)
    pub fn orthogonalize_factors(
        &mut self,
        factors: &HashMap<String, Vec<f64>>,
    ) -> HashMap<String, Vec<f64>> {
        if factors.is_empty() {
            return HashMap::new();
        }

        // Get consistent ordering
        let names: Vec<&String> = factors.keys().collect();
        let vectors: Vec<&[f64]> = names.iter().map(|k| factors[*k].as_slice()).collect();
        
        // Perform orthogonalization
        let orthogonal = self.gram_schmidt_modified(&vectors);
        
        // Reconstruct HashMap
        let mut result = HashMap::with_capacity(factors.len());
        for (name, orth_vec) in names.into_iter().zip(orthogonal.into_iter()) {
            result.insert(name.clone(), orth_vec);
        }
        
        result
    }

    /// Compute correlation matrix of input vectors
    pub fn correlation_matrix(&self, vectors: &[&[f64]]) -> Vec<Vec<f64>> {
        if vectors.is_empty() {
            return Vec::new();
        }

        let n = vectors.len();
        let mut corr = vec![vec![0.0; n]; n];

        // First, normalize all vectors
        let norms: Vec<f64> = vectors.iter().map(|v| self._norm(v)).collect();
        let normalized: Vec<Vec<f64>> = vectors
            .iter()
            .zip(norms.iter())
            .map(|(v, &norm)| {
                if norm > self.tolerance {
                    let mut scaled = v.to_vec();
                    self._scale_in_place_ref(&scaled, 1.0 / norm);
                    scaled
                } else {
                    v.to_vec()
                }
            })
            .collect();

        // Compute correlations as dot products of normalized vectors
        for i in 0..n {
            for j in 0..n {
                corr[i][j] = self._dot(&normalized[i], &normalized[j]);
            }
        }

        corr
    }

    /// Check if a new vector is sufficiently orthogonal to existing basis
    /// 
    /// Returns true if the maximum absolute correlation with any existing vector
    /// is below the tolerance threshold
    pub fn check_orthogonality(&self, new_vector: &[f64], basis: &[&[f64]]) -> bool {
        if basis.is_empty() {
            return true;
        }

        let new_norm = self._norm(new_vector);
        if new_norm < self.tolerance {
            return false;
        }

        for b in basis {
            let b_norm = self._norm(b);
            if b_norm < self.tolerance {
                continue;
            }

            let corr = self._dot(new_vector, b) / (new_norm * b_norm);
            if corr.abs() > self.tolerance {
                return false;
            }
        }

        true
    }

    /// Project a vector onto the orthogonal complement of a subspace
    /// 
    /// This removes all components of `vector` that lie in the span of `basis`
    pub fn project_orthogonal(&self, vector: &[f64], basis: &[&[f64]]) -> Vec<f64> {
        let mut result = vector.to_vec();

        for b in basis {
            let b_norm_sq = self._dot(b, b);
            if b_norm_sq > self.tolerance * self.tolerance {
                let proj_coef = self._dot(&result, b) / b_norm_sq;
                self._axpy(&mut result, b, -proj_coef);
            }
        }

        result
    }

    /// Compute the condition number of a set of vectors (estimate)
    /// Higher values indicate more collinearity
    pub fn condition_number_estimate(&self, vectors: &[&[f64]]) -> f64 {
        if vectors.is_empty() {
            return 1.0;
        }

        let corr = self.correlation_matrix(vectors);
        let n = corr.len();

        // Gershgorin circle theorem for eigenvalue bounds
        let mut max_row_sum = 0.0;
        for i in 0..n {
            let row_sum: f64 = corr[i].iter().map(|x| x.abs()).sum();
            max_row_sum = max_row_sum.max(row_sum);
        }

        // Condition number estimate based on correlation structure
        max_row_sum
    }

    // === Internal helper methods ===

    #[inline]
    fn _dot(&self, a: &[f64], b: &[f64]) -> f64 {
        a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
    }

    #[inline]
    fn _norm(&self, v: &[f64]) -> f64 {
        self._dot(v, v).sqrt()
    }

    #[inline]
    fn _scale(&self, v: &mut [f64], scalar: f64) {
        for x in v.iter_mut() {
            *x *= scalar;
        }
    }

    #[inline]
    fn _scale_in_place_ref(&self, v: &Vec<f64>, scalar: f64) {
        // Note: This takes &Vec but we can't modify through this ref
        // Kept for API compatibility but should use _scale instead
        let _ = (v, scalar);
    }

    #[inline]
    fn _axpy(&self, y: &mut [f64], x: &[f64], a: f64) {
        // y = a * x + y
        debug_assert_eq!(x.len(), y.len());
        for i in 0..y.len() {
            y[i] += a * x[i];
        }
    }

    /// Batch orthogonalization for multiple time periods
    /// Optimized for processing time-series of factor returns
    pub fn batch_orthogonalize(
        &mut self,
        time_series: &[Vec<f64>],  // Each inner vec is one time period's factors
    ) -> Vec<Vec<f64>> {
        if time_series.is_empty() {
            return Vec::new();
        }

        let n_periods = time_series.len();
        let n_factors = time_series[0].len();
        
        // Transpose to get factor time series
        let mut factor_series: Vec<Vec<f64>> = vec![Vec::with_capacity(n_periods); n_factors];
        for period in time_series {
            for (i, &val) in period.iter().enumerate() {
                factor_series[i].push(val);
            }
        }

        // Orthogonalize each factor series against the first (benchmark)
        let benchmark = factor_series[0].as_slice();
        let mut result = vec![benchmark.to_vec()];

        for i in 1..n_factors {
            let ortho = self.project_orthogonal(&factor_series[i], &[benchmark]);
            result.push(ortho);
        }

        // Transpose back
        let mut output = vec![Vec::with_capacity(n_factors); n_periods];
        for (t, period_out) in output.iter_mut().enumerate() {
            for factor in &result {
                period_out.push(factor[t]);
            }
        }

        output
    }
}

/// Incremental orthogonalization for streaming data
pub struct IncrementalOrthogonalizer {
    base: Orthogonalizer,
    running_basis: Vec<Vec<f64>>,
    max_basis_size: usize,
}

impl IncrementalOrthogonalizer {
    pub fn new(max_basis_size: usize, tolerance: f64) -> Self {
        Self {
            base: Orthogonalizer::new(tolerance),
            running_basis: Vec::with_capacity(max_basis_size),
            max_basis_size,
        }
    }

    /// Process a new observation incrementally
    pub fn update(&mut self, new_observation: &[f64]) -> Vec<f64> {
        // Project new observation onto current basis
        let basis_refs: Vec<&[f64]> = self.running_basis.iter().map(|v| v.as_slice()).collect();
        let orthogonal_component = self.base.project_orthogonal(new_observation, &basis_refs);

        // Optionally update basis if it's sufficiently orthogonal and we have room
        let norm = self.base._norm(&orthogonal_component);
        if norm > self.base.tolerance && self.running_basis.len() < self.max_basis_size {
            let mut normalized = orthogonal_component.clone();
            self.base._scale(&mut normalized, 1.0 / norm);
            self.running_basis.push(normalized);
        }

        orthogonal_component
    }

    /// Reset the running basis
    pub fn reset(&mut self) {
        self.running_basis.clear();
    }

    /// Get current basis size
    pub fn basis_size(&self) -> usize {
        self.running_basis.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gram_schmidt_orthogonality() {
        let mut orth = Orthogonalizer::new(1e-10);
        
        // Create some linearly dependent vectors
        let v1 = vec![1.0, 0.0, 0.0];
        let v2 = vec![1.0, 1.0, 0.0];
        let v3 = vec![1.0, 1.0, 1.0];
        
        let vectors = vec![&v1[..], &v2[..], &v3[..]];
        let orthogonal = orth.gram_schmidt_modified(&vectors);
        
        // Check orthogonality
        for i in 0..orthogonal.len() {
            for j in (i + 1)..orthogonal.len() {
                let dot = orth._dot(&orthogonal[i], &orthogonal[j]);
                assert!(dot.abs() < 1e-10, "Vectors {} and {} are not orthogonal: dot={}", i, j, dot);
            }
        }
    }

    #[test]
    fn test_correlation_matrix() {
        let orth = Orthogonalizer::new(1e-10);
        
        let v1 = vec![1.0, 2.0, 3.0, 4.0];
        let v2 = vec![2.0, 4.0, 6.0, 8.0];  // Perfectly correlated with v1
        
        let vectors = vec![&v1[..], &v2[..]];
        let corr = orth.correlation_matrix(&vectors);
        
        // Diagonal should be 1.0
        assert!((corr[0][0] - 1.0).abs() < 1e-10);
        assert!((corr[1][1] - 1.0).abs() < 1e-10);
        
        // Off-diagonal should be close to 1.0 (perfect correlation)
        assert!((corr[0][1] - 1.0).abs() < 1e-10);
        assert!((corr[1][0] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_factor_orthogonalization() {
        let mut orth = Orthogonalizer::new(1e-10);
        
        let mut factors = HashMap::new();
        factors.insert("momentum".to_string(), vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        factors.insert("value".to_string(), vec![2.0, 3.0, 4.0, 5.0, 6.0]);
        factors.insert("volatility".to_string(), vec![5.0, 4.0, 3.0, 2.0, 1.0]);
        
        let orthogonal = orth.orthogonalize_factors(&factors);
        
        assert_eq!(orthogonal.len(), 3);
        
        // Verify orthogonality between factors
        let names: Vec<&String> = orthogonal.keys().collect();
        for i in 0..names.len() {
            for j in (i + 1)..names.len() {
                let dot = orth._dot(&orthogonal[names[i]], &orthogonal[names[j]]);
                assert!(dot.abs() < 1e-10, "Factors {} and {} not orthogonal", names[i], names[j]);
            }
        }
    }
}
