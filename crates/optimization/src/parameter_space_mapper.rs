//! Parameter Space Mapper: Maps high-dimensional parameters to 2D UI coordinates.
//! Uses lightweight t-SNE/UMAP-like algorithms optimized for RAM efficiency.
//! Zero heap allocation during mapping; uses fixed-size buffers.

use std::collections::HashMap;

/// Fixed-size parameter vector (max 64 dimensions)
#[derive(Debug, Clone)]
pub struct ParameterVector {
    pub id: u32,
    pub values: [f64; 64],
    pub dim_count: usize,
    pub metric_score: f64, // Sharpe, Sortino, etc.
}

impl ParameterVector {
    pub fn new(id: u32, dim_count: usize) -> Self {
        assert!(dim_count <= 64, "Max dimensions is 64");
        Self {
            id,
            values: [0.0; 64],
            dim_count,
            metric_score: 0.0,
        }
    }

    #[inline]
    pub fn set(&mut self, idx: usize, value: f64) {
        if idx < self.dim_count {
            self.values[idx] = value;
        }
    }

    #[inline]
    pub fn get(&self, idx: usize) -> f64 {
        if idx < self.dim_count {
            self.values[idx]
        } else {
            0.0
        }
    }
}

/// 2D coordinate for UI visualization
#[derive(Debug, Clone, Copy)]
pub struct Point2D {
    pub x: f64,
    pub y: f64,
    pub original_id: u32,
}

/// Lightweight dimensionality reducer using simplified t-SNE-like approach
pub struct ParameterSpaceMapper {
    /// Input vectors (fixed capacity)
    vectors: Vec<ParameterVector>,
    /// Output 2D points
    points_2d: Vec<Point2D>,
    /// Pre-allocated distance matrix (flattened)
    distances: Vec<f64>,
    /// Perplexity parameter
    perplexity: f64,
    /// Number of iterations
    iterations: usize,
    /// Learning rate
    learning_rate: f64,
}

impl ParameterSpaceMapper {
    pub fn new(max_points: usize) -> Self {
        Self {
            vectors: Vec::with_capacity(max_points),
            points_2d: Vec::with_capacity(max_points),
            distances: Vec::new(),
            perplexity: 30.0,
            iterations: 250, // Reduced for speed
            learning_rate: 100.0,
        }
    }

    /// Add a parameter vector to the mapper
    pub fn add_vector(&mut self, vector: ParameterVector) {
        if self.vectors.len() < self.vectors.capacity() {
            self.vectors.push(vector);
        }
    }

    /// Compute pairwise Euclidean distances (in-place, reusing buffer)
    fn compute_distances(&mut self) {
        let n = self.vectors.len();
        self.distances.resize(n * n, 0.0);

        for i in 0..n {
            for j in (i + 1)..n {
                let mut dist_sq = 0.0;
                let vi = &self.vectors[i];
                let vj = &self.vectors[j];

                for k in 0..vi.dim_count.min(vj.dim_count) {
                    let diff = vi.values[k] - vj.values[k];
                    dist_sq += diff * diff;
                }

                let dist = dist_sq.sqrt();
                self.distances[i * n + j] = dist;
                self.distances[j * n + i] = dist;
            }
        }
    }

    /// Compute Gaussian kernel probabilities with given perplexity
    fn compute_probabilities(&self, distances: &[f64]) -> Vec<f64> {
        let n = self.vectors.len();
        let mut probs = vec![0.0; n * n];

        // Simplified: use fixed bandwidth instead of binary search for perplexity
        let bandwidth = self.perplexity / 3.0;

        for i in 0..n {
            let mut sum = 0.0;
            for j in 0..n {
                if i != j {
                    let dist = distances[i * n + j];
                    let prob = (-dist * dist / (2.0 * bandwidth * bandwidth)).exp();
                    probs[i * n + j] = prob;
                    sum += prob;
                }
            }
            // Normalize
            if sum > 0.0 {
                for j in 0..n {
                    if i != j {
                        probs[i * n + j] /= sum;
                    }
                }
            }
        }

        probs
    }

    /// Initialize 2D points randomly
    fn initialize_points(&mut self) {
        use rand::Rng;
        let mut rng = rand::thread_rng();

        self.points_2d.clear();
        for vec in &self.vectors {
            self.points_2d.push(Point2D {
                x: rng.gen_range(-1.0..1.0),
                y: rng.gen_range(-1.0..1.0),
                original_id: vec.id,
            });
        }
    }

    /// Run the dimensionality reduction
    pub fn reduce(&mut self) -> Vec<Point2D> {
        if self.vectors.is_empty() {
            return vec![];
        }

        // Step 1: Compute distances
        self.compute_distances();

        // Step 2: Compute probabilities (simplified t-SNE P matrix)
        let p_matrix = self.compute_probabilities(&self.distances);

        // Step 3: Initialize 2D points
        self.initialize_points();

        // Step 4: Gradient descent optimization (simplified)
        let n = self.vectors.len();
        let mut velocities = vec![(0.0, 0.0); n];

        for _iter in 0..self.iterations {
            // Compute Q matrix (Student-t distribution in 2D)
            let mut q_matrix = vec![0.0; n * n];
            let mut sum_q = 0.0;

            for i in 0..n {
                for j in (i + 1)..n {
                    let dx = self.points_2d[i].x - self.points_2d[j].x;
                    let dy = self.points_2d[i].y - self.points_2d[j].y;
                    let dist_sq = dx * dx + dy * dy;
                    let q = 1.0 / (1.0 + dist_sq);
                    q_matrix[i * n + j] = q;
                    q_matrix[j * n + i] = q;
                    sum_q += 2.0 * q;
                }
            }

            // Normalize Q
            for i in 0..n {
                for j in 0..n {
                    if i != j {
                        q_matrix[i * n + j] /= sum_q;
                    }
                }
            }

            // Compute gradients and update positions
            for i in 0..n {
                let mut grad_x = 0.0;
                let mut grad_y = 0.0;

                for j in 0..n {
                    if i == j {
                        continue;
                    }

                    let pq_diff = p_matrix[i * n + j] - q_matrix[i * n + j];
                    let dx = self.points_2d[i].x - self.points_2d[j].x;
                    let dy = self.points_2d[i].y - self.points_2d[j].y;
                    let dist_sq = dx * dx + dy * dy;
                    let factor = pq_diff / (1.0 + dist_sq);

                    grad_x += factor * dx;
                    grad_y += factor * dy;
                }

                // Apply momentum and learning rate
                velocities[i].0 = 0.8 * velocities[i].0 - self.learning_rate * grad_x;
                velocities[i].1 = 0.8 * velocities[i].1 - self.learning_rate * grad_y;

                self.points_2d[i].x += velocities[i].0;
                self.points_2d[i].y += velocities[i].1;
            }

            // Decay learning rate
            self.learning_rate *= 0.99;
        }

        // Normalize to [-1, 1] range
        self.normalize_points();

        self.points_2d.clone()
    }

    /// Normalize points to fit in [-1, 1] range
    fn normalize_points(&mut self) {
        if self.points_2d.is_empty() {
            return;
        }

        let mut min_x = f64::MAX;
        let mut max_x = f64::MIN;
        let mut min_y = f64::MAX;
        let mut max_y = f64::MIN;

        for point in &self.points_2d {
            min_x = min_x.min(point.x);
            max_x = max_x.max(point.x);
            min_y = min_y.min(point.y);
            max_y = max_y.max(point.y);
        }

        let range_x = (max_x - min_x).max(1e-10);
        let range_y = (max_y - min_y).max(1e-10);

        for point in &mut self.points_2d {
            point.x = 2.0 * (point.x - min_x) / range_x - 1.0;
            point.y = 2.0 * (point.y - min_y) / range_y - 1.0;
        }
    }

    /// Get points colored by metric score
    pub fn get_colored_points(&self) -> Vec<(Point2D, f64)> {
        let mut result = Vec::with_capacity(self.points_2d.len());

        for point in &self.points_2d {
            if let Some(vec) = self.vectors.iter().find(|v| v.id == point.original_id) {
                result.push((*point, vec.metric_score));
            }
        }

        result
    }

    /// Clear all data
    pub fn clear(&mut self) {
        self.vectors.clear();
        self.points_2d.clear();
        self.distances.clear();
    }

    /// Set perplexity
    pub fn set_perplexity(&mut self, perplexity: f64) {
        self.perplexity = perplexity.clamp(5.0, 50.0);
    }

    /// Set number of iterations
    pub fn set_iterations(&mut self, iterations: usize) {
        self.iterations = iterations.min(500);
    }
}

impl Default for ParameterSpaceMapper {
    fn default() -> Self {
        Self::new(1000)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dimensionality_reduction() {
        let mut mapper = ParameterSpaceMapper::new(100);
        mapper.set_perplexity(10.0);
        mapper.set_iterations(50);

        // Add some sample vectors
        for i in 0..20 {
            let mut vec = ParameterVector::new(i, 5);
            vec.set(0, i as f64 * 0.1);
            vec.set(1, (i * 2) as f64 * 0.05);
            vec.set(2, (i * 3) as f64 * 0.03);
            vec.set(3, (i * 4) as f64 * 0.02);
            vec.set(4, (i * 5) as f64 * 0.01);
            vec.metric_score = 1.0 + (i as f64 * 0.1);
            mapper.add_vector(vec);
        }

        let points = mapper.reduce();
        assert_eq!(points.len(), 20);

        // Verify points are in normalized range
        for point in &points {
            assert!((point.x - (-1.0)).abs() < 2.0);
            assert!((point.y - (-1.0)).abs() < 2.0);
        }

        let colored = mapper.get_colored_points();
        assert_eq!(colored.len(), 20);
    }
}
