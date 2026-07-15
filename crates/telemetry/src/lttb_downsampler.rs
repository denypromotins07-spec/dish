//! Rust implementation of the Largest-Triangle-Three-Buckets (LTTB) algorithm.
//! Downsamples millions of historical ticks into ~1000 visual points for smooth, lag-free browser chart rendering.

/// LTTB downsampler for time-series data
pub struct LttbDownsampler {
    /// Target number of output points
    target_points: usize,
    /// Threshold for switching to simple decimation
    decimation_threshold: usize,
}

impl LttbDownsampler {
    /// Create a new LTTB downsampler
    pub fn new(target_points: usize) -> Self {
        Self {
            target_points,
            decimation_threshold: 100_000, // Use simpler algo for very large datasets
        }
    }

    /// Downsample a time series using LTTB algorithm
    /// Input: Vec of (timestamp_ns, value) tuples
    /// Output: Downsampled Vec of (timestamp_ns, value) tuples
    pub fn downsample(&self, data: &[(u64, f64)]) -> Vec<(u64, f64)> {
        let input_size = data.len();

        if input_size <= self.target_points {
            return data.to_vec();
        }

        // For very large datasets, use bucket-based approach first
        if input_size > self.decimation_threshold {
            return self.downsample_large(data);
        }

        // Standard LTTB
        self.lttb_core(data)
    }

    /// Core LTTB algorithm
    fn lttb_core(&self, data: &[(u64, f64)]) -> Vec<(u64, f64)> {
        let n = data.len();
        let mut result = Vec::with_capacity(self.target_points);

        // Always include first point
        if let Some(&first) = data.first() {
            result.push(first);
        }

        // Calculate bucket sizes
        let bucket_size = (n - 2) as f64 / (self.target_points - 2) as f64;

        // Last point index
        let last_idx = n - 1;

        // Current selected point (starts at first)
        let mut current_idx = 0;

        // Process each bucket
        for i in 0..(self.target_points - 2) {
            // Bucket range
            let bucket_start = ((i as f64 * bucket_size) + 1.0) as usize;
            let bucket_end = (((i as f64 + 1.0) * bucket_size) + 1.0) as usize;
            let bucket_end = bucket_end.min(last_idx);

            // Next bucket range (for average calculation)
            let next_bucket_start = bucket_end;
            let next_bucket_end = (((i as f64 + 2.0) * bucket_size) + 1.0) as usize;
            let next_bucket_end = next_bucket_end.min(last_idx);

            // Calculate average of next bucket
            let mut next_sum_x = 0.0f64;
            let mut next_sum_y = 0.0f64;
            let mut next_count = 0;

            for j in next_bucket_start..next_bucket_end {
                next_sum_x += data[j].0 as f64;
                next_sum_y += data[j].1;
                next_count += 1;
            }

            let next_avg_x = if next_count > 0 { next_sum_x / next_count as f64 } else { 0.0 };
            let next_avg_y = if next_count > 0 { next_sum_y / next_count as f64 } else { 0.0 };

            // Find point in current bucket that maximizes triangle area
            let mut max_area = -1.0f64;
            let mut max_idx = bucket_start;

            let current_point = data[current_idx];

            for j in bucket_start..bucket_end {
                let point = data[j];

                // Calculate triangle area using cross product
                let area = self.triangle_area(
                    current_point,
                    *point,
                    (next_avg_x as u64, next_avg_y),
                );

                if area > max_area {
                    max_area = area;
                    max_idx = j;
                }
            }

            result.push(data[max_idx]);
            current_idx = max_idx;
        }

        // Always include last point
        if let Some(&last) = data.last() {
            result.push(last);
        }

        result
    }

    /// Optimized downsampling for very large datasets
    fn downsample_large(&self, data: &[(u64, f64)]) -> Vec<(u64, f64)> {
        let n = data.len();
        let mut result = Vec::with_capacity(self.target_points);

        // Always include first point
        if let Some(&first) = data.first() {
            result.push(first);
        }

        // Simple bucket averaging for large datasets
        let bucket_size = n / (self.target_points - 2);

        for i in 1..(self.target_points - 1) {
            let start = i * bucket_size;
            let end = ((i + 1) * bucket_size).min(n - 1);

            if start >= n {
                break;
            }

            // Find point with maximum absolute deviation in bucket
            let mut max_deviation = 0.0f64;
            let mut selected_idx = start;

            for j in start..end {
                let deviation = (data[j].1 - data[start].1).abs();
                if deviation > max_deviation {
                    max_deviation = deviation;
                    selected_idx = j;
                }
            }

            result.push(data[selected_idx]);
        }

        // Always include last point
        if let Some(&last) = data.last() {
            result.push(last);
        }

        result
    }

    /// Calculate triangle area given three points
    #[inline]
    fn triangle_area(&self, a: (u64, f64), b: (u64, f64), c: (u64, f64)) -> f64 {
        // Area = 0.5 * |x1(y2 - y3) + x2(y3 - y1) + x3(y1 - y2)|
        // We can skip the 0.5 since we're only comparing areas
        
        let ax = a.0 as f64;
        let ay = a.1;
        let bx = b.0 as f64;
        let by = b.1;
        let cx = c.0 as f64;
        let cy = c.1;

        let area = ax * (by - cy) + bx * (cy - ay) + cx * (ay - by);
        area.abs()
    }

    /// Downsample with additional metrics (for debugging/analysis)
    pub fn downsample_with_metrics(
        &self,
        data: &[(u64, f64)],
    ) -> (Vec<(u64, f64)>, DownsampleMetrics) {
        let input_size = data.len();
        let start_time = std::time::Instant::now();

        let result = self.downsample(data);

        let elapsed = start_time.elapsed();

        let metrics = DownsampleMetrics {
            input_points: input_size,
            output_points: result.len(),
            compression_ratio: if input_size > 0 {
                input_size as f64 / result.len() as f64
            } else {
                1.0
            },
            processing_time_us: elapsed.as_micros() as u64,
        };

        (result, metrics)
    }
}

/// Metrics from downsampling operation
#[derive(Debug, Clone)]
pub struct DownsampleMetrics {
    pub input_points: usize,
    pub output_points: usize,
    pub compression_ratio: f64,
    pub processing_time_us: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_downsample_small_dataset() {
        let downsampler = LttbDownsampler::new(10);

        // Create simple linear data
        let data: Vec<(u64, f64)> = (0..20)
            .map(|i| (i as u64 * 1000, i as f64))
            .collect();

        let result = downsampler.downsample(&data);

        assert!(result.len() <= 10);
        assert_eq!(result.first(), Some(&(0, 0.0)));
        assert_eq!(result.last(), Some(&(19000, 19.0)));
    }

    #[test]
    fn test_downsample_preserves_endpoints() {
        let downsampler = LttbDownsampler::new(5);

        let data = vec![
            (1000, 10.0),
            (2000, 15.0),
            (3000, 12.0),
            (4000, 20.0),
            (5000, 18.0),
        ];

        let result = downsampler.downsample(&data);

        // Small dataset should preserve endpoints
        assert_eq!(result.first(), Some(&(1000, 10.0)));
        assert_eq!(result.last(), Some(&(5000, 18.0)));
    }

    #[test]
    fn test_downsample_returns_input_if_smaller() {
        let downsampler = LttbDownsampler::new(100);

        let data: Vec<(u64, f64)> = (0..10)
            .map(|i| (i as u64, i as f64))
            .collect();

        let result = downsampler.downsample(&data);

        assert_eq!(result.len(), 10);
        assert_eq!(result, data);
    }

    #[test]
    fn test_triangle_area_calculation() {
        let downsampler = LttbDownsampler::new(10);

        // Right triangle with area 0.5
        let a = (0u64, 0.0f64);
        let b = (1u64, 0.0f64);
        let c = (0u64, 1.0f64);

        let area = downsampler.triangle_area(a, b, c);
        assert!((area - 1.0).abs() < 0.0001); // Scaled by 2
    }

    #[test]
    fn test_metrics_generation() {
        let downsampler = LttbDownsampler::new(50);

        let data: Vec<(u64, f64)> = (0..1000)
            .map(|i| (i as u64 * 1000, (i as f64).sin()))
            .collect();

        let (result, metrics) = downsampler.downsample_with_metrics(&data);

        assert_eq!(metrics.input_points, 1000);
        assert_eq!(metrics.output_points, result.len());
        assert!(metrics.compression_ratio > 1.0);
        assert!(metrics.processing_time_us < 1_000_000); // Should be fast
    }

    #[test]
    fn test_large_dataset_handling() {
        let downsampler = LttbDownsampler::new(100);

        // Create dataset larger than threshold
        let data: Vec<(u64, f64)> = (0..200_000)
            .map(|i| (i as u64, (i as f64 * 0.01).sin()))
            .collect();

        let result = downsampler.downsample(&data);

        assert!(result.len() <= 100);
        assert_eq!(result.first().unwrap().0, 0);
        assert_eq!(result.last().unwrap().0, 199_999);
    }
}
