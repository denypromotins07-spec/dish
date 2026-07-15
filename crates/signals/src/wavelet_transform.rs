//! Fast Discrete Wavelet Transform (DWT) in Rust
//! Decomposes high-frequency price action into noise and trend components
//! Uses significantly less memory than standard FFT

use std::collections::VecDeque;

/// Wavelet filter coefficients for DWT
/// Using Daubechies-4 wavelet by default
pub struct WaveletFilters {
    pub low_pass: [f64; 4],
    pub high_pass: [f64; 4],
}

impl WaveletFilters {
    /// Create Daubechies-4 (db4) wavelet filters
    pub fn daubechies_4() -> Self {
        // db4 scaling coefficients
        let h0 = (1.0 + 3.0_f64.sqrt()) / (4.0 * 2.0_f64.sqrt());
        let h1 = (3.0 + 3.0_f64.sqrt()) / (4.0 * 2.0_f64.sqrt());
        let h2 = (3.0 - 3.0_f64.sqrt()) / (4.0 * 2.0_f64.sqrt());
        let h3 = (1.0 - 3.0_f64.sqrt()) / (4.0 * 2.0_f64.sqrt());

        Self {
            low_pass: [h0, h1, h2, h3],
            high_pass: [h3, -h2, h1, -h0], // Quadrature mirror filter
        }
    }

    /// Create Haar wavelet filters (simplest)
    pub fn haar() -> Self {
        let inv_sqrt2 = 1.0 / 2.0_f64.sqrt();
        Self {
            low_pass: [inv_sqrt2, inv_sqrt2, 0.0, 0.0],
            high_pass: [inv_sqrt2, -inv_sqrt2, 0.0, 0.0],
        }
    }
}

/// Discrete Wavelet Transform for signal decomposition
pub struct DiscreteWaveletTransform {
    filters: WaveletFilters,
    /// Maximum decomposition levels
    max_levels: usize,
    /// Reusable buffer to avoid allocations
    work_buffer: Vec<f64>,
}

impl DiscreteWaveletTransform {
    pub fn new(filters: WaveletFilters, max_levels: usize) -> Self {
        Self {
            filters,
            max_levels,
            work_buffer: Vec::with_capacity(4096),
        }
    }

    /// Perform single-level DWT decomposition
    /// Returns (approximation_coefficients, detail_coefficients)
    pub fn decompose_level(&self, signal: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let n = signal.len();
        let half_n = (n + 1) / 2;

        let mut approx = Vec::with_capacity(half_n);
        let mut detail = Vec::with_capacity(half_n);

        // Convolution with downsampling
        for i in 0..half_n {
            let idx = 2 * i;
            
            // Approximation (low-pass)
            let mut a = 0.0;
            for k in 0..4 {
                let sample_idx = (idx + k) % n;
                a += self.filters.low_pass[k] * signal[sample_idx];
            }
            approx.push(a);

            // Detail (high-pass)
            let mut d = 0.0;
            for k in 0..4 {
                let sample_idx = (idx + k) % n;
                d += self.filters.high_pass[k] * signal[sample_idx];
            }
            detail.push(d);
        }

        (approx, detail)
    }

    /// Multi-level DWT decomposition
    /// Returns vector of (level, detail_coefficients) plus final approximation
    pub fn decompose(&mut self, signal: &[f64], levels: usize) -> WaveletDecomposition {
        let actual_levels = levels.min(self.max_levels).min(signal.len().leading_zeros() as usize);
        
        let mut current_signal = signal.to_vec();
        let mut details: Vec<Vec<f64>> = Vec::with_capacity(actual_levels);

        for _ in 0..actual_levels {
            if current_signal.len() < 4 {
                break;
            }

            let (approx, detail) = self.decompose_level(&current_signal);
            details.push(detail);
            current_signal = approx;
        }

        WaveletDecomposition {
            approximation: current_signal,
            details,
        }
    }

    /// Reconstruct signal from wavelet decomposition
    pub fn reconstruct(&self, decomposition: &WaveletDecomposition) -> Vec<f64> {
        let mut signal = decomposition.approximation.clone();

        // Reconstruct from coarsest to finest level
        for level in (0..decomposition.details.len()).rev() {
            signal = self.reconstruct_level(&signal, &decomposition.details[level]);
        }

        signal
    }

    /// Single-level reconstruction
    fn reconstruct_level(&self, approx: &[f64], detail: &[f64]) -> Vec<f64> {
        let n = approx.len() * 2;
        let mut reconstructed = vec![0.0; n];

        // Inverse DWT using reconstruction filters
        let g0 = self.filters.low_pass;
        let g1 = self.filters.high_pass;

        for i in 0..approx.len() {
            for k in 0..4 {
                let idx = (2 * i + k) % n;
                reconstructed[idx] += g0[k] * approx[i];
                reconstructed[idx] += g1[k] * detail[i];
            }
        }

        reconstructed
    }

    /// Extract trend component from signal
    pub fn extract_trend(&mut self, signal: &[f64], levels: usize) -> Vec<f64> {
        let decomp = self.decompose(signal, levels);
        
        // Trend is the final approximation, upsampled to original length
        let mut trend = decomp.approximation;
        
        // Upsample to match original signal length
        while trend.len() < signal.len() {
            let mut upsampled = vec![0.0; trend.len() * 2];
            for (i, &val) in trend.iter().enumerate() {
                upsampled[2 * i] = val;
                upsampled[2 * i + 1] = val;
            }
            trend = upsampled;
        }

        trend.truncate(signal.len());
        trend
    }

    /// Extract noise component from signal
    pub fn extract_noise(&mut self, signal: &[f64], levels: usize) -> Vec<f64> {
        let trend = self.extract_trend(signal, levels);
        
        let noise: Vec<f64> = signal
            .iter()
            .zip(trend.iter())
            .map(|(&s, &t)| s - t)
            .collect();

        noise
    }

    /// Denoise signal using wavelet thresholding
    pub fn denoise(&mut self, signal: &[f64], threshold_type: ThresholdType) -> Vec<f64> {
        let levels = self.max_levels.min(4);
        let mut decomp = self.decompose(signal, levels);

        // Apply thresholding to detail coefficients
        for detail in &mut decomp.details {
            let threshold = self._calculate_threshold(detail, threshold_type);
            for coeff in detail.iter_mut() {
                *coeff = self._apply_threshold(*coeff, threshold, threshold_type);
            }
        }

        self.reconstruct(&decomp)
    }

    fn _calculate_threshold(&self, coeffs: &[f64], threshold_type: ThresholdType) -> f64 {
        match threshold_type {
            ThresholdType::Universal => {
                // Universal threshold: sigma * sqrt(2 * log(n))
                let sigma = self._estimate_noise_level(coeffs);
                let n = coeffs.len() as f64;
                sigma * (2.0 * n.ln()).sqrt()
            }
            ThresholdType::SURE => {
                // Stein's Unbiased Risk Estimate (simplified)
                let sorted: Vec<f64> = coeffs.iter().map(|x| x.powi(2)).collect();
                let mut sorted = sorted;
                sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                
                let n = sorted.len() as f64;
                let mut min_risk = f64::INFINITY;
                let mut best_threshold = 0.0;

                for i in 0..sorted.len() {
                    let threshold = sorted[i].sqrt();
                    let risk = (i as f64 + (n - i as f64) * threshold.powi(2) / n).sqrt();
                    
                    if risk < min_risk {
                        min_risk = risk;
                        best_threshold = threshold;
                    }
                }

                best_threshold
            }
        }
    }

    fn _estimate_noise_level(&self, coeffs: &[f64]) -> f64 {
        // MAD estimator for robustness
        let median = self._median(coeffs);
        let mad: Vec<f64> = coeffs.iter().map(|&x| (x - median).abs()).collect();
        let mad_median = self._median(&mad);
        
        mad_median / 0.6745 // Scale factor for Gaussian
    }

    fn _median(&self, data: &[f64]) -> f64 {
        let mut sorted: Vec<f64> = data.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        
        let n = sorted.len();
        if n % 2 == 0 {
            (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
        } else {
            sorted[n / 2]
        }
    }

    fn _apply_threshold(&self, coeff: f64, threshold: f64, threshold_type: ThresholdType) -> f64 {
        match threshold_type {
            ThresholdType::Soft => {
                if coeff.abs() <= threshold {
                    0.0
                } else {
                    coeff.signum() * (coeff.abs() - threshold)
                }
            }
            ThresholdType::Hard => {
                if coeff.abs() <= threshold {
                    0.0
                } else {
                    coeff
                }
            }
            _ => coeff,
        }
    }
}

/// Type of thresholding for denoising
#[derive(Clone, Copy)]
pub enum ThresholdType {
    Soft,
    Hard,
    Universal,
    SURE,
}

/// Result of wavelet decomposition
#[derive(Clone)]
pub struct WaveletDecomposition {
    pub approximation: Vec<f64>,
    pub details: Vec<Vec<f64>>,
}

/// Rolling wavelet transform for streaming data
pub struct RollingDWT {
    dwt: DiscreteWaveletTransform,
    /// Circular buffer for input data
    buffer: VecDeque<f64>,
    /// Buffer size (power of 2)
    buffer_size: usize,
    /// Levels for decomposition
    levels: usize,
}

impl RollingDWT {
    pub fn new(buffer_size: usize, levels: usize) -> Self {
        // Ensure buffer_size is power of 2
        let buffer_size = buffer_size.next_power_of_two();
        
        Self {
            dwt: DiscreteWaveletTransform::new(WaveletFilters::daubechies_4(), 8),
            buffer: VecDeque::with_capacity(buffer_size),
            buffer_size,
            levels: levels.min(8),
        }
    }

    pub fn update(&mut self, new_value: f64) -> Option<WaveletDecomposition> {
        self.buffer.push_back(new_value);
        
        if self.buffer.len() > self.buffer_size {
            self.buffer.pop_front();
        }

        if self.buffer.len() >= self.buffer_size / 2 {
            let signal: Vec<f64> = self.buffer.iter().copied().collect();
            Some(self.dwt.decompose(&signal, self.levels))
        } else {
            None
        }
    }

    pub fn get_trend(&mut self) -> Option<Vec<f64>> {
        if self.buffer.len() < self.buffer_size / 2 {
            return None;
        }

        let signal: Vec<f64> = self.buffer.iter().copied().collect();
        Some(self.dwt.extract_trend(&signal, self.levels))
    }

    pub fn get_noise(&mut self) -> Option<Vec<f64>> {
        if self.buffer.len() < self.buffer_size / 2 {
            return None;
        }

        let signal: Vec<f64> = self.buffer.iter().copied().collect();
        Some(self.dwt.extract_noise(&signal, self.levels))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dwt_decomposition_reconstruction() {
        let mut dwt = DiscreteWaveletTransform::new(WaveletFilters::haar(), 4);
        
        // Create test signal
        let signal: Vec<f64> = (0..16).map(|i| i as f64 * 0.5).collect();
        
        // Decompose and reconstruct
        let decomp = dwt.decompose(&signal, 3);
        let reconstructed = dwt.reconstruct(&decomp);
        
        // Check reconstruction accuracy (within tolerance)
        for (orig, recon) in signal.iter().zip(reconstructed.iter()) {
            assert!((orig - recon).abs() < 1e-10, "Reconstruction error too large");
        }
    }

    #[test]
    fn test_trend_extraction() {
        let mut dwt = DiscreteWaveletTransform::new(WaveletFilters::daubechies_4(), 4);
        
        // Signal with noise
        let signal: Vec<f64> = (0..32).map(|i| {
            (i as f64 * 0.1).sin() + (i as f64).sin() * 0.1
        }).collect();
        
        let trend = dwt.extract_trend(&signal, 2);
        
        assert_eq!(trend.len(), signal.len());
        
        // Trend should be smoother than original
        let orig_var: f64 = signal.iter().map(|x| x.powi(2)).sum::<f64>() / signal.len() as f64;
        let trend_var: f64 = trend.iter().map(|x| x.powi(2)).sum::<f64>() / trend.len() as f64;
        
        assert!(trend_var <= orig_var, "Trend should have lower variance");
    }

    #[test]
    fn test_rolling_dwt() {
        let mut rolling = RollingDWT::new(64, 3);
        
        for i in 0..100 {
            let value = (i as f64 * 0.1).sin();
            rolling.update(value);
        }
        
        let trend = rolling.get_trend();
        assert!(trend.is_some());
        assert!(trend.unwrap().len() > 0);
    }
}
