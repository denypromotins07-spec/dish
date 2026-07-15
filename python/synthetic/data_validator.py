"""
Statistical validator for synthetic LOB data.
Compares synthetic distributions against real historical data using Wasserstein distance.
Ensures GAN outputs are realistic and not toxic.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy import stats
from dataclasses import dataclass


@dataclass
class ValidationMetrics:
    """Container for validation metrics."""
    wasserstein_distance: float
    kolmogorov_smirnov_stat: float
    mean_absolute_error: float
    correlation_diff: float
    tail_ratio_diff: float
    overall_score: float  # 0-1, higher is better


class DistributionValidator:
    """
    Validates synthetic data distributions against real data.
    Uses multiple statistical tests for comprehensive validation.
    """
    
    def __init__(self, epsilon: float = 1e-8):
        self.epsilon = epsilon
        
    def compute_wasserstein_1d(
        self,
        real: np.ndarray,
        synthetic: np.ndarray,
    ) -> float:
        """Compute 1D Wasserstein distance (Earth Mover's Distance)."""
        return stats.wasserstein_distance(real, synthetic)
    
    def compute_kolmogorov_smirnov(
        self,
        real: np.ndarray,
        synthetic: np.ndarray,
    ) -> float:
        """Compute Kolmogorov-Smirnov statistic."""
        ks_stat, _ = stats.ks_2samp(real, synthetic)
        return ks_stat
    
    def compute_tail_ratio(
        self,
        data: np.ndarray,
        quantile: float = 0.95,
    ) -> float:
        """Compute ratio of tail mass to total mass."""
        threshold = np.quantile(np.abs(data), quantile)
        tail_mass = np.sum(np.abs(data) > threshold)
        return tail_mass / len(data)
    
    def validate_marginals(
        self,
        real: np.ndarray,
        synthetic: np.ndarray,
    ) -> Dict[str, float]:
        """Validate marginal distributions for each feature."""
        if real.ndim == 1:
            real = real.reshape(-1, 1)
            synthetic = synthetic.reshape(-1, 1)
            
        n_features = real.shape[1]
        metrics = {
            'wasserstein': [],
            'ks_statistic': [],
        }
        
        for i in range(n_features):
            real_feat = real[:, i]
            synth_feat = synthetic[:, i]
            
            metrics['wasserstein'].append(
                self.compute_wasserstein_1d(real_feat, synth_feat)
            )
            metrics['ks_statistic'].append(
                self.compute_kolmogorov_smirnov(real_feat, synth_feat)
            )
        
        return {
            'mean_wasserstein': np.mean(metrics['wasserstein']),
            'max_wasserstein': np.max(metrics['wasserstein']),
            'mean_ks': np.mean(metrics['ks_statistic']),
            'max_ks': np.max(metrics['ks_statistic']),
        }
    
    def validate_correlations(
        self,
        real: np.ndarray,
        synthetic: np.ndarray,
    ) -> float:
        """Validate correlation structure preservation."""
        if real.shape[0] < 2:
            return 0.0
            
        real_corr = np.corrcoef(real.T)
        synth_corr = np.corrcoef(synthetic.T)
        
        # Handle NaN correlations
        real_corr = np.nan_to_num(real_corr, nan=0.0)
        synth_corr = np.nan_to_num(synth_corr, nan=0.0)
        
        # Frobenius norm of difference
        corr_diff = np.linalg.norm(real_corr - synth_corr, 'fro')
        
        # Normalize to [0, 1]
        max_diff = np.sqrt(real_corr.size)
        normalized_diff = min(1.0, corr_diff / (max_diff + self.epsilon))
        
        return 1.0 - normalized_diff  # Higher is better
    
    def validate_temporal_structure(
        self,
        real: np.ndarray,
        synthetic: np.ndarray,
        lag: int = 1,
    ) -> float:
        """Validate temporal autocorrelation structure."""
        def autocorr(data: np.ndarray, lag: int) -> np.ndarray:
            if data.ndim == 1:
                data = data.reshape(-1, 1)
            
            n_features = data.shape[1]
            acf = []
            for i in range(n_features):
                if len(data[:, i]) > lag:
                    acf.append(
                        np.corrcoef(data[:-lag, i], data[lag:, i])[0, 1]
                    )
                else:
                    acf.append(0.0)
            return np.array(acf)
        
        real_acf = autocorr(real, lag)
        synth_acf = autocorr(synthetic, lag)
        
        real_acf = np.nan_to_num(real_acf, nan=0.0)
        synth_acf = np.nan_to_num(synth_acf, nan=0.0)
        
        mae = np.mean(np.abs(real_acf - synth_acf))
        return 1.0 - min(1.0, mae)  # Higher is better


class SyntheticDataValidator:
    """
    Comprehensive validator for synthetic LOB data.
    Combines multiple validation metrics into an overall quality score.
    """
    
    def __init__(
        self,
        device: torch.device = None,
        wasserstein_threshold: float = 0.5,
        ks_threshold: float = 0.3,
        correlation_threshold: float = 0.7,
    ):
        self.device = device or (
            torch.device("cuda:0") if torch.cuda.is_available() 
            else torch.device("cpu")
        )
        self.wasserstein_threshold = wasserstein_threshold
        self.ks_threshold = ks_threshold
        self.correlation_threshold = correlation_threshold
        
        self.distribution_validator = DistributionValidator()
        
        # Historical statistics for comparison
        self.real_stats: Optional[Dict] = None
        
    def set_reference_statistics(
        self,
        real_data: np.ndarray,
    ) -> None:
        """Set reference statistics from real data."""
        self.real_stats = {
            'mean': np.mean(real_data, axis=0),
            'std': np.std(real_data, axis=0),
            'skewness': stats.skew(real_data, axis=0),
            'kurtosis': stats.kurtosis(real_data, axis=0),
            'quantiles': np.quantile(real_data, [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99], axis=0),
        }
        
    def validate(
        self,
        synthetic_data: np.ndarray,
        real_data: Optional[np.ndarray] = None,
    ) -> ValidationMetrics:
        """
        Comprehensive validation of synthetic data.
        
        Args:
            synthetic_data: Generated synthetic data
            real_data: Optional real data for comparison (uses stored stats if None)
            
        Returns:
            ValidationMetrics with all scores
        """
        if real_data is None:
            if self.real_stats is None:
                raise ValueError("No reference data provided. Call set_reference_statistics first.")
            # Generate proxy real data from stats for validation
            real_data = np.random.normal(
                self.real_stats['mean'],
                self.real_stats['std'],
                (len(synthetic_data), len(self.real_stats['mean']))
            )
        
        # Ensure same shape
        min_samples = min(len(real_data), len(synthetic_data))
        real_data = real_data[:min_samples]
        synthetic_data = synthetic_data[:min_samples]
        
        # Marginal distributions
        marginal_metrics = self.distribution_validator.validate_marginals(
            real_data, synthetic_data
        )
        
        # Correlation structure
        corr_score = self.distribution_validator.validate_correlations(
            real_data, synthetic_data
        )
        
        # Temporal structure
        temporal_score = self.distribution_validator.validate_temporal_structure(
            real_data, synthetic_data
        )
        
        # Tail behavior
        real_tail = self.distribution_validator.compute_tail_ratio(real_data)
        synth_tail = self.distribution_validator.compute_tail_ratio(synthetic_data)
        tail_diff = abs(real_tail - synth_tail)
        
        # Mean Absolute Error on moments
        if self.real_stats is not None:
            synth_mean = np.mean(synthetic_data, axis=0)
            synth_skew = stats.skew(synthetic_data, axis=0)
            synth_kurt = stats.kurtosis(synthetic_data, axis=0)
            
            mae_mean = np.mean(np.abs(synth_mean - self.real_stats['mean']))
            mae_skew = np.mean(np.abs(synth_skew - self.real_stats['skewness']))
            mae_kurt = np.mean(np.abs(synth_kurt - self.real_stats['kurtosis']))
            mae_total = (mae_mean + mae_skew + mae_kurt) / 3
        else:
            mae_total = 0.0
        
        # Overall score (weighted average)
        overall_score = (
            0.3 * (1.0 - min(1.0, marginal_metrics['mean_wasserstein'] / self.wasserstein_threshold)) +
            0.2 * (1.0 - min(1.0, marginal_metrics['mean_ks'] / self.ks_threshold)) +
            0.25 * corr_score +
            0.15 * temporal_score +
            0.1 * (1.0 - min(1.0, tail_diff * 10))
        )
        
        return ValidationMetrics(
            wasserstein_distance=marginal_metrics['mean_wasserstein'],
            kolmogorov_smirnov_stat=marginal_metrics['mean_ks'],
            mean_absolute_error=mae_total,
            correlation_diff=1.0 - corr_score,
            tail_ratio_diff=tail_diff,
            overall_score=max(0.0, min(1.0, overall_score)),
        )
    
    def is_acceptable(
        self,
        metrics: ValidationMetrics,
    ) -> bool:
        """Check if synthetic data passes quality thresholds."""
        return (
            metrics.wasserstein_distance < self.wasserstein_threshold and
            metrics.kolmogorov_smirnov_stat < self.ks_threshold and
            metrics.overall_score > 0.6
        )
    
    def detect_toxic_patterns(
        self,
        synthetic_data: np.ndarray,
        max_autocorr: float = 0.9,
        min_variance_ratio: float = 0.1,
    ) -> Dict[str, bool]:
        """
        Detect potentially toxic patterns in synthetic data.
        
        Flags:
        - excessive_autocorrelation: Too predictable
        - variance_collapse: Lack of diversity
        - extreme_values: Unrealistic outliers
        """
        flags = {}
        
        # Check for excessive autocorrelation
        if synthetic_data.ndim == 1:
            synthetic_data = synthetic_data.reshape(-1, 1)
            
        max_acf = 0.0
        for i in range(min(5, synthetic_data.shape[1])):
            if len(synthetic_data) > 10:
                acf = np.corrcoef(
                    synthetic_data[:-1, i], 
                    synthetic_data[1:, i]
                )[0, 1]
                max_acf = max(max_acf, abs(acf))
        
        flags['excessive_autocorrelation'] = max_acf > max_autocorr
        
        # Check for variance collapse
        real_var = np.var(synthetic_data)
        if self.real_stats is not None:
            ref_var = np.mean(self.real_stats['std'] ** 2)
            variance_ratio = real_var / (ref_var + 1e-8)
            flags['variance_collapse'] = variance_ratio < min_variance_ratio
        else:
            flags['variance_collapse'] = False
            
        # Check for extreme values
        z_scores = np.abs(
            (synthetic_data - np.mean(synthetic_data, axis=0)) / 
            (np.std(synthetic_data, axis=0) + 1e-8)
        )
        flags['extreme_values'] = np.any(z_scores > 10)
        
        return flags


if __name__ == "__main__":
    # Example usage
    np.random.seed(42)
    
    # Generate dummy real data
    real_data = np.random.normal(0, 1, (1000, 5))
    real_data[:, 0] = real_data[:, 0] * 0.5 + np.random.exponential(0.5, 1000)  # Skewed
    
    # Generate dummy synthetic data
    synthetic_good = real_data + np.random.normal(0, 0.1, real_data.shape)
    synthetic_bad = np.random.uniform(-5, 5, real_data.shape)
    
    # Initialize validator
    validator = SyntheticDataValidator()
    validator.set_reference_statistics(real_data)
    
    # Validate good synthetic data
    print("Validating GOOD synthetic data:")
    metrics_good = validator.validate(synthetic_good)
    print(f"  Wasserstein: {metrics_good.wasserstein_distance:.4f}")
    print(f"  KS Statistic: {metrics_good.kolmogorov_smirnov_stat:.4f}")
    print(f"  Correlation Score: {1 - metrics_good.correlation_diff:.4f}")
    print(f"  Overall Score: {metrics_good.overall_score:.4f}")
    print(f"  Acceptable: {validator.is_acceptable(metrics_good)}")
    
    toxic_flags = validator.detect_toxic_patterns(synthetic_good)
    print(f"  Toxic Flags: {toxic_flags}")
    
    print("\nValidating BAD synthetic data:")
    metrics_bad = validator.validate(synthetic_bad)
    print(f"  Wasserstein: {metrics_bad.wasserstein_distance:.4f}")
    print(f"  KS Statistic: {metrics_bad.kolmogorov_smirnov_stat:.4f}")
    print(f"  Overall Score: {metrics_bad.overall_score:.4f}")
    print(f"  Acceptable: {validator.is_acceptable(metrics_bad)}")
    
    toxic_flags_bad = validator.detect_toxic_patterns(synthetic_bad)
    print(f"  Toxic Flags: {toxic_flags_bad}")
