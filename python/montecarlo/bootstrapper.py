"""
Block-bootstrapping module for time-series data.
Preserves autocorrelation, volatility clustering, and regime shifts of crypto markets when generating synthetic training datasets.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class BootstrapResult:
    """Result of a bootstrap sampling operation."""
    synthetic_data: np.ndarray
    block_count: int
    average_block_length: float
    preserved_autocorrelation: float
    preserved_volatility_clustering: bool


class BlockBootstrap:
    """
    Block bootstrapping for financial time series.
    
    Implements:
    - Moving Block Bootstrap (MBB)
    - Stationary Bootstrap (Politis & Romano)
    - Circular Block Bootstrap
    - Markov-switching bootstrap for regime preservation
    """
    
    def __init__(
        self,
        block_size: int = 20,
        method: str = "stationary",
        preserve_regimes: bool = True,
    ):
        self.block_size = block_size
        self.method = method
        self.preserve_regimes = preserve_regimes
        
        # Regime detection state
        self.regime_labels: Optional[np.ndarray] = None
        self.regime_transitions: Optional[Dict] = None
    
    def bootstrap(
        self,
        data: np.ndarray,
        n_samples: int,
        seed: Optional[int] = None,
    ) -> List[np.ndarray]:
        """
        Generate bootstrap samples from the original data.
        
        Parameters
        ----------
        data : np.ndarray
            Original time series data (1D or 2D).
        n_samples : int
            Number of bootstrap samples to generate.
        seed : Optional[int]
            Random seed for reproducibility.
            
        Returns
        -------
        List[np.ndarray]
            List of bootstrap sample arrays.
        """
        if seed is not None:
            np.random.seed(seed)
        
        if len(data.shape) == 1:
            data = data.reshape(-1, 1)
        
        n_obs = data.shape[0]
        samples = []
        
        if self.method == "moving":
            samples = self._moving_block_bootstrap(data, n_samples, n_obs)
        elif self.method == "stationary":
            samples = self._stationary_bootstrap(data, n_samples, n_obs)
        elif self.method == "circular":
            samples = self._circular_block_bootstrap(data, n_samples, n_obs)
        elif self.method == "markov":
            samples = self._markov_bootstrap(data, n_samples, n_obs)
        else:
            raise ValueError(f"Unknown bootstrap method: {self.method}")
        
        return samples
    
    def _moving_block_bootstrap(
        self,
        data: np.ndarray,
        n_samples: int,
        n_obs: int,
    ) -> List[np.ndarray]:
        """Moving Block Bootstrap (Künsch, 1989)."""
        samples = []
        num_blocks = int(np.ceil(n_obs / self.block_size))
        
        for _ in range(n_samples):
            blocks = []
            for _ in range(num_blocks):
                start_idx = np.random.randint(0, n_obs - self.block_size + 1)
                end_idx = start_idx + self.block_size
                blocks.append(data[start_idx:end_idx])
            
            sample = np.vstack(blocks)[:n_obs]
            samples.append(sample)
        
        return samples
    
    def _stationary_bootstrap(
        self,
        data: np.ndarray,
        n_samples: int,
        n_obs: int,
    ) -> List[np.ndarray]:
        """Stationary Bootstrap (Politis & Romano, 1994)."""
        # Calculate optimal block length parameter
        p = 1 / self.block_size
        
        samples = []
        
        for _ in range(n_samples):
            sample_indices = []
            current_idx = np.random.randint(0, n_obs)
            
            while len(sample_indices) < n_obs:
                sample_indices.append(current_idx)
                
                # Decide whether to continue block or start new one
                if np.random.random() < p:
                    # Start new block at random position
                    current_idx = np.random.randint(0, n_obs)
                else:
                    # Continue current block
                    current_idx = (current_idx + 1) % n_obs
            
            sample = data[sample_indices[:n_obs]]
            samples.append(sample)
        
        return samples
    
    def _circular_block_bootstrap(
        self,
        data: np.ndarray,
        n_samples: int,
        n_obs: int,
    ) -> List[np.ndarray]:
        """Circular Block Bootstrap (Politis & Romano, 1992)."""
        # Pad data circularly
        padded_data = np.vstack([data, data[:self.block_size]])
        
        samples = []
        num_blocks = int(np.ceil(n_obs / self.block_size))
        
        for _ in range(n_samples):
            blocks = []
            for _ in range(num_blocks):
                start_idx = np.random.randint(0, n_obs)
                end_idx = start_idx + self.block_size
                blocks.append(padded_data[start_idx:end_idx])
            
            sample = np.vstack(blocks)[:n_obs]
            samples.append(sample)
        
        return samples
    
    def _markov_bootstrap(
        self,
        data: np.ndarray,
        n_samples: int,
        n_obs: int,
    ) -> List[np.ndarray]:
        """Markov-switching bootstrap for regime preservation."""
        # Detect regimes if not already done
        if self.regime_labels is None:
            self._detect_regimes(data)
        
        samples = []
        
        for _ in range(n_samples):
            sample = self._sample_from_markov_regimes(data, n_obs)
            samples.append(sample)
        
        return samples
    
    def _detect_regimes(self, data: np.ndarray, n_regimes: int = 3):
        """Detect market regimes using volatility clustering."""
        returns = np.diff(data, axis=0).flatten()
        
        # Calculate rolling volatility
        vol_window = min(20, len(returns) // 5)
        rolling_vol = np.std(returns[:vol_window])
        vol_series = []
        
        for i in range(len(returns)):
            start = max(0, i - vol_window)
            vol_series.append(np.std(returns[start:i+1]))
        
        vol_series = np.array(vol_series)
        
        # Simple k-means-like regime assignment
        thresholds = np.percentile(vol_series, [33, 66])
        self.regime_labels = np.digitize(vol_series, thresholds)
        
        # Build transition matrix
        self.regime_transitions = {r: [] for r in range(n_regimes)}
        for i in range(len(self.regime_labels) - 1):
            current = self.regime_labels[i]
            next_regime = self.regime_labels[i + 1]
            self.regime_transitions[current].append(next_regime)
    
    def _sample_from_markov_regimes(self, data: np.ndarray, n_obs: int) -> np.ndarray:
        """Sample from detected regimes preserving transitions."""
        sample = []
        current_regime = np.random.choice(list(self.regime_transitions.keys()))
        
        indices = np.where(self.regime_labels == current_regime)[0]
        if len(indices) > 0:
            current_idx = np.random.choice(indices)
            sample.append(data[current_idx])
        
        while len(sample) < n_obs:
            # Transition to new regime based on historical probabilities
            if self.regime_transitions[current_regime]:
                next_regime = np.random.choice(self.regime_transitions[current_regime])
            else:
                next_regime = np.random.choice(list(self.regime_transitions.keys()))
            
            indices = np.where(self.regime_labels == next_regime)[0]
            if len(indices) > 0:
                current_idx = np.random.choice(indices)
                sample.append(data[current_idx])
                current_regime = next_regime
        
        return np.array(sample[:n_obs])


class TimeSeriesPreservationChecker:
    """Verify that bootstrap samples preserve key time series properties."""
    
    @staticmethod
    def calculate_autocorrelation(data: np.ndarray, lag: int = 1) -> float:
        """Calculate autocorrelation at specified lag."""
        if len(data) <= lag:
            return 0.0
        
        mean = np.mean(data)
        var = np.var(data)
        
        if var == 0:
            return 0.0
        
        autocov = np.mean((data[:-lag] - mean) * (data[lag:] - mean))
        return autocov / var
    
    @staticmethod
    def detect_volatility_clustering(data: np.ndarray) -> bool:
        """Check for presence of volatility clustering using Ljung-Box test approximation."""
        returns = np.diff(data.flatten())
        abs_returns = np.abs(returns)
        
        # Check if absolute returns are autocorrelated
        acf_abs = TimeSeriesPreservationChecker.calculate_autocorrelation(abs_returns, lag=1)
        
        return acf_abs > 0.1  # Threshold for significant clustering
    
    @staticmethod
    def compare_properties(
        original: np.ndarray,
        bootstrap_samples: List[np.ndarray],
    ) -> Dict[str, float]:
        """Compare time series properties between original and bootstrap samples."""
        results = {
            'original_autocorrelation': TimeSeriesPreservationChecker.calculate_autocorrelation(original.flatten()),
            'mean_bootstrap_autocorrelation': 0.0,
            'autocorrelation_preservation_ratio': 0.0,
            'original_volatility_clustering': False,
            'bootstrap_volatility_clustering_ratio': 0.0,
        }
        
        if not bootstrap_samples:
            return results
        
        # Average autocorrelation across samples
        acfs = [TimeSeriesPreservationChecker.calculate_autocorrelation(s.flatten()) 
                for s in bootstrap_samples]
        results['mean_bootstrap_autocorrelation'] = np.mean(acfs)
        
        # Preservation ratio
        orig_ac = results['original_autocorrelation']
        if orig_ac != 0:
            results['autocorrelation_preservation_ratio'] = np.mean(acfs) / orig_ac
        
        # Volatility clustering
        results['original_volatility_clustering'] = TimeSeriesPreservationChecker.detect_volatility_clustering(original)
        
        vc_count = sum(TimeSeriesPreservationChecker.detect_volatility_clustering(s) 
                       for s in bootstrap_samples)
        results['bootstrap_volatility_clustering_ratio'] = vc_count / len(bootstrap_samples)
        
        return results


def generate_synthetic_training_data(
    price_data: np.ndarray,
    volume_data: Optional[np.ndarray] = None,
    n_synthetic_datasets: int = 100,
    block_size: int = 20,
    method: str = "stationary",
    seed: int = 42,
) -> Tuple[List[np.ndarray], Optional[List[np.ndarray]], Dict]:
    """
    Generate synthetic training datasets using block bootstrapping.
    
    Parameters
    ----------
    price_data : np.ndarray
        Historical price data.
    volume_data : Optional[np.ndarray]
        Historical volume data (optional).
    n_synthetic_datasets : int, default 100
        Number of synthetic datasets to generate.
    block_size : int, default 20
        Block size for bootstrapping.
    method : str, default "stationary"
        Bootstrap method ("moving", "stationary", "circular", "markov").
    seed : int, default 42
        Random seed.
        
    Returns
    -------
    Tuple[List[np.ndarray], Optional[List[np.ndarray]], Dict]
        Synthetic price datasets, synthetic volume datasets (if provided), and diagnostics.
    """
    bs = BlockBootstrap(block_size=block_size, method=method)
    
    # Bootstrap prices
    price_samples = bs.bootstrap(price_data, n_synthetic_datasets, seed=seed)
    
    # Bootstrap volumes if provided
    volume_samples = None
    if volume_data is not None:
        # Use same random state for aligned sampling
        np.random.seed(seed)
        volume_bs = BlockBootstrap(block_size=block_size, method=method)
        volume_samples = volume_bs.bootstrap(volume_data, n_synthetic_datasets, seed=seed)
    
    # Check preservation
    checker = TimeSeriesPreservationChecker()
    diagnostics = checker.compare_properties(price_data, price_samples)
    diagnostics['method'] = method
    diagnostics['block_size'] = block_size
    diagnostics['n_samples'] = n_synthetic_datasets
    
    logger.info(f"Generated {n_synthetic_datasets} synthetic datasets")
    logger.info(f"Autocorrelation preservation: {diagnostics['autocorrelation_preservation_ratio']:.2%}")
    logger.info(f"Volatility clustering preserved: {diagnostics['bootstrap_volatility_clustering_ratio']:.2%}")
    
    return price_samples, volume_samples, diagnostics


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Generate sample data with volatility clustering
    np.random.seed(42)
    n = 1000
    
    # Simulate GARCH-like returns with volatility clustering
    returns = np.random.normal(0, 1, n)
    vol = np.ones(n)
    for i in range(1, n):
        vol[i] = 0.9 * vol[i-1] + 0.1 * returns[i-1]**2
        returns[i] = np.random.normal(0, np.sqrt(vol[i]))
    
    prices = 100 * np.cumprod(1 + returns / 100)
    
    # Generate synthetic datasets
    price_samples, _, diagnostics = generate_synthetic_training_data(
        prices,
        n_synthetic_datasets=50,
        block_size=20,
        method="stationary",
    )
    
    print("\nDiagnostics:")
    for key, value in diagnostics.items():
        print(f"  {key}: {value}")
    
    print(f"\nGenerated {len(price_samples)} synthetic price series")
    print(f"Each series has shape: {price_samples[0].shape}")
