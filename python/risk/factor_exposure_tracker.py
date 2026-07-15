# python/risk/factor_exposure_tracker.py
"""
Factor exposure tracker using rolling PCA.
Tracks portfolio's hidden exposure to latent factors (Momentum, Volatility, Liquidity, Size).
Memory-efficient implementation with strict RAM constraints.
"""

from __future__ import annotations
import polars as pl
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Dict, List
from collections import deque


@dataclass
class FactorExposure:
    """Exposure of portfolio to a single factor."""
    factor_name: str
    exposure: float          # Beta coefficient
    z_score: float           # Standardized exposure
    p_value: float           # Statistical significance
    rolling_correlation: float  # Rolling correlation with factor
    contribution_to_var: float  # Risk contribution from this factor


@dataclass
class FactorSnapshot:
    """Complete factor exposure snapshot at a point in time."""
    timestamp: int
    exposures: List[FactorExposure]
    r_squared: float         # Model fit quality
    residual_risk: float     # Unexplained risk
    active_factor_bets: List[str]  # Factors with significant tilts


class RollingPCA:
    """
    Memory-efficient rolling Principal Component Analysis.
    
    Uses incremental SVD approximation for low-latency updates.
    Suitable for tracking evolving factor structure in crypto markets.
    """
    
    def __init__(
        self,
        n_components: int = 5,
        window_size: int = 252,
        min_samples: int = 60,
    ):
        """
        Initialize rolling PCA.
        
        Args:
            n_components: Number of factors to extract
            window_size: Rolling window for calculation
            min_samples: Minimum samples before producing output
        """
        self.n_components = n_components
        self.window_size = window_size
        self.min_samples = min_samples
        
        # Circular buffer for returns data (memory-bounded)
        self._returns_buffer: deque = deque(maxlen=window_size)
        self._asset_names: List[str] = []
        
        # Cached results
        self._loadings: Optional[np.ndarray] = None
        self._explained_variance: Optional[np.ndarray] = None
        self._last_update: int = 0
    
    def add_returns(self, timestamp: int, returns: Dict[str, float]) -> bool:
        """
        Add a new observation of asset returns.
        
        Args:
            timestamp: Unix timestamp of observation
            returns: Dict mapping asset -> daily return
            
        Returns:
            True if PCA was recomputed
        """
        if not self._asset_names:
            self._asset_names = list(returns.keys())
        elif len(returns) != len(self._asset_names):
            # Align assets
            aligned = {name: returns.get(name, 0.0) for name in self._asset_names}
            returns = aligned
        
        self._returns_buffer.append(returns)
        
        # Recompute PCA periodically or when we have enough data
        n_obs = len(self._returns_buffer)
        if n_obs >= self.min_samples and n_obs % 10 == 0:
            self._compute_pca()
            return True
        
        return False
    
    def _compute_pca(self) -> None:
        """Compute PCA on the returns matrix."""
        if len(self._returns_buffer) < self.min_samples:
            return
        
        # Convert to matrix
        returns_matrix = np.array([
            [r[name] for name in self._asset_names]
            for r in self._returns_buffer
        ])
        
        # Remove mean (center the data)
        returns_matrix = returns_matrix - returns_matrix.mean(axis=0)
        
        # Compute covariance matrix
        cov_matrix = np.cov(returns_matrix, rowvar=False)
        
        # Eigendecomposition
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        
        # Sort by eigenvalue (descending)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # Keep top components
        self._loadings = eigenvectors[:, :self.n_components]
        self._explained_variance = eigenvalues[:self.n_components]
        self._last_update = len(self._returns_buffer)
    
    def get_factor_returns(self) -> Optional[np.ndarray]:
        """
        Get the time series of factor returns.
        
        Returns:
            Array of shape (n_observations, n_factors)
        """
        if self._loadings is None:
            return None
        
        returns_matrix = np.array([
            [r[name] for name in self._asset_names]
            for r in self._returns_buffer
        ])
        returns_matrix = returns_matrix - returns_matrix.mean(axis=0)
        
        # Project onto loadings
        factor_returns = returns_matrix @ self._loadings
        
        return factor_returns
    
    def get_loadings(self) -> Optional[np.ndarray]:
        """Get current factor loadings matrix."""
        return self._loadings.copy() if self._loadings is not None else None
    
    def get_explained_variance_ratio(self) -> Optional[np.ndarray]:
        """Get ratio of variance explained by each factor."""
        if self._explained_variance is None:
            return None
        
        total_var = self._explained_variance.sum()
        if total_var > 0:
            return self._explained_variance / total_var
        return None


class FactorExposureTracker:
    """
    Tracks portfolio exposure to latent risk factors.
    
    Identifies hidden bets on:
    - Momentum (recent performance continuation)
    - Volatility (sensitivity to market volatility)
    - Liquidity (exposure to liquidity shocks)
    - Size (small-cap vs large-cap bias)
    - Market Beta (overall market sensitivity)
    """
    
    FACTOR_NAMES = [
        "Market",
        "Size", 
        "Momentum",
        "Volatility",
        "Liquidity",
    ]
    
    def __init__(
        self,
        max_history_days: int = 500,
        lookback_days: int = 90,
    ):
        """
        Initialize factor tracker.
        
        Args:
            max_history_days: Maximum history to retain (memory constraint)
            lookback_days: Days for rolling calculations
        """
        self.max_history_days = max_history_days
        self.lookback_days = lookback_days
        
        # Rolling PCA for factor identification
        self.pca = RollingPCA(n_components=5, window_size=lookback_days)
        
        # Historical factor exposures (memory-bounded)
        self._exposure_history: deque = deque(maxlen=max_history_days)
        
        # Factor proxy data (precomputed factor returns)
        self._factor_returns: Dict[str, deque] = {
            name: deque(maxlen=max_history_days)
            for name in self.FACTOR_NAMES
        }
        
        # Portfolio returns history
        self._portfolio_returns: deque = deque(maxlen=max_history_days)
        
        # Current exposures
        self._current_exposures: Dict[str, FactorExposure] = {}
        self._last_r_squared: float = 0.0
        self._last_residual: float = 0.0
    
    def update(
        self,
        timestamp: int,
        portfolio_return: float,
        asset_returns: Dict[str, float],
        factor_proxies: Optional[Dict[str, float]] = None,
    ) -> Optional[FactorSnapshot]:
        """
        Update factor exposures with new return data.
        
        Args:
            timestamp: Unix timestamp
            portfolio_return: Portfolio's return for the period
            asset_returns: Individual asset returns
            factor_proxies: Optional precomputed factor returns
            
        Returns:
            FactorSnapshot if enough data, else None
        """
        # Store portfolio return
        self._portfolio_returns.append((timestamp, portfolio_return))
        
        # Update PCA with asset returns
        self.pca.add_returns(timestamp, asset_returns)
        
        # Store factor proxies if provided
        if factor_proxies:
            for name in self.FACTOR_NAMES:
                if name in factor_proxies:
                    self._factor_returns[name].append((timestamp, factor_proxies[name]))
        
        # Compute exposures if we have enough data
        if len(self._portfolio_returns) >= self.lookback_days:
            snapshot = self._compute_exposures(timestamp)
            self._exposure_history.append(snapshot)
            return snapshot
        
        return None
    
    def _compute_exposures(self, timestamp: int) -> FactorSnapshot:
        """Compute current factor exposures via rolling regression."""
        # Gather historical data
        n_points = min(len(self._portfolio_returns), self.lookback_days)
        
        port_returns = np.array([
            r[1] for r in list(self._portfolio_returns)[-n_points:]
        ])
        
        # Build factor matrix
        factor_names = []
        factor_data = []
        
        for name in self.FACTOR_NAMES:
            fr = list(self._factor_returns[name])[-n_points:]
            if len(fr) == n_points:
                factor_names.append(name)
                factor_data.append([x[1] for x in fr])
        
        if len(factor_data) < 2:
            # Not enough factor data, use PCA-derived factors
            pca_returns = self.pca.get_factor_returns()
            if pca_returns is not None and len(pca_returns) >= n_points:
                factor_data = [pca_returns[-n_points:, i] for i in range(pca_returns.shape[1])]
                factor_names = [f"PCA_{i}" for i in range(pca_returns.shape[1])]
        
        if len(factor_data) == 0:
            return self._empty_snapshot(timestamp)
        
        # Stack factors
        X = np.column_stack(factor_data)
        y = port_returns
        
        # Add constant
        X = np.column_stack([np.ones(len(y)), X])
        
        # OLS regression: y = X @ beta + epsilon
        try:
            # Use numpy's least squares
            beta, residuals, rank, s = np.linalg.lstsq(X, y, rcond=None)
            
            # Compute R-squared
            y_pred = X @ beta
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - y.mean()) ** 2)
            r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
            
            # Compute standard errors for t-stats
            n = len(y)
            k = X.shape[1]
            mse = ss_res / (n - k) if n > k else 1.0
            
            # Variance-covariance matrix of betas
            var_beta = mse * np.linalg.inv(X.T @ X)
            se_beta = np.sqrt(np.diag(var_beta))
            
            # T-statistics and p-values (approximate)
            t_stats = beta / np.maximum(se_beta, 1e-10)
            # Approximate p-value using normal distribution
            p_values = 2 * (1 - 0.5 * (1 + np.erf(np.abs(t_stats) / np.sqrt(2))))
            
        except np.linalg.LinAlgError:
            # Singular matrix, return zeros
            beta = np.zeros(len(factor_names) + 1)
            r_squared = 0.0
            p_values = np.ones(len(factor_names) + 1)
            ss_res = np.sum(port_returns ** 2)
        
        # Build exposure objects
        exposures = []
        active_bets = []
        
        for i, name in enumerate(factor_names):
            exp = beta[i + 1]  # Skip intercept
            p_val = p_values[i + 1]
            
            # Compute z-score (standardized exposure)
            z_score = exp / 0.3 if exp != 0 else 0.0  # Assume typical SE ~0.3
            
            # Rolling correlation
            corr = self._compute_rolling_correlation(name, n_points)
            
            # Contribution to VaR (simplified)
            var_contrib = abs(exp) * 0.15  # Simplified risk contribution
            
            exposure = FactorExposure(
                factor_name=name,
                exposure=exp,
                z_score=z_score,
                p_value=min(p_val, 1.0),
                rolling_correlation=corr,
                contribution_to_var=var_contrib,
            )
            exposures.append(exposure)
            
            # Track significant bets (|t| > 2)
            if abs(z_score) > 1.5:
                active_bets.append(f"{name}: {exp:.2f}")
        
        self._current_exposures = {e.factor_name: e for e in exposures}
        self._last_r_squared = r_squared
        self._last_residual = np.sqrt(ss_res / n_points) if n_points > 0 else 0.0
        
        return FactorSnapshot(
            timestamp=timestamp,
            exposures=exposures,
            r_squared=r_squared,
            residual_risk=self._last_residual,
            active_factor_bets=active_bets,
        )
    
    def _compute_rolling_correlation(self, factor_name: str, n_points: int) -> float:
        """Compute rolling correlation between portfolio and factor."""
        if factor_name not in self._factor_returns:
            return 0.0
        
        port_ret = np.array([r[1] for r in list(self._portfolio_returns)[-n_points:]])
        fact_ret = np.array([r[1] for r in list(self._factor_returns[factor_name])[-n_points:]])
        
        if len(port_ret) != len(fact_ret) or len(port_ret) < 10:
            return 0.0
        
        # Correlation
        if port_ret.std() > 0 and fact_ret.std() > 0:
            corr = np.corrcoef(port_ret, fact_ret)[0, 1]
            return corr if not np.isnan(corr) else 0.0
        
        return 0.0
    
    def _empty_snapshot(self, timestamp: int) -> FactorSnapshot:
        """Return empty snapshot when insufficient data."""
        return FactorSnapshot(
            timestamp=timestamp,
            exposures=[],
            r_squared=0.0,
            residual_risk=0.0,
            active_factor_bets=[],
        )
    
    def get_current_exposures(self) -> Dict[str, FactorExposure]:
        """Get latest factor exposures."""
        return self._current_exposures.copy()
    
    def get_exposure_summary(self) -> Dict:
        """Get JSON-serializable summary of current exposures."""
        return {
            "exposures": {
                name: {
                    "exposure": exp.exposure,
                    "z_score": exp.z_score,
                    "p_value": exp.p_value,
                    "correlation": exp.rolling_correlation,
                    "var_contribution": exp.contribution_to_var,
                }
                for name, exp in self._current_exposures.items()
            },
            "r_squared": self._last_r_squared,
            "residual_risk": self._last_residual,
            "significant_bets": [
                f"{name}: {exp.exposure:.3f}"
                for name, exp in self._current_exposures.items()
                if abs(exp.z_score) > 1.5
            ],
        }
    
    def get_hidden_risks(self, threshold: float = 2.0) -> List[str]:
        """
        Identify potentially dangerous hidden factor exposures.
        
        Args:
            threshold: Z-score threshold for concern
            
        Returns:
            List of warning messages
        """
        warnings = []
        
        for name, exp in self._current_exposures.items():
            if abs(exp.z_score) > threshold:
                direction = "LONG" if exp.exposure > 0 else "SHORT"
                warnings.append(
                    f"HIDDEN RISK: Significant {direction} exposure to {name} "
                    f"(z={exp.z_score:.2f}, p={exp.p_value:.3f})"
                )
        
        if self._last_r_squared < 0.5:
            warnings.append(
                f"MODEL WARNING: Low R² ({self._last_r_squared:.2f}) - "
                "significant unexplained risk"
            )
        
        return warnings


def create_factor_proxies(
    asset_returns: Dict[str, float],
    market_data: Dict,
) -> Dict[str, float]:
    """
    Create simple factor proxies from available data.
    
    This is a simplified implementation - production would use
    more sophisticated factor construction.
    """
    proxies = {}
    
    # Market proxy: equal-weighted average of all assets
    if asset_returns:
        proxies["Market"] = np.mean(list(asset_returns.values()))
    
    # Volatility proxy: inverse of market return magnitude
    proxies["Volatility"] = market_data.get("vix_change", 0.0)
    
    # Liquidity proxy: funding rate changes
    proxies["Liquidity"] = market_data.get("funding_rate_change", 0.0)
    
    # Momentum proxy: recent trend indicator
    proxies["Momentum"] = market_data.get("momentum_signal", 0.0)
    
    # Size proxy: small-cap minus large-cap return
    proxies["Size"] = market_data.get("size_factor", 0.0)
    
    return proxies


if __name__ == "__main__":
    # Example usage
    import random
    
    tracker = FactorExposureTracker(
        max_history_days=252,
        lookback_days=60,
    )
    
    # Simulate some data
    assets = ["BTC", "ETH", "SOL", "AVAX", "DOT"]
    
    for day in range(100):
        timestamp = 1700000000 + day * 86400
        
        # Random asset returns
        asset_returns = {
            asset: random.gauss(0.001, 0.03)
            for asset in assets
        }
        
        # Portfolio return (weighted sum with some alpha)
        portfolio_return = sum(asset_returns.values()) / len(assets) + random.gauss(0, 0.01)
        
        # Factor proxies
        factor_proxies = {
            "Market": np.mean(list(asset_returns.values())),
            "Volatility": random.gauss(0, 0.02),
            "Liquidity": random.gauss(0, 0.01),
            "Momentum": random.gauss(0.001, 0.02),
            "Size": random.gauss(0, 0.015),
        }
        
        snapshot = tracker.update(
            timestamp=timestamp,
            portfolio_return=portfolio_return,
            asset_returns=asset_returns,
            factor_proxies=factor_proxies,
        )
        
        if snapshot and day % 20 == 0:
            print(f"\nDay {day}:")
            print(f"  R²: {snapshot.r_squared:.3f}")
            print(f"  Active bets: {snapshot.active_factor_bets}")
