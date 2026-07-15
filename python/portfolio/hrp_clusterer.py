"""
Hierarchical Risk Parity (HRP) implementation using fast hierarchical clustering
of crypto assets based on return correlations.

Prevents concentration risk in highly correlated altcoin baskets.
Memory-efficient implementation bounded for <14GB total system RAM.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform


@dataclass(slots=True)
class HRPResult:
    """Result of HRP allocation."""
    weights: np.ndarray
    cluster_assignments: np.ndarray
    dendrogram_linkage: np.ndarray
    total_risk: float
    n_clusters: int


def compute_distance_matrix(correlation: np.ndarray) -> np.ndarray:
    """
    Convert correlation matrix to distance matrix.
    
    Args:
        correlation: Correlation matrix
    
    Returns:
        Distance matrix where d = sqrt((1 - corr) / 2)
    """
    # Ensure correlation is valid
    correlation = np.clip(correlation, -1.0, 1.0)
    
    # Distance metric: d = sqrt((1 - corr) / 2)
    distance = np.sqrt((1.0 - correlation) / 2.0)
    
    # Set diagonal to zero
    np.fill_diagonal(distance, 0.0)
    
    return distance


def hierarchical_clustering(
    returns: np.ndarray,
    method: str = 'ward',
    max_clusters: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform hierarchical clustering on asset returns.
    
    Args:
        returns: T x N matrix of returns
        method: Clustering method ('ward', 'single', 'complete', 'average')
        max_clusters: Maximum number of clusters
    
    Returns:
        linkage_matrix, cluster_assignments
    """
    n_assets = returns.shape[1]
    
    # Compute correlation matrix
    correlation = np.corrcoef(returns.T)
    
    # Handle NaN correlations
    correlation = np.nan_to_num(correlation, nan=0.0)
    
    # Convert to distance matrix
    distance = compute_distance_matrix(correlation)
    
    # Convert to condensed form for scipy
    condensed_dist = squareform(distance, checks=False)
    
    # Perform hierarchical clustering
    linkage_matrix = linkage(condensed_dist, method=method)
    
    # Get cluster assignments
    if max_clusters is not None:
        cluster_assignments = fcluster(linkage_matrix, max_clusters, criterion='maxclust')
    else:
        # Default: cut at midpoint of dendrogram height
        max_height = np.max(linkage_matrix[:, 2])
        cluster_assignments = fcluster(linkage_matrix, max_height / 2, criterion='distance')
    
    return linkage_matrix, cluster_assignments


def quasi_diagonal_matrix(linkage: np.ndarray, n: int) -> np.ndarray:
    """
    Compute quasi-diagonal matrix from linkage.
    Reorders assets so that similar assets are adjacent.
    
    Args:
        linkage: Hierarchical clustering linkage matrix
        n: Number of assets
    
    Returns:
        Permutation vector
    """
    # Get leaf order from dendrogram
    from scipy.cluster.hierarchy import leaves_list
    return leaves_list(linkage)


def recursive_bisection(
    covariance: np.ndarray,
    linkage: np.ndarray,
    cluster_ids: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Recursive bisection algorithm for HRP weight calculation.
    
    Args:
        covariance: Asset covariance matrix
        linkage: Hierarchical clustering linkage
        cluster_ids: Initial cluster IDs (optional)
    
    Returns:
        Portfolio weights
    """
    n = covariance.shape[0]
    weights = np.ones(n)
    
    # Get quasi-diagonal ordering
    perm = quasi_diagonal_matrix(linkage, n)
    inv_perm = np.argsort(perm)
    
    # Reorder covariance
    cov_perm = covariance[perm, :][:, perm]
    
    # Recursive bisection
    def bisect(cov: np.ndarray, indices: np.ndarray) -> np.ndarray:
        if len(indices) == 1:
            return np.array([1.0])
        
        if len(indices) == 2:
            # Simple inverse variance weighting for 2 assets
            var = np.diag(cov)
            inv_var = 1.0 / np.maximum(var, 1e-10)
            w = inv_var / np.sum(inv_var)
            return w
        
        # Split cluster into two sub-clusters
        mid = len(indices) // 2
        left_idx = indices[:mid]
        right_idx = indices[mid:]
        
        # Calculate cluster variances
        left_cov = cov[left_idx, :][:, left_idx]
        right_cov = cov[right_idx, :][:, right_idx]
        
        # Inverse variance for each cluster
        left_var = np.trace(left_cov) / len(left_idx)
        right_var = np.trace(right_cov) / len(right_idx)
        
        # Allocate between clusters
        alpha = 1.0 - left_var / (left_var + right_var)
        
        # Recursively allocate within clusters
        left_weights = bisect(cov, left_idx)
        right_weights = bisect(cov, right_idx)
        
        # Combine
        full_weights = np.zeros(len(indices))
        full_weights[:mid] = alpha * left_weights
        full_weights[mid:] = (1.0 - alpha) * right_weights
        
        return full_weights
    
    # Start with all indices in sorted order
    indices = np.arange(n)
    weights_perm = bisect(cov_perm, indices)
    
    # Restore original ordering
    weights = np.zeros(n)
    weights[perm] = weights_perm
    
    return weights


def get_cluster_variance(contributions: np.ndarray, covariance: np.ndarray) -> float:
    """Calculate variance of a cluster given weight contributions."""
    return contributions @ covariance @ contributions


def hrp_allocation(
    returns: np.ndarray,
    covariance: Optional[np.ndarray] = None,
    method: str = 'ward',
    max_clusters: Optional[int] = None
) -> HRPResult:
    """
    Full HRP allocation pipeline.
    
    Args:
        returns: T x N matrix of returns
        covariance: Pre-computed covariance (optional, computed from returns if not provided)
        method: Clustering method
        max_clusters: Maximum number of clusters
    
    Returns:
        HRPResult with weights and cluster info
    """
    n_assets = returns.shape[1]
    
    # Compute covariance if not provided
    if covariance is None:
        covariance = np.cov(returns.T)
    
    # Ensure positive semi-definite
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    eigenvalues = np.maximum(eigenvalues, 1e-10)
    covariance = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
    
    # Perform clustering
    linkage_matrix, cluster_assignments = hierarchical_clustering(
        returns, method=method, max_clusters=max_clusters
    )
    
    # Calculate weights via recursive bisection
    weights = recursive_bisection(covariance, linkage_matrix)
    
    # Normalize
    weights = weights / np.sum(weights)
    
    # Calculate portfolio risk
    port_var = weights @ covariance @ weights
    port_vol = np.sqrt(port_var)
    
    n_clusters = len(np.unique(cluster_assignments))
    
    return HRPResult(
        weights=weights,
        cluster_assignments=cluster_assignments,
        dendrogram_linkage=linkage_matrix,
        total_risk=port_vol,
        n_clusters=n_clusters
    )


class HierarchicalRiskParity:
    """
    Incremental HRP allocator with caching and online updates.
    Optimized for crypto portfolios with rapidly changing correlations.
    """
    
    __slots__ = (
        'n_assets', 'asset_names', 'returns_buffer', 'max_history',
        'last_weights', 'last_clustering', 'cache_valid', 'min_history'
    )
    
    def __init__(
        self,
        n_assets: int,
        asset_names: Optional[List[str]] = None,
        max_history: int = 252,
        min_history: int = 30
    ):
        self.n_assets = n_assets
        self.asset_names = asset_names or [f"asset_{i}" for i in range(n_assets)]
        self.returns_buffer: Optional[np.ndarray] = None
        self.max_history = max_history
        self.min_history = min_history
        self.last_weights: Optional[np.ndarray] = None
        self.last_clustering: Optional[HRPResult] = None
        self.cache_valid = False
    
    def add_returns(self, new_returns: np.ndarray) -> None:
        """Add new returns observation(s)."""
        new_returns = np.atleast_2d(new_returns)
        
        if new_returns.shape[1] != self.n_assets:
            raise ValueError(f"Expected {self.n_assets} assets, got {new_returns.shape[1]}")
        
        if self.returns_buffer is None:
            self.returns_buffer = new_returns[-self.max_history:]
        else:
            self.returns_buffer = np.vstack([self.returns_buffer, new_returns])
            # Trim buffer
            if len(self.returns_buffer) > self.max_history:
                self.returns_buffer = self.returns_buffer[-self.max_history:]
        
        self.cache_valid = False
    
    def compute(self, force_recompute: bool = False) -> Optional[HRPResult]:
        """Compute HRP allocation."""
        if self.returns_buffer is None or len(self.returns_buffer) < self.min_history:
            return None
        
        if self.cache_valid and self.last_clustering is not None and not force_recompute:
            return self.last_clustering
        
        result = hrp_allocation(self.returns_buffer)
        
        self.last_weights = result.weights.copy()
        self.last_clustering = result
        self.cache_valid = True
        
        return result
    
    def get_cluster_summary(self) -> Dict[str, any]:
        """Get summary of current cluster structure."""
        if self.last_clustering is None:
            return {}
        
        clusters = self.last_clustering.cluster_assignments
        weights = self.last_clustering.weights
        
        summary = {
            'n_clusters': self.last_clustering.n_clusters,
            'clusters': {}
        }
        
        for cluster_id in np.unique(clusters):
            mask = clusters == cluster_id
            cluster_assets = [self.asset_names[i] for i in range(self.n_assets) if mask[i]]
            cluster_weight = np.sum(weights[mask])
            
            summary['clusters'][int(cluster_id)] = {
                'assets': cluster_assets,
                'weight': float(cluster_weight),
                'size': int(np.sum(mask))
            }
        
        return summary
    
    def rebalance_needed(self, threshold: float = 0.05) -> bool:
        """Check if rebalancing is needed based on weight drift."""
        if self.last_weights is None:
            return True
        
        # Quick estimate: check if any single asset weight drifted significantly
        current_result = self.compute(force_recompute=False)
        if current_result is None:
            return True
        
        weight_diff = np.abs(current_result.weights - self.last_weights)
        return np.max(weight_diff) > threshold


def nested_clusters_hrp(
    returns: np.ndarray,
    n_macro_clusters: int = 3,
    n_micro_per_macro: int = 5
) -> HRPResult:
    """
    Nested HRP: First cluster into macro groups, then apply HRP within each.
    Useful for crypto with distinct sectors (L1, L2, DeFi, etc.).
    """
    n_assets = returns.shape[1]
    
    # First level: macro clustering
    linkage_macro, macro_clusters = hierarchical_clustering(
        returns, method='ward', max_clusters=n_macro_clusters
    )
    
    # Initialize weights
    weights = np.zeros(n_assets)
    
    # Covariance
    covariance = np.cov(returns.T)
    
    # For each macro cluster, apply HRP
    macro_weights = np.ones(n_macro_clusters) / n_macro_clusters
    
    for macro_id in range(n_macro_clusters):
        mask = macro_clusters == (macro_id + 1)  # fcluster uses 1-indexed
        indices = np.where(mask)[0]
        
        if len(indices) == 0:
            continue
        
        # Sub-covariance
        sub_cov = covariance[np.ix_(indices, indices)]
        sub_returns = returns[:, indices]
        
        # Apply HRP to this cluster
        if len(indices) > 1:
            sub_result = hrp_allocation(sub_returns, sub_cov)
            weights[indices] = macro_weights[macro_id] * sub_result.weights
        else:
            weights[indices] = macro_weights[macro_id]
    
    # Normalize
    weights /= np.sum(weights)
    
    # Portfolio risk
    port_var = weights @ covariance @ weights
    port_vol = np.sqrt(port_var)
    
    return HRPResult(
        weights=weights,
        cluster_assignments=macro_clusters,
        dendrogram_linkage=linkage_macro,
        total_risk=port_vol,
        n_clusters=n_macro_clusters
    )


if __name__ == '__main__':
    # Test HRP allocation
    np.random.seed(42)
    
    # Simulate returns for 10 assets over 100 days
    # Create some correlation structure
    n_assets = 10
    n_days = 100
    
    # Factor model: 3 factors driving returns
    factors = np.random.randn(n_days, 3)
    factor_loadings = np.random.randn(n_assets, 3)
    idiosyncratic = np.random.randn(n_days, n_assets) * 0.5
    
    returns = factors @ factor_loadings.T + idiosyncratic
    
    print("Testing HRP Allocation:")
    result = hrp_allocation(returns)
    print(f"  Number of clusters: {result.n_clusters}")
    print(f"  Cluster assignments: {result.cluster_assignments}")
    print(f"  Weights: {result.weights}")
    print(f"  Portfolio volatility: {result.total_risk:.4f}")
    
    # Compare to equal weight
    cov = np.cov(returns.T)
    eq_weights = np.ones(n_assets) / n_assets
    eq_vol = np.sqrt(eq_weights @ cov @ eq_weights)
    print(f"\nEqual weight volatility: {eq_vol:.4f}")
    print(f"HRP diversification ratio: {eq_vol / result.total_risk:.3f}")
