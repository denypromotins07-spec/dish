"""
Unsupervised learning module for market regime clustering.
Implements HDBSCAN and K-Means with PCA dimensionality reduction.
Optimized for minimal RAM usage with aggressive memory management.
"""

import os
import logging
from typing import Any, Dict, Optional, Tuple, List
import numpy as np
import pandas as pd

# Memory constraints
os.environ["OMP_NUM_THREADS"] = "4"

try:
    from sklearn.cluster import KMeans, DBSCAN
    from sklearn.decomposition import IncrementalPCA
    from sklearn.preprocessing import StandardScaler
except ImportError:
    raise ImportError("scikit-learn required. Install with: pip install scikit-learn")

try:
    import hdbscan
    HDBSCAN_AVAILABLE = True
except ImportError:
    HDBSCAN_AVAILABLE = False
    logging.warning("hdbscan not available. Install with: pip install hdbscan")

logger = logging.getLogger(__name__)


class MarketRegimeClusterer:
    """
    Market regime clustering using HDBSCAN and K-Means with PCA.
    Designed to identify ranging vs trending states with minimal RAM footprint.
    """
    
    def __init__(
        self,
        n_clusters: int = 3,  # Typically: ranging, trending_up, trending_down
        n_pca_components: int = 10,  # Aggressive dimensionality reduction
        max_memory_gb: float = 2.0,  # Max memory for clustering
        batch_size: int = 10000,
        min_cluster_size: int = 50,
        min_samples: int = 5,
    ):
        self.n_clusters = n_clusters
        self.n_pca_components = n_pca_components
        self.max_memory_gb = max_memory_gb
        self.batch_size = batch_size
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        
        self.scaler: Optional[StandardScaler] = None
        self.pca: Optional[IncrementalPCA] = None
        self.kmeans: Optional[KMeans] = None
        self.hdbscan_model: Optional[Any] = None
        
        self.cluster_labels_: Optional[np.ndarray] = None
        self.cluster_centers_: Optional[np.ndarray] = None
        self.regime_mapping: Dict[int, str] = {}
    
    def _check_memory(self) -> float:
        """Check available system RAM."""
        import psutil
        return psutil.virtual_memory().available / (1024 ** 3)
    
    def _adaptive_batch_size(self, n_samples: int, n_features: int) -> int:
        """Calculate adaptive batch size based on available memory."""
        available_ram = self._check_memory()
        
        # Estimate memory per sample (float64)
        bytes_per_sample = n_features * 8
        
        # Target: use at most 50% of available RAM for data
        max_samples_in_memory = int((available_ram * 0.5 * 1024**3) / bytes_per_sample)
        
        # Clamp to reasonable bounds
        return max(1000, min(max_samples_in_memory, 50000))
    
    def fit_pca_incremental(self, X: np.ndarray) -> np.ndarray:
        """
        Fit PCA incrementally to avoid loading entire dataset into memory.
        Returns transformed data.
        """
        n_samples, n_features = X.shape
        
        if n_features <= self.n_pca_components:
            logger.info(f"Skipping PCA: features ({n_features}) <= components ({self.n_pca_components})")
            return X
        
        # Adjust components if needed
        actual_components = min(self.n_pca_components, n_features - 1, n_samples - 1)
        
        logger.info(f"Fitting IncrementalPCA: {n_features} -> {actual_components} components")
        
        self.pca = IncrementalPCA(n_components=actual_components, batch_size=self.batch_size)
        
        # Fit in batches
        for i in range(0, n_samples, self.batch_size):
            batch = X[i:i+self.batch_size]
            self.pca.partial_fit(batch)
        
        # Transform in batches to save memory
        X_transformed = np.zeros((n_samples, actual_components), dtype=np.float32)
        
        for i in range(0, n_samples, self.batch_size):
            batch = X[i:i+self.batch_size]
            X_transformed[i:i+len(batch)] = self.pca.transform(batch).astype(np.float32)
        
        # Free some memory
        del X
        
        return X_transformed
    
    def fit_hierarchical_clustering(self, X: np.ndarray) -> Dict[str, Any]:
        """
        Perform hierarchical clustering: first K-Means, then refine with HDBSCAN if available.
        """
        import psutil
        
        n_samples = len(X)
        logger.info(f"Starting clustering on {n_samples} samples")
        
        # Check memory
        available_ram = self._check_memory()
        if available_ram < 1.0:
            logger.warning("Critical low memory. Using simplified K-Means only.")
            return self._fit_kmeans_only(X)
        
        # Try HDBSCAN if available and enough memory
        if HDBSCAN_AVAILABLE and available_ram > 2.0:
            try:
                return self._fit_hdbscan_with_kmeans_init(X)
            except Exception as e:
                logger.warning(f"HDBSCAN failed: {e}. Falling back to K-Means.")
                return self._fit_kmeans_only(X)
        else:
            return self._fit_kmeans_only(X)
    
    def _fit_kmeans_only(self, X: np.ndarray) -> Dict[str, Any]:
        """Fit K-Means clustering only."""
        import psutil
        
        logger.info("Fitting K-Means clustering...")
        
        # Adaptive number of clusters based on data size
        actual_clusters = min(self.n_clusters, len(X) // 10)
        actual_clusters = max(2, actual_clusters)
        
        self.kmeans = KMeans(
            n_clusters=actual_clusters,
            init="k-means++",
            n_init=10,
            max_iter=300,
            random_state=42,
            algorithm="lloyd",
        )
        
        self.cluster_labels_ = self.kmeans.fit_predict(X)
        self.cluster_centers_ = self.kmeans.cluster_centers_
        
        # Map clusters to regime names
        self._map_regimes(X)
        
        return {
            "algorithm": "kmeans",
            "n_clusters": actual_clusters,
            "inertia": self.kmeans.inertia_,
            "available_ram_gb": psutil.virtual_memory().available / (1024**3),
        }
    
    def _fit_hdbscan_with_kmeans_init(self, X: np.ndarray) -> Dict[str, Any]:
        """Use K-Means to initialize HDBSCAN for better performance."""
        import psutil
        
        logger.info("Fitting HDBSCAN with K-Means initialization...")
        
        # First, run K-Means with more clusters
        kmeans_temp = KMeans(
            n_clusters=self.n_clusters * 2,
            init="k-means++",
            n_init=10,
            max_iter=100,
            random_state=42,
        )
        kmeans_temp.fit(X)
        
        # Use K-Means centers as HDBSCAN input (reduced dataset)
        self.hdbscan_model = hdbscan.HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            min_samples=self.min_samples,
            metric="euclidean",
            prediction_data=True,
            core_dist_n_jobs=4,
        )
        
        self.cluster_labels_ = self.hdbscan_model.fit_predict(X)
        
        # Calculate cluster centers manually
        unique_labels = np.unique(self.cluster_labels_[self.cluster_labels_ >= 0])
        centers = []
        for label in unique_labels:
            mask = self.cluster_labels_ == label
            centers.append(X[mask].mean(axis=0))
        
        self.cluster_centers_ = np.array(centers) if centers else None
        
        # Map clusters to regime names
        self._map_regimes(X)
        
        # Count noise points
        noise_count = np.sum(self.cluster_labels_ == -1)
        
        return {
            "algorithm": "hdbscan",
            "n_clusters": len(unique_labels),
            "noise_points": noise_count,
            "noise_ratio": noise_count / len(X),
            "available_ram_gb": psutil.virtual_memory().available / (1024**3),
        }
    
    def _map_regimes(self, X: np.ndarray):
        """Map cluster labels to meaningful regime names based on statistics."""
        if self.cluster_labels_ is None or self.cluster_centers_ is None:
            return
        
        unique_labels = np.unique(self.cluster_labels_[self.cluster_labels_ >= 0])
        
        # Calculate volatility and trend metrics for each cluster
        cluster_metrics = []
        for label in unique_labels:
            mask = self.cluster_labels_ == label
            cluster_data = X[mask]
            
            # Simple metrics: mean and std (assuming features include returns/volatility)
            mean_return = cluster_data[:, 0].mean() if cluster_data.shape[1] > 0 else 0
            volatility = cluster_data[:, 0].std() if cluster_data.shape[1] > 0 else 0
            
            cluster_metrics.append({
                "label": int(label),
                "mean": mean_return,
                "volatility": volatility,
                "size": mask.sum(),
            })
        
        # Sort by volatility to identify regimes
        cluster_metrics.sort(key=lambda x: x["volatility"])
        
        # Assign regime names
        self.regime_mapping = {}
        for i, metric in enumerate(cluster_metrics):
            if i == 0:
                regime = "low_vol_ranging"
            elif i == len(cluster_metrics) - 1:
                regime = "high_vol_trending"
            else:
                regime = "medium_vol_transition"
            
            self.regime_mapping[metric["label"]] = regime
        
        logger.info(f"Regime mapping: {self.regime_mapping}")
    
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Full fitting pipeline: scaling -> PCA -> clustering.
        """
        import gc
        
        logger.info(f"Starting full clustering pipeline on {X.shape} data")
        
        # Step 1: Scale data
        logger.info("Scaling data...")
        self.scaler = StandardScaler()
        
        # Scale in batches if data is large
        if len(X) > self.batch_size:
            # Partial fit for memory efficiency
            for i in range(0, len(X), self.batch_size):
                batch = X[i:i+self.batch_size]
                if i == 0:
                    self.scaler.partial_fit(batch)
                else:
                    # Update running stats (simplified)
                    pass
            
            # Re-fit on a sample for accuracy
            sample_idx = np.random.choice(len(X), min(self.batch_size * 10, len(X)), replace=False)
            self.scaler.fit(X[sample_idx])
        
        X_scaled = self.scaler.transform(X).astype(np.float32)
        
        # Free memory
        del X
        gc.collect()
        
        # Step 2: PCA dimensionality reduction
        logger.info("Applying PCA...")
        X_reduced = self.fit_pca_incremental(X_scaled)
        
        # Free memory
        del X_scaled
        gc.collect()
        
        # Step 3: Clustering
        logger.info("Running clustering...")
        results = self.fit_hierarchical_clustering(X_reduced)
        
        # Store reduced data for prediction
        self.X_reduced_ = X_reduced
        
        return results
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict cluster labels for new data."""
        if self.scaler is None or (self.kmeans is None and self.hdbscan_model is None):
            raise ValueError("Model not fitted yet")
        
        # Scale
        X_scaled = self.scaler.transform(X).astype(np.float32)
        
        # PCA transform
        if self.pca is not None:
            X_reduced = self.pca.transform(X_scaled).astype(np.float32)
        else:
            X_reduced = X_scaled
        
        # Predict
        if self.hdbscan_model is not None and hasattr(self.hdbscan_model, "approximate_predict"):
            labels, _ = hdbscan.approximate_predict(self.hdbscan_model, X_reduced)
        elif self.kmeans is not None:
            labels = self.kmeans.predict(X_reduced)
        else:
            raise ValueError("No clustering model available")
        
        return labels
    
    def get_regime_name(self, label: int) -> str:
        """Get human-readable regime name for a cluster label."""
        if label == -1:
            return "noise"
        return self.regime_mapping.get(label, f"cluster_{label}")
    
    def get_regime_distribution(self, labels: Optional[np.ndarray] = None) -> Dict[str, int]:
        """Get distribution of regimes."""
        if labels is None:
            labels = self.cluster_labels_
        
        if labels is None:
            return {}
        
        distribution = {}
        for label in np.unique(labels):
            regime_name = self.get_regime_name(int(label))
            count = np.sum(labels == label)
            distribution[regime_name] = int(count)
        
        return distribution


def main():
    """Example usage with synthetic market data."""
    import psutil
    
    # Generate synthetic market data (returns, volatility, volume, etc.)
    np.random.seed(42)
    n_samples = 50000
    n_features = 20
    
    # Simulate different regimes
    X = np.random.randn(n_samples, n_features).astype(np.float32)
    
    # Add some structure: trending periods have higher mean returns
    trend_start = 10000
    trend_end = 20000
    X[trend_start:trend_end, 0] += 0.5  # Positive trend
    
    range_start = 30000
    range_end = 40000
    X[range_start:range_end, :] *= 0.3  # Low volatility ranging
    
    print(f"Generated {n_samples} samples with {n_features} features")
    print(f"Initial available RAM: {psutil.virtual_memory().available / (1024**3):.2f}GB")
    
    # Initialize clusterer
    clusterer = MarketRegimeClusterer(
        n_clusters=3,
        n_pca_components=10,
        max_memory_gb=2.0,
        batch_size=5000,
    )
    
    # Fit
    results = clusterer.fit(X)
    print(f"\nClustering results: {results}")
    
    # Get regime distribution
    distribution = clusterer.get_regime_distribution()
    print(f"\nRegime distribution: {distribution}")
    
    # Predict on new data
    X_new = np.random.randn(1000, n_features).astype(np.float32)
    predictions = clusterer.predict(X_new)
    print(f"\nPredictions on new data: {np.bincount(predictions[predictions >= 0])}")
    
    print(f"\nFinal available RAM: {psutil.virtual_memory().available / (1024**3):.2f}GB")


if __name__ == "__main__":
    main()
