"""
Automated feature selection using lightweight Mutual Information and Variance Thresholding.
Drops highly correlated or useless features to reduce ML input dimensionality.
Designed for low memory footprint on 16GB systems.
"""

import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from collections import defaultdict


class VarianceThresholdSelector:
    """
    Remove features with variance below threshold.
    Low variance features provide little information.
    """
    
    __slots__ = ('_threshold', '_variances', '_support')
    
    def __init__(self, threshold: float = 1e-4):
        self._threshold = threshold
        self._variances: Optional[np.ndarray] = None
        self._support: Optional[np.ndarray] = None
    
    def fit(self, X: np.ndarray) -> 'VarianceThresholdSelector':
        """Compute variances and determine which features to keep."""
        self._variances = np.var(X, axis=0)
        self._support = self._variances > self._threshold
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform data by removing low-variance features."""
        if self._support is None:
            raise ValueError("Must call fit() before transform()")
        return X[:, self._support]
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit and transform in one step."""
        self.fit(X)
        return self.transform(X)
    
    def get_support(self, indices: bool = False) -> np.ndarray:
        """Get boolean mask or indices of selected features."""
        if indices:
            return np.where(self._support)[0]
        return self._support
    
    def get_variances(self) -> np.ndarray:
        """Get computed variances."""
        return self._variances.copy() if self._variances is not None else None


class CorrelationFilter:
    """
    Remove highly correlated features to reduce multicollinearity.
    Keeps one feature from each highly correlated pair.
    """
    
    __slots__ = ('_threshold', '_correlation_matrix', '_selected_indices')
    
    def __init__(self, threshold: float = 0.95):
        self._threshold = threshold
        self._correlation_matrix: Optional[np.ndarray] = None
        self._selected_indices: Optional[np.ndarray] = None
    
    def fit(self, X: np.ndarray) -> 'CorrelationFilter':
        """Identify and mark highly correlated features for removal."""
        n_features = X.shape[1]
        
        # Compute correlation matrix (memory-efficient for large feature sets)
        # Process in chunks if needed
        chunk_size = min(500, n_features)
        self._correlation_matrix = np.zeros((n_features, n_features))
        
        for i in range(0, n_features, chunk_size):
            end_i = min(i + chunk_size, n_features)
            for j in range(i, n_features, chunk_size):
                end_j = min(j + chunk_size, n_features)
                
                # Compute correlation for this chunk
                chunk_i = X[:, i:end_i]
                chunk_j = X[:, j:end_j]
                
                # Standardize
                mean_i = np.mean(chunk_i, axis=0, keepdims=True)
                std_i = np.std(chunk_i, axis=0, keepdims=True) + 1e-10
                mean_j = np.mean(chunk_j, axis=0, keepdims=True)
                std_j = np.std(chunk_j, axis=0, keepdims=True) + 1e-10
                
                norm_i = (chunk_i - mean_i) / std_i
                norm_j = (chunk_j - mean_j) / std_j
                
                corr_chunk = np.dot(norm_i.T, norm_j) / (X.shape[0] - 1)
                self._correlation_matrix[i:end_i, j:end_j] = corr_chunk
        
        # Fill lower triangle (symmetric)
        self._correlation_matrix = (
            self._correlation_matrix + self._correlation_matrix.T
        ) / 2
        np.fill_diagonal(self._correlation_matrix, 1.0)
        
        # Select features: keep one from each correlated pair
        self._selected_indices = self._select_uncorrelated_features()
        
        return self
    
    def _select_uncorrelated_features(self) -> np.ndarray:
        """Greedy selection of uncorrelated features."""
        n_features = self._correlation_matrix.shape[0]
        selected = []
        rejected = set()
        
        # Sort features by average absolute correlation (prefer less correlated)
        avg_corr = np.mean(np.abs(self._correlation_matrix), axis=1)
        sorted_indices = np.argsort(avg_corr)
        
        for idx in sorted_indices:
            if idx in rejected:
                continue
            
            selected.append(idx)
            
            # Mark highly correlated features as rejected
            correlations = np.abs(self._correlation_matrix[idx])
            rejected.update(np.where(correlations > self._threshold)[0])
        
        return np.array(sorted(selected))
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform data by removing correlated features."""
        if self._selected_indices is None:
            raise ValueError("Must call fit() before transform()")
        return X[:, self._selected_indices]
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit and transform in one step."""
        self.fit(X)
        return self.transform(X)
    
    def get_support(self, indices: bool = False) -> np.ndarray:
        """Get boolean mask or indices of selected features."""
        if self._selected_indices is None:
            raise ValueError("Must call fit() first")
        
        if indices:
            return self._selected_indices
        
        mask = np.zeros(self._correlation_matrix.shape[0], dtype=bool)
        mask[self._selected_indices] = True
        return mask
    
    def get_correlation_matrix(self) -> Optional[np.ndarray]:
        """Get the computed correlation matrix."""
        return self._correlation_matrix.copy() if self._correlation_matrix is not None else None


class MutualInformationSelector:
    """
    Lightweight mutual information estimation for feature selection.
    Uses histogram-based approximation for speed and low memory.
    """
    
    __slots__ = ('_k', '_bins', '_mi_scores', '_selected_indices')
    
    def __init__(self, k: int = 100, bins: int = 20):
        """
        Args:
            k: Number of top features to select
            bins: Number of bins for histogram discretization
        """
        self._k = k
        self._bins = bins
        self._mi_scores: Optional[np.ndarray] = None
        self._selected_indices: Optional[np.ndarray] = None
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'MutualInformationSelector':
        """Estimate mutual information between each feature and target."""
        n_samples, n_features = X.shape
        self._mi_scores = np.zeros(n_features)
        
        # Discretize target
        y_discrete = self._discretize(y)
        
        for i in range(n_features):
            x_discrete = self._discretize(X[:, i])
            self._mi_scores[i] = self._compute_mi(x_discrete, y_discrete)
        
        # Select top k features
        self._selected_indices = np.argsort(self._mi_scores)[-self._k:]
        self._selected_indices = np.sort(self._selected_indices)
        
        return self
    
    def _discretize(self, x: np.ndarray) -> np.ndarray:
        """Discretize continuous variable into bins."""
        min_val, max_val = np.min(x), np.max(x)
        if max_val - min_val < 1e-10:
            return np.zeros(len(x), dtype=int)
        
        bin_width = (max_val - min_val) / self._bins
        discrete = ((x - min_val) / bin_width).astype(int)
        discrete = np.clip(discrete, 0, self._bins - 1)
        return discrete
    
    def _compute_mi(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        Compute mutual information using histogram method.
        MI(X;Y) = H(X) + H(Y) - H(X,Y)
        """
        n = len(x)
        
        # Joint histogram
        joint_hist = np.zeros((self._bins, self._bins))
        for xi, yi in zip(x, y):
            joint_hist[xi, yi] += 1
        
        # Normalize to probabilities
        joint_prob = joint_hist / n
        
        # Marginal probabilities
        px = np.sum(joint_prob, axis=1)
        py = np.sum(joint_prob, axis=0)
        
        # Compute MI
        mi = 0.0
        for i in range(self._bins):
            for j in range(self._bins):
                if joint_prob[i, j] > 0 and px[i] > 0 and py[j] > 0:
                    mi += joint_prob[i, j] * np.log(
                        joint_prob[i, j] / (px[i] * py[j]) + 1e-10
                    )
        
        return max(0, mi)
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform data by selecting top MI features."""
        if self._selected_indices is None:
            raise ValueError("Must call fit() before transform()")
        return X[:, self._selected_indices]
    
    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Fit and transform in one step."""
        self.fit(X, y)
        return self.transform(X)
    
    def get_support(self, indices: bool = False) -> np.ndarray:
        """Get boolean mask or indices of selected features."""
        if self._selected_indices is None:
            raise ValueError("Must call fit() first")
        
        if indices:
            return self._selected_indices
        
        mask = np.zeros(self._mi_scores.shape[0], dtype=bool)
        mask[self._selected_indices] = True
        return mask
    
    def get_mi_scores(self) -> Optional[np.ndarray]:
        """Get computed MI scores."""
        return self._mi_scores.copy() if self._mi_scores is not None else None


class FeatureSelectionPipeline:
    """
    Combined feature selection pipeline.
    Applies variance threshold, correlation filter, and MI selection sequentially.
    """
    
    __slots__ = (
        '_variance_selector', 
        '_correlation_filter', 
        '_mi_selector',
        '_feature_names',
        '_selected_feature_names',
    )
    
    def __init__(
        self,
        variance_threshold: float = 1e-4,
        correlation_threshold: float = 0.95,
        max_features: int = 100,
    ):
        self._variance_selector = VarianceThresholdSelector(variance_threshold)
        self._correlation_filter = CorrelationFilter(correlation_threshold)
        self._mi_selector = MutualInformationSelector(k=max_features)
        self._feature_names: List[str] = []
        self._selected_feature_names: List[str] = []
    
    def fit(
        self, 
        X: np.ndarray, 
        y: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None
    ) -> 'FeatureSelectionPipeline':
        """Fit all selectors sequentially."""
        self._feature_names = feature_names if feature_names else [f'f{i}' for i in range(X.shape[1])]
        
        # Step 1: Variance threshold
        X_filtered = self._variance_selector.fit_transform(X)
        variance_mask = self._variance_selector.get_support(indices=False)
        names_after_variance = [
            name for name, keep in zip(self._feature_names, variance_mask) if keep
        ]
        
        # Step 2: Correlation filter (only if enough samples)
        if X_filtered.shape[0] >= 100 and X_filtered.shape[1] > 1:
            X_filtered = self._correlation_filter.fit_transform(X_filtered)
            corr_indices = self._correlation_filter.get_support(indices=True)
            names_after_corr = [names_after_variance[i] for i in corr_indices]
        else:
            names_after_corr = names_after_variance
        
        # Step 3: MI selection (if target provided)
        if y is not None and X_filtered.shape[1] > self._mi_selector._k:
            X_filtered = self._mi_selector.fit_transform(X_filtered, y)
            mi_indices = self._mi_selector.get_support(indices=True)
            self._selected_feature_names = [names_after_corr[i] for i in mi_indices]
        else:
            self._selected_feature_names = names_after_corr
        
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform data through all selection steps."""
        X_out = self._variance_selector.transform(X)
        
        if hasattr(self._correlation_filter, '_selected_indices') and self._correlation_filter._selected_indices is not None:
            X_out = self._correlation_filter.transform(X_out)
        
        if hasattr(self._mi_selector, '_selected_indices') and self._mi_selector._selected_indices is not None:
            X_out = self._mi_selector.transform(X_out)
        
        return X_out
    
    def fit_transform(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None
    ) -> np.ndarray:
        """Fit and transform in one step."""
        self.fit(X, y, feature_names)
        return self.transform(X)
    
    def get_selected_features(self) -> List[str]:
        """Get names of selected features."""
        return self._selected_feature_names
    
    def get_selection_stats(self) -> Dict[str, Any]:
        """Get statistics about feature selection."""
        original_count = len(self._feature_names)
        selected_count = len(self._selected_feature_names)
        
        return {
            'original_features': original_count,
            'selected_features': selected_count,
            'reduction_ratio': 1 - selected_count / original_count if original_count > 0 else 0,
            'selected_names': self._selected_feature_names,
        }


def select_features(
    X: np.ndarray,
    y: Optional[np.ndarray] = None,
    feature_names: Optional[List[str]] = None,
    max_features: int = 100,
) -> Tuple[np.ndarray, List[str]]:
    """
    Convenience function for feature selection.
    
    Returns:
        Tuple of (selected_features, selected_feature_names)
    """
    pipeline = FeatureSelectionPipeline(max_features=max_features)
    X_selected = pipeline.fit_transform(X, y, feature_names)
    return X_selected, pipeline.get_selected_features()
