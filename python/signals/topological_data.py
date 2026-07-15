"""
Lightweight Topological Data Analysis (TDA) using persistent homology.
Detects early warning signals of market crashes, liquidity evaporation, and phase transitions.
Memory-efficient implementation for real-time analysis.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from collections import defaultdict
import warnings

warnings.filterwarnings('ignore')


@dataclass
class PersistenceDiagram:
    """Represents a persistence diagram from TDA."""
    birth_times: np.ndarray
    death_times: np.ndarray
    dimensions: np.ndarray
    
    @property
    def lifetimes(self) -> np.ndarray:
        return self.death_times - self.birth_times
    
    @property
    def persistence(self) -> np.ndarray:
        """Persistence = lifetime normalized by midpoint."""
        midpoints = (self.birth_times + self.death_times) / 2
        with np.errstate(divide='ignore', invalid='ignore'):
            persistence = self.lifetimes / (midpoints + 1e-10)
        return np.nan_to_num(persistence, nan=0.0)
    
    def filter_by_dimension(self, dim: int) -> 'PersistenceDiagram':
        mask = self.dimensions == dim
        return PersistenceDiagram(
            birth_times=self.birth_times[mask],
            death_times=self.death_times[mask],
            dimensions=self.dimensions[mask],
        )
    
    def to_numpy(self) -> np.ndarray:
        """Convert to Nx3 array [birth, death, dimension]."""
        return np.column_stack([self.birth_times, self.death_times, self.dimensions])


@dataclass
class BettiCurve:
    """Betti numbers as function of filtration parameter."""
    filtration_values: np.ndarray
    betti_0: np.ndarray  # Connected components
    betti_1: np.ndarray  # Loops/cycles
    betti_2: np.ndarray  # Voids (if applicable)
    
    def total_betti(self) -> np.ndarray:
        return self.betti_0 + self.betti_1 + self.betti_2


@dataclass
class TDAFeatures:
    """Feature vector extracted from persistence diagram."""
    persistence_entropy: float
    persistence_landscape_mean: float
    persistence_landscape_max: float
    betti_number_sum: float
    longest_lifetime: float
    n_significant_features: int
    bottleneck_distance_prev: float = 0.0


class PersistentHomologyCalculator:
    """
    Computes persistent homology using union-find algorithm.
    Simplified implementation optimized for financial time series.
    """
    
    def __init__(self, max_points: int = 500):
        self.max_points = max_points
    
    def compute_vietoris_rips(
        self,
        point_cloud: np.ndarray,
        max_epsilon: float = None,
        n_steps: int = 50,
    ) -> PersistenceDiagram:
        """
        Compute Vietoris-Rips persistent homology.
        
        Args:
            point_cloud: NxM array of N points in M dimensions
            max_epsilon: Maximum filtration value
            n_steps: Number of epsilon steps
        
        Returns:
            PersistenceDiagram with births, deaths, and dimensions
        """
        if len(point_cloud) > self.max_points:
            # Subsample for efficiency
            indices = np.random.choice(len(point_cloud), self.max_points, replace=False)
            point_cloud = point_cloud[indices]
        
        n_points = len(point_cloud)
        
        # Compute pairwise distances
        distances = self._compute_pairwise_distances(point_cloud)
        
        if max_epsilon is None:
            max_epsilon = np.percentile(distances, 90)
        
        epsilons = np.linspace(0, max_epsilon, n_steps)
        
        births_0 = []
        deaths_0 = []
        births_1 = []
        deaths_1 = []
        
        # Track connected components using union-find
        parent = list(range(n_points))
        
        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]
        
        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py
                return True
            return False
        
        # Track edges for H1 computation
        edge_added = set()
        
        prev_n_components = n_points
        prev_n_edges = 0
        
        for eps in epsilons:
            # Find all pairs within epsilon
            pairs = np.argwhere(distances <= eps)
            
            # Count new connections
            n_components = n_points
            for i, j in pairs:
                if i < j and union(i, j):
                    n_components -= 1
                    
                    # Check if this creates a cycle (H1 feature)
                    if (i, j) in edge_added or (j, i) in edge_added:
                        births_1.append(eps)
                        deaths_1.append(eps + 0.01)  # Approximate death
            
            # Record H0 death when components merge
            if n_components < prev_n_components:
                deaths_0.extend([eps] * (prev_n_components - n_components))
            
            prev_n_components = n_components
            prev_n_edges = len(pairs) // 2
        
        # Fill in remaining births/deaths
        if not births_0:
            births_0 = [0.0] * n_points
        if not deaths_0:
            deaths_0 = [max_epsilon] * n_points
        
        # Ensure equal lengths
        min_len = min(len(births_0), len(deaths_0))
        births_0 = births_0[:min_len]
        deaths_0 = deaths_0[:min_len]
        
        min_len_1 = min(len(births_1), len(deaths_1))
        births_1 = births_1[:min_len_1]
        deaths_1 = deaths_1[:min_len_1]
        
        # Combine into single diagram
        all_births = np.array(births_0 + births_1)
        all_deaths = np.array(deaths_0 + deaths_1)
        all_dims = np.array([0] * len(births_0) + [1] * len(births_1))
        
        return PersistenceDiagram(
            birth_times=all_births,
            death_times=all_deaths,
            dimensions=all_dims,
        )
    
    def _compute_pairwise_distances(self, points: np.ndarray) -> np.ndarray:
        """Compute pairwise Euclidean distances."""
        n = len(points)
        distances = np.zeros((n, n))
        
        for i in range(n):
            diff = points[i+1:] - points[i]
            distances[i, i+1:] = np.sqrt(np.sum(diff ** 2, axis=1))
        
        distances += distances.T
        return distances
    
    def compute_from_time_series(
        self,
        time_series: np.ndarray,
        embedding_dim: int = 3,
        delay: int = 1,
    ) -> PersistenceDiagram:
        """
        Convert time series to point cloud using Takens embedding,
        then compute persistent homology.
        """
        n = len(time_series)
        n_embedded = n - (embedding_dim - 1) * delay
        
        if n_embedded < 10:
            # Not enough points, return empty diagram
            return PersistenceDiagram(
                birth_times=np.array([]),
                death_times=np.array([]),
                dimensions=np.array([]),
            )
        
        # Takens embedding
        point_cloud = np.zeros((n_embedded, embedding_dim))
        for i in range(embedding_dim):
            point_cloud[:, i] = time_series[i * delay:i * delay + n_embedded]
        
        # Normalize
        point_cloud = (point_cloud - point_cloud.mean(axis=0)) / (point_cloud.std(axis=0) + 1e-10)
        
        return self.compute_vietoris_rips(point_cloud)


class TDAMarketMonitor:
    """
    Monitors market state using topological features.
    Detects phase transitions and crash预警 signals.
    """
    
    def __init__(
        self,
        embedding_dim: int = 3,
        delay: int = 1,
        window_size: int = 200,
    ):
        self.embedding_dim = embedding_dim
        self.delay = delay
        self.window_size = window_size
        
        self.calculator = PersistentHomologyCalculator(max_points=window_size)
        
        # History of TDA features
        self.feature_history: List[TDAFeatures] = []
        self.diagram_history: List[PersistenceDiagram] = []
        
        # Baseline features for comparison
        self.baseline_features: Optional[TDAFeatures] = None
        
        # Price buffer
        self.price_buffer: List[float] = []
    
    def update(self, price: float, timestamp: int = 0) -> Optional[TDAFeatures]:
        """
        Update monitor with new price observation.
        Returns TDA features if enough data accumulated.
        """
        self.price_buffer.append(price)
        
        if len(self.price_buffer) < self.window_size:
            return None
        
        # Trim buffer
        if len(self.price_buffer) > self.window_size:
            self.price_buffer = self.price_buffer[-self.window_size:]
        
        # Compute returns for stationarity
        prices = np.array(self.price_buffer)
        returns = np.diff(prices) / prices[:-1]
        
        # Compute persistent homology
        diagram = self.calculator.compute_from_time_series(
            returns,
            embedding_dim=self.embedding_dim,
            delay=self.delay,
        )
        
        # Extract features
        features = self._extract_features(diagram)
        
        # Store history
        self.feature_history.append(features)
        self.diagram_history.append(diagram)
        
        # Limit history size
        if len(self.feature_history) > 100:
            self.feature_history.pop(0)
            self.diagram_history.pop(0)
        
        # Set baseline on first run
        if self.baseline_features is None:
            self.baseline_features = features
        
        return features
    
    def _extract_features(self, diagram: PersistenceDiagram) -> TDAFeatures:
        """Extract feature vector from persistence diagram."""
        # Persistence entropy
        lifetimes = diagram.lifetimes
        total_lifetime = lifetimes.sum() + 1e-10
        probs = lifetimes / total_lifetime
        with np.errstate(divide='ignore'):
            entropy = -np.sum(probs * np.log(probs + 1e-10))
        
        # Persistence landscape statistics
        persistence = diagram.persistence
        landscape_mean = np.mean(persistence)
        landscape_max = np.max(persistence)
        
        # Betti number sum (total topological complexity)
        betti_sum = len(diagram.birth_times)
        
        # Longest living feature
        longest = np.max(lifetimes) if len(lifetimes) > 0 else 0.0
        
        # Count significant features (lifetime > threshold)
        threshold = np.median(lifetimes) if len(lifetimes) > 0 else 0
        n_significant = int(np.sum(lifetimes > threshold * 2))
        
        # Bottleneck distance from baseline
        bottleneck = 0.0
        if self.diagram_history:
            bottleneck = self._bottleneck_distance(diagram, self.diagram_history[-1])
        
        return TDAFeatures(
            persistence_entropy=entropy,
            persistence_landscape_mean=landscape_mean,
            persistence_landscape_max=landscape_max,
            betti_number_sum=betti_sum,
            longest_lifetime=longest,
            n_significant_features=n_significant,
            bottleneck_distance_prev=bottleneck,
        )
    
    def _bottleneck_distance(
        self,
        diag1: PersistenceDiagram,
        diag2: PersistenceDiagram,
    ) -> float:
        """Approximate bottleneck distance between diagrams."""
        # Simplified: use Wasserstein-like approximation
        points1 = np.column_stack([diag1.birth_times, diag1.death_times])
        points2 = np.column_stack([diag2.birth_times, diag2.death_times])
        
        if len(points1) == 0 or len(points2) == 0:
            return 0.0
        
        # Match each point in diag1 to nearest in diag2
        distances = []
        for p1 in points1:
            dists = np.sqrt(np.sum((points2 - p1) ** 2, axis=1))
            distances.append(np.min(dists))
        
        return np.max(distances) if distances else 0.0
    
    def detect_phase_transition(self, z_threshold: float = 2.5) -> bool:
        """
        Detect if market is undergoing a phase transition.
        Based on sudden changes in topological features.
        """
        if len(self.feature_history) < 20:
            return False
        
        recent = self.feature_history[-5:]
        historical = self.feature_history[:-5]
        
        # Check for significant change in key features
        features_to_check = ['persistence_entropy', 'betti_number_sum', 'longest_lifetime']
        
        for feat_name in features_to_check:
            hist_values = [getattr(f, feat_name) for f in historical]
            mean_val = np.mean(hist_values)
            std_val = np.std(hist_values) + 1e-10
            
            recent_values = [getattr(f, feat_name) for f in recent]
            recent_mean = np.mean(recent_values)
            
            z_score = abs(recent_mean - mean_val) / std_val
            
            if z_score > z_threshold:
                return True
        
        return False
    
    def detect_crash_warning(self) -> Dict[str, Any]:
        """
        Detect early warning signals of potential market crash.
        Returns dict with warning indicators.
        """
        warnings = {
            'is_warning': False,
            'indicators': [],
            'severity': 0.0,
        }
        
        if len(self.feature_history) < 30:
            return warnings
        
        severity = 0.0
        
        # Indicator 1: Sudden increase in topological complexity
        recent_betti = [f.betti_number_sum for f in self.feature_history[-10:]]
        hist_betti = [f.betti_number_sum for f in self.feature_history[:-10]]
        
        if np.mean(recent_betti) > np.mean(hist_betti) + 2 * np.std(hist_betti):
            warnings['indicators'].append('increased_complexity')
            severity += 0.3
        
        # Indicator 2: High persistence entropy (instability)
        recent_entropy = [f.persistence_entropy for f in self.feature_history[-10:]]
        if np.mean(recent_entropy) > np.percentile([f.persistence_entropy for f in self.feature_history], 90):
            warnings['indicators'].append('high_entropy')
            severity += 0.3
        
        # Indicator 3: Large bottleneck distance (structural change)
        recent_bottleneck = [f.bottleneck_distance_prev for f in self.feature_history[-10:]]
        if np.mean(recent_bottleneck) > 0.5:
            warnings['indicators'].append('structural_change')
            severity += 0.4
        
        warnings['severity'] = min(severity, 1.0)
        warnings['is_warning'] = severity > 0.5
        
        return warnings
    
    def get_market_state_summary(self) -> Dict[str, Any]:
        """Get summary of current market topological state."""
        if not self.feature_history:
            return {'status': 'insufficient_data'}
        
        latest = self.feature_history[-1]
        
        return {
            'status': 'active',
            'persistence_entropy': latest.persistence_entropy,
            'topological_complexity': latest.betti_number_sum,
            'longest_feature_lifetime': latest.longest_lifetime,
            'n_significant_features': latest.n_significant_features,
            'recent_structural_change': latest.bottleneck_distance_prev,
            'phase_transition_detected': self.detect_phase_transition(),
            'crash_warning': self.detect_crash_warning(),
        }


if __name__ == "__main__":
    # Example usage
    np.random.seed(42)
    
    # Simulate market data
    n_points = 500
    base_signal = np.cumsum(np.random.randn(n_points) * 0.01) + 100
    
    # Add a "crash" in the middle
    base_signal[300:] -= 0.05 * np.arange(n_points - 300)
    
    monitor = TDAMarketMonitor(window_size=100, embedding_dim=3)
    
    print("Processing market data...")
    for i, price in enumerate(base_signal):
        features = monitor.update(price, i)
        
        if i % 50 == 0 and features is not None:
            print(f"t={i}: entropy={features.persistence_entropy:.3f}, "
                  f"complexity={features.betti_number_sum}")
    
    # Get final summary
    summary = monitor.get_market_state_summary()
    print(f"\nMarket State Summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    
    # Check for warnings
    warning = monitor.detect_crash_warning()
    if warning['is_warning']:
        print(f"\n⚠️ CRASH WARNING DETECTED!")
        print(f"  Indicators: {warning['indicators']}")
        print(f"  Severity: {warning['severity']:.2f}")
