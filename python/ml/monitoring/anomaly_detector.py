"""
Anomaly detection for live trading data streams.
Implements Isolation Forest and lightweight Autoencoder for real-time detection of:
- Market microstructure breakdowns
- Stop hunts and liquidity sweeps
- Data feed glitches and stale prices
Optimized for low-latency, memory-efficient operation.
"""

import os
import logging
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from collections import deque

try:
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler
except ImportError:
    raise ImportError("scikit-learn required. Install with: pip install scikit-learn")

logger = logging.getLogger(__name__)


class StreamingIsolationForest:
    """
    Memory-efficient streaming Isolation Forest for anomaly detection.
    Uses a fixed-size buffer and incremental model updates.
    """
    
    def __init__(
        self,
        contamination: float = 0.01,
        n_estimators: int = 50,
        max_samples: int = 256,
        buffer_size: int = 10000,
        retrain_interval: int = 1000,
        random_state: int = 42,
    ):
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.buffer_size = buffer_size
        self.retrain_interval = retrain_interval
        self.random_state = random_state
        
        # Data buffer (circular)
        self.data_buffer: deque = deque(maxlen=buffer_size)
        
        # Model
        self.model: Optional[IsolationForest] = None
        self.scaler: Optional[StandardScaler] = None
        
        # State
        self.samples_seen = 0
        self.last_retrain = 0
        self.is_fitted = False
    
    def partial_fit(self, X: np.ndarray):
        """Add samples to buffer and retrain if needed."""
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
        
        for sample in X:
            self.data_buffer.append(sample.copy())
            self.samples_seen += 1
        
        # Retrain periodically
        if (self.samples_seen - self.last_retrain) >= self.retrain_interval:
            self._retrain()
    
    def _retrain(self):
        """Retrain model on buffered data."""
        if len(self.data_buffer) < self.max_samples * 2:
            return
        
        data = np.array(list(self.data_buffer))
        
        # Fit scaler
        self.scaler = StandardScaler()
        data_scaled = self.scaler.fit_transform(data)
        
        # Fit Isolation Forest
        self.model = IsolationForest(
            n_estimators=self.n_estimators,
            max_samples=min(self.max_samples, len(data)),
            contamination=self.contamination,
            random_state=self.random_state,
            n_jobs=1,  # Single thread for memory efficiency
        )
        
        self.model.fit(data_scaled)
        self.is_fitted = True
        self.last_retrain = self.samples_seen
        
        logger.debug(f"Isolation Forest retrained on {len(data)} samples")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict anomaly labels (-1 for anomaly, 1 for normal)."""
        if not self.is_fitted:
            # Not trained yet: assume all normal
            return np.ones(len(X) if len(X.shape) > 1 else 1, dtype=int)
        
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
        
        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled)
        
        return predictions
    
    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """Get anomaly scores (more negative = more anomalous)."""
        if not self.is_fitted:
            return np.zeros(len(X) if len(X.shape) > 1 else 1)
        
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
        
        X_scaled = self.scaler.transform(X)
        scores = self.model.score_samples(X_scaled)
        
        return scores


class LightweightAutoencoder:
    """
    Minimal autoencoder for anomaly detection using pure NumPy.
    No deep learning framework dependencies for reduced memory footprint.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 32,
        latent_dim: int = 8,
        learning_rate: float = 0.001,
        buffer_size: int = 5000,
    ):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.learning_rate = learning_rate
        self.buffer_size = buffer_size
        
        # Initialize weights (Xavier initialization)
        self._initialize_weights()
        
        # Data buffer for online learning
        self.data_buffer: deque = deque(maxlen=buffer_size)
        
        # Running statistics for normalization
        self.running_mean = np.zeros(input_dim)
        self.running_var = np.ones(input_dim)
        self.n_samples = 0
        
        # Training state
        self.is_trained = False
    
    def _initialize_weights(self):
        """Initialize network weights."""
        # Encoder
        self.W1 = np.random.randn(self.input_dim, self.hidden_dim) * np.sqrt(2.0 / self.input_dim)
        self.b1 = np.zeros(self.hidden_dim)
        
        self.W2 = np.random.randn(self.hidden_dim, self.latent_dim) * np.sqrt(2.0 / self.hidden_dim)
        self.b2 = np.zeros(self.latent_dim)
        
        # Decoder
        self.W3 = np.random.randn(self.latent_dim, self.hidden_dim) * np.sqrt(2.0 / self.latent_dim)
        self.b3 = np.zeros(self.hidden_dim)
        
        self.W4 = np.random.randn(self.hidden_dim, self.input_dim) * np.sqrt(2.0 / self.hidden_dim)
        self.b4 = np.zeros(self.input_dim)
    
    def _relu(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)
    
    def _relu_derivative(self, x: np.ndarray) -> np.ndarray:
        return (x > 0).astype(float)
    
    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Forward pass through autoencoder."""
        # Encoder
        z1 = x @ self.W1 + self.b1
        a1 = self._relu(z1)
        
        z2 = a1 @ self.W2 + self.b2
        a2 = self._sigmoid(z2)  # Latent representation
        
        # Decoder
        z3 = a2 @ self.W3 + self.b3
        a3 = self._relu(z3)
        
        z4 = a3 @ self.W4 + self.b4
        output = self._sigmoid(z4)  # Reconstructed input
        
        return output, (z1, a1, z2, a2, z3, a3, z4)
    
    def compute_loss(self, x: np.ndarray, x_reconstructed: np.ndarray) -> float:
        """Compute reconstruction loss (MSE)."""
        return np.mean((x - x_reconstructed) ** 2)
    
    def partial_fit(self, X: np.ndarray, n_epochs: int = 1):
        """Online training step."""
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
        
        # Update running statistics
        batch_size = len(X)
        total_samples = self.n_samples + batch_size
        
        delta = X - self.running_mean
        self.running_mean += delta * batch_size / total_samples
        self.running_var = (
            self.n_samples / total_samples * self.running_var +
            np.var(X, axis=0) * batch_size / total_samples +
            batch_size / total_samples * (delta ** 2) * self.n_samples / total_samples
        )
        self.n_samples = total_samples
        
        # Add to buffer
        for sample in X:
            self.data_buffer.append(sample.copy())
        
        # Normalize
        epsilon = 1e-8
        X_norm = (X - self.running_mean) / np.sqrt(self.running_var + epsilon)
        
        # Train on buffer periodically
        if len(self.data_buffer) >= min(100, self.buffer_size // 10):
            buffer_data = np.array(list(self.data_buffer))
            buffer_norm = (buffer_data - self.running_mean) / np.sqrt(self.running_var + epsilon)
            
            # Mini-batch training
            for epoch in range(n_epochs):
                indices = np.random.choice(len(buffer_norm), min(32, len(buffer_norm)), replace=False)
                batch = buffer_norm[indices]
                
                # Forward pass
                output, cache = self.forward(batch)
                
                # Backward pass (simplified gradient descent)
                z1, a1, z2, a2, z3, a3, z4 = cache
                
                # Output layer gradient
                d_output = 2 * (output - batch) / len(batch) * self._sigmoid(z4) * (1 - self._sigmoid(z4))
                
                # Layer 3
                d_W4 = a3.T @ d_output
                d_b4 = np.sum(d_output, axis=0)
                d_a3 = d_output @ self.W4.T
                
                d_z3 = d_a3 * self._relu_derivative(z3)
                d_W3 = a2.T @ d_z3
                d_b3 = np.sum(d_z3, axis=0)
                d_a2 = d_z3 @ self.W3.T
                
                # Layer 2 (latent)
                d_z2 = d_a2 * self._sigmoid(z2) * (1 - self._sigmoid(z2))
                d_W2 = a1.T @ d_z2
                d_b2 = np.sum(d_z2, axis=0)
                d_a1 = d_z2 @ self.W2.T
                
                # Layer 1
                d_z1 = d_a1 * self._relu_derivative(z1)
                d_W1 = batch.T @ d_z1
                d_b1 = np.sum(d_z1, axis=0)
                
                # Update weights
                self.W4 -= self.learning_rate * d_W4
                self.b4 -= self.learning_rate * d_b4
                self.W3 -= self.learning_rate * d_W3
                self.b3 -= self.learning_rate * d_b3
                self.W2 -= self.learning_rate * d_W2
                self.b2 -= self.learning_rate * d_b2
                self.W1 -= self.learning_rate * d_W1
                self.b1 -= self.learning_rate * d_b1
            
            self.is_trained = True
    
    def reconstruct(self, X: np.ndarray) -> np.ndarray:
        """Reconstruct input through autoencoder."""
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
        
        # Normalize
        epsilon = 1e-8
        X_norm = (X - self.running_mean) / np.sqrt(self.running_var + epsilon)
        
        output, _ = self.forward(X_norm)
        return output
    
    def anomaly_score(self, X: np.ndarray) -> np.ndarray:
        """Get reconstruction error as anomaly score."""
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
        
        X_reconstructed = self.reconstruct(X)
        
        # Per-sample MSE
        mse = np.mean((X - X_reconstructed) ** 2, axis=1)
        
        return mse


class AnomalyDetector:
    """
    Combined anomaly detector using both Isolation Forest and Autoencoder.
    Provides ensemble anomaly scoring for robust detection.
    """
    
    def __init__(
        self,
        input_dim: int,
        contamination: float = 0.01,
        use_isolation_forest: bool = True,
        use_autoencoder: bool = True,
        if_weight: float = 0.5,
        ae_weight: float = 0.5,
        anomaly_threshold: float = 0.7,  # Score above this = anomaly
    ):
        self.input_dim = input_dim
        self.contamination = contamination
        self.use_isolation_forest = use_isolation_forest
        self.use_autoencoder = use_autoencoder
        self.if_weight = if_weight
        self.ae_weight = ae_weight
        self.anomaly_threshold = anomaly_threshold
        
        # Initialize detectors
        self.if_detector = StreamingIsolationForest(
            contamination=contamination,
        ) if use_isolation_forest else None
        
        self.ae_detector = LightweightAutoencoder(
            input_dim=input_dim,
        ) if use_autoencoder else None
        
        # Anomaly history
        self.anomaly_history: deque = deque(maxlen=1000)
        self.alert_cooldown = 10  # Minimum steps between alerts
    
    def partial_fit(self, X: np.ndarray):
        """Update detectors with new data."""
        if self.if_detector:
            self.if_detector.partial_fit(X)
        
        if self.ae_detector:
            self.ae_detector.partial_fit(X)
    
    def detect(self, X: np.ndarray) -> Dict[str, Any]:
        """
        Detect anomalies in input data.
        
        Returns:
            Dictionary with anomaly scores and flags
        """
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
        
        scores = {}
        is_anomaly = False
        
        # Isolation Forest score
        if self.if_detector and self.if_detector.is_fitted:
            if_scores = -self.if_detector.score_samples(X)  # Negate so higher = more anomalous
            if_scores_normalized = (if_scores - if_scores.min()) / (if_scores.max() - if_scores.min() + 1e-8)
            scores['isolation_forest'] = if_scores_normalized
            
            if_labels = self.if_detector.predict(X)
            scores['if_anomaly_flag'] = (if_labels == -1).astype(int)
        
        # Autoencoder score
        if self.ae_detector and self.ae_detector.is_trained:
            ae_scores = self.ae_detector.anomaly_score(X)
            ae_scores_normalized = ae_scores / (np.max(ae_scores) + 1e-8)
            scores['autoencoder'] = ae_scores_normalized
        
        # Ensemble score
        if len(scores) > 0:
            ensemble_score = 0.0
            weight_sum = 0.0
            
            if 'isolation_forest' in scores and self.use_isolation_forest:
                ensemble_score += self.if_weight * scores['isolation_forest']
                weight_sum += self.if_weight
            
            if 'autoencoder' in scores and self.use_autoencoder:
                ensemble_score += self.ae_weight * scores['autoencoder']
                weight_sum += self.ae_weight
            
            scores['ensemble'] = ensemble_score / weight_sum if weight_sum > 0 else 0.0
            is_anomaly = scores['ensemble'] > self.anomaly_threshold
        
        # Store in history
        self.anomaly_history.append({
            'score': scores.get('ensemble', 0.0),
            'is_anomaly': is_anomaly,
        })
        
        return {
            'scores': scores,
            'is_anomaly': is_anomaly,
            'anomaly_type': self._classify_anomaly(X, scores),
        }
    
    def _classify_anomaly(self, X: np.ndarray, scores: Dict) -> str:
        """Classify the type of anomaly detected."""
        if not scores:
            return "unknown"
        
        ensemble_score = scores.get('ensemble', 0.0)
        
        if ensemble_score < self.anomaly_threshold:
            return "normal"
        
        # Simple classification based on score patterns
        if_score = scores.get('isolation_forest', 0.0)
        ae_score = scores.get('autoencoder', 0.0)
        
        if if_score > ae_score * 1.5:
            return "structural_anomaly"  # Detected mainly by IF
        elif ae_score > if_score * 1.5:
            return "reconstruction_anomaly"  # Detected mainly by AE
        else:
            return "combined_anomaly"  # Both detectors agree


def main():
    """Test anomaly detection."""
    import psutil
    
    print(f"Initial RAM: {psutil.virtual_memory().available / (1024**3):.2f}GB")
    
    # Generate synthetic market data
    np.random.seed(42)
    n_normal = 1000
    n_anomalies = 20
    
    # Normal data
    X_normal = np.random.randn(n_normal, 10) * 0.5 + np.array([1.0, 0.5, 0.3, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.001])
    
    # Anomalies (extreme values)
    X_anomalies = np.random.randn(n_anomalies, 10) * 3.0 + np.array([5.0, -3.0, 10.0, -5.0, 8.0, -4.0, 6.0, -2.0, 4.0, -1.0])
    
    X = np.vstack([X_normal, X_anomalies])
    true_labels = np.array([0] * n_normal + [1] * n_anomalies)
    
    print(f"\nDataset: {len(X)} samples ({n_anomalies} anomalies)")
    
    # Create detector
    detector = AnomalyDetector(
        input_dim=10,
        contamination=n_anomalies / len(X),
        anomaly_threshold=0.6,
    )
    
    # Train incrementally
    print("\nTraining detector...")
    for i in range(0, len(X_normal), 100):
        batch = X_normal[i:i+100]
        detector.partial_fit(batch)
    
    # Detect
    print("\nDetecting anomalies...")
    detected = []
    for i in range(len(X)):
        result = detector.detect(X[i:i+1])
        detected.append(1 if result['is_anomaly'] else 0)
    
    detected = np.array(detected)
    
    # Calculate metrics
    tp = np.sum((detected == 1) & (true_labels == 1))
    fp = np.sum((detected == 1) & (true_labels == 0))
    fn = np.sum((detected == 0) & (true_labels == 1))
    tn = np.sum((detected == 0) & (true_labels == 0))
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"\nDetection Results:")
    print(f"  True Positives: {tp}")
    print(f"  False Positives: {fp}")
    print(f"  Precision: {precision:.2%}")
    print(f"  Recall: {recall:.2%}")
    print(f"  F1 Score: {f1:.2%}")
    
    print(f"\nFinal RAM: {psutil.virtual_memory().available / (1024**3):.2f}GB")


if __name__ == "__main__":
    main()
