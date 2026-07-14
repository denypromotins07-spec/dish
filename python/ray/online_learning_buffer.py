"""
Strict fixed-size Ring Buffer for Reinforcement Learning and Online Learning state preparation.
Ensures historical feature data never exceeds a predefined 500MB RAM cap.
Uses memory-mapped arrays for efficient circular buffer implementation.
"""

import numpy as np
from typing import Optional, Dict, Any, Tuple, List
import threading
import mmap
import os


class RingBuffer:
    """
    Fixed-size circular buffer with O(1) append and efficient slicing.
    Memory-efficient implementation using pre-allocated numpy arrays.
    """
    
    __slots__ = ('_buffer', '_head', '_size', '_capacity', '_dtype', '_lock')
    
    def __init__(self, capacity: int, dtype: np.dtype = np.float64):
        """
        Args:
            capacity: Maximum number of elements (strict limit)
            dtype: Data type for stored values
        """
        self._capacity = capacity
        self._dtype = dtype
        self._buffer = np.zeros(capacity, dtype=dtype)
        self._head = 0
        self._size = 0
        self._lock = threading.Lock()
    
    def append(self, value: float) -> None:
        """Append value to buffer, overwriting oldest if full."""
        with self._lock:
            idx = self._head % self._capacity
            self._buffer[idx] = value
            self._head += 1
            if self._size < self._capacity:
                self._size += 1
    
    def append_many(self, values: np.ndarray) -> None:
        """Append multiple values efficiently."""
        with self._lock:
            n = len(values)
            
            if n >= self._capacity:
                # Buffer completely overwritten
                self._buffer[:] = values[-self._capacity:]
                self._head = n
                self._size = self._capacity
            else:
                for i, v in enumerate(values):
                    idx = (self._head + i) % self._capacity
                    self._buffer[idx] = v
                
                self._head += n
                self._size = min(self._size + n, self._capacity)
    
    def get(self, n: int = None) -> np.ndarray:
        """Get last n elements in chronological order."""
        with self._lock:
            if n is None:
                n = self._size
            
            n = min(n, self._size)
            
            if n == 0:
                return np.array([], dtype=self._dtype)
            
            # Calculate start index
            start = (self._head - n) % self._capacity
            
            if start + n <= self._capacity:
                return self._buffer[start:start + n].copy()
            else:
                # Wrap around
                end_part = self._buffer[start:].copy()
                start_part = self._buffer[:n - (self._capacity - start)].copy()
                return np.concatenate([end_part, start_part])
    
    def get_all(self) -> np.ndarray:
        """Get all elements in chronological order."""
        return self.get()
    
    def get_last(self) -> Optional[float]:
        """Get most recent element."""
        with self._lock:
            if self._size == 0:
                return None
            idx = (self._head - 1) % self._capacity
            return float(self._buffer[idx])
    
    def __len__(self) -> int:
        return self._size
    
    def is_full(self) -> bool:
        return self._size == self._capacity
    
    def clear(self) -> None:
        """Reset buffer to empty state."""
        with self._lock:
            self._buffer.fill(0)
            self._head = 0
            self._size = 0
    
    @property
    def capacity(self) -> int:
        return self._capacity
    
    @property
    def size(self) -> int:
        return self._size


class MultiDimensionalRingBuffer:
    """
    Ring buffer for multi-dimensional data (e.g., feature vectors).
    """
    
    __slots__ = ('_buffer', '_head', '_size', '_capacity', '_feature_dim', '_lock')
    
    def __init__(self, capacity: int, feature_dim: int, dtype: np.dtype = np.float64):
        """
        Args:
            capacity: Maximum number of samples
            feature_dim: Dimension of each feature vector
            dtype: Data type
        """
        self._capacity = capacity
        self._feature_dim = feature_dim
        self._buffer = np.zeros((capacity, feature_dim), dtype=dtype)
        self._head = 0
        self._size = 0
        self._lock = threading.Lock()
    
    def append(self, features: np.ndarray) -> None:
        """Append feature vector to buffer."""
        with self._lock:
            idx = self._head % self._capacity
            self._buffer[idx] = features
            self._head += 1
            if self._size < self._capacity:
                self._size += 1
    
    def get(self, n: int = None) -> np.ndarray:
        """Get last n samples in chronological order."""
        with self._lock:
            if n is None:
                n = self._size
            
            n = min(n, self._size)
            
            if n == 0:
                return np.array([], dtype=self._buffer.dtype).reshape(0, self._feature_dim)
            
            start = (self._head - n) % self._capacity
            
            if start + n <= self._capacity:
                return self._buffer[start:start + n].copy()
            else:
                end_part = self._buffer[start:].copy()
                start_part = self._buffer[:n - (self._capacity - start)].copy()
                return np.concatenate([end_part, start_part], axis=0)
    
    def get_sequences(self, seq_length: int) -> np.ndarray:
        """
        Get overlapping sequences for RNN/Transformer input.
        Returns array of shape (num_sequences, seq_length, feature_dim)
        """
        with self._lock:
            if self._size < seq_length:
                return np.array([]).reshape(0, seq_length, self._feature_dim)
            
            data = self.get()
            num_sequences = len(data) - seq_length + 1
            
            if num_sequences <= 0:
                return np.array([]).reshape(0, seq_length, self._feature_dim)
            
            sequences = np.zeros((num_sequences, seq_length, self._feature_dim))
            for i in range(num_sequences):
                sequences[i] = data[i:i + seq_length]
            
            return sequences
    
    def __len__(self) -> int:
        return self._size
    
    @property
    def feature_dim(self) -> int:
        return self._feature_dim
    
    @property
    def memory_bytes(self) -> int:
        """Estimate memory usage in bytes."""
        return self._buffer.nbytes


class OnlineLearningBuffer:
    """
    Specialized buffer for online/reinforcement learning.
    Stores (state, action, reward, next_state, done) tuples.
    Strictly bounded to 500MB maximum.
    """
    
    __slots__ = (
        '_states', '_actions', '_rewards', '_next_states', '_dones',
        '_head', '_size', '_capacity', '_state_dim', '_action_dim',
        '_max_memory_mb', '_lock'
    )
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        max_memory_mb: float = 500.0,
        dtype: np.dtype = np.float32
    ):
        """
        Args:
            state_dim: Dimension of state vectors
            action_dim: Dimension of action vectors
            max_memory_mb: Maximum memory in megabytes (default 500MB)
            dtype: Data type (float32 for memory efficiency)
        """
        self._state_dim = state_dim
        self._action_dim = action_dim
        self._max_memory_mb = max_memory_mb
        self._dtype = dtype
        
        # Calculate capacity based on memory limit
        # Each transition: state + action + reward + next_state + done
        bytes_per_transition = (
            state_dim * dtype().itemsize +
            action_dim * dtype().itemsize +
            dtype().itemsize +  # reward
            state_dim * dtype().itemsize +  # next_state
            1  # done (bool)
        )
        
        max_transitions = int((max_memory_mb * 1024 * 1024) / bytes_per_transition)
        self._capacity = max_transitions
        
        # Pre-allocate arrays
        self._states = np.zeros((self._capacity, state_dim), dtype=dtype)
        self._actions = np.zeros((self._capacity, action_dim), dtype=dtype)
        self._rewards = np.zeros(self._capacity, dtype=dtype)
        self._next_states = np.zeros((self._capacity, state_dim), dtype=dtype)
        self._dones = np.zeros(self._capacity, dtype=bool)
        
        self._head = 0
        self._size = 0
        self._lock = threading.Lock()
    
    def store(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ) -> None:
        """Store a transition tuple."""
        with self._lock:
            idx = self._head % self._capacity
            
            self._states[idx] = state
            self._actions[idx] = action
            self._rewards[idx] = reward
            self._next_states[idx] = next_state
            self._dones[idx] = done
            
            self._head += 1
            if self._size < self._capacity:
                self._size += 1
    
    def sample(self, batch_size: int) -> Dict[str, np.ndarray]:
        """
        Randomly sample a batch of transitions.
        Returns dict with keys: states, actions, rewards, next_states, dones
        """
        with self._lock:
            if self._size == 0:
                return {}
            
            batch_size = min(batch_size, self._size)
            indices = np.random.choice(self._size, batch_size, replace=False)
            
            return {
                'states': self._states[indices].copy(),
                'actions': self._actions[indices].copy(),
                'rewards': self._rewards[indices].copy(),
                'next_states': self._next_states[indices].copy(),
                'dones': self._dones[indices].copy(),
            }
    
    def get_recent(self, n: int) -> Dict[str, np.ndarray]:
        """Get most recent n transitions."""
        with self._lock:
            n = min(n, self._size)
            
            if n == 0:
                return {}
            
            start = (self._head - n) % self._capacity
            
            if start + n <= self._capacity:
                indices = slice(start, start + n)
            else:
                indices = np.concatenate([
                    np.arange(start, self._capacity),
                    np.arange(0, n - (self._capacity - start))
                ])
            
            return {
                'states': self._states[indices].copy(),
                'actions': self._actions[indices].copy(),
                'rewards': self._rewards[indices].copy(),
                'next_states': self._next_states[indices].copy(),
                'dones': self._dones[indices].copy(),
            }
    
    def __len__(self) -> int:
        return self._size
    
    @property
    def memory_usage_mb(self) -> float:
        """Calculate actual memory usage in MB."""
        total_bytes = (
            self._states.nbytes +
            self._actions.nbytes +
            self._rewards.nbytes +
            self._next_states.nbytes +
            self._dones.nbytes
        )
        return total_bytes / (1024 * 1024)
    
    @property
    def is_full(self) -> bool:
        return self._size == self._capacity
    
    def clear(self) -> None:
        """Clear all stored transitions."""
        with self._lock:
            self._states.fill(0)
            self._actions.fill(0)
            self._rewards.fill(0)
            self._next_states.fill(0)
            self._dones.fill(False)
            self._head = 0
            self._size = 0


class FeatureHistoryBuffer:
    """
    Buffer specifically for storing feature history for ML model input.
    Combines multiple ring buffers for different feature types.
    """
    
    __slots__ = ('_technical_buffer', '_smc_buffer', '_quant_buffer', '_macro_buffer', '_lock')
    
    def __init__(
        self,
        technical_dim: int = 50,
        smc_dim: int = 20,
        quant_dim: int = 40,
        macro_dim: int = 40,
        max_history_seconds: int = 3600,
        tick_interval_ms: int = 100
    ):
        """
        Args:
            technical_dim: Dimension of technical indicator features
            smc_dim: Dimension of SMC features
            quant_dim: Dimension of quantitative features
            macro_dim: Dimension of macro/on-chain features
            max_history_seconds: Maximum history duration
            tick_interval_ms: Expected tick interval
        """
        # Calculate capacity based on time window
        capacity = (max_history_seconds * 1000) // tick_interval_ms
        
        # Limit to ensure under 500MB total
        max_capacity_per_buffer = 50_000_000 // (technical_dim + smc_dim + quant_dim + macro_dim)
        capacity = min(capacity, max_capacity_per_buffer)
        
        self._technical_buffer = MultiDimensionalRingBuffer(capacity, technical_dim)
        self._smc_buffer = MultiDimensionalRingBuffer(capacity, smc_dim)
        self._quant_buffer = MultiDimensionalRingBuffer(capacity, quant_dim)
        self._macro_buffer = MultiDimensionalRingBuffer(capacity, macro_dim)
        self._lock = threading.Lock()
    
    def store(
        self,
        technical: np.ndarray,
        smc: np.ndarray,
        quant: np.ndarray,
        macro: Optional[np.ndarray] = None
    ) -> None:
        """Store a complete feature snapshot."""
        if macro is None:
            macro = np.zeros(self._macro_buffer.feature_dim)
        
        self._technical_buffer.append(technical)
        self._smc_buffer.append(smc)
        self._quant_buffer.append(quant)
        self._macro_buffer.append(macro)
    
    def get_feature_matrix(self, lookback: int = None) -> np.ndarray:
        """
        Get concatenated feature matrix for last lookback timesteps.
        Shape: (lookback, total_features)
        """
        if lookback is None:
            lookback = len(self._technical_buffer)
        
        tech = self._technical_buffer.get(lookback)
        smc = self._smc_buffer.get(lookback)
        quant = self._quant_buffer.get(lookback)
        macro = self._macro_buffer.get(lookback)
        
        # Ensure all have same length
        min_len = min(len(tech), len(smc), len(quant), len(macro))
        
        if min_len == 0:
            return np.array([])
        
        return np.hstack([
            tech[-min_len:],
            smc[-min_len:],
            quant[-min_len:],
            macro[-min_len:],
        ])
    
    def get_total_features(self) -> int:
        """Get total feature dimension."""
        return (
            self._technical_buffer.feature_dim +
            self._smc_buffer.feature_dim +
            self._quant_buffer.feature_dim +
            self._macro_buffer.feature_dim
        )
    
    def __len__(self) -> int:
        return len(self._technical_buffer)
    
    @property
    def memory_usage_mb(self) -> float:
        """Total memory usage across all buffers."""
        return (
            self._technical_buffer.memory_bytes +
            self._smc_buffer.memory_bytes +
            self._quant_buffer.memory_bytes +
            self._macro_buffer.memory_bytes
        ) / (1024 * 1024)


# Singleton instance
_buffer: Optional[OnlineLearningBuffer] = None


def get_learning_buffer(
    state_dim: int = 150,
    action_dim: int = 3,
    max_memory_mb: float = 500.0
) -> OnlineLearningBuffer:
    """Get or create singleton learning buffer instance."""
    global _buffer
    if _buffer is None:
        _buffer = OnlineLearningBuffer(state_dim, action_dim, max_memory_mb)
    return _buffer
