"""
Highly compressed state-space encoder for RL agents.
Maps raw L2 order book data, queue positions, and CVD into dense tensors.
Optimized for minimal VRAM/RAM usage.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


# Compression constants
MAX_ORDERBOOK_LEVELS = 10
PRICE_BUCKET_BITS = 8
TIME_DECAY_FACTOR = 0.95


@dataclass
class CompressedState:
    """Compressed state representation."""
    orderbook_features: np.ndarray
    flow_features: np.ndarray
    position_features: np.ndarray
    temporal_features: np.ndarray
    full_vector: np.ndarray


class OrderBookCompressor:
    """Compresses L2 order book data into fixed-size representation."""
    
    def __init__(self, max_levels: int = MAX_ORDERBOOK_LEVELS):
        self.max_levels = max_levels
        
    def compress_level2(
        self,
        bids: np.ndarray,
        asks: np.ndarray,
        mid_price: float,
    ) -> np.ndarray:
        """Compress L2 order book to fixed-size vector (40,)."""
        features = np.zeros(4 * self.max_levels, dtype=np.float32)
        
        for i in range(min(len(bids), self.max_levels)):
            price_offset = (bids[i, 0] - mid_price) / mid_price
            size = bids[i, 1]
            idx = i * 4
            features[idx] = np.clip(price_offset * 100, -1, 1)
            features[idx + 1] = np.log1p(size) / 10
            features[idx + 2] = price_offset
            features[idx + 3] = size / (np.sum(bids[:, 1]) + 1e-8)
            
        for i in range(min(len(asks), self.max_levels)):
            price_offset = (asks[i, 0] - mid_price) / mid_price
            size = asks[i, 1]
            idx = (i + self.max_levels) * 4
            features[idx] = np.clip(price_offset * 100, -1, 1)
            features[idx + 1] = np.log1p(size) / 10
            features[idx + 2] = price_offset
            features[idx + 3] = size / (np.sum(asks[:, 1]) + 1e-8)
            
        return features
    
    def compute_imbalance(self, bids: np.ndarray, asks: np.ndarray) -> float:
        bid_vol = np.sum(bids[:, 1])
        ask_vol = np.sum(asks[:, 1])
        return (bid_vol - ask_vol) / (bid_vol + ask_vol + 1e-8)


class FlowFeatureExtractor:
    """Extracts compressed flow features from trade/CVD data."""
    
    def __init__(self, window_size: int = 20):
        self.window_size = window_size
        self.cvd_history = np.zeros(window_size)
        self.trade_history = np.zeros(window_size)
        self.pos = 0
        
    def update(self, cvd: float, trade_flow: float) -> None:
        self.cvd_history[self.pos] = cvd
        self.trade_history[self.pos] = trade_flow
        self.pos = (self.pos + 1) % self.window_size
        
    def extract(self) -> np.ndarray:
        features = np.zeros(8, dtype=np.float32)
        valid_cvd = self.cvd_history[:self.pos] if self.pos > 0 else self.cvd_history
        valid_trade = self.trade_history[:self.pos] if self.pos > 0 else self.trade_history
        
        if len(valid_cvd) > 0:
            features[0] = valid_cvd[-1]
            features[1] = np.mean(valid_cvd)
            features[2] = np.std(valid_cvd)
            features[3] = valid_cvd[-1] - valid_cvd[0]
            
        if len(valid_trade) > 0:
            features[4] = valid_trade[-1]
            features[5] = np.mean(valid_trade)
            features[6] = np.std(valid_trade)
            features[7] = np.sum(valid_trade)
            
        return features


class PositionFeatureExtractor:
    """Extracts compressed position-related features."""
    
    def extract(
        self,
        queue_position: float,
        fill_rate: float,
        remaining_qty_ratio: float,
        avg_fill_slippage: float,
    ) -> np.ndarray:
        return np.array([
            queue_position,
            fill_rate,
            remaining_qty_ratio,
            avg_fill_slippage,
        ], dtype=np.float32)


class TemporalFeatureExtractor:
    """Extracts compressed temporal features."""
    
    def __init__(self):
        self.activity_decay = 0.0
        self.last_update_ns = 0
        
    def update(self, current_time_ns: int) -> None:
        dt = (current_time_ns - self.last_update_ns) / 1e9 if self.last_update_ns > 0 else 0
        self.activity_decay *= TIME_DECAY_FACTOR
        self.activity_decay += 1.0
        self.last_update_ns = current_time_ns
        
    def extract(self, time_to_deadline: float) -> np.ndarray:
        return np.array([
            time_to_deadline,
            self.activity_decay,
            np.sin(2 * np.pi * time_to_deadline),
            np.cos(2 * np.pi * time_to_deadline),
        ], dtype=np.float32)


class StateSpaceEncoder:
    """
    Main encoder that combines all feature extractors.
    Produces a dense tensor suitable for RL agents.
    """
    
    def __init__(self, device: torch.device = None):
        self.device = device or (
            torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        )
        
        self.ob_compressor = OrderBookCompressor(MAX_ORDERBOOK_LEVELS)
        self.flow_extractor = FlowFeatureExtractor()
        self.position_extractor = PositionFeatureExtractor()
        self.temporal_extractor = TemporalFeatureExtractor()
        
        # Total feature dimension: 40 + 8 + 4 + 4 = 56
        self.feature_dim = 40 + 8 + 4 + 4
        
        # Neural compression layer for further reduction
        self.compression_net = nn.Sequential(
            nn.Linear(self.feature_dim, 32),
            nn.LayerNorm(32),
            nn.ReLU(),
            nn.Linear(32, 24),  # Final compressed size
            nn.LayerNorm(24),
        ).to(self.device)
        
    def encode(
        self,
        bids: np.ndarray,
        asks: np.ndarray,
        mid_price: float,
        cvd: float,
        trade_flow: float,
        queue_position: float,
        fill_rate: float,
        remaining_qty_ratio: float,
        avg_fill_slippage: float,
        time_to_deadline: float,
        current_time_ns: int,
    ) -> CompressedState:
        """Encode all inputs into compressed state."""
        self.temporal_extractor.update(current_time_ns)
        self.flow_extractor.update(cvd, trade_flow)
        
        ob_features = self.ob_compressor.compress_level2(bids, asks, mid_price)
        flow_features = self.flow_extractor.extract()
        position_features = self.position_extractor.extract(
            queue_position, fill_rate, remaining_qty_ratio, avg_fill_slippage
        )
        temporal_features = self.temporal_extractor.extract(time_to_deadline)
        
        full_vector = np.concatenate([
            ob_features,
            flow_features,
            position_features,
            temporal_features,
        ])
        
        return CompressedState(
            orderbook_features=ob_features,
            flow_features=flow_features,
            position_features=position_features,
            temporal_features=temporal_features,
            full_vector=full_vector,
        )
    
    def encode_tensor(
        self,
        state: CompressedState,
    ) -> torch.Tensor:
        """Convert compressed state to neural network tensor."""
        x = torch.FloatTensor(state.full_vector).unsqueeze(0).to(self.device)
        return self.compression_net(x)
    
    def get_memory_footprint_mb(self) -> float:
        total_params = sum(p.numel() for p in self.compression_net.parameters())
        return (total_params * 4 * 4) / (1024 ** 2)


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    encoder = StateSpaceEncoder(device=device)
    print(f"Encoder memory footprint: {encoder.get_memory_footprint_mb():.2f} MB")
    print(f"Feature dimension: {encoder.feature_dim}")
    print(f"Compressed dimension: 24")
    
    # Test encoding
    bids = np.array([[99.9, 100], [99.8, 200], [99.7, 150]])
    asks = np.array([[100.1, 120], [100.2, 180], [100.3, 160]])
    
    state = encoder.encode(
        bids=bids,
        asks=asks,
        mid_price=100.0,
        cvd=500.0,
        trade_flow=50.0,
        queue_position=0.3,
        fill_rate=0.5,
        remaining_qty_ratio=0.5,
        avg_fill_slippage=0.0005,
        time_to_deadline=0.7,
        current_time_ns=1234567890,
    )
    
    print(f"\nEncoded state shape: {state.full_vector.shape}")
    
    tensor = encoder.encode_tensor(state)
    print(f"Compressed tensor shape: {tensor.shape}")
    print(f"Compressed tensor: {tensor.cpu().detach().numpy()[0]}")
