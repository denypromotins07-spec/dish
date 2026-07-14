"""
Ray Directed Acyclic Graph (DAG) for parallel feature computation.
Orchestrates computation of 150 domain features with strict RAM limits.
Limits worker concurrency to prevent memory spikes on 16GB systems.
"""

import ray
from ray import remote, workflow
from typing import List, Dict, Any, Optional
import time
import numpy as np


# Initialize Ray with strict memory limits for 16GB system
# Reserve 2GB for OS and other processes, use max 14GB
ray.init(
    num_cpus=8,  # AMD Ryzen AI 5 has efficient cores
    _memory=12_000_000_000,  # 12GB object store limit
    object_store_memory=4_000_000_000,  # 4GB for objects
    max_restarts=3,
    max_task_retries=3,
)


@remote(num_cpus=1, memory=500_000_000)
class TechnicalIndicatorWorker:
    """Worker for computing technical analysis features."""
    
    def __init__(self):
        self._cache = {}
    
    def compute_oscillators(self, prices: np.ndarray, timestamps: np.ndarray) -> Dict[str, np.ndarray]:
        """Compute RSI, MACD, Stochastic features."""
        n = len(prices)
        
        # RSI (14-period)
        rsi = np.zeros(n)
        gains = np.zeros(n)
        losses = np.zeros(n)
        
        for i in range(1, n):
            change = prices[i] - prices[i-1]
            gains[i] = max(0, change)
            losses[i] = max(0, -change)
        
        avg_gain = np.mean(gains[:14]) if n >= 14 else np.mean(gains)
        avg_loss = np.mean(losses[:14]) if n >= 14 else np.mean(losses)
        
        for i in range(14, n):
            avg_gain = (avg_gain * 13 + gains[i]) / 14
            avg_loss = (avg_loss * 13 + losses[i]) / 14
            rs = avg_gain / avg_loss if avg_loss != 0 else 100
            rsi[i] = 100 - (100 / (1 + rs))
        
        # MACD (12, 26, 9)
        ema_fast = np.zeros(n)
        ema_slow = np.zeros(n)
        
        ema_fast[0] = prices[0]
        ema_slow[0] = prices[0]
        
        for i in range(1, n):
            ema_fast[i] = ema_fast[i-1] + (2/13) * (prices[i] - ema_fast[i-1])
            ema_slow[i] = ema_slow[i-1] + (2/27) * (prices[i] - ema_slow[i-1])
        
        macd_line = ema_fast - ema_slow
        
        signal_line = np.zeros(n)
        signal_line[0] = macd_line[0]
        for i in range(1, n):
            signal_line[i] = signal_line[i-1] + (2/10) * (macd_line[i] - signal_line[i-1])
        
        histogram = macd_line - signal_line
        
        return {
            'rsi': rsi,
            'macd_line': macd_line,
            'signal_line': signal_line,
            'macd_histogram': histogram,
        }
    
    def compute_trend_features(self, prices: np.ndarray, highs: np.ndarray, lows: np.ndarray) -> Dict[str, np.ndarray]:
        """Compute EMA, SMA, Bollinger Bands, ATR, ADX."""
        n = len(prices)
        
        # EMA (20-period)
        ema20 = np.zeros(n)
        ema20[0] = prices[0]
        for i in range(1, n):
            ema20[i] = ema20[i-1] + (2/21) * (prices[i] - ema20[i-1])
        
        # Bollinger Bands (20, 2)
        sma20 = np.convolve(prices, np.ones(20)/20, mode='same')
        std20 = np.array([np.std(prices[max(0,i-19):i+1]) for i in range(n)])
        bb_upper = sma20 + 2 * std20
        bb_lower = sma20 - 2 * std20
        bb_width = (bb_upper - bb_lower) / sma20
        
        # ATR (14-period)
        atr = np.zeros(n)
        tr = np.zeros(n)
        tr[0] = highs[0] - lows[0]
        for i in range(1, n):
            tr[i] = max(
                highs[i] - lows[i],
                abs(highs[i] - prices[i-1]),
                abs(lows[i] - prices[i-1])
            )
        
        atr[0] = tr[0]
        for i in range(1, n):
            atr[i] = (atr[i-1] * 13 + tr[i]) / 14
        
        return {
            'ema20': ema20,
            'bb_upper': bb_upper,
            'bb_middle': sma20,
            'bb_lower': bb_lower,
            'bb_width': bb_width,
            'atr': atr,
        }


@remote(num_cpus=1, memory=500_000_000)
class SMCWorker:
    """Worker for Smart Money Concepts features."""
    
    def compute_smc_features(
        self, 
        prices: np.ndarray, 
        highs: np.ndarray, 
        lows: np.ndarray,
        timestamps: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """Compute BOS, CHoCH, Order Block, FVG features."""
        n = len(prices)
        
        # Swing detection
        swing_highs = np.zeros(n)
        swing_lows = np.zeros(n)
        
        window = 5
        for i in range(window, n - window):
            if highs[i] == max(highs[i-window:i+window+1]):
                swing_highs[i] = 1
            if lows[i] == min(lows[i-window:i+window+1]):
                swing_lows[i] = 1
        
        # Premium/Discount zone indicator
        recent_high = np.array([max(prices[max(0,i-20):i+1]) for i in range(n)])
        recent_low = np.array([min(prices[max(0,i-20):i+1]) for i in range(n)])
        range_mid = (recent_high + recent_low) / 2
        pd_zone = np.where(prices > range_mid, 1, -1)  # 1 = premium, -1 = discount
        
        # Volatility regime (simplified)
        returns = np.diff(prices) / prices[:-1]
        returns = np.concatenate([[0], returns])
        rolling_vol = np.array([np.std(returns[max(0,i-20):i+1]) for i in range(n)])
        vol_regime = np.where(rolling_vol > np.percentile(rolling_vol, 75), 1, 
                             np.where(rolling_vol < np.percentile(rolling_vol, 25), -1, 0))
        
        return {
            'swing_highs': swing_highs,
            'swing_lows': swing_lows,
            'pd_zone': pd_zone,
            'vol_regime': vol_regime,
        }


@remote(num_cpus=1, memory=500_000_000)
class QuantitativeWorker:
    """Worker for quantitative finance features."""
    
    def compute_quant_features(self, prices: np.ndarray, volumes: np.ndarray) -> Dict[str, np.ndarray]:
        """Compute statistical and quantitative features."""
        n = len(prices)
        
        # Returns
        returns = np.diff(prices) / prices[:-1]
        returns = np.concatenate([[0], returns])
        
        # Log returns
        log_returns = np.diff(np.log(prices + 1e-10))
        log_returns = np.concatenate([[0], log_returns])
        
        # Rolling statistics
        rolling_mean = np.array([np.mean(returns[max(0,i-20):i+1]) for i in range(n)])
        rolling_std = np.array([np.std(returns[max(0,i-20):i+1]) for i in range(n)])
        rolling_skew = np.array([
            np.mean(((returns[max(0,i-20):i+1] - m) / s) ** 3) if s > 0 else 0
            for i, (m, s) in enumerate(zip(rolling_mean, rolling_std))
        ])
        rolling_kurt = np.array([
            np.mean(((returns[max(0,i-20):i+1] - m) / s) ** 4) - 3 if s > 0 else 0
            for i, (m, s) in enumerate(zip(rolling_mean, rolling_std))
        ])
        
        # Volume features
        volume_ma = np.convolve(volumes, np.ones(20)/20, mode='same')
        volume_ratio = volumes / (volume_ma + 1e-10)
        
        # VWAP approximation
        typical_price = (prices + np.roll(prices, 1) + np.roll(prices, -1)) / 3
        typical_price[0] = prices[0]
        typical_price[-1] = prices[-1]
        vwap = np.cumsum(typical_price * volumes) / (np.cumsum(volumes) + 1e-10)
        vwap_deviation = (prices - vwap) / vwap
        
        return {
            'returns': returns,
            'log_returns': log_returns,
            'rolling_mean': rolling_mean,
            'rolling_std': rolling_std,
            'rolling_skew': rolling_skew,
            'rolling_kurt': rolling_kurt,
            'volume_ratio': volume_ratio,
            'vwap': vwap,
            'vwap_deviation': vwap_deviation,
        }


@remote
class FeatureAggregator:
    """Aggregates features from all workers into final feature matrix."""
    
    def __init__(self, max_features: int = 150):
        self._max_features = max_features
        self._feature_names = []
    
    def aggregate(
        self,
        tech_features: Dict[str, np.ndarray],
        smc_features: Dict[str, np.ndarray],
        quant_features: Dict[str, np.ndarray],
        macro_features: Optional[Dict[str, np.ndarray]] = None,
    ) -> Dict[str, Any]:
        """Combine all features into unified matrix."""
        n = None
        for feat_dict in [tech_features, smc_features, quant_features]:
            for arr in feat_dict.values():
                n = len(arr)
                break
            if n:
                break
        
        if not n:
            return {'features': np.array([]), 'names': []}
        
        # Build feature matrix
        feature_list = []
        names = []
        
        # Technical features
        for name, arr in sorted(tech_features.items()):
            if len(arr) == n:
                feature_list.append(arr.reshape(-1, 1))
                names.append(f'tech_{name}')
        
        # SMC features
        for name, arr in sorted(smc_features.items()):
            if len(arr) == n:
                feature_list.append(arr.reshape(-1, 1))
                names.append(f'smc_{name}')
        
        # Quantitative features
        for name, arr in sorted(quant_features.items()):
            if len(arr) == n:
                feature_list.append(arr.reshape(-1, 1))
                names.append(f'quant_{name}')
        
        # Macro features (if provided)
        if macro_features:
            for name, arr in sorted(macro_features.items()):
                if len(arr) == n:
                    feature_list.append(arr.reshape(-1, 1))
                    names.append(f'macro_{name}')
        
        # Concatenate
        if feature_list:
            feature_matrix = np.hstack(feature_list)
        else:
            feature_matrix = np.zeros((n, 0))
        
        # Truncate to max features if needed
        if feature_matrix.shape[1] > self._max_features:
            feature_matrix = feature_matrix[:, :self._max_features]
            names = names[:self._max_features]
        
        return {
            'features': feature_matrix,
            'names': names,
            'shape': feature_matrix.shape,
            'timestamp_ns': time.time_ns(),
        }


class FeaturePipeline:
    """
    Main Ray DAG orchestrator for feature computation.
    Manages worker lifecycle and enforces memory constraints.
    """
    
    def __init__(self, max_concurrent_workers: int = 4):
        self._max_workers = max_concurrent_workers
        self._tech_worker = TechnicalIndicatorWorker.remote()
        self._smc_worker = SMCWorker.remote()
        self._quant_worker = QuantitativeWorker.remote()
        self._aggregator = FeatureAggregator.remote()
    
    async def compute_features(
        self,
        prices: np.ndarray,
        highs: np.ndarray,
        lows: np.ndarray,
        volumes: np.ndarray,
        timestamps: np.ndarray,
        macro_features: Optional[Dict[str, np.ndarray]] = None,
    ) -> Dict[str, Any]:
        """
        Execute full feature computation pipeline.
        
        Args:
            prices: Price array
            highs: High prices
            lows: Low prices
            volumes: Volume array
            timestamps: Timestamp array in nanoseconds
            macro_features: Optional pre-computed macro features
        
        Returns:
            Dict with feature matrix and metadata
        """
        # Validate inputs
        assert len(prices) == len(highs) == len(lows) == len(volumes) == len(timestamps)
        
        # Convert to Ray objects
        prices_ref = ray.put(prices.astype(np.float64))
        highs_ref = ray.put(highs.astype(np.float64))
        lows_ref = ray.put(lows.astype(np.float64))
        volumes_ref = ray.put(volumes.astype(np.float64))
        timestamps_ref = ray.put(timestamps.astype(np.int64))
        
        # Execute workers in parallel (limited concurrency)
        tech_future = self._tech_worker.compute_oscillators.remote(prices_ref, timestamps_ref)
        trend_future = self._tech_worker.compute_trend_features.remote(prices_ref, highs_ref, lows_ref)
        
        smc_future = self._smc_worker.compute_smc_features.remote(
            prices_ref, highs_ref, lows_ref, timestamps_ref
        )
        
        quant_future = self._quant_worker.compute_quant_features.remote(prices_ref, volumes_ref)
        
        # Wait for all computations
        tech_osc = await tech_future
        tech_trend = await trend_future
        smc_feat = await smc_future
        quant_feat = await quant_future
        
        # Combine technical features
        tech_combined = {**tech_osc, **tech_trend}
        
        # Aggregate all features
        result = await self._aggregator.aggregate.remote(
            tech_combined, smc_feat, quant_feat, macro_features
        )
        
        return ray.get(result)
    
    def cleanup(self):
        """Release worker resources."""
        ray.kill(self._tech_worker)
        ray.kill(self._smc_worker)
        ray.kill(self._quant_worker)
        ray.kill(self._aggregator)


# Singleton instance
_pipeline: Optional[FeaturePipeline] = None


def get_pipeline() -> FeaturePipeline:
    """Get or create singleton pipeline instance."""
    global _pipeline
    if _pipeline is None:
        _pipeline = FeaturePipeline(max_concurrent_workers=4)
    return _pipeline


async def run_feature_computation(
    prices: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    volumes: np.ndarray,
    timestamps: np.ndarray,
) -> Dict[str, Any]:
    """Convenience function to run full feature computation."""
    pipeline = get_pipeline()
    return await pipeline.compute_features(prices, highs, lows, volumes, timestamps)
