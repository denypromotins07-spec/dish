"""
Automated feature store updater that recalculates rolling statistics,
cointegration vectors, and normalizations using latest market data
before retraining cycles. Memory-bounded with efficient Polars operations.
"""

import polars as pl
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from collections import deque
import threading
from datetime import datetime

# Memory bounds
MAX_FEATURE_ROWS = 100_000
MAX_PAIRS_FOR_COINT = 20


@dataclass
class FeatureConfig:
    """Configuration for a single feature."""
    name: str
    calculation_type: str  # 'rolling', 'lag', 'diff', 'ratio', 'coint'
    window_size: int
    symbol: Optional[str] = None
    pair: Optional[Tuple[str, str]] = None


class FeatureStoreUpdater:
    """
    Manages feature calculations and updates the feature store
    before model retraining cycles.
    """

    def __init__(self, symbols: List[str]):
        self.symbols = symbols
        self._feature_store: Dict[str, pl.DataFrame] = {}
        self._raw_data: Dict[str, deque] = {s: deque(maxlen=MAX_FEATURE_ROWS) for s in symbols}
        self._feature_configs: List[FeatureConfig] = []
        self._lock = threading.RLock()
        
        # Pre-computed cointegration matrix
        self._coint_matrix: Optional[np.ndarray] = None
        self._coint_symbols: List[str] = []

    def add_raw_data(self, symbol: str, timestamp_ns: int, price: float, volume: float) -> None:
        """Add raw market data point."""
        if symbol not in self._raw_data:
            self._raw_data[symbol] = deque(maxlen=MAX_FEATURE_ROWS)
        
        self._raw_data[symbol].append({
            'timestamp': timestamp_ns,
            'price': price,
            'volume': volume,
        })

    def add_batch_data(self, symbol: str, df: pl.DataFrame) -> None:
        """Add batch of raw data."""
        for row in df.iter_rows(named=True):
            self.add_raw_data(symbol, row['timestamp'], row['price'], row.get('volume', 0))

    def register_feature(self, config: FeatureConfig) -> None:
        """Register a feature configuration."""
        self._feature_configs.append(config)

    def compute_rolling_features(
        self,
        symbol: str,
        windows: List[int],
        feature_name: str = 'returns',
    ) -> pl.DataFrame:
        """Compute rolling statistics for a symbol."""
        with self._lock:
            if symbol not in self._raw_data or len(self._raw_data[symbol]) < max(windows):
                return pl.DataFrame()
            
            # Convert to DataFrame
            df = pl.DataFrame(list(self._raw_data[symbol]))
            
            # Calculate returns
            df = df.with_columns([
                (pl.col('price').pct_change()).alias('returns'),
            ])
            
            # Compute rolling stats for each window
            for window in windows:
                df = df.with_columns([
                    pl.col('returns').rolling_mean(window).alias(f'{feature_name}_ma_{window}'),
                    pl.col('returns').rolling_std(window).alias(f'{feature_name}_std_{window}'),
                    pl.col('returns').rolling_min(window).alias(f'{feature_name}_min_{window}'),
                    pl.col('returns').rolling_max(window).alias(f'{feature_name}_max_{window}'),
                    pl.col('volume').rolling_mean(window).alias(f'volume_ma_{window}'),
                ])
            
            return df

    def compute_lag_features(
        self,
        symbol: str,
        lags: List[int],
        feature_name: str = 'returns',
    ) -> pl.DataFrame:
        """Compute lag features for a symbol."""
        with self._lock:
            if symbol not in self._raw_data or len(self._raw_data[symbol]) < max(lags):
                return pl.DataFrame()
            
            df = pl.DataFrame(list(self._raw_data[symbol]))
            df = df.with_columns([
                (pl.col('price').pct_change()).alias('returns'),
            ])
            
            for lag in lags:
                df = df.with_columns([
                    pl.col('returns').shift(lag).alias(f'{feature_name}_lag_{lag}'),
                ])
            
            return df

    def compute_cross_symbol_features(
        self,
        base_symbol: str,
        quote_symbols: List[str],
    ) -> pl.DataFrame:
        """Compute ratio features between symbols."""
        with self._lock:
            if base_symbol not in self._raw_data:
                return pl.DataFrame()
            
            base_df = pl.DataFrame(list(self._raw_data[base_symbol])).select(['timestamp', 'price']).rename({'price': 'base_price'})
            
            result = base_df
            for quote in quote_symbols:
                if quote not in self._raw_data:
                    continue
                
                quote_df = pl.DataFrame(list(self._raw_data[quote])).select(['timestamp', 'price']).rename({'price': f'{quote}_price'})
                
                # Join on timestamp (approximate)
                result = result.join(quote_df, on='timestamp', how='left')
                
                # Calculate ratio
                result = result.with_columns([
                    (pl.col('base_price') / pl.col(f'{quote}_price')).alias(f'ratio_{base_symbol}_{quote}'),
                ])
            
            return result

    def compute_cointegration_matrix(self, max_pairs: int = MAX_PAIRS_FOR_COINT) -> np.ndarray:
        """
        Compute cointegration matrix for pairs trading signals.
        Uses Engle-Granger two-step method approximation.
        """
        with self._lock:
            # Select top symbols by data availability
            available = [s for s in self.symbols if len(self._raw_data.get(s, [])) > 100]
            symbols_to_use = available[:max_pairs]
            
            if len(symbols_to_use) < 2:
                return np.array([])
            
            # Build price matrix
            n = len(symbols_to_use)
            min_len = min(len(self._raw_data[s]) for s in symbols_to_use)
            
            prices = np.zeros((min_len, n))
            for i, symbol in enumerate(symbols_to_use):
                data = list(self._raw_data[symbol])[-min_len:]
                prices[:, i] = [d['price'] for d in data]
            
            # Compute log prices
            log_prices = np.log(prices)
            
            # Simplified cointegration score (correlation of residuals)
            coint_scores = np.zeros((n, n))
            for i in range(n):
                for j in range(i + 1, n):
                    # Simple OLS residual correlation as proxy
                    pi = log_prices[:, i]
                    pj = log_prices[:, j]
                    
                    # Residuals from regression
                    coef = np.cov(pi, pj)[0, 1] / np.var(pj)
                    residuals = pi - coef * pj
                    
                    # ADF-like score (variance ratio test proxy)
                    var_ratio = np.var(np.diff(residuals)) / np.var(residuals)
                    coint_scores[i, j] = var_ratio
                    coint_scores[j, i] = var_ratio
            
            self._coint_matrix = coint_scores
            self._coint_symbols = symbols_to_use
            
            return coint_scores

    def normalize_features(self, df: pl.DataFrame, method: str = 'zscore') -> pl.DataFrame:
        """Normalize features in DataFrame."""
        numeric_cols = df.select(pl.col(pl.Float64)).columns
        
        if method == 'zscore':
            return df.with_columns([
                (pl.col(col) - pl.col(col).mean()) / pl.col(col).std().clip(1e-8)
                for col in numeric_cols
            ])
        elif method == 'minmax':
            return df.with_columns([
                (pl.col(col) - pl.col(col).min()) / (pl.col(col).max() - pl.col(col).min()).clip(1e-8)
                for col in numeric_cols
            ])
        elif method == 'robust':
            return df.with_columns([
                (pl.col(col) - pl.col(col).median()) / 
                (pl.col(col).quantile(0.75) - pl.col(col).quantile(0.25)).clip(1e-8)
                for col in numeric_cols
            ])
        
        return df

    def update_all_features(self) -> Dict[str, pl.DataFrame]:
        """
        Update all registered features for all symbols.
        Returns dictionary of symbol -> feature DataFrame.
        """
        results = {}
        
        for symbol in self.symbols:
            features_list = []
            
            # Rolling features
            rolling_df = self.compute_rolling_features(symbol, windows=[5, 10, 20, 50])
            if len(rolling_df) > 0:
                features_list.append(rolling_df)
            
            # Lag features
            lag_df = self.compute_lag_features(symbol, lags=[1, 2, 3, 5])
            if len(lag_df) > 0:
                features_list.append(lag_df)
            
            # Cross-symbol features
            other_symbols = [s for s in self.symbols if s != symbol][:3]
            if other_symbols:
                cross_df = self.compute_cross_symbol_features(symbol, other_symbols)
                if len(cross_df) > 0:
                    features_list.append(cross_df)
            
            if features_list:
                # Join all features
                combined = features_list[0]
                for df in features_list[1:]:
                    combined = combined.join(df, on='timestamp', how='left')
                
                # Normalize
                combined = self.normalize_features(combined, method='zscore')
                
                # Drop nulls
                combined = combined.drop_nulls()
                
                results[symbol] = combined
                self._feature_store[symbol] = combined
        
        return results

    def get_feature_matrix(self, symbols: Optional[List[str]] = None) -> Tuple[np.ndarray, List[str]]:
        """Get combined feature matrix for ML training."""
        if symbols is None:
            symbols = list(self._feature_store.keys())
        
        if not symbols:
            return np.array([]), []
        
        # Get common timestamps
        dfs = [self._feature_store[s] for s in symbols if s in self._feature_store]
        if not dfs:
            return np.array([]), []
        
        # Concatenate features (not rows - each symbol gets separate columns)
        feature_cols = []
        for df in dfs:
            cols = [c for c in df.columns if c not in ('timestamp', 'price', 'volume')]
            feature_cols.extend(cols)
        
        # Merge on timestamp
        merged = dfs[0]
        for df in dfs[1:]:
            merged = merged.join(df, on='timestamp', how='inner', suffix='_dup')
        
        # Remove duplicate columns and convert to numpy
        feature_df = merged.select(feature_cols)
        
        return feature_df.to_numpy(), feature_cols

    def export_for_training(self, output_path: str) -> None:
        """Export feature store to parquet for training pipeline."""
        if not self._feature_store:
            self.update_all_features()
        
        for symbol, df in self._feature_store.items():
            safe_symbol = symbol.replace('/', '_').replace('-', '_')
            path = f"{output_path}/{safe_symbol}_features.parquet"
            df.write_parquet(path)


if __name__ == "__main__":
    # Example usage
    updater = FeatureStoreUpdater(['BTC-USDT', 'ETH-USDT'])
    
    # Add sample data
    base_time = int(datetime.now().timestamp() * 1e9)
    for i in range(200):
        updater.add_raw_data('BTC-USDT', base_time + i * 60_000_000_000, 50000 + np.random.randn() * 100, np.random.rand() * 1000)
        updater.add_raw_data('ETH-USDT', base_time + i * 60_000_000_000, 3000 + np.random.randn() * 50, np.random.rand() * 500)
    
    # Register features
    updater.register_feature(FeatureConfig('returns_ma', 'rolling', window_size=20, symbol='BTC-USDT'))
    
    # Update all features
    features = updater.update_all_features()
    
    for symbol, df in features.items():
        print(f"{symbol}: {len(df)} rows, {len(df.columns)} features")
    
    # Get feature matrix
    X, cols = updater.get_feature_matrix()
    print(f"Feature matrix shape: {X.shape}")
    print(f"Features: {cols[:10]}...")
