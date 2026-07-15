"""
Parquet UI Reader: Fast Parquet reader for historical backtest logs and trade journals.
Uses zero-copy Polars DataFrames to keep RAM usage near zero even with years of tick data.
Memory-mapped file access with chunked reading.
"""

import polars as pl
from typing import Dict, List, Optional, Tuple, Iterator, Any
from dataclasses import dataclass
import json
import threading
from pathlib import Path


# Memory bounds
MAX_ROWS_PER_CHUNK = 100_000
MAX_MEMORY_MB = 512


@dataclass
class QueryResult:
    """Result of a parquet query."""
    row_count: int
    column_names: List[str]
    memory_bytes: int
    data_json: str  # Compact JSON for UI


class ParquetUiReader:
    """
    Zero-copy Parquet reader using Polars lazy evaluation.
    Supports chunked reading for large files without loading everything into RAM.
    """
    
    def __init__(self, max_memory_mb: int = MAX_MEMORY_MB):
        self.max_memory_mb = max_memory_mb
        self._lock = threading.RLock()
        self._file_cache: Dict[str, pl.LazyFrame] = {}
        self._cache_max_size = 10
        
    def open_file(self, path: str) -> pl.LazyFrame:
        """
        Open a Parquet file as a LazyFrame (zero-copy until collect).
        Caches the LazyFrame for repeated queries.
        """
        with self._lock:
            if path in self._file_cache:
                return self._file_cache[path]
            
            # Evict oldest if cache is full
            if len(self._file_cache) >= self._cache_max_size:
                oldest = next(iter(self._file_cache))
                del self._file_cache[oldest]
            
            # Scan lazily (no data loaded yet)
            lf = pl.scan_parquet(path, low_memory=True)
            self._file_cache[path] = lf
            
            return lf
    
    def get_schema(self, path: str) -> Dict[str, str]:
        """Get schema of a Parquet file without loading data."""
        lf = self.open_file(path)
        schema = lf.collect_schema()
        return {name: str(dtype) for name, dtype in schema.items()}
    
    def get_row_count(self, path: str) -> int:
        """Get row count without loading data."""
        lf = self.open_file(path)
        return lf.select(pl.count()).collect().item()
    
    def query(
        self,
        path: str,
        columns: Optional[List[str]] = None,
        filters: Optional[Dict[str, Any]] = None,
        limit: Optional[int] = None,
        offset: int = 0
    ) -> QueryResult:
        """
        Execute a query on a Parquet file with optional filters.
        Uses streaming to minimize memory usage.
        """
        lf = self.open_file(path)
        
        # Select columns
        if columns:
            lf = lf.select(columns)
        
        # Apply filters
        if filters:
            for col, value in filters.items():
                if isinstance(value, tuple):
                    # Range filter: (min, max)
                    min_val, max_val = value
                    lf = lf.filter((pl.col(col) >= min_val) & (pl.col(col) <= max_val))
                elif isinstance(value, list):
                    # In filter
                    lf = lf.filter(pl.col(col).is_in(value))
                else:
                    # Equality filter
                    lf = lf.filter(pl.col(col) == value)
        
        # Apply offset and limit
        if offset > 0:
            lf = lf.slice(offset, limit if limit else MAX_ROWS_PER_CHUNK)
        elif limit:
            lf = lf.limit(limit)
        
        # Collect with streaming
        df = lf.collect(streaming=True)
        
        # Convert to JSON
        data_json = df.write_json(row_oriented=True)
        
        return QueryResult(
            row_count=len(df),
            column_names=df.columns,
            memory_bytes=df.estimated_size(),
            data_json=data_json
        )
    
    def query_chunked(
        self,
        path: str,
        columns: Optional[List[str]] = None,
        filters: Optional[Dict[str, Any]] = None,
        chunk_size: int = MAX_ROWS_PER_CHUNK
    ) -> Iterator[QueryResult]:
        """
        Iterate over a Parquet file in chunks.
        Yields QueryResult for each chunk without loading entire file.
        """
        lf = self.open_file(path)
        
        if columns:
            lf = lf.select(columns)
        
        if filters:
            for col, value in filters.items():
                if isinstance(value, tuple):
                    min_val, max_val = value
                    lf = lf.filter((pl.col(col) >= min_val) & (pl.col(col) <= max_val))
                elif isinstance(value, list):
                    lf = lf.filter(pl.col(col).is_in(value))
                else:
                    lf = lf.filter(pl.col(col) == value)
        
        # Get total rows
        total_rows = lf.select(pl.count()).collect().item()
        
        # Iterate in chunks
        offset = 0
        while offset < total_rows:
            chunk_lf = lf.slice(offset, chunk_size)
            df = chunk_lf.collect(streaming=True)
            
            if len(df) == 0:
                break
            
            yield QueryResult(
                row_count=len(df),
                column_names=df.columns,
                memory_bytes=df.estimated_size(),
                data_json=df.write_json(row_oriented=True)
            )
            
            offset += chunk_size
    
    def aggregate(
        self,
        path: str,
        aggregations: Dict[str, str],
        group_by: Optional[List[str]] = None,
        filters: Optional[Dict[str, Any]] = None
    ) -> QueryResult:
        """
        Perform aggregations on Parquet data.
        Supports: sum, mean, std, min, max, count, first, last
        """
        lf = self.open_file(path)
        
        if filters:
            for col, value in filters.items():
                if isinstance(value, tuple):
                    min_val, max_val = value
                    lf = lf.filter((pl.col(col) >= min_val) & (pl.col(col) <= max_val))
                elif isinstance(value, list):
                    lf = lf.filter(pl.col(col).is_in(value))
                else:
                    lf = lf.filter(pl.col(col) == value)
        
        # Build aggregation expressions
        agg_exprs = []
        for col, agg_func in aggregations.items():
            if agg_func == 'sum':
                agg_exprs.append(pl.col(col).sum().alias(f"{col}_sum"))
            elif agg_func == 'mean':
                agg_exprs.append(pl.col(col).mean().alias(f"{col}_mean"))
            elif agg_func == 'std':
                agg_exprs.append(pl.col(col).std().alias(f"{col}_std"))
            elif agg_func == 'min':
                agg_exprs.append(pl.col(col).min().alias(f"{col}_min"))
            elif agg_func == 'max':
                agg_exprs.append(pl.col(col).max().alias(f"{col}_max"))
            elif agg_func == 'count':
                agg_exprs.append(pl.col(col).count().alias(f"{col}_count"))
            elif agg_func == 'first':
                agg_exprs.append(pl.col(col).first().alias(f"{col}_first"))
            elif agg_func == 'last':
                agg_exprs.append(pl.col(col).last().alias(f"{col}_last"))
        
        if group_by:
            lf = lf.group_by(group_by).agg(agg_exprs)
        else:
            lf = lf.select(agg_exprs)
        
        df = lf.collect(streaming=True)
        
        return QueryResult(
            row_count=len(df),
            column_names=df.columns,
            memory_bytes=df.estimated_size(),
            data_json=df.write_json(row_oriented=True)
        )
    
    def get_time_range(self, path: str, timestamp_col: str = "timestamp") -> Tuple[int, int]:
        """Get min/max timestamps from a Parquet file."""
        lf = self.open_file(path)
        result = lf.select([
            pl.col(timestamp_col).min().alias('min_ts'),
            pl.col(timestamp_col).max().alias('max_ts')
        ]).collect()
        
        return (result['min_ts'].item(), result['max_ts'].item())
    
    def sample(
        self,
        path: str,
        n: int = 1000,
        fraction: Optional[float] = None
    ) -> QueryResult:
        """Get a random sample of rows."""
        lf = self.open_file(path)
        
        if fraction:
            df = lf.sample(fraction=fraction).collect(streaming=True)
        else:
            df = lf.sample(n=n).collect(streaming=True)
        
        return QueryResult(
            row_count=len(df),
            column_names=df.columns,
            memory_bytes=df.estimated_size(),
            data_json=df.write_json(row_oriented=True)
        )
    
    def close_file(self, path: str):
        """Remove a file from cache."""
        with self._lock:
            self._file_cache.pop(path, None)
    
    def clear_cache(self):
        """Clear all cached LazyFrames."""
        with self._lock:
            self._file_cache.clear()
    
    def get_cache_info(self) -> Dict:
        """Get information about cached files."""
        with self._lock:
            info = {}
            for path, lf in self._file_cache.items():
                try:
                    row_count = lf.select(pl.count()).collect().item()
                    info[path] = {'rows': row_count}
                except Exception:
                    info[path] = {'rows': 'unknown'}
            
            return {
                'cached_files': len(info),
                'max_cached': self._cache_max_size,
                'files': info
            }


# Singleton instance
_reader_instance: Optional[ParquetUiReader] = None
_instance_lock = threading.Lock()


def get_parquet_reader(max_memory_mb: int = MAX_MEMORY_MB) -> ParquetUiReader:
    """Get or create the singleton ParquetUiReader instance."""
    global _reader_instance
    if _reader_instance is None:
        with _instance_lock:
            if _reader_instance is None:
                _reader_instance = ParquetUiReader(max_memory_mb)
    return _reader_instance


if __name__ == '__main__':
    # Example usage with synthetic data
    import tempfile
    import numpy as np
    
    # Create a sample Parquet file
    n_rows = 1_000_000
    data = {
        'timestamp': np.arange(n_rows) * 1_000_000,
        'symbol': np.random.choice(['BTC-PERP', 'ETH-PERP', 'SOL-PERP'], n_rows),
        'price': np.random.uniform(1000, 50000, n_rows),
        'volume': np.random.exponential(1.0, n_rows),
        'pnl': np.random.normal(0, 10, n_rows),
    }
    
    df = pl.DataFrame(data)
    
    with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as f:
        df.write_parquet(f.name)
        temp_path = f.name
    
    try:
        reader = get_parquet_reader()
        
        # Get schema
        schema = reader.get_schema(temp_path)
        print(f"Schema: {schema}")
        
        # Get row count
        count = reader.get_row_count(temp_path)
        print(f"Row count: {count}")
        
        # Query with filter
        result = reader.query(
            temp_path,
            columns=['symbol', 'price', 'pnl'],
            filters={'symbol': 'BTC-PERP'},
            limit=10
        )
        print(f"Query result: {result.row_count} rows, {result.memory_bytes} bytes")
        
        # Aggregation
        agg_result = reader.aggregate(
            temp_path,
            aggregations={'pnl': 'sum', 'pnl': 'mean', 'volume': 'sum'},
            group_by=['symbol']
        )
        print(f"Aggregation: {agg_result.data_json[:200]}...")
        
        # Cache info
        cache_info = reader.get_cache_info()
        print(f"Cache: {cache_info}")
        
    finally:
        import os
        os.unlink(temp_path)
