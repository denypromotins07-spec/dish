# python/telemetry/journal_query_engine.py
"""
In-memory DuckDB/Polars hybrid query engine for trade journal.
Allows instant search, filter, and aggregation of millions of trades.
Memory-efficient: queries parquet files without loading entire dataset into RAM.
"""

from __future__ import annotations
import duckdb
import polars as pl
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Dict, List, Any, Tuple
from collections import deque
import time


@dataclass
class QueryResult:
    """Result of a journal query."""
    row_count: int
    execution_time_ms: float
    memory_used_mb: float
    data: pl.DataFrame
    aggregations: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> dict:
        return {
            "row_count": self.row_count,
            "execution_time_ms": self.execution_time_ms,
            "memory_used_mb": self.memory_used_mb,
            "aggregations": self.aggregations,
        }


@dataclass
class FilterSpec:
    """Specification for filtering trades."""
    tag_name: Optional[str] = None
    tag_value: Optional[str] = None
    strategy_id: Optional[int] = None
    asset_id: Optional[int] = None
    side: Optional[int] = None
    date_from: Optional[str] = None  # YYYY-MM-DD
    date_to: Optional[str] = None
    pnl_min: Optional[float] = None
    pnl_max: Optional[float] = None
    execution_quality: Optional[str] = None


class JournalQueryEngine:
    """
    High-performance query engine for trade journal data.
    
    Features:
    - DuckDB for SQL queries on Parquet files (zero-copy)
    - Polars for in-memory transformations
    - Memory-bounded result caching
    - Tag-based filtering
    - Aggregation pushdown to DuckDB
    """
    
    # Memory limit for cached results (MB)
    MAX_CACHE_MEMORY_MB = 50.0
    
    def __init__(
        self,
        parquet_directory: str,
        max_cache_rows: int = 100_000,
    ):
        """
        Initialize the query engine.
        
        Args:
            parquet_directory: Directory containing parquet files
            max_cache_rows: Maximum rows to cache in memory
        """
        self.parquet_dir = Path(parquet_directory)
        self.max_cache_rows = max_cache_rows
        
        # Initialize DuckDB connection
        self._conn = duckdb.connect(":memory:")
        
        # Configure DuckDB for low memory usage
        self._conn.execute("SET memory_limit='2GB'")
        self._conn.execute("SET threads TO 4")
        
        # Result cache (memory-bounded)
        self._cache: deque[Tuple[str, pl.DataFrame]] = deque()
        self._cache_memory_bytes: int = 0
        
        # Statistics
        self._query_count: int = 0
        self._cache_hits: int = 0
    
    def _register_parquet_files(self) -> None:
        """Register parquet files with DuckDB."""
        if not self.parquet_dir.exists():
            return
        
        parquet_files = list(self.parquet_dir.glob("**/*.parquet"))
        if not parquet_files:
            return
        
        # Create a view over all parquet files
        file_patterns = [str(f) for f in parquet_files]
        
        # Use glob pattern if files are in same structure
        self._conn.execute(f"""
            CREATE VIEW IF NOT EXISTS trades AS
            SELECT * FROM read_parquet('{self.parquet_dir}/**/*.parquet')
        """)
    
    def search_trades(
        self,
        filters: FilterSpec,
        limit: int = 10000,
        use_cache: bool = True,
    ) -> QueryResult:
        """
        Search trades with filters.
        
        Args:
            filters: Filter specification
            limit: Maximum rows to return
            use_cache: Whether to use result cache
            
        Returns:
            QueryResult with matching trades
        """
        start_time = time.perf_counter()
        
        # Build cache key
        cache_key = self._build_cache_key(filters, limit)
        
        # Check cache
        if use_cache:
            cached = self._get_from_cache(cache_key)
            if cached is not None:
                self._cache_hits += 1
                return QueryResult(
                    row_count=len(cached),
                    execution_time_ms=(time.perf_counter() - start_time) * 1000,
                    memory_used_mb=0.0,  # From cache
                    data=cached,
                )
        
        # Ensure parquet files are registered
        self._register_parquet_files()
        
        # Build WHERE clause
        where_clauses = []
        params = {}
        
        if filters.tag_name and filters.tag_value:
            where_clauses.append("""
                tags IS NOT NULL AND 
                array_length(list_filter(tags, x -> x.name = ?)) > 0 AND
                list_filter(tags, x -> x.name = ?)[1].value = ?
            """)
            params["tag_name"] = filters.tag_name
            params["tag_value"] = filters.tag_value
        
        if filters.strategy_id is not None:
            where_clauses.append("strategy_id = ?")
            params["strategy_id"] = filters.strategy_id
        
        if filters.asset_id is not None:
            where_clauses.append("asset_id = ?")
            params["asset_id"] = filters.asset_id
        
        if filters.side is not None:
            where_clauses.append("side = ?")
            params["side"] = filters.side
        
        if filters.date_from:
            where_clauses.append("date >= ?")
            params["date_from"] = filters.date_from
        
        if filters.date_to:
            where_clauses.append("date <= ?")
            params["date_to"] = filters.date_to
        
        if filters.pnl_min is not None:
            where_clauses.append("pnl >= ?")
            params["pnl_min"] = filters.pnl_min
        
        if filters.pnl_max is not None:
            where_clauses.append("pnl <= ?")
            params["pnl_max"] = filters.pnl_max
        
        if filters.execution_quality:
            where_clauses.append("""
                tags IS NOT NULL AND
                list_filter(tags, x -> x.name = 'execution_quality')[1].value = ?
            """)
            params["exec_quality"] = filters.execution_quality
        
        # Build query
        where_clause = " AND ".join(where_clauses) if where_clauses else "1=1"
        
        query = f"""
            SELECT * FROM trades
            WHERE {where_clause}
            ORDER BY timestamp_ns DESC
            LIMIT ?
        """
        params["limit"] = limit
        
        # Execute query
        try:
            result = self._conn.execute(query, list(params.values())).fetch_df()
            df = pl.from_pandas(result)
        except Exception as e:
            # Fallback: return empty DataFrame
            df = pl.DataFrame()
        
        execution_time = (time.perf_counter() - start_time) * 1000
        
        # Cache result if small enough
        self._add_to_cache(cache_key, df)
        
        self._query_count += 1
        
        return QueryResult(
            row_count=len(df),
            execution_time_ms=execution_time,
            memory_used_mb=self._estimate_df_memory(df),
            data=df,
        )
    
    def aggregate_trades(
        self,
        filters: FilterSpec,
        group_by: List[str],
        aggregations: List[str],
    ) -> QueryResult:
        """
        Aggregate trades with grouping.
        
        Args:
            filters: Filter specification
            group_by: Columns to group by
            aggregations: Aggregation expressions (SQL style)
            
        Returns:
            QueryResult with aggregated data
        """
        start_time = time.perf_counter()
        
        self._register_parquet_files()
        
        # Build WHERE clause
        where_clauses = []
        params = []
        
        if filters.strategy_id is not None:
            where_clauses.append("strategy_id = ?")
            params.append(filters.strategy_id)
        
        if filters.asset_id is not None:
            where_clauses.append("asset_id = ?")
            params.append(filters.asset_id)
        
        if filters.date_from:
            where_clauses.append("date >= ?")
            params.append(filters.date_from)
        
        if filters.date_to:
            where_clauses.append("date <= ?")
            params.append(filters.date_to)
        
        where_clause = " AND ".join(where_clauses) if where_clauses else "1=1"
        
        # Build aggregation query
        agg_exprs = ", ".join(aggregations)
        group_exprs = ", ".join(group_by)
        
        query = f"""
            SELECT {group_exprs}, {agg_exprs}
            FROM trades
            WHERE {where_clause}
            GROUP BY {group_exprs}
        """
        
        try:
            result = self._conn.execute(query, params).fetch_df()
            df = pl.from_pandas(result)
            
            # Compute additional stats
            agg_stats = {}
            for col in df.columns:
                if col not in group_by:
                    agg_stats[col] = {
                        "min": float(df[col].min()) if df[col].dtype.is_numeric() else None,
                        "max": float(df[col].max()) if df[col].dtype.is_numeric() else None,
                        "sum": float(df[col].sum()) if df[col].dtype.is_numeric() else None,
                    }
            
        except Exception as e:
            df = pl.DataFrame()
            agg_stats = {}
        
        execution_time = (time.perf_counter() - start_time) * 1000
        
        return QueryResult(
            row_count=len(df),
            execution_time_ms=execution_time,
            memory_used_mb=self._estimate_df_memory(df),
            data=df,
            aggregations=agg_stats,
        )
    
    def get_tag_statistics(self) -> Dict[str, Any]:
        """Get statistics about tags in the journal."""
        self._register_parquet_files()
        
        query = """
            SELECT 
                COUNT(*) as total_trades,
                COUNT(DISTINCT tags) as trades_with_tags
            FROM trades
            WHERE tags IS NOT NULL
        """
        
        try:
            result = self._conn.execute(query).fetchone()
            if result:
                return {
                    "total_trades": result[0],
                    "trades_with_tags": result[1],
                }
        except Exception:
            pass
        
        return {"total_trades": 0, "trades_with_tags": 0}
    
    def get_available_tags(self) -> Dict[str, List[str]]:
        """Get all unique tag names and values."""
        self._register_parquet_files()
        
        # Extract distinct tag names and values
        query = """
            SELECT DISTINCT 
                unnest(list_extract(tags, 1).name) as tag_name,
                unnest(list_extract(tags, 1).value) as tag_value
            FROM trades
            WHERE tags IS NOT NULL
        """
        
        try:
            result = self._conn.execute(query).fetch_df()
            df = pl.from_pandas(result)
            
            # Group by tag name
            tags_by_name = {}
            for row in df.iter_rows():
                name, value = row
                if name not in tags_by_name:
                    tags_by_name[name] = set()
                tags_by_name[name].add(value)
            
            return {k: sorted(list(v)) for k, v in tags_by_name.items()}
            
        except Exception:
            return {}
    
    def _build_cache_key(self, filters: FilterSpec, limit: int) -> str:
        """Build cache key from filters."""
        parts = [
            f"tag={filters.tag_name}:{filters.tag_value}" if filters.tag_name else "",
            f"strategy={filters.strategy_id}" if filters.strategy_id is not None else "",
            f"asset={filters.asset_id}" if filters.asset_id is not None else "",
            f"side={filters.side}" if filters.side is not None else "",
            f"date={filters.date_from}:{filters.date_to}" if filters.date_from else "",
            f"limit={limit}",
        ]
        return "|".join(p for p in parts if p)
    
    def _get_from_cache(self, key: str) -> Optional[pl.DataFrame]:
        """Get DataFrame from cache."""
        for cached_key, df in self._cache:
            if cached_key == key:
                return df
        return None
    
    def _add_to_cache(self, key: str, df: pl.DataFrame) -> None:
        """Add DataFrame to cache with memory management."""
        df_memory = self._estimate_df_memory(df)
        
        # Evict old entries if needed
        while self._cache_memory_bytes + df_memory > self.MAX_CACHE_MEMORY_MB * 1024 * 1024:
            if not self._cache:
                break
            _, old_df = self._cache.popleft()
            self._cache_memory_bytes -= self._estimate_df_memory(old_df)
        
        # Add new entry
        self._cache.append((key, df))
        self._cache_memory_bytes += df_memory
    
    def _estimate_df_memory(self, df: pl.DataFrame) -> int:
        """Estimate memory usage of DataFrame in bytes."""
        try:
            return df.estimated_size()
        except Exception:
            # Fallback estimate
            return len(df) * 100  # Rough estimate of 100 bytes per row
    
    def get_query_stats(self) -> Dict[str, Any]:
        """Get query engine statistics."""
        return {
            "total_queries": self._query_count,
            "cache_hits": self._cache_hits,
            "cache_hit_rate": self._cache_hits / self._query_count if self._query_count > 0 else 0,
            "cache_memory_mb": self._cache_memory_bytes / (1024 * 1024),
            "cache_entries": len(self._cache),
        }
    
    def close(self) -> None:
        """Close database connection."""
        self._conn.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def create_sample_filters(
    strategy_id: Optional[int] = None,
    tag: Optional[Tuple[str, str]] = None,
    date_range: Optional[Tuple[str, str]] = None,
) -> FilterSpec:
    """Create filter specification from common parameters."""
    return FilterSpec(
        strategy_id=strategy_id,
        tag_name=tag[0] if tag else None,
        tag_value=tag[1] if tag else None,
        date_from=date_range[0] if date_range else None,
        date_to=date_range[1] if date_range else None,
    )


if __name__ == "__main__":
    # Example usage
    import tempfile
    
    # Create temporary directory for demo
    with tempfile.TemporaryDirectory() as tmpdir:
        engine = JournalQueryEngine(tmpdir)
        
        # Get available tags (will be empty without data)
        tags = engine.get_available_tags()
        print(f"Available tags: {tags}")
        
        # Search with filters
        filters = FilterSpec(strategy_id=1)
        result = engine.search_trades(filters, limit=100)
        
        print(f"\nQuery result:")
        print(f"  Rows: {result.row_count}")
        print(f"  Time: {result.execution_time_ms:.2f}ms")
        print(f"  Memory: {result.memory_used_mb:.2f}MB")
        
        # Stats
        stats = engine.get_query_stats()
        print(f"\nEngine stats: {stats}")
        
        engine.close()
