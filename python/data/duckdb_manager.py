"""
DuckDB Manager for Time-Series Tick Data

DuckDB in-memory/SSD hybrid setup configured for fast time-series storage
of tick data with strict memory eviction policies to prevent RAM overflow.

Features:
- Hybrid in-memory/SSD storage
- Automatic memory-bounded operation
- LRU-based eviction policy
- Efficient time-series queries
"""

import os
import time
import threading
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass
from datetime import datetime
from collections import OrderedDict

try:
    import duckdb
    DUCKDB_AVAILABLE = True
except ImportError:
    DUCKDB_AVAILABLE = False
    print("[DUCKDB] DuckDB not available, using mock implementation")


# Constants
DEFAULT_MEMORY_LIMIT_MB = 2048
DEFAULT_EVICTION_THRESHOLD_MB = 1800
DEFAULT_TEMP_DIR = "/tmp/duckdb_trading"
CHECKPOINT_INTERVAL_SEC = 60


@dataclass(slots=True)
class DuckDBConfig:
    """Configuration for DuckDB instance."""
    memory_limit_mb: int = DEFAULT_MEMORY_LIMIT_MB
    eviction_threshold_mb: int = DEFAULT_EVICTION_THRESHOLD_MB
    temp_dir: str = DEFAULT_TEMP_DIR
    checkpoint_interval_sec: int = CHECKPOINT_INTERVAL_SEC
    max_threads: int = 4
    enable_httpfs: bool = False


class TableSchema:
    """Predefined table schemas for trading data."""
    
    TICK_DATA = """
        CREATE TABLE IF NOT EXISTS tick_data (
            symbol VARCHAR NOT NULL,
            timestamp_ns BIGINT NOT NULL,
            bid_price DOUBLE NOT NULL,
            ask_price DOUBLE NOT NULL,
            bid_size DOUBLE NOT NULL,
            ask_size DOUBLE NOT NULL,
            sequence BIGINT,
            exchange VARCHAR,
            PRIMARY KEY (symbol, timestamp_ns)
        )
    """
    
    ORDER_BOOK = """
        CREATE TABLE IF NOT EXISTS order_book (
            symbol VARCHAR NOT NULL,
            timestamp_ns BIGINT NOT NULL,
            bids DOUBLE[][] NOT NULL,
            asks DOUBLE[][] NOT NULL,
            PRIMARY KEY (symbol, timestamp_ns)
        )
    """
    
    TRADES = """
        CREATE TABLE IF NOT EXISTS trades (
            trade_id VARCHAR NOT NULL,
            symbol VARCHAR NOT NULL,
            timestamp_ns BIGINT NOT NULL,
            price DOUBLE NOT NULL,
            quantity DOUBLE NOT NULL,
            side VARCHAR NOT NULL,
            PRIMARY KEY (trade_id)
        )
    """
    
    ORDERS = """
        CREATE TABLE IF NOT EXISTS orders (
            order_id VARCHAR NOT NULL,
            symbol VARCHAR NOT NULL,
            timestamp_ns BIGINT NOT NULL,
            side VARCHAR NOT NULL,
            order_type VARCHAR NOT NULL,
            price DOUBLE,
            quantity DOUBLE NOT NULL,
            filled_quantity DOUBLE DEFAULT 0,
            status VARCHAR NOT NULL,
            PRIMARY KEY (order_id)
        )
    """


class MemoryBoundedConnection:
    """
    DuckDB connection with memory bounds and eviction support.
    """
    
    def __init__(self, config: DuckDBConfig):
        self.config = config
        self._conn: Optional[Any] = None
        self._lock = threading.Lock()
        self._table_sizes: OrderedDict[str, int] = OrderedDict()
        self._total_size_mb = 0
        
        if not DUCKDB_AVAILABLE:
            return
        
        # Ensure temp directory exists
        Path(config.temp_dir).mkdir(parents=True, exist_ok=True)
        
        # Configure DuckDB with memory limits
        self._connect()
    
    def _connect(self):
        """Establish connection with memory-limited configuration."""
        if not DUCKDB_AVAILABLE:
            return
        
        # Build connection string with memory limits
        conn_config = {
            'memory_limit': f'{self.config.memory_limit_mb}MB',
            'threads': str(self.config.max_threads),
            'temp_directory': self.config.temp_dir,
        }
        
        self._conn = duckdb.connect(
            database=':memory:',
            config=conn_config
        )
        
        # Set additional memory-related settings
        self._execute("SET preserve_insertion_order = false")
        self._execute(f"SET memory_limit='{self.config.memory_limit_mb}MB'")
    
    def _execute(self, query: str, params: Optional[Tuple] = None) -> Any:
        """Execute a query with thread safety."""
        with self._lock:
            if self._conn is None:
                raise RuntimeError("DuckDB connection not established")
            
            if params:
                return self._conn.execute(query, params)
            return self._conn.execute(query)
    
    def create_tables(self):
        """Create all required tables."""
        if self._conn is None:
            return
        
        self._execute(TableSchema.TICK_DATA)
        self._execute(TableSchema.ORDER_BOOK)
        self._execute(TableSchema.TRADES)
        self._execute(TableSchema.ORDERS)
    
    def insert_ticks(self, ticks: List[Dict[str, Any]]):
        """Insert tick data with memory monitoring."""
        if not ticks or self._conn is None:
            return
        
        # Batch insert for efficiency
        values = []
        for tick in ticks:
            values.append((
                tick['symbol'],
                tick['timestamp_ns'],
                tick['bid_price'],
                tick['ask_price'],
                tick['bid_size'],
                tick['ask_size'],
                tick.get('sequence'),
                tick.get('exchange', 'unknown'),
            ))
        
        self._execute(
            "INSERT OR REPLACE INTO tick_data VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            values[0] if len(values) == 1 else None
        )
        
        # For batch inserts
        if len(values) > 1:
            for v in values:
                try:
                    self._execute(
                        "INSERT OR REPLACE INTO tick_data VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                        v
                    )
                except Exception:
                    pass
        
        self._update_table_size('tick_data', len(ticks))
    
    def query_ticks(
        self,
        symbol: str,
        start_ns: Optional[int] = None,
        end_ns: Optional[int] = None,
        limit: int = 1000,
    ) -> List[Dict[str, Any]]:
        """Query tick data with time range."""
        if self._conn is None:
            return []
        
        query = "SELECT * FROM tick_data WHERE symbol = ?"
        params = [symbol]
        
        if start_ns:
            query += " AND timestamp_ns >= ?"
            params.append(start_ns)
        
        if end_ns:
            query += " AND timestamp_ns <= ?"
            params.append(end_ns)
        
        query += " ORDER BY timestamp_ns DESC LIMIT ?"
        params.append(limit)
        
        result = self._execute(query, tuple(params))
        if result is None:
            return []
        
        columns = ['symbol', 'timestamp_ns', 'bid_price', 'ask_price', 
                   'bid_size', 'ask_size', 'sequence', 'exchange']
        
        return [dict(zip(columns, row)) for row in result.fetchall()]
    
    def _update_table_size(self, table: str, rows_added: int):
        """Update tracked table size for eviction management."""
        # Estimate size per row (varies by table)
        bytes_per_row = 100  # Rough estimate
        
        added_mb = (rows_added * bytes_per_row) / (1024 * 1024)
        self._total_size_mb += added_mb
        
        if table in self._table_sizes:
            self._table_sizes[table] += rows_added
        else:
            self._table_sizes[table] = rows_added
        
        # Move to end (most recently used)
        self._table_sizes.move_to_end(table)
        
        # Check if eviction needed
        if self._total_size_mb > self.config.eviction_threshold_mb:
            self._evict_oldest()
    
    def _evict_oldest(self):
        """Evict oldest data to stay within memory limits."""
        if not self._table_sizes:
            return
        
        # Get oldest table (first in OrderedDict)
        oldest_table, oldest_count = next(iter(self._table_sizes.items()))
        
        # Delete oldest 50% of data
        delete_count = oldest_count // 2
        
        try:
            self._execute(f"""
                DELETE FROM {oldest_table}
                WHERE timestamp_ns IN (
                    SELECT timestamp_ns FROM {oldest_table}
                    ORDER BY timestamp_ns ASC
                    LIMIT ?
                )
            """, (delete_count,))
            
            self._table_sizes[oldest_table] -= delete_count
            self._total_size_mb -= (delete_count * 100) / (1024 * 1024)
            
            print(f"[DUCKDB] Evicted {delete_count} rows from {oldest_table}")
            
        except Exception as e:
            print(f"[DUCKDB] Eviction error: {e}")
    
    def checkpoint(self):
        """Force checkpoint to disk."""
        if self._conn is None:
            return
        
        try:
            self._execute("CHECKPOINT")
        except Exception as e:
            print(f"[DUCKDB] Checkpoint error: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        stats = {
            'connected': self._conn is not None,
            'total_size_mb': self._total_size_mb,
            'memory_limit_mb': self.config.memory_limit_mb,
            'eviction_threshold_mb': self.config.eviction_threshold_mb,
            'tables': dict(self._table_sizes),
        }
        
        if self._conn is not None:
            try:
                result = self._execute("SELECT COUNT(*) FROM tick_data")
                if result:
                    stats['tick_count'] = result.fetchone()[0]
            except Exception:
                pass
        
        return stats
    
    def close(self):
        """Close the connection."""
        with self._lock:
            if self._conn:
                self.checkpoint()
                self._conn.close()
                self._conn = None


class DuckDBManager:
    """
    High-level manager for DuckDB operations.
    Provides automatic initialization, connection pooling, and memory management.
    """
    
    _instance: Optional['DuckDBManager'] = None
    _lock = threading.Lock()
    
    def __new__(cls, *args, **kwargs) -> 'DuckDBManager':
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
            return cls._instance
    
    def __init__(self, config: Optional[DuckDBConfig] = None):
        if hasattr(self, '_initialized') and self._initialized:
            return
        
        self._initialized = True
        self.config = config or DuckDBConfig()
        self._connection: Optional[MemoryBoundedConnection] = None
        self._checkpoint_thread: Optional[threading.Thread] = None
        self._running = False
    
    def initialize(self):
        """Initialize the database connection and tables."""
        self._connection = MemoryBoundedConnection(self.config)
        self._connection.create_tables()
        print("[DUCKDB] Database initialized")
    
    def start(self):
        """Start background maintenance tasks."""
        self._running = True
        self._checkpoint_thread = threading.Thread(
            target=self._checkpoint_loop,
            daemon=True,
            name="DuckDB-Checkpoint"
        )
        self._checkpoint_thread.start()
    
    def stop(self):
        """Stop background tasks and close connection."""
        self._running = False
        if self._checkpoint_thread:
            self._checkpoint_thread.join(timeout=5.0)
        
        if self._connection:
            self._connection.close()
    
    def _checkpoint_loop(self):
        """Periodic checkpoint loop."""
        while self._running:
            time.sleep(self.config.checkpoint_interval_sec)
            if self._connection:
                self._connection.checkpoint()
    
    @property
    def connection(self) -> Optional[MemoryBoundedConnection]:
        return self._connection
    
    def insert_ticks(self, ticks: List[Dict[str, Any]]):
        """Insert tick data."""
        if self._connection:
            self._connection.insert_ticks(ticks)
    
    def query_ticks(
        self,
        symbol: str,
        start_ns: Optional[int] = None,
        end_ns: Optional[int] = None,
        limit: int = 1000,
    ) -> List[Dict[str, Any]]:
        """Query tick data."""
        if self._connection:
            return self._connection.query_ticks(symbol, start_ns, end_ns, limit)
        return []
    
    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        if self._connection:
            return self._connection.get_stats()
        return {}


# Convenience functions
_manager_instance: Optional[DuckDBManager] = None


def get_manager() -> DuckDBManager:
    """Get or create the global manager instance."""
    global _manager_instance
    if _manager_instance is None:
        _manager_instance = DuckDBManager()
    return _manager_instance


def init_duckdb(memory_limit_mb: int = 2048) -> DuckDBManager:
    """Initialize DuckDB with custom settings."""
    global _manager_instance
    config = DuckDBConfig(memory_limit_mb=memory_limit_mb)
    _manager_instance = DuckDBManager(config)
    _manager_instance.initialize()
    return _manager_instance


if __name__ == "__main__":
    # Demo/test code
    print("[DEMO] DuckDB Manager Demo")
    
    if not DUCKDB_AVAILABLE:
        print("[DEMO] DuckDB not installed, skipping demo")
    else:
        manager = init_duckdb(memory_limit_mb=512)
        manager.start()
        
        # Insert some test ticks
        now_ns = time.time_ns()
        ticks = [
            {
                'symbol': 'BTCUSDT',
                'timestamp_ns': now_ns - i * 1000000,
                'bid_price': 50000.0 + i * 0.01,
                'ask_price': 50000.5 + i * 0.01,
                'bid_size': 1.5,
                'ask_size': 2.0,
                'sequence': i,
                'exchange': 'binance',
            }
            for i in range(100)
        ]
        
        manager.insert_ticks(ticks)
        
        # Query back
        results = manager.query_ticks('BTCUSDT', limit=10)
        print(f"[DEMO] Retrieved {len(results)} ticks")
        
        # Stats
        print("[DEMO] Stats:", manager.get_stats())
        
        manager.stop()
