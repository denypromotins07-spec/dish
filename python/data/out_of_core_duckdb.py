"""
Out-of-Core Data Processing with DuckDB for Windows

This module configures DuckDB to use NVMe SSD for spilling and processing
massive historical datasets without exceeding RAM limits.

Key Features:
- Streaming SQL queries that never load more than 200MB into RAM
- Automatic spilling to SSD when memory threshold is reached
- Optimized for Windows NVMe SSD performance
- Replaces heavy in-memory Pandas/Ray processing

Target: Process GBs/TBs of tick data while staying under 200MB RAM usage
"""

import os
import logging
from pathlib import Path
from typing import Optional, Generator, Any, Dict, List
from contextlib import contextmanager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class OutOfCoreDuckDBConfig:
    """Configuration for out-of-core DuckDB processing"""
    
    # Memory limit in bytes (200MB strict limit)
    DEFAULT_MEMORY_LIMIT = 200 * 1024 * 1024
    
    # Temp directory on NVMe SSD for spilling
    DEFAULT_TEMP_DIR = r"C:\crypto_bot\data\duckdb_temp"
    
    def __init__(
        self,
        memory_limit_bytes: int = DEFAULT_MEMORY_LIMIT,
        temp_directory: str = DEFAULT_TEMP_DIR,
        threads: int = 8,  # Leave cores for Rust/Python
        enable_compression: bool = True,
    ):
        self.memory_limit = memory_limit_bytes
        self.temp_dir = Path(temp_directory)
        self.threads = threads
        self.enable_compression = enable_compression
    
    def to_duckdb_settings(self) -> Dict[str, str]:
        """Convert to DuckDB configuration parameters"""
        return {
            "memory_limit": f"{self.memory_limit // (1024 * 1024)}MB",
            "temp_directory": str(self.temp_dir),
            "threads": str(self.threads),
            "enable_object_cache": "false",  # Disable object cache for lower RAM
            "autoinstall_known_extensions": "false",  # Prevent extension downloads
        }


@contextmanager
def get_duckdb_connection(config: Optional[OutOfCoreDuckDBConfig] = None):
    """
    Context manager for DuckDB connection with out-of-core settings.
    
    Args:
        config: Configuration object (uses defaults if None)
    
    Yields:
        Configured DuckDB connection
    """
    import duckdb
    
    if config is None:
        config = OutOfCoreDuckDBConfig()
    
    # Ensure temp directory exists on NVMe SSD
    config.temp_dir.mkdir(parents=True, exist_ok=True)
    
    conn = duckdb.connect(database=":memory:")
    
    # Apply memory-limiting settings
    settings = config.to_duckdb_settings()
    for key, value in settings.items():
        conn.execute(f"SET {key}='{value}'")
        logger.debug(f"DuckDB: SET {key}={value}")
    
    logger.info(
        f"DuckDB configured: {config.memory_limit // (1024*1024)}MB limit, "
        f"temp dir: {config.temp_dir}"
    )
    
    try:
        yield conn
    finally:
        conn.close()


class StreamingDataProcessor:
    """
    Processes massive datasets using streaming SQL queries.
    
    Never loads more than 200MB into RAM at a time by using:
    - Chunked reading from Parquet/CSV
    - SQL-based filtering before loading
    - Generator-based result iteration
    """
    
    def __init__(self, config: Optional[OutOfCoreDuckDBConfig] = None):
        self.config = config or OutOfCoreDuckDBConfig()
        self._chunk_size = 100000  # Rows per chunk
    
    def process_parquet_streaming(
        self,
        parquet_path: str,
        sql_query: str,
        batch_size: int = 100000,
    ) -> Generator[List[tuple], None, None]:
        """
        Process a Parquet file using streaming SQL.
        
        Args:
            parquet_path: Path to input Parquet file
            sql_query: SQL query to execute (use 'read_parquet' function)
            batch_size: Number of rows per batch
        
        Yields:
            Batches of result rows
        """
        with get_duckdb_connection(self.config) as conn:
            # Use DuckDB's built-in Parquet reader with streaming
            query = f"""
                SELECT * FROM (
                    {sql_query}
                ) LIMIT {batch_size} OFFSET ?
            """
            
            offset = 0
            while True:
                result = conn.execute(query, [offset]).fetchall()
                
                if not result:
                    break
                
                yield result
                offset += batch_size
                
                logger.debug(f"Processed {offset} rows from {parquet_path}")
    
    def aggregate_large_dataset(
        self,
        parquet_paths: List[str],
        aggregation_query: str,
    ) -> List[tuple]:
        """
        Perform aggregation on large datasets without loading all data.
        
        Args:
            parquet_paths: List of Parquet file paths
            aggregation_query: SQL aggregation query
        
        Returns:
            Aggregation results
        """
        with get_duckdb_connection(self.config) as conn:
            # Create view over all parquet files
            files_str = ", ".join(f"'{p}'" for p in parquet_paths)
            
            create_view = f"""
                CREATE VIEW tick_data AS
                SELECT * FROM read_parquet([{files_str}])
            """
            conn.execute(create_view)
            
            # Execute aggregation (DuckDB handles spilling automatically)
            result = conn.execute(aggregation_query).fetchall()
            
            logger.info(f"Aggregation complete: {len(result)} result rows")
            return result
    
    def join_historical_data(
        self,
        trades_path: str,
        orderbook_path: str,
        join_query: str,
    ) -> Generator[List[tuple], None, None]:
        """
        Join large historical datasets using streaming.
        
        Args:
            trades_path: Path to trades Parquet file
            orderbook_path: Path to order book Parquet file
            join_query: SQL join query
        
        Yields:
            Batches of joined rows
        """
        with get_duckdb_connection(self.config) as conn:
            # Create views for both datasets
            conn.execute(f"""
                CREATE VIEW trades AS
                SELECT * FROM read_parquet('{trades_path}')
            """)
            
            conn.execute(f"""
                CREATE VIEW orderbook AS
                SELECT * FROM read_parquet('{orderbook_path}')
            """)
            
            # Execute join with streaming results
            offset = 0
            while True:
                query = f"""
                    {join_query}
                    LIMIT {self._chunk_size} OFFSET {offset}
                """
                result = conn.execute(query).fetchall()
                
                if not result:
                    break
                
                yield result
                offset += self._chunk_size


class HistoricalDataLoader:
    """
    Loads historical tick data for backtesting with strict memory limits.
    
    Uses DuckDB's ability to query Parquet files directly without loading
    entire files into memory.
    """
    
    def __init__(self, data_directory: str, config: Optional[OutOfCoreDuckDBConfig] = None):
        self.data_dir = Path(data_directory)
        self.config = config or OutOfCoreDuckDBConfig()
    
    def get_available_symbols(self) -> List[str]:
        """Get list of available trading symbols"""
        symbols = set()
        
        for parquet_file in self.data_dir.glob("**/*.parquet"):
            # Extract symbol from filename (e.g., BTCUSDT_2024.parquet)
            symbol = parquet_file.stem.split("_")[0]
            symbols.add(symbol)
        
        return sorted(list(symbols))
    
    def get_date_range(self, symbol: str) -> tuple:
        """
        Get the available date range for a symbol.
        
        Returns:
            Tuple of (min_date, max_date) as strings
        """
        with get_duckdb_connection(self.config) as conn:
            pattern = str(self.data_dir / f"{symbol}_*.parquet")
            
            query = f"""
                SELECT 
                    MIN(timestamp) as min_ts,
                    MAX(timestamp) as max_ts
                FROM read_parquet('{pattern}')
            """
            
            result = conn.execute(query).fetchone()
            return (str(result[0]), str(result[1])) if result else (None, None)
    
    def stream_ticks(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        columns: Optional[List[str]] = None,
    ) -> Generator[Dict[str, Any], None, None]:
        """
        Stream tick data for a specific symbol and date range.
        
        Args:
            symbol: Trading symbol (e.g., 'BTCUSDT')
            start_date: Start date (ISO format)
            end_date: End date (ISO format)
            columns: Specific columns to select (None for all)
        
        Yields:
            Tick data as dictionaries
        """
        pattern = str(self.data_dir / f"{symbol}_*.parquet")
        
        col_str = "*" if columns is None else ", ".join(columns)
        
        query = f"""
            SELECT {col_str}
            FROM read_parquet('{pattern}')
            WHERE timestamp >= '{start_date}'
              AND timestamp <= '{end_date}'
            ORDER BY timestamp
        """
        
        with get_duckdb_connection(self.config) as conn:
            result = conn.execute(query)
            
            # Get column names
            columns = [desc[0] for desc in result.description]
            
            # Stream rows as dictionaries
            while True:
                rows = result.fetchmany(10000)
                if not rows:
                    break
                
                for row in rows:
                    yield dict(zip(columns, row))


def optimize_duckdb_for_nvme(ssd_path: str) -> Dict[str, str]:
    """
    Generate optimal DuckDB settings for NVMe SSD performance.
    
    Args:
        ssd_path: Path to NVMe SSD directory
    
    Returns:
        Dictionary of DuckDB settings
    """
    return {
        # Memory settings
        "memory_limit": "200MB",
        "max_memory": "200MB",
        
        # Temp/spill settings (point to fast NVMe)
        "temp_directory": ssd_path,
        
        # Threading (leave cores for other processes)
        "threads": "8",
        
        # Disable features that increase RAM usage
        "enable_object_cache": "false",
        "enable_external_tables": "false",
        
        # Compression for smaller disk footprint
        "compression": "zstd",
        
        # Batch size for efficient I/O
        "batch_size": "2048",
    }


if __name__ == "__main__":
    # Example usage
    print("=== Out-of-Core DuckDB Demo ===\n")
    
    # Create configuration
    config = OutOfCoreDuckDBConfig(
        memory_limit_bytes=200 * 1024 * 1024,  # 200MB
        temp_directory=r"C:\crypto_bot\data\duckdb_temp",
    )
    
    print(f"Memory Limit: {config.memory_limit // (1024*1024)}MB")
    print(f"Temp Directory: {config.temp_dir}")
    print(f"Settings: {config.to_duckdb_settings()}")
    
    # Example: Create processor
    processor = StreamingDataProcessor(config)
    print("\nStreamingDataProcessor initialized")
    
    # Example: Get available symbols (would need actual data)
    # loader = HistoricalDataLoader(r"C:\crypto_bot\data\ticks", config)
    # symbols = loader.get_available_symbols()
    # print(f"Available symbols: {symbols}")
    
    print("\n=== Configuration Complete ===")
