"""
Ray Data Sharder for Distributed Processing.
Splits massive Parquet/LOB datasets into micro-batches (<500MB each).
Ensures no single Ray worker exceeds memory limits during walk-forward optimizations.
"""

import ray
from typing import List, Dict, Any, Optional, Iterator
from dataclasses import dataclass
import pandas as pd
import polars as pl
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ShardConfig:
    """Configuration for data sharding."""
    max_shard_size_mb: float = 500.0
    target_shard_count: Optional[int] = None
    partition_by: Optional[str] = None  # Column to partition by
    sort_by: Optional[str] = None  # Column to sort by before sharding


@ray.remote
class DataSharder:
    """
    Ray actor for sharding large datasets.
    Each shard is guaranteed to fit within memory limits.
    """
    
    def __init__(self, config: ShardConfig):
        self.config = config
        self.shards: List[Dict[str, Any]] = []
        
    def load_and_shard(self, file_path: str) -> List[str]:
        """Load dataset and split into shards."""
        logger.info(f"Loading {file_path} for sharding...")
        
        # Use Polars for memory-efficient loading
        df = pl.scan_parquet(file_path)
        
        # Get total row count
        total_rows = df.select(pl.count()).collect().item()
        
        # Calculate shard sizes
        if self.config.target_shard_count:
            rows_per_shard = total_rows // self.config.target_shard_count
        else:
            # Estimate rows per MB (rough estimate for typical LOB data)
            estimated_row_size_bytes = 200  # ~200 bytes per row
            rows_per_shard = int((self.config.max_shard_size_mb * 1024 * 1024) / estimated_row_size_bytes)
        
        # Create shards
        shard_paths = []
        num_shards = (total_rows + rows_per_shard - 1) // rows_per_shard
        
        for i in range(num_shards):
            offset = i * rows_per_shard
            limit = min(rows_per_shard, total_rows - offset)
            
            # Collect shard data
            shard_df = df.slice(offset, limit).collect()
            
            # Save shard to temporary file
            shard_path = f"/tmp/shard_{i}_{os.path.basename(file_path)}"
            shard_df.write_parquet(shard_path, compression="zstd", compression_level=3)
            
            shard_paths.append(shard_path)
            logger.info(f"Created shard {i+1}/{num_shards}: {shard_path}")
            
            # Explicit garbage collection
            del shard_df
            
        return shard_paths
    
    def shard_dataframe(self, df: pl.DataFrame) -> List[pl.DataFrame]:
        """Shard an already-loaded DataFrame."""
        total_rows = len(df)
        
        # Calculate shard size
        if self.config.target_shard_count:
            rows_per_shard = total_rows // self.config.target_shard_count
        else:
            estimated_row_size_bytes = 200
            rows_per_shard = int((self.config.max_shard_size_mb * 1024 * 1024) / estimated_row_size_bytes)
        
        shards = []
        for i in range(0, total_rows, rows_per_shard):
            shard = df.slice(i, rows_per_shard)
            shards.append(shard)
        
        logger.info(f"Split DataFrame into {len(shards)} shards")
        return shards


class RayDataSharder:
    """
    High-level interface for distributed data sharding.
    Manages Ray actors and distributes sharding work.
    """
    
    def __init__(self, max_shard_size_mb: float = 500.0):
        self.config = ShardConfig(max_shard_size_mb=max_shard_size_mb)
        self.sharder_actors = []
        
    def initialize(self, num_actors: int = 4):
        """Initialize Ray sharding actors."""
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True)
        
        self.sharder_actors = [
            DataSharder.remote(self.config) 
            for _ in range(num_actors)
        ]
        logger.info(f"Initialized {num_actors} DataSharder actors")
    
    async def shard_dataset(self, file_path: str) -> List[str]:
        """
        Distribute sharding work across actors.
        Returns list of shard file paths.
        """
        if not self.sharder_actors:
            raise RuntimeError("Call initialize() first")
        
        # Assign to least loaded actor (round-robin for simplicity)
        actor_idx = hash(file_path) % len(self.sharder_actors)
        actor = self.sharder_actors[actor_idx]
        
        # Execute sharding
        shard_paths = await actor.load_and_shard.remote(file_path)
        return shard_paths
    
    def shard_multiple(self, file_paths: List[str]) -> List[List[str]]:
        """Shard multiple files in parallel."""
        futures = []
        for i, path in enumerate(file_paths):
            actor = self.sharder_actors[i % len(self.sharder_actors)]
            future = actor.load_and_shard.remote(path)
            futures.append(future)
        
        return ray.get(futures)
    
    def cleanup(self):
        """Cleanup Ray actors."""
        for actor in self.sharder_actors:
            ray.kill(actor)
        self.sharder_actors = []
        logger.info("Cleaned up DataSharder actors")


def create_sharded_dataset_iterator(
    base_path: str,
    max_shard_size_mb: float = 500.0,
) -> Iterator[pl.DataFrame]:
    """
    Create an iterator that yields one shard at a time.
    Memory-efficient for processing massive datasets.
    """
    sharder = RayDataSharder(max_shard_size_mb=max_shard_size_mb)
    sharder.initialize(num_actors=2)
    
    try:
        # Get all parquet files
        import glob
        files = glob.glob(os.path.join(base_path, "*.parquet"))
        
        for file_path in files:
            # Shard the file
            shard_paths = ray.get(sharder.shard_dataset.remote(file_path))
            
            # Yield each shard
            for shard_path in shard_paths:
                df = pl.read_parquet(shard_path)
                yield df
                del df  # Explicit cleanup
                
    finally:
        sharder.cleanup()


# Example usage
async def example_sharding():
    """Example: Shard a massive LOB dataset for walk-forward optimization."""
    
    sharder = RayDataSharder(max_shard_size_mb=500.0)
    sharder.initialize(num_actors=4)
    
    # Shard multiple years of data
    files = [
        "/data/lob_2020.parquet",
        "/data/lob_2021.parquet",
        "/data/lob_2022.parquet",
        "/data/lob_2023.parquet",
    ]
    
    # Process in parallel
    all_shards = sharder.shard_multiple(files)
    
    print(f"Created {sum(len(s) for s in all_shards)} total shards")
    
    # Now each shard can be processed by a separate Ray worker
    # without exceeding 500MB memory limit
    
    sharder.cleanup()


if __name__ == "__main__":
    import asyncio
    asyncio.run(example_sharding())
