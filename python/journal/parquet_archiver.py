# python/journal/parquet_archiver.py
"""
Asynchronous, low-footprint archiver for trade journal events.
Flushes Rust ring buffer data into compressed Parquet files.
Strict memory limit: active journal RAM footprint < 100MB.
"""

from __future__ import annotations
import asyncio
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, AsyncIterator, Dict, Any
from collections import deque
import time
import mmap


@dataclass
class ArchiverConfig:
    """Configuration for the parquet archiver."""
    # Target file size before rotation (MB)
    target_file_size_mb: int = 64
    # Maximum rows per file
    max_rows_per_file: int = 1_000_000
    # Flush interval in seconds
    flush_interval_seconds: float = 5.0
    # Compression codec
    compression: str = "zstd"
    # Compression level (1-9 for zstd)
    compression_level: int = 3
    # Output directory
    output_dir: str = "./journal_archive"
    # Maximum memory buffer size (MB) - strict limit
    max_buffer_memory_mb: float = 50.0
    # Row group size for parquet
    row_group_size: int = 100_000


@dataclass
class TradeEventBatch:
    """Batch of trade events ready for archival."""
    timestamps_ns: list[int]
    order_ids: list[int]
    stages: list[int]
    prices: list[float]
    quantities: list[float]
    strategy_ids: list[int]
    asset_ids: list[int]
    sides: list[int]
    venue_ids: list[int]
    stage_latencies_ns: list[int]
    sequences: list[int]
    flags: list[int]
    
    @property
    def row_count(self) -> int:
        return len(self.timestamps_ns)
    
    def to_pyarrow_table(self) -> pa.Table:
        """Convert to PyArrow table for parquet writing."""
        return pa.table({
            "timestamp_ns": pa.array(self.timestamps_ns, type=pa.int64()),
            "order_id": pa.array(self.order_ids, type=pa.int64()),
            "stage": pa.array(self.stages, type=pa.uint8()),
            "price": pa.array(self.prices, type=pa.float64()),
            "quantity": pa.array(self.quantities, type=pa.float64()),
            "strategy_id": pa.array(self.strategy_ids, type=pa.uint16()),
            "asset_id": pa.array(self.asset_ids, type=pa.uint16()),
            "side": pa.array(self.sides, type=pa.uint8()),
            "venue_id": pa.array(self.venue_ids, type=pa.uint8()),
            "stage_latency_ns": pa.array(self.stage_latencies_ns, type=pa.uint32()),
            "sequence": pa.array(self.sequences, type=pa.uint16()),
            "flags": pa.array(self.flags, type=pa.uint8()),
        })


class MemoryBoundedBuffer:
    """
    Memory-bounded buffer for accumulating events before flush.
    Enforces strict RAM limits to stay under 100MB total journal footprint.
    """
    
    def __init__(self, max_memory_mb: float = 50.0):
        """
        Initialize with memory limit.
        
        Args:
            max_memory_mb: Maximum memory to use in MB
        """
        self.max_memory_bytes = int(max_memory_mb * 1024 * 1024)
        self._batches: deque[TradeEventBatch] = deque()
        self._current_memory_bytes: int = 0
        self._total_rows: int = 0
    
    def add_batch(self, batch: TradeEventBatch) -> bool:
        """
        Add a batch to the buffer.
        
        Returns:
            True if successful, False if memory limit exceeded
        """
        # Estimate memory usage (~100 bytes per row estimate)
        estimated_bytes = batch.row_count * 100
        
        if self._current_memory_bytes + estimated_bytes > self.max_memory_bytes:
            return False
        
        self._batches.append(batch)
        self._current_memory_bytes += estimated_bytes
        self._total_rows += batch.row_count
        return True
    
    def drain_all(self) -> list[TradeEventBatch]:
        """Drain all batches and reset."""
        batches = list(self._batches)
        self._batches.clear()
        self._current_memory_bytes = 0
        self._total_rows = 0
        return batches
    
    @property
    def row_count(self) -> int:
        return self._total_rows
    
    @property
    def memory_usage_bytes(self) -> int:
        return self._current_memory_bytes
    
    def should_flush(self, min_rows: int = 100_000) -> bool:
        """Check if we should flush based on row count."""
        return self._total_rows >= min_rows


class ParquetArchiver:
    """
    Asynchronous parquet archiver for trade journal events.
    
    Features:
    - Memory-bounded buffering (< 100MB total)
    - Automatic file rotation
    - ZSTD compression for space efficiency
    - Async I/O for non-blocking operation
    - Partitioning by date for efficient queries
    """
    
    # PyArrow schema for trade events
    SCHEMA = pa.schema([
        ("timestamp_ns", pa.int64()),
        ("order_id", pa.int64()),
        ("stage", pa.uint8()),
        ("price", pa.float64()),
        ("quantity", pa.float64()),
        ("strategy_id", pa.uint16()),
        ("asset_id", pa.uint16()),
        ("side", pa.uint8()),
        ("venue_id", pa.uint8()),
        ("stage_latency_ns", pa.uint32()),
        ("sequence", pa.uint16()),
        ("flags", pa.uint8()),
        # Derived columns added during write
        ("timestamp_us", pa.int64()),
        ("date", pa.string()),
        ("hour", pa.uint8()),
    ])
    
    def __init__(self, config: Optional[ArchiverConfig] = None):
        """
        Initialize the archiver.
        
        Args:
            config: Archiver configuration
        """
        self.config = config or ArchiverConfig()
        self.buffer = MemoryBoundedBuffer(self.config.max_buffer_memory_mb)
        
        # File management
        self._current_file_index: int = 0
        self._current_file_rows: int = 0
        self._current_writer: Optional[pq.ParquetWriter] = None
        self._output_path: Path = Path(self.config.output_dir)
        
        # Statistics
        self._files_written: int = 0
        self._total_rows_archived: int = 0
        self._last_flush_time: float = 0
        
        # Async control
        self._running: bool = False
        self._flush_task: Optional[asyncio.Task] = None
    
    async def start(self) -> None:
        """Start the background flush task."""
        self._running = True
        self._output_path.mkdir(parents=True, exist_ok=True)
        self._flush_task = asyncio.create_task(self._flush_loop())
    
    async def stop(self) -> None:
        """Stop the archiver and flush remaining data."""
        self._running = False
        
        if self._flush_task:
            self._flush_task.cancel()
            try:
                await self._flush_task
            except asyncio.CancelledError:
                pass
        
        # Final flush
        await self._flush_to_parquet()
        
        # Close current writer
        if self._current_writer:
            self._current_writer.close()
            self._current_writer = None
    
    def ingest_batch(self, batch: TradeEventBatch) -> bool:
        """
        Ingest a batch of events from the Rust ring buffer.
        
        Args:
            batch: Trade event batch
            
        Returns:
            True if accepted, False if rejected (memory pressure)
        """
        return self.buffer.add_batch(batch)
    
    async def _flush_loop(self) -> None:
        """Background loop to periodically flush data."""
        while self._running:
            try:
                await asyncio.sleep(self.config.flush_interval_seconds)
                
                # Check if flush is needed
                now = time.time()
                should_flush = (
                    self.buffer.should_flush(self.config.max_rows_per_file // 10)
                    or (now - self._last_flush_time) > self.config.flush_interval_seconds
                )
                
                if should_flush and self.buffer.row_count > 0:
                    await self._flush_to_parquet()
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                # Log error but continue
                print(f"Flush loop error: {e}")
    
    async def _flush_to_parquet(self) -> None:
        """Flush buffered data to parquet files."""
        batches = self.buffer.drain_all()
        if not batches:
            return
        
        # Combine batches into single table
        tables = [batch.to_pyarrow_table() for batch in batches]
        combined = pa.concat_tables(tables)
        
        # Add derived columns
        timestamps_us = [ts // 1000 for ts in combined["timestamp_ns"].to_pylist()]
        dates = [
            time.strftime("%Y-%m-%d", time.gmtime(ts // 1_000_000_000))
            for ts in combined["timestamp_ns"].to_pylist()
        ]
        hours = [
            int(time.strftime("%H", time.gmtime(ts // 1_000_000_000)))
            for ts in combined["timestamp_ns"].to_pylist()
        ]
        
        combined = combined.append_column(
            "timestamp_us", pa.array(timestamps_us, type=pa.int64())
        )
        combined = combined.append_column(
            "date", pa.array(dates, type=pa.string())
        )
        combined = combined.append_column(
            "hour", pa.array(hours, type=pa.uint8())
        )
        
        # Write with partitioning by date
        await self._write_table(combined)
        
        self._last_flush_time = time.time()
    
    async def _write_table(self, table: pa.Table) -> None:
        """Write table to parquet with rotation."""
        # Use run_in_executor for blocking I/O
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._write_table_sync, table)
    
    def _write_table_sync(self, table: pa.Table) -> None:
        """Synchronous parquet write."""
        n_rows = table.num_rows
        
        # Determine partition path
        if "date" in table.column_names:
            dates = table["date"].to_pylist()
            if dates:
                # Use first date for partition (batches should be same day)
                date_partition = dates[0]
                partition_path = self._output_path / f"date={date_partition}"
                partition_path.mkdir(parents=True, exist_ok=True)
            else:
                partition_path = self._output_path
        else:
            partition_path = self._output_path
        
        # Generate filename
        timestamp = int(time.time() * 1000)
        filename = f"trades_{timestamp}_{self._current_file_index:06d}.parquet"
        filepath = partition_path / filename
        
        # Write settings
        write_options = pq.ParquetWriter(
            filepath,
            table.schema,
            compression=self.config.compression,
            compression_level=self.config.compression_level,
            row_group_size=self.config.row_group_size,
            use_dictionary=True,
            write_statistics=True,
        )
        
        try:
            write_options.write_table(table)
        finally:
            write_options.close()
        
        self._files_written += 1
        self._total_rows_archived += n_rows
        self._current_file_rows += n_rows
        
        # Rotate file if needed
        if self._current_file_rows >= self.config.max_rows_per_file:
            self._current_file_index += 1
            self._current_file_rows = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get archiver statistics."""
        return {
            "files_written": self._files_written,
            "total_rows_archived": self._total_rows_archived,
            "buffer_rows": self.buffer.row_count,
            "buffer_memory_bytes": self.buffer.memory_usage_bytes,
            "max_buffer_memory_bytes": self.buffer.max_memory_bytes,
            "is_running": self._running,
        }


class SharedMemoryReader:
    """
    Reader for shared memory ring buffer exported by Rust.
    Uses memory-mapped files for zero-copy access.
    """
    
    def __init__(self, shm_path: str, event_size: int = 64):
        """
        Initialize shared memory reader.
        
        Args:
            shm_path: Path to shared memory file
            event_size: Size of each event in bytes
        """
        self.shm_path = Path(shm_path)
        self.event_size = event_size
        self._mm: Optional[mmap.mmap] = None
        self._last_read_pos: int = 0
    
    def open(self) -> bool:
        """Open the shared memory file."""
        if not self.shm_path.exists():
            return False
        
        try:
            fd = open(self.shm_path, "r+b")
            self._mm = mmap.mmap(fd.fileno(), 0, mmap.MAP_SHARED)
            return True
        except Exception:
            return False
    
    def close(self) -> None:
        """Close the shared memory mapping."""
        if self._mm:
            self._mm.close()
            self._mm = None
    
    def read_events(self, max_events: int = 10000) -> list[bytes]:
        """Read events from shared memory."""
        if not self._mm:
            return []
        
        events = []
        pos = self._last_read_pos
        
        for _ in range(max_events):
            if pos + self.event_size > len(self._mm):
                break
            
            event_data = self._mm[pos:pos + self.event_size]
            
            # Check if this is a valid event (non-zero timestamp)
            timestamp = int.from_bytes(event_data[:8], "little")
            if timestamp == 0:
                break
            
            events.append(event_data)
            pos += self.event_size
        
        self._last_read_pos = pos
        return events


async def main():
    """Example usage."""
    config = ArchiverConfig(
        target_file_size_mb=64,
        max_rows_per_file=500_000,
        flush_interval_seconds=2.0,
        output_dir="./test_journal_archive",
    )
    
    archiver = ParquetArchiver(config)
    await archiver.start()
    
    try:
        # Simulate ingesting batches
        for i in range(10):
            batch = TradeEventBatch(
                timestamps_ns=[time.time_ns() + j for j in range(1000)],
                order_ids=list(range(i * 1000, (i + 1) * 1000)),
                stages=[0] * 1000,
                prices=[100.0 + j * 0.01 for j in range(1000)],
                quantities=[1.0] * 1000,
                strategy_ids=[1] * 1000,
                asset_ids=[42] * 1000,
                sides=[0] * 1000,
                venue_ids=[1] * 1000,
                stage_latencies_ns=[1000] * 1000,
                sequences=list(range(1000)),
                flags=[0] * 1000,
            )
            
            if not archiver.ingest_batch(batch):
                print(f"Batch {i} rejected - memory pressure")
            
            await asyncio.sleep(0.1)
        
        # Wait for flushes
        await asyncio.sleep(5)
        
        print(f"Stats: {archiver.get_stats()}")
        
    finally:
        await archiver.stop()


if __name__ == "__main__":
    asyncio.run(main())
