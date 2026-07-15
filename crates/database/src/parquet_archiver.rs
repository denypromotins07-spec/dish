//! Parquet Archiver with Async I/O
//! Background thread flushing ring buffers to compressed Parquet files
//! Strict 500MB RAM ceiling enforcement

use arrow::array::{Float64Array, Int64Array, StringArray};
use arrow::datatypes::{DataType, Field, Schema, TimeUnit};
use arrow::record_batch::RecordBatch;
use parquet::arrow::ArrowWriter;
use parquet::basic::{Compression, Encoding};
use parquet::file::properties::WriterProperties;
use std::fs::File;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::mpsc;
use tokio::task::spawn_blocking;

/// Ring buffer entry for tick data
#[derive(Clone, Debug)]
pub struct TickEntry {
    pub timestamp_ns: i64,
    pub symbol: String,
    pub exchange: String,
    pub price: f64,
    pub volume: f64,
}

/// Ring buffer entry for orderbook snapshot
#[derive(Clone, Debug)]
pub struct OrderbookEntry {
    pub timestamp_ns: i64,
    pub symbol: String,
    pub exchange: String,
    pub bid_price: f64,
    pub ask_price: f64,
    pub bid_volume: f64,
    pub ask_volume: f64,
}

/// Unified archive entry
#[derive(Clone, Debug)]
pub enum ArchiveEntry {
    Tick(TickEntry),
    Orderbook(OrderbookEntry),
}

/// Memory-bounded ring buffer
pub struct BoundedRingBuffer<T> {
    buffer: Vec<Option<T>>,
    capacity: usize,
    head: usize,
    tail: usize,
    size: usize,
    max_memory_bytes: usize,
    current_memory_bytes: usize,
}

impl<T: Clone> BoundedRingBuffer<T> {
    pub fn new(capacity: usize, max_memory_bytes: usize) -> Self {
        Self {
            buffer: vec![None; capacity],
            capacity,
            head: 0,
            tail: 0,
            size: 0,
            max_memory_bytes,
            current_memory_bytes: 0,
        }
    }

    #[inline]
    pub fn push(&mut self, item: T, estimated_size: usize) -> bool {
        // Check memory limit before pushing
        if self.current_memory_bytes + estimated_size > self.max_memory_bytes {
            // Drop oldest entries until we have space
            while self.current_memory_bytes + estimated_size > self.max_memory_bytes && self.size > 0 {
                self.pop();
            }
        }

        if self.size == self.capacity {
            // Buffer full, drop oldest
            self.pop();
        }

        self.buffer[self.tail] = Some(item);
        self.tail = (self.tail + 1) % self.capacity;
        self.size += 1;
        self.current_memory_bytes += estimated_size;
        
        true
    }

    #[inline]
    pub fn pop(&mut self) -> Option<T> {
        if self.size == 0 {
            return None;
        }

        let item = self.buffer[self.head].take();
        if let Some(ref val) = item {
            // Rough estimate of memory freed
            self.current_memory_bytes = self.current_memory_bytes.saturating_sub(100);
        }
        
        self.head = (self.head + 1) % self.capacity;
        self.size -= 1;
        
        item
    }

    #[inline]
    pub fn drain(&mut self) -> Vec<T> {
        let mut result = Vec::with_capacity(self.size);
        while let Some(item) = self.pop() {
            result.push(item);
        }
        result
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.size
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.size == 0
    }

    #[inline]
    pub fn memory_usage(&self) -> usize {
        self.current_memory_bytes
    }
}

/// Parquet archiver configuration
pub struct ArchiverConfig {
    pub output_dir: PathBuf,
    pub flush_interval: Duration,
    pub max_rows_per_file: usize,
    pub compression: Compression,
    pub max_memory_mb: usize,
}

impl Default for ArchiverConfig {
    fn default() -> Self {
        Self {
            output_dir: PathBuf::from("./parquet_archive"),
            flush_interval: Duration::from_secs(60),
            max_rows_per_file: 1_000_000,
            compression: Compression::ZSTD(Default::default()),
            max_memory_mb: 500, // Strict 500MB ceiling
        }
    }
}

/// Parquet archiver with async background flushing
pub struct ParquetArchiver {
    config: ArchiverConfig,
    tick_buffer: BoundedRingBuffer<TickEntry>,
    orderbook_buffer: BoundedRingBuffer<OrderbookEntry>,
    rx: mpsc::Receiver<ArchiveEntry>,
    shutdown: bool,
    last_flush: Instant,
    rows_written: u64,
    files_created: u64,
}

impl ParquetArchiver {
    pub fn new(config: ArchiverConfig, rx: mpsc::Receiver<ArchiveEntry>) -> Self {
        // Calculate buffer capacities based on memory limit
        let max_memory_bytes = config.max_memory_mb * 1024 * 1024;
        let tick_buffer_size = max_memory_bytes / 2 / 100; // ~100 bytes per tick entry
        let orderbook_buffer_size = max_memory_bytes / 2 / 150; // ~150 bytes per orderbook entry

        Self {
            config,
            tick_buffer: BoundedRingBuffer::new(tick_buffer_size, max_memory_bytes / 2),
            orderbook_buffer: BoundedRingBuffer::new(orderbook_buffer_size, max_memory_bytes / 2),
            rx,
            shutdown: false,
            last_flush: Instant::now(),
            rows_written: 0,
            files_created: 0,
        }
    }

    pub async fn run(mut self) -> Result<(), Box<dyn std::error::Error>> {
        // Ensure output directory exists
        tokio::fs::create_dir_all(&self.config.output_dir).await?;

        while !self.shutdown {
            tokio::select! {
                msg = self.rx.recv() => {
                    match msg {
                        Some(entry) => self.process_entry(entry),
                        None => {
                            // Channel closed, shutdown gracefully
                            self.flush_all().await?;
                            break;
                        }
                    }
                }
                _ = tokio::time::sleep(Duration::from_millis(100)) => {
                    // Periodic flush check
                    if self.last_flush.elapsed() >= self.config.flush_interval {
                        self.maybe_flush().await?;
                    }
                }
            }
        }

        Ok(())
    }

    #[inline]
    fn process_entry(&mut self, entry: ArchiveEntry) {
        match entry {
            ArchiveEntry::Tick(tick) => {
                self.tick_buffer.push(tick, 100);
            }
            ArchiveEntry::Orderbook(ob) => {
                self.orderbook_buffer.push(ob, 150);
            }
        }
    }

    async fn maybe_flush(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        // Flush if buffers are getting full or interval exceeded
        if self.tick_buffer.len() > self.tick_buffer.capacity / 2
            || self.orderbook_buffer.len() > self.orderbook_buffer.capacity / 2
            || self.last_flush.elapsed() >= self.config.flush_interval
        {
            self.flush_all().await?;
        }
        Ok(())
    }

    async fn flush_all(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        let ticks = self.tick_buffer.drain();
        let orderbooks = self.orderbook_buffer.drain();

        if !ticks.is_empty() {
            self.write_ticks_parquet(ticks).await?;
        }

        if !orderbooks.is_empty() {
            self.write_orderbooks_parquet(orderbooks).await?;
        }

        self.last_flush = Instant::now();
        Ok(())
    }

    async fn write_ticks_parquet(&mut self, ticks: Vec<TickEntry>) -> Result<(), Box<dyn std::error::Error>> {
        if ticks.is_empty() {
            return Ok(());
        }

        let path = self.generate_filename("ticks");
        let rows = ticks.len() as u64;

        // Spawn blocking task for file I/O
        let config = self.config.clone();
        spawn_blocking(move || {
            Self::write_ticks_to_file(&path, &ticks, &config)
        })
        .await??;

        self.rows_written += rows;
        self.files_created += 1;

        log::info!("Archived {} ticks to {:?}", rows, path);
        Ok(())
    }

    async fn write_orderbooks_parquet(&mut self, orderbooks: Vec<OrderbookEntry>) -> Result<(), Box<dyn std::error::Error>> {
        if orderbooks.is_empty() {
            return Ok(());
        }

        let path = self.generate_filename("orderbooks");
        let rows = orderbooks.len() as u64;

        let config = self.config.clone();
        spawn_blocking(move || {
            Self::write_orderbooks_to_file(&path, &orderbooks, &config)
        })
        .await??;

        self.rows_written += rows;
        self.files_created += 1;

        log::info!("Archived {} orderbooks to {:?}", rows, path);
        Ok(())
    }

    fn generate_filename(&self, prefix: &str) -> PathBuf {
        let timestamp = chrono::Utc::now().format("%Y%m%d_%H%M%S_%f");
        self.config.output_dir.join(format!("{}_{}.parquet", prefix, timestamp))
    }

    fn write_ticks_to_file(
        path: &Path,
        ticks: &[TickEntry],
        config: &ArchiverConfig,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let schema = Arc::new(Schema::new(vec![
            Field::new("timestamp_ns", DataType::Int64, false),
            Field::new("symbol", DataType::Utf8, false),
            Field::new("exchange", DataType::Utf8, false),
            Field::new("price", DataType::Float64, false),
            Field::new("volume", DataType::Float64, false),
        ]));

        let timestamps: Int64Array = ticks.iter().map(|t| t.timestamp_ns).collect();
        let symbols: StringArray = ticks.iter().map(|t| t.symbol.as_str()).collect();
        let exchanges: StringArray = ticks.iter().map(|t| t.exchange.as_str()).collect();
        let prices: Float64Array = ticks.iter().map(|t| t.price).collect();
        let volumes: Float64Array = ticks.iter().map(|t| t.volume).collect();

        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(timestamps),
                Arc::new(symbols),
                Arc::new(exchanges),
                Arc::new(prices),
                Arc::new(volumes),
            ],
        )?;

        let props = WriterProperties::builder()
            .set_compression(config.compression)
            .set_encoding(Encoding::PLAIN)
            .build();

        let file = File::create(path)?;
        let mut writer = ArrowWriter::try_new(file, schema, Some(props))?;
        writer.write(&batch)?;
        writer.close()?;

        Ok(())
    }

    fn write_orderbooks_to_file(
        path: &Path,
        orderbooks: &[OrderbookEntry],
        config: &ArchiverConfig,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let schema = Arc::new(Schema::new(vec![
            Field::new("timestamp_ns", DataType::Int64, false),
            Field::new("symbol", DataType::Utf8, false),
            Field::new("exchange", DataType::Utf8, false),
            Field::new("bid_price", DataType::Float64, false),
            Field::new("ask_price", DataType::Float64, false),
            Field::new("bid_volume", DataType::Float64, false),
            Field::new("ask_volume", DataType::Float64, false),
        ]));

        let timestamps: Int64Array = orderbooks.iter().map(|o| o.timestamp_ns).collect();
        let symbols: StringArray = orderbooks.iter().map(|o| o.symbol.as_str()).collect();
        let exchanges: StringArray = orderbooks.iter().map(|o| o.exchange.as_str()).collect();
        let bid_prices: Float64Array = orderbooks.iter().map(|o| o.bid_price).collect();
        let ask_prices: Float64Array = orderbooks.iter().map(|o| o.ask_price).collect();
        let bid_volumes: Float64Array = orderbooks.iter().map(|o| o.bid_volume).collect();
        let ask_volumes: Float64Array = orderbooks.iter().map(|o| o.ask_volume).collect();

        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(timestamps),
                Arc::new(symbols),
                Arc::new(exchanges),
                Arc::new(bid_prices),
                Arc::new(ask_prices),
                Arc::new(bid_volumes),
                Arc::new(ask_volumes),
            ],
        )?;

        let props = WriterProperties::builder()
            .set_compression(config.compression)
            .set_encoding(Encoding::PLAIN)
            .build();

        let file = File::create(path)?;
        let mut writer = ArrowWriter::try_new(file, schema, Some(props))?;
        writer.write(&batch)?;
        writer.close()?;

        Ok(())
    }

    pub fn get_stats(&self) -> (u64, u64, usize, usize) {
        (
            self.rows_written,
            self.files_created,
            self.tick_buffer.memory_usage(),
            self.orderbook_buffer.memory_usage(),
        )
    }

    pub fn shutdown(&mut self) {
        self.shutdown = true;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ring_buffer_memory_limit() {
        let mut buffer: BoundedRingBuffer<TickEntry> = BoundedRingBuffer::new(1000, 10000);
        
        for i in 0..100 {
            let entry = TickEntry {
                timestamp_ns: i,
                symbol: "BTCUSDT".to_string(),
                exchange: "binance".to_string(),
                price: 45000.0,
                volume: 1.0,
            };
            buffer.push(entry, 100);
        }

        assert!(buffer.memory_usage() <= 10000);
        assert!(buffer.len() < 100); // Should have dropped some entries
    }
}
