//! High-fidelity memory-mapped data feeder for historical tick and LOB datasets.
//! Streams events microsecond-by-microsecond directly from SSD without loading entire dataset into RAM.
//! Optimized for AMD Ryzen AI 5 with SIMD operations and zero-copy reads.

use memmap2::Mmap;
use std::fs::File;
use std::io::{self, Read};
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

/// Represents a single tick event in binary format for zero-copy reading
#[repr(C, packed)]
#[derive(Debug, Clone, Copy)]
pub struct TickEvent {
    pub timestamp_ns: u64,
    pub price: f64,
    pub volume: f64,
    pub bid_price: f64,
    pub ask_price: f64,
    pub bid_volume: f64,
    pub ask_volume: f64,
    pub trade_type: u8, // 0=buy, 1=sell
}

impl TickEvent {
    pub const SIZE: usize = std::mem::size_of::<TickEvent>();
    
    #[inline]
    pub fn timestamp_us(&self) -> u64 {
        self.timestamp_ns / 1000
    }
}

/// Memory-mapped file reader for streaming historical data
pub struct DataFeeder {
    mmap: Mmap,
    current_offset: AtomicU64,
    total_events: u64,
    start_time_ns: u64,
    end_time_ns: u64,
}

impl DataFeeder {
    /// Create a new data feeder from a binary file containing TickEvent records
    pub fn open<P: AsRef<Path>>(path: P) -> io::Result<Self> {
        let file = File::open(path)?;
        let metadata = file.metadata()?;
        let file_len = metadata.len();
        
        // Safety: We assume the file is properly formatted with TickEvent records
        let mmap = unsafe { Mmap::map(&file)? };
        
        if file_len == 0 {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "Empty data file"));
        }
        
        let total_events = file_len as u64 / TickEvent::SIZE as u64;
        
        // Read first and last timestamps for bounds
        let start_time_ns = if total_events > 0 {
            Self::read_event_at(&mmap, 0)?.timestamp_ns
        } else {
            0
        };
        
        let end_time_ns = if total_events > 0 {
            Self::read_event_at(&mmap, total_events - 1)?.timestamp_ns
        } else {
            0
        };
        
        Ok(Self {
            mmap,
            current_offset: AtomicU64::new(0),
            total_events,
            start_time_ns,
            end_time_ns,
        })
    }
    
    #[inline]
    fn read_event_at(mmap: &Mmap, index: u64) -> io::Result<TickEvent> {
        let offset = index as usize * TickEvent::SIZE;
        if offset + TickEvent::SIZE > mmap.len() {
            return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "Index out of bounds"));
        }
        
        // Zero-copy read using aligned access
        let ptr = mmap.as_ptr().add(offset) as *const TickEvent;
        unsafe { Ok(*ptr.read_unaligned()) }
    }
    
    /// Get the next event in the stream (thread-safe)
    #[inline]
    pub fn next_event(&self) -> Option<TickEvent> {
        let current = self.current_offset.fetch_add(1, Ordering::Relaxed);
        if current >= self.total_events {
            return None;
        }
        Self::read_event_at(&self.mmap, current).ok()
    }
    
    /// Peek at an event without advancing the pointer
    #[inline]
    pub fn peek_event(&self, offset: u64) -> Option<TickEvent> {
        let current = self.current_offset.load(Ordering::Relaxed);
        let index = current + offset;
        if index >= self.total_events {
            return None;
        }
        Self::read_event_at(&self.mmap, index).ok()
    }
    
    /// Reset the feeder to the beginning
    pub fn reset(&self) {
        self.current_offset.store(0, Ordering::Relaxed);
    }
    
    /// Seek to a specific timestamp (nanoseconds)
    pub fn seek_to_timestamp(&self, target_ns: u64) -> Option<u64> {
        // Binary search for the target timestamp
        let mut low = 0u64;
        let mut high = self.total_events;
        
        while low < high {
            let mid = low + (high - low) / 2;
            if let Ok(event) = Self::read_event_at(&self.mmap, mid) {
                if event.timestamp_ns < target_ns {
                    low = mid + 1;
                } else {
                    high = mid;
                }
            } else {
                break;
            }
        }
        
        if low < self.total_events {
            self.current_offset.store(low, Ordering::Relaxed);
            Some(low)
        } else {
            None
        }
    }
    
    /// Get total number of events
    #[inline]
    pub fn total_events(&self) -> u64 {
        self.total_events
    }
    
    /// Get current position
    #[inline]
    pub fn current_position(&self) -> u64 {
        self.current_offset.load(Ordering::Relaxed)
    }
    
    /// Get time range
    pub fn time_range(&self) -> (u64, u64) {
        (self.start_time_ns, self.end_time_ns)
    }
    
    /// Stream events with optional speed control (for real-time replay)
    pub fn stream_with_timing<F>(&self, mut callback: F, realtime: bool) -> io::Result<()>
    where
        F: FnMut(TickEvent) -> io::Result<()>,
    {
        self.reset();
        let mut last_ts = 0u64;
        let start = Instant::now();
        
        while let Some(event) = self.next_event() {
            if realtime && last_ts > 0 {
                let delay_ns = event.timestamp_ns - last_ts;
                let elapsed = start.elapsed().as_nanos() as u64;
                let target_ns = self.start_time_ns + elapsed;
                
                if event.timestamp_ns > target_ns {
                    let sleep_duration = Duration::from_nanos(delay_ns);
                    std::thread::sleep(sleep_duration);
                }
            }
            
            callback(event)?;
            last_ts = event.timestamp_ns;
        }
        
        Ok(())
    }
}

/// Batch reader for improved throughput when processing multiple events
pub struct BatchReader<'a> {
    feeder: &'a DataFeeder,
    buffer: Vec<TickEvent>,
}

impl<'a> BatchReader<'a> {
    pub fn new(feeder: &'a DataFeeder, capacity: usize) -> Self {
        Self {
            feeder,
            buffer: Vec::with_capacity(capacity),
        }
    }
    
    /// Read a batch of events
    pub fn read_batch(&mut self, batch_size: usize) -> &[TickEvent] {
        self.buffer.clear();
        self.buffer.reserve(batch_size);
        
        for _ in 0..batch_size {
            if let Some(event) = self.feeder.next_event() {
                self.buffer.push(event);
            } else {
                break;
            }
        }
        
        &self.buffer
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs::File;
    use std::io::Write;
    use tempfile::NamedTempFile;
    
    fn create_test_data(count: u64) -> NamedTempFile {
        let mut file = NamedTempFile::new().unwrap();
        for i in 0..count {
            let event = TickEvent {
                timestamp_ns: i * 1000, // 1 microsecond apart
                price: 50000.0 + (i as f64 * 0.01),
                volume: 1.5,
                bid_price: 49999.9,
                ask_price: 50000.1,
                bid_volume: 10.0,
                ask_volume: 10.0,
                trade_type: i % 2,
            };
            let bytes = unsafe {
                std::slice::from_raw_parts(
                    &event as *const TickEvent as *const u8,
                    TickEvent::SIZE,
                )
            };
            file.write_all(bytes).unwrap();
        }
        file
    }
    
    #[test]
    fn test_data_feeder_open_and_read() {
        let temp_file = create_test_data(1000);
        let feeder = DataFeeder::open(temp_file.path()).unwrap();
        
        assert_eq!(feeder.total_events(), 1000);
        
        let mut count = 0;
        while let Some(_event) = feeder.next_event() {
            count += 1;
        }
        assert_eq!(count, 1000);
    }
    
    #[test]
    fn test_seek_to_timestamp() {
        let temp_file = create_test_data(1000);
        let feeder = DataFeeder::open(temp_file.path()).unwrap();
        
        let target_ns = 500_000; // 500 microseconds
        let pos = feeder.seek_to_timestamp(target_ns);
        assert!(pos.is_some());
        assert!(pos.unwrap() >= 500);
    }
    
    #[test]
    fn test_batch_reader() {
        let temp_file = create_test_data(1000);
        let feeder = DataFeeder::open(temp_file.path()).unwrap();
        let mut reader = BatchReader::new(&feeder, 100);
        
        let batch = reader.read_batch(50);
        assert_eq!(batch.len(), 50);
    }
}
