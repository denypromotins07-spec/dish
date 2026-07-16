//! Memory-Mapped File Engine for NVMe SSD
//! 
//! This module provides high-performance mmap access to:
//! - Deep Limit Order Book (L3) data stored on NVMe SSD
//! - Historical tick data for backtesting
//! 
//! Design: Keep only top 10 levels in RAM, use SSD's PCIe bandwidth
//! for microsecond lookups without bloating the 10GB RAM limit.

use std::fs::{File, OpenOptions};
use std::io::{self, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use memmap2::{Mmap, MmapMut};

/// Configuration for the mmap SSD engine
#[derive(Clone, Debug)]
pub struct MmapConfig {
    /// Base directory for data files on NVMe SSD
    pub data_directory: PathBuf,
    /// Maximum file size per mmap region (1GB default)
    pub max_file_size: u64,
    /// Number of pre-allocated mmap regions
    pub preallocate_regions: usize,
}

impl Default for MmapConfig {
    fn default() -> Self {
        Self {
            data_directory: PathBuf::from("C:\\crypto_bot\\data\\nvme"),
            max_file_size: 1024 * 1024 * 1024, // 1GB
            preallocate_regions: 10,
        }
    }
}

/// Represents a single order book level
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct OrderBookLevel {
    pub price: i64,      // Price in fixed-point (multiply by 1e8 for actual value)
    pub quantity: i64,   // Quantity in fixed-point
    pub order_count: u32,
    pub _padding: u32,   // Alignment padding
}

/// Top 10 levels of the order book kept in RAM
#[repr(C)]
#[derive(Clone, Debug)]
pub struct RamOrderBook {
    pub bids: [OrderBookLevel; 10],
    pub asks: [OrderBookLevel; 10],
    pub timestamp_ns: u64,
    pub symbol_id: u32,
    pub _padding: u32,
}

impl Default for RamOrderBook {
    fn default() -> Self {
        Self {
            bids: [OrderBookLevel::default(); 10],
            asks: [OrderBookLevel::default(); 10],
            timestamp_ns: 0,
            symbol_id: 0,
            _padding: 0,
        }
    }
}

/// Memory-mapped file handle for historical data
pub struct MappedFile {
    file: File,
    mmap: Mmap,
    path: PathBuf,
    size: usize,
}

impl MappedFile {
    /// Opens or creates a memory-mapped file
    pub fn open(path: &Path, size: usize) -> io::Result<Self> {
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .open(path)?;

        // Extend file if needed
        if file.metadata()?.len() < size as u64 {
            file.set_len(size as u64)?;
        }

        let mmap = unsafe { Mmap::map(&file)? };

        Ok(Self {
            file,
            mmap,
            path: path.to_path_buf(),
            size,
        })
    }

    /// Reads data at a specific offset
    pub fn read_at(&self, offset: usize, len: usize) -> io::Result<&[u8]> {
        if offset + len > self.size {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "Read exceeds file bounds",
            ));
        }
        Ok(&self.mmap[offset..offset + len])
    }

    /// Writes data at a specific offset
    pub fn write_at(&mut self, offset: usize, data: &[u8]) -> io::Result<()> {
        if offset + data.len() > self.size {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "Write exceeds file bounds",
            ));
        }

        let mut mmap_mut = unsafe { MmapMut::map_mut(&self.file)? };
        mmap_mut[offset..offset + data.len()].copy_from_slice(data);
        mmap_mut.flush()?;

        Ok(())
    }

    /// Flushes changes to disk
    pub fn flush(&self) -> io::Result<()> {
        self.mmap.flush()
    }
}

/// High-performance L3 order book storage engine
pub struct L3OrderBookEngine {
    config: Arc<MmapConfig>,
    ram_book: RamOrderBook,
    depth_maps: Vec<MappedFile>,
    current_file_idx: usize,
}

impl L3OrderBookEngine {
    /// Creates a new L3 order book engine
    pub fn new(config: Arc<MmapConfig>) -> io::Result<Self> {
        // Ensure data directory exists
        std::fs::create_dir_all(&config.data_directory)?;

        let mut engine = Self {
            config,
            ram_book: RamOrderBook::default(),
            depth_maps: Vec::new(),
            current_file_idx: 0,
        };

        // Pre-allocate mmap regions
        for i in 0..engine.config.preallocate_regions {
            let path = engine.config.data_directory.join(format!("l3_depth_{}.bin", i));
            let file = MappedFile::open(&path, engine.config.max_file_size as usize)?;
            engine.depth_maps.push(file);
        }

        Ok(engine)
    }

    /// Updates the top 10 levels in RAM
    pub fn update_ram_book(&mut self, book: RamOrderBook) {
        self.ram_book = book;
    }

    /// Gets the current top 10 levels from RAM (microsecond access)
    pub fn get_ram_book(&self) -> &RamOrderBook {
        &self.ram_book
    }

    /// Appends deep L3 data to the current mmap file
    pub fn append_l3_data(&mut self, data: &[u8]) -> io::Result<(usize, usize)> {
        let current_file = &mut self.depth_maps[self.current_file_idx];
        
        // Find next available offset (simplified - real impl needs metadata tracking)
        let offset = 0; // Placeholder
        
        current_file.write_at(offset, data)?;
        
        Ok((self.current_file_idx, offset))
    }

    /// Reads historical L3 data from mmap (microsecond SSD access)
    pub fn read_l3_data(&self, file_idx: usize, offset: usize, len: usize) -> io::Result<&[u8]> {
        if file_idx >= self.depth_maps.len() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "Invalid file index",
            ));
        }
        
        self.depth_maps[file_idx].read_at(offset, len)
    }

    /// Rotates to the next mmap file when current is full
    pub fn rotate_file(&mut self) -> io::Result<()> {
        self.current_file_idx = (self.current_file_idx + 1) % self.depth_maps.len();
        Ok(())
    }

    /// Flushes all pending writes to NVMe SSD
    pub fn flush_all(&self) -> io::Result<()> {
        for file in &self.depth_maps {
            file.flush()?;
        }
        Ok(())
    }
}

/// Historical tick data reader using mmap
pub struct TickDataReader {
    mapped_file: MappedFile,
    tick_size: usize, // Size of each tick record in bytes
}

impl TickDataReader {
    /// Opens a tick data file for reading
    pub fn open(path: &Path, tick_size: usize) -> io::Result<Self> {
        let file_size = std::fs::metadata(path)?.len() as usize;
        let mapped_file = MappedFile::open(path, file_size)?;

        Ok(Self {
            mapped_file,
            tick_size,
        })
    }

    /// Reads a specific tick by index
    pub fn read_tick(&self, index: usize) -> io::Result<&[u8]> {
        let offset = index * self.tick_size;
        self.mapped_file.read_at(offset, self.tick_size)
    }

    /// Returns total number of ticks in the file
    pub fn tick_count(&self) -> usize {
        self.mapped_file.size / self.tick_size
    }

    /// Iterates over all ticks (streaming to avoid RAM bloat)
    pub fn iter_ticks(&self) -> TickIterator<'_> {
        TickIterator {
            reader: self,
            current_index: 0,
        }
    }
}

/// Iterator for streaming tick data
pub struct TickIterator<'a> {
    reader: &'a TickDataReader,
    current_index: usize,
}

impl<'a> Iterator for TickIterator<'a> {
    type Item = io::Result<&'a [u8]>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_index >= self.reader.tick_count() {
            return None;
        }

        let result = self.reader.read_tick(self.current_index);
        self.current_index += 1;
        Some(result)
    }
}

/// Represents a single tick record
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct TickRecord {
    pub timestamp_ns: u64,
    pub price: i64,
    pub quantity: i64,
    pub trade_type: u8, // 0=buy, 1=sell
    pub _padding: [u8; 7],
}

impl TickRecord {
    pub const SIZE: usize = std::mem::size_of::<TickRecord>();
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;

    #[test]
    fn test_mmap_file() {
        let temp_dir = env::temp_dir().join("mmap_test");
        let path = temp_dir.join("test.bin");
        
        let mut file = MappedFile::open(&path, 1024).unwrap();
        let data = b"Hello, mmap!";
        file.write_at(0, data).unwrap();
        
        let read_data = file.read_at(0, data.len()).unwrap();
        assert_eq!(read_data, data);
        
        // Cleanup
        std::fs::remove_file(path).ok();
    }

    #[test]
    fn test_tick_record_size() {
        assert_eq!(TickRecord::SIZE, 32); // Should be 32 bytes with padding
    }
}
