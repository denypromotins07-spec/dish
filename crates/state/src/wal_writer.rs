//! Write-Ahead Logging (WAL) Implementation
//! Ensures critical state changes are persisted before execution
//! Enables exact replay of execution events after OOM or crash

use std::fs::{File, OpenOptions};
use std::io::{BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{mpsc, Mutex};

/// WAL entry types
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum WalEntryType {
    OrderNew,
    OrderCancel,
    OrderModify,
    OrderFill,
    PositionOpen,
    PositionClose,
    PositionUpdate,
    StrategyStateChange,
    Checkpoint,
}

/// Single WAL entry with sequence number
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct WalEntry {
    pub seq_num: u64,
    pub entry_type: WalEntryType,
    pub timestamp_ns: u64,
    pub data: Vec<u8>,  // Serialized payload
    pub checksum: u32,
}

impl WalEntry {
    /// Calculate CRC32 checksum for data integrity
    pub fn calculate_checksum(data: &[u8]) -> u32 {
        crc32fast::hash(data)
    }

    /// Verify entry integrity
    pub fn verify(&self) -> bool {
        Self::calculate_checksum(&self.data) == self.checksum
    }
}

/// WAL writer configuration
#[derive(Clone)]
pub struct WalConfig {
    pub path: PathBuf,
    pub max_file_size_mb: usize,
    pub sync_mode: SyncMode,
    pub rotation_enabled: bool,
}

#[derive(Clone, Copy)]
pub enum SyncMode {
    Full,      // fsync after every write
    Normal,    // fsync periodically
    None,      // No explicit sync (OS controls)
}

impl Default for WalConfig {
    fn default() -> Self {
        Self {
            path: PathBuf::from("./wal"),
            max_file_size_mb: 100,
            sync_mode: SyncMode::Normal,
            rotation_enabled: true,
        }
    }
}

/// Write-Ahead Log
pub struct WalWriter {
    config: WalConfig,
    current_seq: u64,
    current_file: Option<BufWriter<File>>,
    current_path: PathBuf,
    tx: mpsc::Sender<WalEntry>,
    shutdown: Arc<Mutex<bool>>,
    entries_written: u64,
    bytes_written: u64,
    last_sync: Instant,
}

impl WalWriter {
    /// Create new WAL writer
    pub async fn new(config: WalConfig) -> Result<Self, WalError> {
        let (tx, rx) = mpsc::channel(1000);
        
        // Ensure directory exists
        tokio::fs::create_dir_all(&config.path).await?;
        
        let mut writer = Self {
            config: config.clone(),
            current_seq: 0,
            current_file: None,
            current_path: PathBuf::new(),
            tx,
            shutdown: Arc::new(Mutex::new(false)),
            entries_written: 0,
            bytes_written: 0,
            last_sync: Instant::now(),
        };
        
        // Open initial WAL file
        writer.open_new_file().await?;
        
        // Start background writer task
        tokio::spawn(async move {
            writer.background_writer(rx).await;
        });
        
        Ok(Self {
            config,
            current_seq: 0,
            current_file: None,
            current_path: PathBuf::new(),
            tx,
            shutdown: Arc::new(Mutex::new(false)),
            entries_written: 0,
            bytes_written: 0,
            last_sync: Instant::now(),
        })
    }

    /// Open a new WAL file
    async fn open_new_file(&mut self) -> Result<(), WalError> {
        let timestamp = chrono::Utc::now().format("%Y%m%d_%H%M%S_%f");
        let filename = format!("wal_{}.log", timestamp);
        let path = self.config.path.join(&filename);
        
        let file = OpenOptions::new()
            .create(true)
            .write(true)
            .append(true)
            .open(&path)?;
        
        self.current_file = Some(BufWriter::new(file));
        self.current_path = path;
        
        Ok(())
    }

    /// Background writer task
    async fn background_writer(mut self, mut rx: mpsc::Receiver<WalEntry>) {
        let mut sync_interval = tokio::time::interval(Duration::from_millis(100));
        
        loop {
            tokio::select! {
                msg = rx.recv() => {
                    match msg {
                        Some(entry) => {
                            if let Err(e) = self.write_entry_internal(&entry) {
                                log::error!("WAL write error: {}", e);
                            }
                        }
                        None => {
                            // Channel closed, shutdown
                            break;
                        }
                    }
                }
                _ = sync_interval.tick() => {
                    if matches!(self.config.sync_mode, SyncMode::Full) {
                        if let Some(ref mut writer) = self.current_file {
                            let _ = writer.flush();
                        }
                    }
                }
            }
        }
    }

    /// Write entry to current file
    fn write_entry_internal(&mut self, entry: &WalEntry) -> Result<(), WalError> {
        if let Some(ref mut writer) = self.current_file {
            // Serialize entry
            let serialized = bincode::serialize(entry)?;
            
            // Write length prefix + data
            let len = serialized.len() as u32;
            writer.write_all(&len.to_le_bytes())?;
            writer.write_all(&serialized)?;
            
            self.entries_written += 1;
            self.bytes_written += (4 + serialized.len()) as u64;
            
            // Check if rotation needed
            if self.config.rotation_enabled 
                && self.bytes_written >= (self.config.max_file_size_mb as u64 * 1024 * 1024)
            {
                writer.flush()?;
                self.open_new_file().await?;
            } else if matches!(self.config.sync_mode, SyncMode::Full) {
                writer.flush()?;
            }
        }
        
        Ok(())
    }

    /// Append order creation to WAL
    pub async fn append_order_new(&self, order_data: &[u8]) -> Result<u64, WalError> {
        self.append(WalEntryType::OrderNew, order_data).await
    }

    /// Append order cancellation to WAL
    pub async fn append_order_cancel(&self, order_data: &[u8]) -> Result<u64, WalError> {
        self.append(WalEntryType::OrderCancel, order_data).await
    }

    /// Append order fill to WAL
    pub async fn append_order_fill(&self, fill_data: &[u8]) -> Result<u64, WalError> {
        self.append(WalEntryType::OrderFill, fill_data).await
    }

    /// Append position update to WAL
    pub async fn append_position_update(&self, position_data: &[u8]) -> Result<u64, WalError> {
        self.append(WalEntryType::PositionUpdate, position_data).await
    }

    /// Generic append method
    pub async fn append(&self, entry_type: WalEntryType, data: &[u8]) -> Result<u64, WalError> {
        let seq_num = self.current_seq;
        self.current_seq += 1;
        
        let entry = WalEntry {
            seq_num,
            entry_type,
            timestamp_ns: chrono::Utc::now().timestamp_nanos() as u64,
            data: data.to_vec(),
            checksum: WalEntry::calculate_checksum(data),
        };
        
        // Send to background writer
        self.tx.send(entry).await.map_err(|_| WalError::ChannelClosed)?;
        
        Ok(seq_num)
    }

    /// Create checkpoint - special entry marking consistent state
    pub async fn checkpoint(&self, state_snapshot: &[u8]) -> Result<u64, WalError> {
        self.append(WalEntryType::Checkpoint, state_snapshot).await
    }

    /// Force sync to disk
    pub async fn sync(&mut self) -> Result<(), WalError> {
        if let Some(ref mut writer) = self.current_file {
            writer.flush()?;
            writer.get_ref().sync_all()?;
            self.last_sync = Instant::now();
        }
        Ok(())
    }

    /// Get current sequence number
    pub fn get_current_seq(&self) -> u64 {
        self.current_seq
    }

    /// Get statistics
    pub fn get_stats(&self) -> WalStats {
        WalStats {
            entries_written: self.entries_written,
            bytes_written: self.bytes_written,
            current_seq: self.current_seq,
            last_sync: self.last_sync.elapsed(),
        }
    }

    /// Shutdown writer gracefully
    pub async fn shutdown(&self) {
        let mut shutdown = self.shutdown.lock().await;
        *shutdown = true;
    }
}

/// WAL reader for replay
pub struct WalReader {
    files: Vec<PathBuf>,
}

impl WalReader {
    /// Find all WAL files in directory
    pub fn find_wal_files<P: AsRef<Path>>(dir: P) -> Result<Vec<PathBuf>, WalError> {
        let mut files: Vec<PathBuf> = std::fs::read_dir(dir)?
            .filter_map(|e| e.ok())
            .filter(|e| {
                e.file_name()
                    .to_string_lossy()
                    .starts_with("wal_")
                    && e.file_name()
                    .to_string_lossy()
                    .ends_with(".log")
            })
            .map(|e| e.path())
            .collect();
        
        files.sort();
        Ok(files)
    }

    /// Create reader from files
    pub fn new(files: Vec<PathBuf>) -> Self {
        Self { files }
    }

    /// Iterate through all WAL entries
    pub fn iter_entries<F>(&self, mut callback: F) -> Result<(), WalError>
    where
        F: FnMut(WalEntry) -> Result<(), WalError>,
    {
        for file_path in &self.files {
            self.read_file_entries(file_path, &mut callback)?;
        }
        Ok(())
    }

    /// Read entries from single file
    fn read_file_entries<F>(
        &self,
        path: &Path,
        callback: &mut F,
    ) -> Result<(), WalError>
    where
        F: FnMut(WalEntry) -> Result<(), WalError>,
    {
        let file = File::open(path)?;
        let mut reader = BufReader::new(file);
        
        loop {
            // Read length prefix
            let mut len_buf = [0u8; 4];
            match reader.read_exact(&mut len_buf) {
                Ok(_) => {}
                Err(std::io::ErrorKind::UnexpectedEof) => break,
                Err(e) => return Err(WalError::Io(e)),
            }
            
            let len = u32::from_le_bytes(len_buf) as usize;
            
            // Read entry data
            let mut data = vec![0u8; len];
            reader.read_exact(&mut data)?;
            
            // Deserialize and validate
            let entry: WalEntry = bincode::deserialize(&data)?;
            
            if !entry.verify() {
                log::warn!("WAL entry checksum mismatch at seq {}", entry.seq_num);
                continue; // Skip corrupted entry
            }
            
            callback(entry)?;
        }
        
        Ok(())
    }

    /// Replay entries up to a checkpoint
    pub fn replay_to_checkpoint<F>(
        &self,
        mut callback: F,
    ) -> Result<Option<u64>, WalError>
    where
        F: FnMut(WalEntry) -> Result<(), WalError>,
    {
        let mut last_checkpoint: Option<u64> = None;
        
        self.iter_entries(|entry| {
            if entry.entry_type == WalEntryType::Checkpoint {
                last_checkpoint = Some(entry.seq_num);
            }
            callback(entry)
        })?;
        
        Ok(last_checkpoint)
    }

    /// Get latest valid sequence number
    pub fn get_latest_seq(&self) -> Result<u64, WalError> {
        let mut max_seq = 0u64;
        
        self.iter_entries(|entry| {
            if entry.seq_num > max_seq && entry.verify() {
                max_seq = entry.seq_num;
            }
            Ok(())
        })?;
        
        Ok(max_seq)
    }
}

/// WAL statistics
#[derive(Debug, Clone)]
pub struct WalStats {
    pub entries_written: u64,
    pub bytes_written: u64,
    pub current_seq: u64,
    pub last_sync: Duration,
}

/// WAL errors
#[derive(Debug)]
pub enum WalError {
    Io(std::io::Error),
    Serialize(bincode::Error),
    ChannelClosed,
    CorruptedEntry,
}

impl From<std::io::Error> for WalError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}

impl From<bincode::Error> for WalError {
    fn from(e: bincode::Error) -> Self {
        Self::Serialize(e)
    }
}

impl std::fmt::Display for WalError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(e) => write!(f, "IO error: {}", e),
            Self::Serialize(e) => write!(f, "Serialization error: {}", e),
            Self::ChannelClosed => write!(f, "Channel closed"),
            Self::CorruptedEntry => write!(f, "Corrupted entry"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[tokio::test]
    async fn test_wal_write_read() {
        let temp_dir = TempDir::new().unwrap();
        let config = WalConfig {
            path: temp_dir.path().to_path_buf(),
            max_file_size_mb: 10,
            sync_mode: SyncMode::Full,
            rotation_enabled: false,
        };

        let mut writer = WalWriter::new(config.clone()).await.unwrap();
        
        // Write some entries
        let seq1 = writer.append_order_new(b"order1").await.unwrap();
        let seq2 = writer.append_order_fill(b"fill1").await.unwrap();
        
        // Sync
        writer.sync().await.unwrap();
        
        // Read back
        let files = WalReader::find_wal_files(temp_dir.path()).unwrap();
        let reader = WalReader::new(files);
        
        let mut entries = Vec::new();
        reader.iter_entries(|e| {
            entries.push(e);
            Ok(())
        }).unwrap();
        
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].seq_num, seq1);
        assert_eq!(entries[1].seq_num, seq2);
    }
}
