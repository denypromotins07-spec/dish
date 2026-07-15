//! Rust state synchronizer for cross-chain events.
//! Aligns events with different block times and finality into a unified timeline.

use std::cmp::Ordering;
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering as AtomicOrdering};
use std::sync::Arc;
use parking_lot::RwLock;

/// Unified timestamp in microseconds since epoch
pub type MicroTimestamp = u128;

/// Chain identifier
pub type ChainId = u64;

/// Cross-chain event with unified timing
#[derive(Debug, Clone)]
pub struct CrossChainEvent {
    pub event_id: [u8; 32],
    pub chain_id: ChainId,
    pub chain_block_number: u64,
    pub chain_timestamp: u64,
    pub unified_timestamp_us: MicroTimestamp,
    pub event_type: EventType,
    pub payload: Vec<u8>,
    pub finality_status: FinalityStatus,
    pub confidence_score: f64,
}

/// Types of cross-chain events
#[derive(Debug, Clone, PartialEq)]
pub enum EventType {
    BridgeTransfer,
    MessagePassing,
    TokenMint,
    TokenBurn,
    LiquidityAdd,
    LiquidityRemove,
    OracleUpdate,
    Custom(String),
}

/// Finality status of an event
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum FinalityStatus {
    Pending,
    Confirmed,
    Finalized,
    ReorgSafe,
}

impl FinalityStatus {
    pub fn as_u8(&self) -> u8 {
        match self {
            Self::Pending => 0,
            Self::Confirmed => 1,
            Self::Finalized => 2,
            Self::ReorgSafe => 3,
        }
    }
}

/// Chain-specific finality parameters
#[derive(Debug, Clone)]
pub struct ChainFinalityParams {
    pub chain_id: ChainId,
    pub block_time_ms: u64,
    pub confirmations_required: u64,
    pub finality_blocks: u64,
    pub reorg_safe_blocks: u64,
}

impl ChainFinalityParams {
    pub fn ethereum_mainnet() -> Self {
        Self {
            chain_id: 1,
            block_time_ms: 12000,
            confirmations_required: 12,
            finality_blocks: 64,
            reorg_safe_blocks: 128,
        }
    }

    pub fn arbitrum() -> Self {
        Self {
            chain_id: 42161,
            block_time_ms: 250,
            confirmations_required: 20,
            finality_blocks: 100,
            reorg_safe_blocks: 200,
        }
    }

    pub fn optimism() -> Self {
        Self {
            chain_id: 10,
            block_time_ms: 2000,
            confirmations_required: 50,
            finality_blocks: 100,
            reorg_safe_blocks: 200,
        }
    }
}

/// State synchronizer for cross-chain event alignment
pub struct StateSynchronizer {
    /// Events sorted by unified timestamp
    events_by_time: RwLock<BTreeMap<MicroTimestamp, Vec<CrossChainEvent>>>,
    /// Events sorted by event ID for deduplication
    events_by_id: RwLock<std::collections::HashMap<[u8; 32], MicroTimestamp>>,
    /// Chain finality parameters
    chain_params: RwLock<std::collections::HashMap<ChainId, ChainFinalityParams>>,
    /// Current watermarks per chain (highest processed block)
    watermarks: RwLock<std::collections::HashMap<ChainId, u64>>,
    /// Global watermark (lowest common denominator)
    global_watermark: AtomicU64,
    /// Maximum events to keep in memory
    max_events: usize,
    /// Event count
    event_count: AtomicU64,
}

impl StateSynchronizer {
    /// Create a new state synchronizer
    pub fn new(max_events: usize) -> Self {
        let mut chain_params = std::collections::HashMap::new();
        chain_params.insert(1, ChainFinalityParams::ethereum_mainnet());
        chain_params.insert(42161, ChainFinalityParams::arbitrum());
        chain_params.insert(10, ChainFinalityParams::optimism());

        Self {
            events_by_time: RwLock::new(BTreeMap::new()),
            events_by_id: RwLock::new(std::collections::HashMap::new()),
            chain_params: RwLock::new(chain_params),
            watermarks: RwLock::new(std::collections::HashMap::new()),
            global_watermark: AtomicU64::new(0),
            max_events,
            event_count: AtomicU64::new(0),
        }
    }

    /// Register or update chain finality parameters
    pub fn register_chain_params(&self, params: ChainFinalityParams) {
        let mut chain_params = self.chain_params.write();
        chain_params.insert(params.chain_id, params);
    }

    /// Convert chain timestamp to unified microsecond timestamp
    pub fn chain_to_unified_timestamp(
        &self,
        chain_id: ChainId,
        chain_timestamp_seconds: u64,
    ) -> MicroTimestamp {
        // Convert seconds to microseconds
        (chain_timestamp_seconds as u128) * 1_000_000
    }

    /// Add a cross-chain event to the synchronizer
    pub fn add_event(&self, mut event: CrossChainEvent) -> Result<(), SyncError> {
        // Check for duplicates
        {
            let events_by_id = self.events_by_id.read();
            if events_by_id.contains_key(&event.event_id) {
                return Err(SyncError::DuplicateEvent);
            }
        }

        // Calculate unified timestamp if not set
        if event.unified_timestamp_us == 0 {
            event.unified_timestamp_us = 
                self.chain_to_unified_timestamp(event.chain_id, event.chain_timestamp);
        }

        // Update finality status based on chain params
        event.finality_status = self.calculate_finality_status(
            event.chain_id,
            event.chain_block_number,
        );

        // Insert into maps
        {
            let mut events_by_id = self.events_by_id.write();
            events_by_id.insert(event.event_id, event.unified_timestamp_us);
        }

        {
            let mut events_by_time = self.events_by_time.write();
            events_by_time
                .entry(event.unified_timestamp_us)
                .or_insert_with(Vec::new)
                .push(event);
        }

        // Update event count
        let count = self.event_count.fetch_add(1, AtomicOrdering::Relaxed);

        // Enforce max size
        if count as usize >= self.max_events {
            self.prune_old_events();
        }

        Ok(())
    }

    /// Calculate finality status for an event
    fn calculate_finality_status(
        &self,
        chain_id: ChainId,
        block_number: u64,
    ) -> FinalityStatus {
        let chain_params = self.chain_params.read();
        let watermarks = self.watermarks.read();

        let params = match chain_params.get(&chain_id) {
            Some(p) => p,
            None => return FinalityStatus::Pending,
        };

        let watermark = watermarks.get(&chain_id).copied().unwrap_or(0);

        if watermark == 0 {
            return FinalityStatus::Pending;
        }

        let confirmations = watermark.saturating_sub(block_number);

        if confirmations >= params.reorg_safe_blocks {
            FinalityStatus::ReorgSafe
        } else if confirmations >= params.finality_blocks {
            FinalityStatus::Finalized
        } else if confirmations >= params.confirmations_required {
            FinalityStatus::Confirmed
        } else {
            FinalityStatus::Pending
        }
    }

    /// Update the watermark for a chain
    pub fn update_watermark(&self, chain_id: ChainId, block_number: u64) {
        let mut watermarks = self.watermarks.write();
        
        let prev = watermarks.entry(chain_id).or_insert(0);
        if block_number > *prev {
            *prev = block_number;
        }

        // Update global watermark (minimum across all chains)
        self.update_global_watermark(&watermarks);
    }

    fn update_global_watermark(&self, watermarks: &std::collections::HashMap<ChainId, u64>) {
        if watermarks.is_empty() {
            return;
        }

        let min_watermark = *watermarks.values().min().unwrap_or(&0);
        self.global_watermark.store(min_watermark, AtomicOrdering::Relaxed);

        // Update finality status for all events
        self.refresh_finality_statuses();
    }

    fn refresh_finality_statuses(&self) {
        let mut events_by_time = self.events_by_time.write();
        
        for (_timestamp, events) in events_by_time.iter_mut() {
            for event in events.iter_mut() {
                event.finality_status = self.calculate_finality_status(
                    event.chain_id,
                    event.chain_block_number,
                );
            }
        }
    }

    /// Get events within a time range
    pub fn get_events_in_range(
        &self,
        start_us: MicroTimestamp,
        end_us: MicroTimestamp,
        min_finality: Option<FinalityStatus>,
    ) -> Vec<CrossChainEvent> {
        let events_by_time = self.events_by_time.read();
        let mut result = Vec::new();

        for (timestamp, events) in events_by_time.range(start_us..=end_us) {
            for event in events {
                if let Some(min_final) = min_finality {
                    if event.finality_status >= min_final {
                        result.push(event.clone());
                    }
                } else {
                    result.push(event.clone());
                }
            }
        }

        // Sort by timestamp then by chain_id for deterministic ordering
        result.sort_by(|a, b| {
            a.unified_timestamp_us
                .cmp(&b.unified_timestamp_us)
                .then_with(|| a.chain_id.cmp(&b.chain_id))
        });

        result
    }

    /// Get the unified timeline up to the global watermark
    pub fn get_unified_timeline(&self, limit: usize) -> Vec<CrossChainEvent> {
        let global_wm = self.global_watermark.load(AtomicOrdering::Relaxed);
        
        // Get all finalized events
        let mut events = Vec::new();
        {
            let events_by_time = self.events_by_time.read();
            for (_timestamp, batch) in events_by_time.iter() {
                for event in batch {
                    if event.finality_status >= FinalityStatus::Finalized {
                        events.push(event.clone());
                    }
                }
            }
        }

        // Sort and limit
        events.sort_by(|a, b| {
            a.unified_timestamp_us.cmp(&b.unified_timestamp_us)
        });
        events.truncate(limit);

        events
    }

    /// Prune old events to stay within memory bounds
    fn prune_old_events(&self) {
        let mut events_by_time = self.events_by_time.write();
        let mut events_by_id = self.events_by_id.write();

        // Remove oldest entries until under limit
        while events_by_time.len() > self.max_events / 10 {
            if let Some((oldest_key, _)) = events_by_time.iter().next() {
                let key = *oldest_key;
                if let Some(events) = events_by_time.remove(&key) {
                    for event in events {
                        events_by_id.remove(&event.event_id);
                    }
                }
            } else {
                break;
            }
        }
    }

    /// Get statistics about the synchronizer state
    pub fn get_stats(&self) -> SyncStats {
        let events_by_time = self.events_by_time.read();
        let events_by_id = self.events_by_id.read();
        let watermarks = self.watermarks.read();

        let total_events: usize = events_by_time.values().map(|v| v.len()).sum();
        
        let mut by_finality = std::collections::HashMap::new();
        for (_ts, events) in events_by_time.iter() {
            for event in events {
                *by_finality.entry(event.finality_status).or_insert(0) += 1;
            }
        }

        SyncStats {
            total_events,
            unique_event_ids: events_by_id.len(),
            time_buckets: events_by_time.len(),
            chains_tracked: watermarks.len(),
            global_watermark: self.global_watermark.load(AtomicOrdering::Relaxed),
            by_finality,
            max_events: self.max_events,
        }
    }

    /// Export events for external consumption
    pub fn export_events(&self, min_finality: FinalityStatus) -> Vec<CrossChainEvent> {
        self.get_unified_timeline(self.max_events)
            .into_iter()
            .filter(|e| e.finality_status >= min_finality)
            .collect()
    }
}

#[derive(Debug, Clone)]
pub struct SyncStats {
    pub total_events: usize,
    pub unique_event_ids: usize,
    pub time_buckets: usize,
    pub chains_tracked: usize,
    pub global_watermark: u64,
    pub by_finality: std::collections::HashMap<FinalityStatus, usize>,
    pub max_events: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub enum SyncError {
    DuplicateEvent,
    InvalidTimestamp,
    ChainNotRegistered,
    MemoryLimitReached,
}

impl Default for StateSynchronizer {
    fn default() -> Self {
        Self::new(100_000)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_synchronizer_creation() {
        let sync = StateSynchronizer::new(1000);
        let stats = sync.get_stats();
        
        assert_eq!(stats.total_events, 0);
        assert_eq!(stats.chains_tracked, 3); // Default chains
    }

    #[test]
    fn test_add_and_query_events() {
        let sync = StateSynchronizer::new(1000);
        
        // Add some events
        let event1 = CrossChainEvent {
            event_id: [1u8; 32],
            chain_id: 1,
            chain_block_number: 100,
            chain_timestamp: 1000,
            unified_timestamp_us: 1_000_000_000,
            event_type: EventType::BridgeTransfer,
            payload: vec![],
            finality_status: FinalityStatus::Pending,
            confidence_score: 1.0,
        };

        let event2 = CrossChainEvent {
            event_id: [2u8; 32],
            chain_id: 42161,
            chain_block_number: 200,
            chain_timestamp: 1001,
            unified_timestamp_us: 1_001_000_000,
            event_type: EventType::MessagePassing,
            payload: vec![],
            finality_status: FinalityStatus::Pending,
            confidence_score: 1.0,
        };

        sync.add_event(event1.clone()).unwrap();
        sync.add_event(event2.clone()).unwrap();

        // Query events
        let events = sync.get_events_in_range(0, 2_000_000_000, None);
        assert_eq!(events.len(), 2);

        // Test duplicate rejection
        assert!(matches!(sync.add_event(event1), Err(SyncError::DuplicateEvent)));
    }

    #[test]
    fn test_watermark_and_finality() {
        let sync = StateSynchronizer::new(1000);
        
        // Update watermark to simulate chain progress
        sync.update_watermark(1, 200); // Ethereum
        
        let status = sync.calculate_finality_status(1, 100);
        // With 100 confirmations, should be at least Finalized
        assert!(status >= FinalityStatus::Finalized);
    }
}
