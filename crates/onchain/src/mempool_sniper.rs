//! Pending transaction listener and decoder for EVM mempool analysis.
//! Detects large DEX swaps, liquidations, and MEV opportunities.

use bytes::Bytes;
use dashmap::DashMap;
use std::sync::atomic::{AtomicU64, AtomicBool, Ordering};
use std::sync::Arc;
use tokio::sync::mpsc;
use tracing::{debug, error, info, warn};

/// Decoded pending transaction with enriched metadata
#[derive(Debug, Clone)]
pub struct MempoolTx {
    pub hash: [u8; 32],
    pub from: [u8; 20],
    pub to: Option<[u8; 20]>,
    pub value: u128,
    pub gas_price: u128,
    pub max_fee_per_gas: Option<u128>,
    pub max_priority_fee_per_gas: Option<u128>,
    pub gas_limit: u64,
    pub nonce: u64,
    pub input_data: Bytes,
    pub tx_type: TxType,
    pub dex_info: Option<DexInfo>,
    pub liquidation_info: Option<LiquidationInfo>,
    pub timestamp_ns: u128,
}

/// Classified transaction type
#[derive(Debug, Clone, PartialEq)]
pub enum TxType {
    Unknown,
    Transfer,
    DexSwap,
    Liquidation,
    ContractDeployment,
    MEVArbitrage,
    SandwichAttack,
}

/// DEX-specific swap information
#[derive(Debug, Clone)]
pub struct DexInfo {
    pub dex_name: DexName,
    pub token_in: [u8; 20],
    pub token_out: [u8; 20],
    pub amount_in: u128,
    pub min_amount_out: u128,
    pub recipient: [u8; 20],
    pub deadline: u64,
}

#[derive(Debug, Clone, PartialEq)]
pub enum DexName {
    UniswapV2,
    UniswapV3,
    SushiSwap,
    Curve,
    Balancer,
    Unknown,
}

/// Liquidation detection information
#[derive(Debug, Clone)]
pub struct LiquidationInfo {
    pub borrower: [u8; 20],
    pub collateral_token: [u8; 20],
    pub debt_token: [u8; 20],
    pub debt_amount: u128,
    pub collateral_amount: u128,
    pub protocol: LendingProtocol,
}

#[derive(Debug, Clone, PartialEq)]
pub enum LendingProtocol {
    Aave,
    Compound,
    MakerDAO,
    Unknown,
}

/// Mempool sniper configuration
pub struct MempoolConfig {
    pub min_value_threshold: u128,
    pub min_gas_price: u128,
    pub tracked_dexes: Vec<DexName>,
    pub track_liquidations: bool,
    pub track_mev: bool,
    pub max_pending_txs: usize,
}

impl Default for MempoolConfig {
    fn default() -> Self {
        Self {
            min_value_threshold: 1_000_000_000_000_000_000u128, // 1 ETH
            min_gas_price: 50_000_000_000, // 50 gwei
            tracked_dexes: vec![DexName::UniswapV2, DexName::UniswapV3, DexName::SushiSwap],
            track_liquidations: true,
            track_mev: true,
            max_pending_txs: 10_000,
        }
    }
}

/// High-performance mempool analyzer
pub struct MempoolSniper {
    config: MempoolConfig,
    pending_txs: DashMap<[u8; 32], MempoolTx>,
    tx_sender: mpsc::Sender<MempoolEvent>,
    is_running: AtomicBool,
    processed_count: AtomicU64,
    filtered_count: AtomicU64,
}

/// Events emitted by the mempool sniper
#[derive(Debug)]
pub enum MempoolEvent {
    NewPendingTx(MempoolTx),
    LargeSwapDetected(MempoolTx),
    LiquidationDetected(MempoolTx),
    MEVOpportunity(MempoolTx),
    ToxicFlowWarning(MempoolTx),
}

impl MempoolSniper {
    pub fn new(config: MempoolConfig, tx_sender: mpsc::Sender<MempoolEvent>) -> Self {
        Self {
            config,
            pending_txs: DashMap::with_capacity(config.max_pending_txs / 2),
            tx_sender,
            is_running: AtomicBool::new(false),
            processed_count: AtomicU64::new(0),
            filtered_count: AtomicU64::new(0),
        }
    }

    /// Start processing pending transactions
    pub async fn start(&self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        self.is_running.store(true, Ordering::Relaxed);
        info!("Mempool sniper started");
        Ok(())
    }

    /// Stop processing
    pub fn stop(&self) {
        self.is_running.store(false, Ordering::Relaxed);
        info!("Mempool sniper stopped");
    }

    /// Process a raw pending transaction from the mempool
    pub async fn process_pending_tx(
        &self,
        raw_tx: &[u8],
    ) -> Result<Option<MempoolTx>, Box<dyn std::error::Error + Send + Sync>> {
        if !self.is_running.load(Ordering::Relaxed) {
            return Ok(None);
        }

        let timestamp_ns = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();

        // Decode transaction (RLP decoding in production)
        let decoded = self.decode_transaction(raw_tx)?;

        // Apply filters
        if !self.should_process(&decoded) {
            self.filtered_count.fetch_add(1, Ordering::Relaxed);
            return Ok(None);
        }

        // Classify transaction type
        let mut tx = decoded;
        tx.tx_type = self.classify_tx_type(&tx);
        tx.timestamp_ns = timestamp_ns;

        // Enrich with DEX or liquidation info
        if tx.tx_type == TxType::DexSwap {
            tx.dex_info = self.extract_dex_info(&tx);
        } else if tx.tx_type == TxType::Liquidation {
            tx.liquidation_info = self.extract_liquidation_info(&tx);
        }

        // Store in pending map
        self.pending_txs.insert(tx.hash, tx.clone());

        // Enforce max size
        if self.pending_txs.len() > self.config.max_pending_txs {
            self.prune_oldest();
        }

        self.processed_count.fetch_add(1, Ordering::Relaxed);

        // Emit events based on classification
        self.emit_events(&tx).await?;

        Ok(Some(tx))
    }

    /// Check if transaction should be processed based on filters
    fn should_process(&self, tx: &MempoolTx) -> bool {
        if tx.gas_price < self.config.min_gas_price {
            return false;
        }

        if tx.value < self.config.min_value_threshold {
            // Check if it's a DEX swap which might have 0 ETH value but high token value
            if !self.is_potential_dex_swap(&tx.input_data) {
                return false;
            }
        }

        true
    }

    /// Classify transaction type based on input data
    fn classify_tx_type(&self, tx: &MempoolTx) -> TxType {
        if tx.input_data.is_empty() || tx.input_data.len() < 4 {
            return TxType::Transfer;
        }

        let function_selector = &tx.input_data[0..4];

        // Check for known DEX functions
        if self.is_dex_function(function_selector) {
            return TxType::DexSwap;
        }

        // Check for liquidation functions
        if self.is_liquidation_function(function_selector) {
            return TxType::Liquidation;
        }

        // Check for MEV patterns (multi-call, flash loans)
        if self.is_mev_pattern(&tx.input_data) {
            return TxType::MEVArbitrage;
        }

        // Contract deployment
        if tx.to.is_none() {
            return TxType::ContractDeployment;
        }

        TxType::Unknown
    }

    /// Check if input data matches known DEX function selectors
    fn is_dex_function(&self, selector: &[u8]) -> bool {
        // Uniswap V2/V3 swap functions
        let swap_selectors = [
            &[0x38, 0xed, 0x22, 0xf0], // swapExactTokensForTokens
            &[0x79, 0x1a, 0xc9, 0xe5], // swapExactETHForTokens
            &[0x88, 0x03, 0xd4, 0xb3], // swapExactTokensForETH
            &[0xac, 0x96, 0x50, 0xd8], // multicall (Uniswap V3)
            &[0x12, 0x8a, 0xac, 0x0b], // swap (Uniswap V3 router)
        ];

        swap_selectors.iter().any(|s| s.as_slice() == selector)
    }

    /// Check for liquidation function selectors
    fn is_liquidation_function(&self, selector: &[u8]) -> bool {
        let liquidation_selectors = [
            &[0x2e, 0xd5, 0x9c, 0x6b], // liquidate (Aave)
            &[0xfd, 0x3d, 0xb4, 0xa8], // liquidateBorrow (Compound)
            &[0x4d, 0xdd, 0x66, 0xf8], // liquidate (generic)
        ];

        liquidation_selectors.iter().any(|s| s.as_slice() == selector)
    }

    /// Detect MEV patterns in input data
    fn is_mev_pattern(&self, input: &[u8]) -> bool {
        // Look for flash loan patterns
        if input.contains(&[0x1a, 0x41, 0x5e, 0x24]) {
            // flashLoan selector
            return true;
        }

        // Look for multi-call patterns indicating arbitrage
        if input.windows(4).any(|w| w == [0xac, 0x96, 0x50, 0xd8]) {
            return true;
        }

        false
    }

    /// Check if input data suggests a DEX swap
    fn is_potential_dex_swap(&self, input: &[u8]) -> bool {
        if input.len() < 4 {
            return false;
        }
        self.is_dex_function(&input[0..4])
    }

    /// Extract DEX-specific information from transaction
    fn extract_dex_info(&self, tx: &MempoolTx) -> Option<DexInfo> {
        if tx.input_data.len() < 4 {
            return None;
        }

        let selector = &tx.input_data[0..4];

        // Simple extraction - production would use proper ABI decoding
        if selector == &[0x38, 0xed, 0x22, 0xf0] {
            // swapExactTokensForTokens
            Some(DexInfo {
                dex_name: DexName::UniswapV2,
                token_in: [0u8; 20], // Would extract from calldata
                token_out: [0u8; 20],
                amount_in: 0,
                min_amount_out: 0,
                recipient: [0u8; 20],
                deadline: 0,
            })
        } else if selector == &[0xac, 0x96, 0x50, 0xd8] {
            Some(DexInfo {
                dex_name: DexName::UniswapV3,
                token_in: [0u8; 20],
                token_out: [0u8; 20],
                amount_in: 0,
                min_amount_out: 0,
                recipient: [0u8; 20],
                deadline: 0,
            })
        } else {
            Some(DexInfo {
                dex_name: DexName::Unknown,
                token_in: [0u8; 20],
                token_out: [0u8; 20],
                amount_in: 0,
                min_amount_out: 0,
                recipient: [0u8; 20],
                deadline: 0,
            })
        }
    }

    /// Extract liquidation information from transaction
    fn extract_liquidation_info(&self, tx: &MempoolTx) -> Option<LiquidationInfo> {
        Some(LiquidationInfo {
            borrower: [0u8; 20],
            collateral_token: [0u8; 20],
            debt_token: [0u8; 20],
            debt_amount: 0,
            collateral_amount: 0,
            protocol: LendingProtocol::Unknown,
        })
    }

    /// Emit appropriate events based on transaction classification
    async fn emit_events(&self, tx: &MempoolTx) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        // Always emit new pending tx event
        self.tx_sender.send(MempoolEvent::NewPendingTx(tx.clone())).await?;

        match tx.tx_type {
            TxType::DexSwap => {
                if let Some(dex) = &tx.dex_info {
                    if dex.amount_in >= self.config.min_value_threshold {
                        self.tx_sender.send(MempoolEvent::LargeSwapDetected(tx.clone())).await?;
                    }
                }
            }
            TxType::Liquidation => {
                if self.config.track_liquidations {
                    self.tx_sender.send(MempoolEvent::LiquidationDetected(tx.clone())).await?;
                }
            }
            TxType::MEVArbitrage => {
                if self.config.track_mev {
                    self.tx_sender.send(MempoolEvent::MEVOpportunity(tx.clone())).await?;
                }
            }
            _ => {}
        }

        Ok(())
    }

    /// Prune oldest transactions when cache is full
    fn prune_oldest(&self) {
        // Keep most recent transactions by removing oldest entries
        // In production, would use timestamps to determine oldest
        if let Some((key, _)) = self.pending_txs.iter().next() {
            self.pending_txs.remove(key.key());
        }
    }

    /// Decode raw RLP-encoded transaction
    fn decode_transaction(&self, raw: &[u8]) -> Result<MempoolTx, Box<dyn std::error::Error + Send + Sync>> {
        // Production implementation would use alloy-rlp or similar
        // For now, return a mock structure
        Ok(MempoolTx {
            hash: [0u8; 32],
            from: [0u8; 20],
            to: None,
            value: 0,
            gas_price: 0,
            max_fee_per_gas: None,
            max_priority_fee_per_gas: None,
            gas_limit: 0,
            nonce: 0,
            input_data: Bytes::from(raw.to_vec()),
            tx_type: TxType::Unknown,
            dex_info: None,
            liquidation_info: None,
            timestamp_ns: 0,
        })
    }

    /// Get current pending transaction count
    pub fn pending_count(&self) -> usize {
        self.pending_txs.len()
    }

    /// Get statistics
    pub fn get_stats(&self) -> MempoolStats {
        MempoolStats {
            processed: self.processed_count.load(Ordering::Relaxed),
            filtered: self.filtered_count.load(Ordering::Relaxed),
            pending: self.pending_count(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct MempoolStats {
    pub processed: u64,
    pub filtered: u64,
    pub pending: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::sync::mpsc;

    #[tokio::test]
    async fn test_mempool_sniper_creation() {
        let config = MempoolConfig::default();
        let (tx, _rx) = mpsc::channel(100);
        let sniper = MempoolSniper::new(config, tx);

        assert!(!sniper.is_running.load(Ordering::Relaxed));
        assert_eq!(sniper.pending_count(), 0);
    }

    #[tokio::test]
    async fn test_start_stop() {
        let config = MempoolConfig::default();
        let (tx, _rx) = mpsc::channel(100);
        let sniper = MempoolSniper::new(config, tx);

        sniper.start().await.unwrap();
        assert!(sniper.is_running.load(Ordering::Relaxed));

        sniper.stop();
        assert!(!sniper.is_running.load(Ordering::Relaxed));
    }
}
