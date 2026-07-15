//! Smart transaction splitter for on-chain/DEX execution.
//! Breaks large orders into randomized, micro-sized limit orders across multiple wallets.
//! Hides institutional footprints from MEV bots and front-runners.

use std::sync::Arc;
use tokio::sync::RwLock;
use rand::Rng;
use std::time::{Duration, Instant};

/// Configuration for transaction splitting.
#[derive(Clone, Debug)]
pub struct TxSplitConfig {
    /// Minimum number of child orders to split into
    pub min_splits: usize,
    /// Maximum number of child orders to split into
    pub max_splits: usize,
    /// Minimum size for each child order (in base units)
    pub min_order_size: f64,
    /// Randomization factor for order sizes (0.0 - 1.0)
    pub size_randomization: f64,
    /// Randomization factor for timing jitter (0.0 - 1.0)
    pub time_jitter_factor: f64,
    /// Minimum delay between child orders in milliseconds
    pub min_delay_ms: u64,
    /// Maximum delay between child orders in milliseconds
    pub max_delay_ms: u64,
}

impl Default for TxSplitConfig {
    fn default() -> Self {
        TxSplitConfig {
            min_splits: 5,
            max_splits: 20,
            min_order_size: 0.001,
            size_randomization: 0.3,
            time_jitter_factor: 0.5,
            min_delay_ms: 100,
            max_delay_ms: 2000,
        }
    }
}

/// Represents a single child order in a split sequence.
#[derive(Clone, Debug)]
pub struct ChildOrder {
    pub sequence: usize,
    pub size: f64,
    pub wallet_index: usize,
    pub delay_ms: u64,
    pub order_type: OrderType,
}

/// Type of order to place.
#[derive(Clone, Debug)]
pub enum OrderType {
    Limit { price: f64 },
    Market,
}

/// Result of executing a child order.
#[derive(Clone, Debug)]
pub struct ChildOrderResult {
    pub sequence: usize,
    pub success: bool,
    pub executed_size: Option<f64>,
    pub tx_hash: Option<String>,
    pub error: Option<String>,
    pub execution_time: Duration,
}

/// Smart transaction splitter for MEV protection.
pub struct TxSplitter {
    config: TxSplitConfig,
    wallets: Arc<RwLock<Vec<String>>>, // Wallet addresses
}

impl TxSplitter {
    /// Creates a new transaction splitter.
    pub fn new(config: TxSplitConfig) -> Self {
        TxSplitter {
            config,
            wallets: Arc::new(RwLock::new(Vec::new())),
        }
    }

    /// Adds a wallet address to the rotation pool.
    pub async fn add_wallet(&self, address: String) {
        let mut wallets = self.wallets.write().await;
        wallets.push(address);
    }

    /// Splits a large order into multiple randomized child orders.
    pub async fn split_order(
        &self,
        total_size: f64,
        order_type: OrderType,
    ) -> Vec<ChildOrder> {
        let mut rng = rand::thread_rng();
        let wallets = self.wallets.read().await;
        
        // Determine number of splits
        let num_splits = if wallets.is_empty() {
            rng.gen_range(self.config.min_splits..=self.config.max_splits)
        } else {
            // Prefer using all available wallets
            wallets.len().max(self.config.min_splits).min(self.config.max_splits)
        };

        // Calculate base size per split
        let base_size = total_size / num_splits as f64;
        
        let mut child_orders = Vec::with_capacity(num_splits);
        let mut remaining_size = total_size;

        for i in 0..num_splits {
            // Apply randomization to size
            let randomization = if i == num_splits - 1 {
                // Last order takes remaining size to ensure exact total
                remaining_size
            } else {
                let variance = base_size * self.config.size_randomization;
                let random_factor = rng.gen_range(-variance..=variance);
                let size = (base_size + random_factor).max(self.config.min_order_size);
                remaining_size -= size;
                size
            };

            // Select wallet (round-robin or random if fewer wallets than splits)
            let wallet_index = if wallets.is_empty() {
                0 // Placeholder if no wallets configured
            } else {
                i % wallets.len()
            };

            // Calculate random delay
            let delay_ms = rng.gen_range(self.config.min_delay_ms..=self.config.max_delay_ms);

            child_orders.push(ChildOrder {
                sequence: i,
                size: randomization,
                wallet_index,
                delay_ms,
                order_type: order_type.clone(),
            });
        }

        child_orders
    }

    /// Executes the split orders with randomized timing.
    /// The `execute_fn` callback is called for each child order.
    pub async fn execute_split<F, Fut>(
        &self,
        child_orders: Vec<ChildOrder>,
        execute_fn: F,
    ) -> Vec<ChildOrderResult>
    where
        F: Fn(ChildOrder) -> Fut + Sync,
        Fut: Future<Output = ChildOrderResult>,
    {
        let mut results = Vec::with_capacity(child_orders.len());
        let start_time = Instant::now();

        for (idx, order) in child_orders.into_iter().enumerate() {
            // Wait for the specified delay
            if idx > 0 {
                tokio::time::sleep(Duration::from_millis(order.delay_ms)).await;
            }

            // Execute the order
            let exec_start = Instant::now();
            let result = execute_fn(order).await;
            
            // Record execution time
            let mut result = result;
            result.execution_time = exec_start.elapsed();
            
            results.push(result);
        }

        let total_time = start_time.elapsed();
        log_execution_summary(&results, total_time);
        
        results
    }

    /// Gets the list of configured wallets.
    pub async fn get_wallets(&self) -> Vec<String> {
        self.wallets.read().await.clone()
    }
}

fn log_execution_summary(results: &[ChildOrderResult], total_time: Duration) {
    let total_orders = results.len();
    let successful = results.iter().filter(|r| r.success).count();
    let failed = total_orders - successful;
    
    let total_executed: f64 = results
        .iter()
        .filter_map(|r| r.executed_size)
        .sum();

    eprintln!(
        "[TX_SPLITTER] Execution complete: {}/{} successful, {} failed. \
         Total executed: {:.8}, Total time: {:?}",
        successful, total_orders, failed, total_executed, total_time
    );
}

// Required for the trait bound in execute_split
use std::future::Future;

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_order_splitting() {
        let config = TxSplitConfig {
            min_splits: 3,
            max_splits: 5,
            min_order_size: 0.001,
            size_randomization: 0.2,
            time_jitter_factor: 0.5,
            min_delay_ms: 50,
            max_delay_ms: 200,
        };

        let splitter = TxSplitter::new(config);
        
        // Add some test wallets
        splitter.add_wallet("0xWallet1".to_string()).await;
        splitter.add_wallet("0xWallet2".to_string()).await;
        splitter.add_wallet("0xWallet3".to_string()).await;

        let total_size = 1.0;
        let order_type = OrderType::Limit { price: 50000.0 };
        
        let child_orders = splitter.split_order(total_size, order_type).await;
        
        assert!(child_orders.len() >= 3);
        assert!(child_orders.len() <= 5);
        
        // Verify total size is preserved
        let total: f64 = child_orders.iter().map(|o| o.size).sum();
        assert!((total - total_size).abs() < 0.0001);
        
        // Verify delays are within range
        for order in &child_orders {
            assert!(order.delay_ms >= 50);
            assert!(order.delay_ms <= 200);
        }
    }

    #[tokio::test]
    async fn test_execute_split() {
        let config = TxSplitConfig {
            min_splits: 2,
            max_splits: 2,
            min_order_size: 0.001,
            size_randomization: 0.0,
            time_jitter_factor: 0.0,
            min_delay_ms: 10,
            max_delay_ms: 10,
        };

        let splitter = TxSplitter::new(config);
        
        let child_orders = vec![
            ChildOrder {
                sequence: 0,
                size: 0.5,
                wallet_index: 0,
                delay_ms: 10,
                order_type: OrderType::Market,
            },
            ChildOrder {
                sequence: 1,
                size: 0.5,
                wallet_index: 0,
                delay_ms: 10,
                order_type: OrderType::Market,
            },
        ];

        let results = splitter
            .execute_split(child_orders, |order| async move {
                // Mock execution
                ChildOrderResult {
                    sequence: order.sequence,
                    success: true,
                    executed_size: Some(order.size),
                    tx_hash: Some(format!("0xtx{}", order.sequence)),
                    error: None,
                    execution_time: Duration::from_millis(5),
                }
            })
            .await;

        assert_eq!(results.len(), 2);
        assert!(results.iter().all(|r| r.success));
    }
}
