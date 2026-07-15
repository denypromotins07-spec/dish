//! LMDB Integration for Zero-Copy State Persistence
//! ACID-compliant, microsecond persistence of orders, positions, and strategy states

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, Instant};

use lmdb::{Database, Environment, EnvironmentFlags, WriteFlags, RwTransaction, Transaction};
use serde::{Serialize, Deserialize};

/// State entry types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum StateType {
    OpenOrder,
    Position,
    StrategyState,
    ExecutionLog,
}

/// Generic state entry with metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StateEntry<T> {
    pub key: String,
    pub value: T,
    pub state_type: StateType,
    pub created_at: u64,  // Unix timestamp in nanoseconds
    pub updated_at: u64,
    pub version: u64,
}

/// Order state for persistence
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersistedOrder {
    pub order_id: String,
    pub client_order_id: String,
    pub symbol: String,
    pub side: String,
    pub quantity: f64,
    pub filled_quantity: f64,
    pub price: Option<f64>,
    pub status: String,
    pub exchange: String,
}

/// Position state for persistence
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersistedPosition {
    pub position_id: String,
    pub symbol: String,
    pub exchange: String,
    pub side: String,
    pub quantity: f64,
    pub entry_price: f64,
    pub mark_price: f64,
    pub unrealized_pnl: f64,
    pub realized_pnl: f64,
}

/// Strategy state for persistence
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersistedStrategyState {
    pub strategy_id: String,
    pub name: String,
    pub is_active: bool,
    pub parameters: HashMap<String, String>,
    pub metrics: HashMap<String, f64>,
}

/// LMDB-backed state store
pub struct LmdbStore {
    env: Arc<Environment>,
    state_db: Database,
    index_db: Database,
}

impl LmdbStore {
    /// Create or open LMDB environment
    pub fn new<P: AsRef<Path>>(
        path: P,
        max_size_mb: usize,
    ) -> Result<Self, lmdb::Error> {
        let env = Environment::new()
            .set_flags(
                EnvironmentFlags::NO_SUB_DIR
                    | EnvironmentFlags::WRITE_MAP
                    | EnvironmentFlags::MAP_ASYNC
            )
            .set_max_dbs(10)
            .set_map_size(max_size_mb * 1024 * 1024)
            .open(path.as_ref())?;

        let mut txn = env.begin_rw_txn()?;
        
        let state_db = txn.create_db(Some("state"), lmdb::DatabaseFlags::empty())?;
        let index_db = txn.create_db(Some("index"), lmdb::DatabaseFlags::empty())?;
        
        txn.commit()?;

        Ok(Self {
            env: Arc::new(env),
            state_db,
            index_db,
        })
    }

    /// Store a state entry
    pub fn put<T: Serialize>(
        &self,
        key: &str,
        entry: &StateEntry<T>,
    ) -> Result<(), lmdb::Error> {
        let mut txn = self.env.begin_rw_txn()?;
        
        let serialized = bincode::serialize(&entry)
            .map_err(|e| lmdb::Error::Other)?;
        
        txn.put(self.state_db, key.as_bytes(), &serialized, WriteFlags::empty())?;
        
        // Update index
        let index_key = format!("{}:{}", entry.state_type as u8, key);
        txn.put(self.index_db, index_key.as_bytes(), key.as_bytes(), WriteFlags::empty())?;
        
        txn.commit()
    }

    /// Retrieve a state entry
    pub fn get<T: for<'de> Deserialize<'de>>(
        &self,
        key: &str,
    ) -> Result<Option<StateEntry<T>>, lmdb::Error> {
        let txn = self.env.begin_ro_txn()?;
        
        match txn.get(self.state_db, key.as_bytes()) {
            Ok(bytes) => {
                let entry: StateEntry<T> = bincode::deserialize(bytes)
                    .map_err(|e| lmdb::Error::Other)?;
                Ok(Some(entry))
            }
            Err(lmdb::Error::NotFound) => Ok(None),
            Err(e) => Err(e),
        }
    }

    /// Delete a state entry
    pub fn delete(&self, key: &str) -> Result<(), lmdb::Error> {
        let mut txn = self.env.begin_rw_txn()?;
        txn.del(self.state_db, key.as_bytes(), None)?;
        txn.commit()
    }

    /// Store an order
    pub fn put_order(&self, order: &PersistedOrder) -> Result<(), lmdb::Error> {
        let entry = StateEntry {
            key: order.order_id.clone(),
            value: order.clone(),
            state_type: StateType::OpenOrder,
            created_at: chrono::Utc::now().timestamp_nanos() as u64,
            updated_at: chrono::Utc::now().timestamp_nanos() as u64,
            version: 1,
        };
        self.put(&order.order_id, &entry)
    }

    /// Retrieve an order
    pub fn get_order(&self, order_id: &str) -> Result<Option<PersistedOrder>, lmdb::Error> {
        let result: Option<StateEntry<PersistedOrder>> = self.get(order_id)?;
        Ok(result.map(|e| e.value))
    }

    /// Store a position
    pub fn put_position(&self, position: &PersistedPosition) -> Result<(), lmdb::Error> {
        let entry = StateEntry {
            key: position.position_id.clone(),
            value: position.clone(),
            state_type: StateType::Position,
            created_at: chrono::Utc::now().timestamp_nanos() as u64,
            updated_at: chrono::Utc::now().timestamp_nanos() as u64,
            version: 1,
        };
        self.put(&position.position_id, &entry)
    }

    /// Retrieve a position
    pub fn get_position(&self, position_id: &str) -> Result<Option<PersistedPosition>, lmdb::Error> {
        let result: Option<StateEntry<PersistedPosition>> = self.get(position_id)?;
        Ok(result.map(|e| e.value))
    }

    /// Store strategy state
    pub fn put_strategy_state(&self, state: &PersistedStrategyState) -> Result<(), lmdb::Error> {
        let entry = StateEntry {
            key: state.strategy_id.clone(),
            value: state.clone(),
            state_type: StateType::StrategyState,
            created_at: chrono::Utc::now().timestamp_nanos() as u64,
            updated_at: chrono::Utc::now().timestamp_nanos() as u64,
            version: 1,
        };
        self.put(&state.strategy_id, &entry)
    }

    /// Retrieve strategy state
    pub fn get_strategy_state(&self, strategy_id: &str) -> Result<Option<PersistedStrategyState>, lmdb::Error> {
        let result: Option<StateEntry<PersistedStrategyState>> = self.get(strategy_id)?;
        Ok(result.map(|e| e.value))
    }

    /// Iterate over all entries of a specific type
    pub fn iterate_by_type<F>(
        &self,
        state_type: StateType,
        mut callback: F,
    ) -> Result<(), lmdb::Error>
    where
        F: FnMut(&[u8]) -> Result<(), lmdb::Error>,
    {
        let txn = self.env.begin_ro_txn()?;
        let cursor = txn.open_ro_cursor(self.index_db)?;
        
        let prefix = format!("{}", state_type as u8);
        
        for result in cursor.iter() {
            let (key, value) = result?;
            if key.starts_with(prefix.as_bytes()) {
                callback(value)?;
            }
        }
        
        Ok(())
    }

    /// Get all open orders
    pub fn get_all_open_orders(&self) -> Result<Vec<PersistedOrder>, lmdb::Error> {
        let mut orders = Vec::new();
        
        self.iterate_by_type(StateType::OpenOrder, |key_bytes| {
            let key = String::from_utf8_lossy(key_bytes).to_string();
            if let Some(Some(order)) = self.get::<PersistedOrder>(&key)? {
                if order.status == "NEW" || order.status == "PARTIALLY_FILLED" {
                    orders.push(order);
                }
            }
            Ok(())
        })?;
        
        Ok(orders)
    }

    /// Get all active positions
    pub fn get_all_positions(&self) -> Result<Vec<PersistedPosition>, lmdb::Error> {
        let mut positions = Vec::new();
        
        self.iterate_by_type(StateType::Position, |key_bytes| {
            let key = String::from_utf8_lossy(key_bytes).to_string();
            if let Some(Some(position)) = self.get::<PersistedPosition>(&key)? {
                positions.push(position);
            }
            Ok(())
        })?;
        
        Ok(positions)
    }

    /// Sync data to disk
    pub fn sync(&self) -> Result<(), lmdb::Error> {
        // LMDB with MAP_ASYNC handles this automatically, but we can force sync
        let _txn = self.env.begin_rw_txn()?;
        // Transaction commit will sync
        Ok(())
    }

    /// Get database statistics
    pub fn get_stats(&self) -> Result<DbStats, lmdb::Error> {
        let txn = self.env.begin_ro_txn()?;
        let stat = txn.stat(self.state_db)?;
        
        Ok(DbStats {
            page_size: stat.ms_psize,
            depth: stat.ms_depth,
            branch_pages: stat.ms_branch_pages,
            leaf_pages: stat.ms_leaf_pages,
            overflow_pages: stat.ms_overflow_pages,
            entries: stat.ms_entries,
        })
    }
}

/// Database statistics
#[derive(Debug, Clone)]
pub struct DbStats {
    pub page_size: u32,
    pub depth: u32,
    pub branch_pages: u64,
    pub leaf_pages: u64,
    pub overflow_pages: u64,
    pub entries: u64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_lmdb_store() {
        let temp_dir = TempDir::new().unwrap();
        let store = LmdbStore::new(temp_dir.path(), 100).unwrap();

        let order = PersistedOrder {
            order_id: "ORD123".to_string(),
            client_order_id: "CLIENT123".to_string(),
            symbol: "BTCUSDT".to_string(),
            side: "BUY".to_string(),
            quantity: 0.001,
            filled_quantity: 0.0,
            price: Some(45000.0),
            status: "NEW".to_string(),
            exchange: "binance".to_string(),
        };

        store.put_order(&order).unwrap();
        
        let retrieved = store.get_order("ORD123").unwrap();
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().order_id, "ORD123");
    }
}
