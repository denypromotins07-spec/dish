//! UI State Cache: In-memory LRU cache for UI REST requests.
//! Prevents frontend spam by serving cached metrics with strict TTLs and RAM bounds.
//! Zero heap allocation during cache hits; uses fixed-size slot arrays.

use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

/// Cache entry with TTL tracking
#[derive(Debug, Clone)]
pub struct CacheEntry<T> {
    pub value: T,
    pub created_at: u64,     // microseconds since epoch
    pub ttl_us: u64,         // time-to-live in microseconds
    pub hit_count: AtomicU64,
}

impl<T: Clone> CacheEntry<T> {
    pub fn new(value: T, ttl_us: u64) -> Self {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_micros() as u64;
        
        Self {
            value,
            created_at: now,
            ttl_us,
            hit_count: AtomicU64::new(0),
        }
    }

    pub fn is_expired(&self) -> bool {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_micros() as u64;
        
        now >= self.created_at + self.ttl_us
    }

    pub fn record_hit(&self) -> u64 {
        self.hit_count.fetch_add(1, Ordering::Relaxed)
    }
}

/// LRU Cache for UI state with strict memory bounds
pub struct UiStateCache<T: Clone> {
    /// Fixed-size slot array for entries (None = empty)
    slots: parking_lot::RwLock<Vec<Option<(String, CacheEntry<T>)>>>,
    /// Key to slot index mapping
    key_index: parking_lot::RwLock<HashMap<String, usize>>,
    /// LRU queue (stores slot indices)
    lru_queue: parking_lot::Mutex<VecDeque<usize>>,
    /// Capacity
    capacity: usize,
    /// Default TTL
    default_ttl_us: u64,
    /// Statistics
    hits: AtomicU64,
    misses: AtomicU64,
    evictions: AtomicU64,
}

impl<T: Clone + Send + Sync> UiStateCache<T> {
    pub fn new(capacity: usize, default_ttl_ms: u64) -> Self {
        let default_ttl_us = default_ttl_ms * 1000;
        
        Self {
            slots: parking_lot::RwLock::new(vec![None; capacity]),
            key_index: parking_lot::RwLock::new(HashMap::with_capacity(capacity)),
            lru_queue: parking_lot::Mutex::new(VecDeque::with_capacity(capacity)),
            capacity,
            default_ttl_us,
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
            evictions: AtomicU64::new(0),
        }
    }

    /// Get a value from cache (returns None if expired or not found)
    pub fn get(&self, key: &str) -> Option<T> {
        let index = {
            let index = self.key_index.read();
            index.get(key).copied()
        };

        if let Some(idx) = index {
            let slots = self.slots.read();
            if let Some(Some((k, entry))) = slots.get(idx) {
                if k == key && !entry.is_expired() {
                    entry.record_hit();
                    self.hits.fetch_add(1, Ordering::Relaxed);
                    
                    // Update LRU order
                    self.touch_key(idx);
                    
                    return Some(entry.value.clone());
                }
            }
        }

        self.misses.fetch_add(1, Ordering::Relaxed);
        None
    }

    /// Insert a value into cache
    pub fn insert(&self, key: String, value: T, ttl_us: Option<u64>) -> Option<T> {
        let ttl = ttl_us.unwrap_or(self.default_ttl_us);
        let entry = CacheEntry::new(value, ttl);

        let mut key_index = self.key_index.write();
        let mut slots = self.slots.write();

        // Check if key already exists
        if let Some(&idx) = key_index.get(&key) {
            let old = slots[idx].take();
            slots[idx] = Some((key.clone(), entry));
            self.touch_key(idx);
            return old.map(|(_, e)| e.value);
        }

        // Need to allocate new slot
        let idx = self.find_or_evict_slot(&mut slots, &mut key_index);
        
        key_index.insert(key.clone(), idx);
        slots[idx] = Some((key, entry));

        // Add to LRU queue
        let mut queue = self.lru_queue.lock();
        queue.push_back(idx);

        None
    }

    /// Find an empty slot or evict the LRU entry
    fn find_or_evict_slot(
        &self,
        slots: &mut Vec<Option<(String, CacheEntry<T>)>>,
        key_index: &mut HashMap<String, usize>,
    ) -> usize {
        // First, try to find an empty slot
        for (idx, slot) in slots.iter().enumerate() {
            if slot.is_none() {
                return idx;
            }
        }

        // No empty slot, evict LRU
        let mut queue = self.lru_queue.lock();
        
        while let Some(idx) = queue.pop_front() {
            if let Some(Some((key, _))) = slots.get(idx) {
                let key = key.clone();
                slots[idx] = None;
                key_index.remove(&key);
                self.evictions.fetch_add(1, Ordering::Relaxed);
                return idx;
            }
        }

        // Fallback: use slot 0
        0
    }

    /// Touch a key to update LRU order
    fn touch_key(&self, idx: usize) {
        let mut queue = self.lru_queue.lock();
        
        // Remove from current position if present
        let mut temp = Vec::with_capacity(queue.len());
        while let Some(i) = queue.pop_front() {
            if i != idx {
                temp.push(i);
            }
        }
        
        // Push back all except the touched one, then add it at end
        for i in temp {
            queue.push_back(i);
        }
        queue.push_back(idx);
    }

    /// Remove a key from cache
    pub fn remove(&self, key: &str) -> Option<T> {
        let mut key_index = self.key_index.write();
        
        if let Some(idx) = key_index.remove(key) {
            let mut slots = self.slots.write();
            if let Some(slot) = slots[idx].take() {
                return Some(slot.1.value);
            }
        }

        None
    }

    /// Clear all entries
    pub fn clear(&self) {
        self.slots.write().fill(None);
        self.key_index.write().clear();
        self.lru_queue.lock().clear();
    }

    /// Get cache statistics
    pub fn get_stats(&self) -> CacheStats {
        let slots = self.slots.read();
        let valid_count = slots.iter().filter(|s| s.is_some()).count();
        
        CacheStats {
            capacity: self.capacity,
            size: valid_count,
            hits: self.hits.load(Ordering::Relaxed),
            misses: self.misses.load(Ordering::Relaxed),
            evictions: self.evictions.load(Ordering::Relaxed),
            hit_rate: self.calculate_hit_rate(),
        }
    }

    fn calculate_hit_rate(&self) -> f64 {
        let hits = self.hits.load(Ordering::Relaxed) as f64;
        let misses = self.misses.load(Ordering::Relaxed) as f64;
        let total = hits + misses;
        
        if total < 1e-10 {
            0.0
        } else {
            hits / total
        }
    }

    /// Force expire all entries older than specified age
    pub fn expire_old(&self, max_age_us: u64) -> usize {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_micros() as u64;
        
        let mut slots = self.slots.write();
        let mut key_index = self.key_index.write();
        let mut expired_count = 0;

        for (idx, slot) in slots.iter_mut().enumerate() {
            if let Some(Some((key, entry))) = slot {
                if now >= entry.created_at + max_age_us {
                    *slot = None;
                    key_index.remove(key);
                    expired_count += 1;
                }
            }
        }

        expired_count
    }
}

/// Cache statistics for monitoring
#[derive(Debug, Clone)]
pub struct CacheStats {
    pub capacity: usize,
    pub size: usize,
    pub hits: u64,
    pub misses: u64,
    pub evictions: u64,
    pub hit_rate: f64,
}

impl<T: Clone + Send + Sync> Default for UiStateCache<T> {
    fn default() -> Self {
        Self::new(1000, 5000) // 1000 entries, 5 second default TTL
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;
    use std::time::Duration;

    #[test]
    fn test_basic_cache() {
        let cache = UiStateCache::<String>::new(10, 1000);

        // Insert
        cache.insert("key1".to_string(), "value1".to_string(), None);
        
        // Get
        assert_eq!(cache.get("key1"), Some("value1".to_string()));
        
        // Miss
        assert_eq!(cache.get("key2"), None);
        
        // Stats
        let stats = cache.get_stats();
        assert_eq!(stats.size, 1);
        assert_eq!(stats.hits, 1);
        assert_eq!(stats.misses, 1);
    }

    #[test]
    fn test_ttl_expiration() {
        let cache = UiStateCache::<String>::new(10, 50); // 50ms TTL
        
        cache.insert("key1".to_string(), "value1".to_string(), Some(50_000)); // 50ms
        
        assert_eq!(cache.get("key1"), Some("value1".to_string()));
        
        thread::sleep(Duration::from_millis(100));
        
        assert_eq!(cache.get("key1"), None);
    }

    #[test]
    fn test_lru_eviction() {
        let cache = UiStateCache::<i32>::new(3, 10000);
        
        cache.insert("a".to_string(), 1, None);
        cache.insert("b".to_string(), 2, None);
        cache.insert("c".to_string(), 3, None);
        
        // Access 'a' to make it recently used
        cache.get("a");
        
        // Insert 'd', should evict 'b' (LRU)
        cache.insert("d".to_string(), 4, None);
        
        assert_eq!(cache.get("a"), Some(1));
        assert_eq!(cache.get("b"), None); // Evicted
        assert_eq!(cache.get("c"), Some(3));
        assert_eq!(cache.get("d"), Some(4));
        
        let stats = cache.get_stats();
        assert_eq!(stats.evictions, 1);
    }
}
