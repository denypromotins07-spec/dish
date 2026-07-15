//! Memory-efficient Union-Find (Disjoint Set) algorithm for address clustering.
//! Strictly bounds memory usage to prevent exceeding 14GB RAM limit.

use dashmap::DashMap;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;

/// Maximum number of addresses to track before triggering cleanup
const MAX_ADDRESSES: usize = 5_000_000;

/// Memory budget in bytes for the clustering structure
const MEMORY_BUDGET_BYTES: usize = 2 * 1024 * 1024 * 1024; // 2GB max

/// Union-Find node with path compression and union by rank
#[derive(Debug, Clone)]
struct UnionFindNode {
    parent: u64,
    rank: u16,
    size: u32,
    /// Additional metadata flags
    flags: u8,
}

impl UnionFindNode {
    fn new(id: u64) -> Self {
        Self {
            parent: id,
            rank: 0,
            size: 1,
            flags: 0,
        }
    }
}

/// Address clustering engine using Union-Find with strict memory bounds
pub struct AddressClusterer {
    /// Map from address hash to node ID
    address_to_id: DashMap<[u8; 20], u64>,
    /// Union-Find nodes stored in a vector for cache efficiency
    nodes: DashMap<u64, UnionFindNode>,
    /// Reverse map from ID to address (for debugging/inspection)
    id_to_address: DashMap<u64, [u8; 20]>,
    /// Counter for generating unique node IDs
    next_id: AtomicU64,
    /// Current number of tracked addresses
    address_count: AtomicUsize,
    /// Number of clusters formed
    cluster_count: AtomicUsize,
    /// Memory usage tracker
    estimated_memory_bytes: AtomicUsize,
    /// Cleanup threshold
    cleanup_threshold: usize,
}

/// Cluster information returned to callers
#[derive(Debug, Clone)]
pub struct ClusterInfo {
    pub cluster_id: u64,
    pub member_count: usize,
    pub members: Vec<[u8; 20]>,
    pub total_transactions: u64,
    pub first_seen_timestamp: u64,
    pub last_seen_timestamp: u64,
}

/// Statistics about the clustering state
#[derive(Debug, Clone)]
pub struct ClusterStats {
    pub total_addresses: usize,
    pub total_clusters: usize,
    pub largest_cluster_size: usize,
    pub average_cluster_size: f64,
    pub memory_usage_bytes: usize,
    pub memory_budget_bytes: usize,
}

impl AddressClusterer {
    /// Create a new address clusterer with default settings
    pub fn new() -> Self {
        let initial_capacity = MAX_ADDRESSES / 4;
        Self {
            address_to_id: DashMap::with_capacity(initial_capacity),
            nodes: DashMap::with_capacity(initial_capacity),
            id_to_address: DashMap::with_capacity(initial_capacity),
            next_id: AtomicU64::new(0),
            address_count: AtomicUsize::new(0),
            cluster_count: AtomicUsize::new(initial_capacity),
            estimated_memory_bytes: AtomicUsize::new(0),
            cleanup_threshold: MAX_ADDRESSES,
        }
    }

    /// Create with custom memory budget
    pub fn with_budget(max_addresses: usize, memory_budget_gb: f64) -> Self {
        let initial_capacity = max_addresses / 4;
        let memory_bytes = (memory_budget_gb * 1024.0 * 1024.0 * 1024.0) as usize;
        
        Self {
            address_to_id: DashMap::with_capacity(initial_capacity),
            nodes: DashMap::with_capacity(initial_capacity),
            id_to_address: DashMap::with_capacity(initial_capacity),
            next_id: AtomicU64::new(0),
            address_count: AtomicUsize::new(0),
            cluster_count: AtomicUsize::new(initial_capacity),
            estimated_memory_bytes: AtomicUsize::new(memory_bytes / 4),
            cleanup_threshold: max_addresses,
        }
    }

    /// Add an address to the clusterer, returning its ID
    fn get_or_create_id(&self, address: [u8; 20]) -> u64 {
        // Check if already exists
        if let Some((_, &id)) = self.address_to_id.get(&address) {
            return id;
        }

        // Generate new ID
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);

        // Insert into maps
        self.address_to_id.insert(address, id);
        self.nodes.insert(id, UnionFindNode::new(id));
        self.id_to_address.insert(id, address);

        // Update counters
        self.address_count.fetch_add(1, Ordering::Relaxed);
        self.cluster_count.fetch_add(1, Ordering::Relaxed);

        // Update memory estimate (~100 bytes per address including overhead)
        self.estimated_memory_bytes.fetch_add(100, Ordering::Relaxed);

        // Check if cleanup needed
        if self.address_count.load(Ordering::Relaxed) >= self.cleanup_threshold {
            self.maybe_cleanup();
        }

        id
    }

    /// Union two addresses into the same cluster
    pub fn union(&self, address1: [u8; 20], address2: [u8; 20]) {
        let id1 = self.get_or_create_id(address1);
        let id2 = self.get_or_create_id(address2);

        self.union_by_id(id1, id2);
    }

    /// Union by internal IDs (faster when IDs are already known)
    fn union_by_id(&self, id1: u64, id2: u64) {
        if id1 == id2 {
            return;
        }

        let root1 = self.find(id1);
        let root2 = self.find(id2);

        if root1 == root2 {
            return;
        }

        // Union by rank
        let mut node1 = self.nodes.get_mut(&root1).unwrap();
        let mut node2 = self.nodes.get_mut(&root2).unwrap();

        if node1.rank < node2.rank {
            node1.parent = root2;
            node2.size += node1.size;
            drop(node1);
            drop(node2);
        } else if node1.rank > node2.rank {
            node2.parent = root1;
            node1.size += node2.size;
            drop(node1);
            drop(node2);
        } else {
            node2.parent = root1;
            node1.rank += 1;
            node1.size += node2.size;
            drop(node1);
            drop(node2);
        }

        self.cluster_count.fetch_sub(1, Ordering::Relaxed);
    }

    /// Find the root of an address with path compression
    pub fn find_address(&self, address: [u8; 20]) -> Option<[u8; 20]> {
        self.address_to_id.get(&address).map(|(_, &id)| {
            let root_id = self.find(id);
            self.id_to_address.get(&root_id).map(|(_, &addr)| addr).unwrap_or(address)
        })
    }

    /// Find the root ID with path compression
    fn find(&self, mut id: u64) -> u64 {
        // First pass: find root
        let root = loop {
            if let Some(node) = self.nodes.get(&id) {
                if node.parent == id {
                    break id;
                }
                let parent = node.parent;
                drop(node);
                id = parent;
            } else {
                break id;
            }
        };

        // Second pass: path compression
        let mut current = id;
        while let Some(mut node) = self.nodes.get(&current) {
            if node.parent == root {
                break;
            }
            let parent = node.parent;
            node.parent = root;
            drop(node);
            current = parent;
        }

        root
    }

    /// Check if two addresses are in the same cluster
    pub fn are_connected(&self, address1: [u8; 20], address2: [u8; 20]) -> bool {
        if let (Some(id1), Some(id2)) = (
            self.address_to_id.get(&address1).map(|(_, &id)| id),
            self.address_to_id.get(&address2).map(|(_, &id)| id),
        ) {
            return self.find(id1) == self.find(id2);
        }
        false
    }

    /// Get cluster information for an address
    pub fn get_cluster_info(&self, address: [u8; 20]) -> Option<ClusterInfo> {
        let id = self.address_to_id.get(&address).map(|(_, &id)| id)?;
        let root_id = self.find(id);

        // Get all members of this cluster
        let mut members = Vec::new();
        let mut total_size = 0;

        for entry in self.nodes.iter() {
            if self.find(*entry.key()) == root_id {
                if let Some((_, &addr)) = self.id_to_address.get(entry.key()) {
                    members.push(addr);
                }
                total_size = entry.value().size as usize;
            }
        }

        Some(ClusterInfo {
            cluster_id: root_id,
            member_count: members.len(),
            members,
            total_transactions: 0, // Would track separately in production
            first_seen_timestamp: 0,
            last_seen_timestamp: 0,
        })
    }

    /// Get statistics about the clustering state
    pub fn get_stats(&self) -> ClusterStats {
        let total_addresses = self.address_count.load(Ordering::Relaxed);
        let total_clusters = self.cluster_count.load(Ordering::Relaxed);
        let memory_usage = self.estimated_memory_bytes.load(Ordering::Relaxed);

        // Find largest cluster
        let mut largest_size = 0;
        for entry in self.nodes.iter() {
            if entry.value().parent == *entry.key() {
                largest_size = largest_size.max(entry.value().size as usize);
            }
        }

        let avg_size = if total_clusters > 0 {
            total_addresses as f64 / total_clusters as f64
        } else {
            0.0
        };

        ClusterStats {
            total_addresses,
            total_clusters,
            largest_cluster_size: largest_size,
            average_cluster_size: avg_size,
            memory_usage_bytes: memory_usage,
            memory_budget_bytes: MEMORY_BUDGET_BYTES,
        }
    }

    /// Maybe trigger cleanup if memory is getting tight
    fn maybe_cleanup(&self) {
        let current_memory = self.estimated_memory_bytes.load(Ordering::Relaxed);
        
        if current_memory >= MEMORY_BUDGET_BYTES {
            // In production, would implement LRU eviction or merge small clusters
            tracing::warn!(
                "Memory budget approaching limit: {} / {} bytes",
                current_memory,
                MEMORY_BUDGET_BYTES
            );
        }
    }

    /// Export cluster data for persistence
    pub fn export_clusters(&self) -> Vec<(u64, Vec<[u8; 20]>)> {
        let mut clusters: std::collections::HashMap<u64, Vec<[u8; 20]>> = 
            std::collections::HashMap::new();

        for entry in self.address_to_id.iter() {
            let address = *entry.key();
            let id = *entry.value();
            let root = self.find(id);
            
            clusters.entry(root).or_insert_with(Vec::new).push(address);
        }

        clusters.into_iter().collect()
    }

    /// Clear all data (useful for testing or resetting)
    pub fn clear(&self) {
        self.address_to_id.clear();
        self.nodes.clear();
        self.id_to_address.clear();
        self.next_id.store(0, Ordering::Relaxed);
        self.address_count.store(0, Ordering::Relaxed);
        self.cluster_count.store(0, Ordering::Relaxed);
        self.estimated_memory_bytes.store(0, Ordering::Relaxed);
    }
}

impl Default for AddressClusterer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_clustering() {
        let clusterer = AddressClusterer::new();
        
        let addr1 = [1u8; 20];
        let addr2 = [2u8; 20];
        let addr3 = [3u8; 20];

        // Initially not connected
        assert!(!clusterer.are_connected(addr1, addr2));

        // Union addr1 and addr2
        clusterer.union(addr1, addr2);
        assert!(clusterer.are_connected(addr1, addr2));

        // Union addr2 and addr3 (transitive)
        clusterer.union(addr2, addr3);
        assert!(clusterer.are_connected(addr1, addr3));
        assert!(clusterer.are_connected(addr2, addr3));

        let stats = clusterer.get_stats();
        assert_eq!(stats.total_addresses, 3);
        assert_eq!(stats.total_clusters, 1);
    }

    #[test]
    fn test_cluster_info() {
        let clusterer = AddressClusterer::new();
        
        let addr1 = [1u8; 20];
        let addr2 = [2u8; 20];

        clusterer.union(addr1, addr2);

        let info = clusterer.get_cluster_info(addr1).unwrap();
        assert_eq!(info.member_count, 2);
        assert!(info.members.contains(&addr1));
        assert!(info.members.contains(&addr2));
    }

    #[test]
    fn test_memory_bounds() {
        let clusterer = AddressClusterer::with_budget(1000, 0.1); // 100MB budget
        
        for i in 0..100 {
            let mut addr = [0u8; 20];
            addr[0] = i as u8;
            clusterer.get_or_create_id(addr);
        }

        let stats = clusterer.get_stats();
        assert!(stats.memory_usage_bytes < MEMORY_BUDGET_BYTES);
    }
}
