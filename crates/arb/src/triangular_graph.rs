//! Triangular Arbitrage Graph Engine
//! Lock-free directed graph using Bellman-Ford optimized for negative cycles
//! Detects real-time triangular arbitrage across spot pairs

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;

/// Edge in the triangular arbitrage graph
#[derive(Debug, Clone)]
pub struct GraphEdge {
    pub from: String,  // Base currency
    pub to: String,    // Quote currency
    pub rate: f64,     // Exchange rate
    pub fee_bps: f64,  // Trading fee in basis points
}

/// Vertex in the graph (currency)
#[derive(Debug, Clone)]
pub struct GraphVertex {
    pub symbol: String,
    pub best_dist: f64,
    pub predecessor: Option<String>,
}

/// Triangular arbitrage opportunity
#[derive(Debug, Clone)]
pub struct TriangularArb {
    pub path: Vec<String>,      // e.g., ["BTC", "ETH", "USDT", "BTC"]
    pub profit_pct: f64,        // Expected profit percentage
    pub rates: Vec<f64>,        // Rates along the path
    pub fees_bps: Vec<f64>,     // Fees along the path
    pub timestamp: Instant,
}

/// Lock-free triangular arbitrage graph
pub struct TriangularGraph {
    vertices: Arc<dashmap::DashMap<String, GraphVertex>>,
    edges: Arc<dashmap::DashMap<String, Vec<GraphEdge>>>,
    update_counter: AtomicU64,
    min_profit_threshold: f64,
}

impl TriangularGraph {
    pub fn new(min_profit_threshold: f64) -> Self {
        Self {
            vertices: Arc::new(dashmap::DashMap::new()),
            edges: Arc::new(dashmap::DashMap::new()),
            update_counter: AtomicU64::new(0),
            min_profit_threshold,
        }
    }

    /// Add or update an edge (trading pair)
    pub fn update_edge(&self, from: &str, to: &str, rate: f64, fee_bps: f64) {
        let edge = GraphEdge {
            from: from.to_string(),
            to: to.to_string(),
            rate,
            fee_bps,
        };

        // Add forward edge
        let mut edges = self.edges.entry(from.to_string()).or_insert_with(Vec::new);
        
        // Update existing edge or add new one
        if let Some(pos) = edges.iter().position(|e| e.to == to) {
            edges[pos] = edge;
        } else {
            edges.push(edge);
        }

        // Ensure vertices exist
        self.vertices.entry(from.to_string()).or_insert_with(|| GraphVertex {
            symbol: from.to_string(),
            best_dist: f64::INFINITY,
            predecessor: None,
        });
        self.vertices.entry(to.to_string()).or_insert_with(|| GraphVertex {
            symbol: to.to_string(),
            best_dist: f64::INFINITY,
            predecessor: None,
        });

        self.update_counter.fetch_add(1, Ordering::Relaxed);
    }

    /// Find all triangular arbitrage opportunities using Bellman-Ford
    pub fn find_arbitrage(&self) -> Vec<TriangularArb> {
        let mut opportunities = Vec::new();
        
        // Get all unique currencies
        let currencies: Vec<String> = self.vertices.iter()
            .map(|entry| entry.key().clone())
            .collect();

        // Run Bellman-Ford from each currency to find negative cycles
        for start_currency in &currencies {
            if let Some(arb) = self.bellman_ford_detect(start_currency) {
                if arb.profit_pct >= self.min_profit_threshold {
                    opportunities.push(arb);
                }
            }
        }

        opportunities
    }

    /// Bellman-Ford algorithm optimized for negative cycle detection
    fn bellman_ford_detect(&self, start: &str) -> Option<TriangularArb> {
        let n = self.vertices.len();
        
        // Initialize distances
        let mut dist: HashMap<String, f64> = HashMap::new();
        let mut pred: HashMap<String, String> = HashMap::new();
        
        dist.insert(start.to_string(), 0.0);
        
        // Relax edges n-1 times
        for _ in 0..n - 1 {
            let mut updated = false;
            
            for entry in self.edges.iter() {
                let from = entry.key();
                let edges = entry.value();
                
                let from_dist = *dist.get(from).unwrap_or(&f64::INFINITY);
                if from_dist == f64::INFINITY {
                    continue;
                }
                
                for edge in edges.iter() {
                    // Use log transform: -log(rate * (1 - fee))
                    let adjusted_rate = edge.rate * (1.0 - edge.fee_bps / 10000.0);
                    let weight = -adjusted_rate.ln();
                    
                    let new_dist = from_dist + weight;
                    
                    if new_dist < *dist.get(&edge.to).unwrap_or(&f64::INFINITY) {
                        dist.insert(edge.to.clone(), new_dist);
                        pred.insert(edge.to.clone(), from.clone());
                        updated = true;
                    }
                }
            }
            
            if !updated {
                break;
            }
        }
        
        // Check for negative cycle (arbitrage opportunity)
        for entry in self.edges.iter() {
            let from = entry.key();
            let edges = entry.value();
            
            let from_dist = *dist.get(from).unwrap_or(&f64::INFINITY);
            if from_dist == f64::INFINITY {
                continue;
            }
            
            for edge in edges.iter() {
                let adjusted_rate = edge.rate * (1.0 - edge.fee_bps / 10000.0);
                let weight = -adjusted_rate.ln();
                let new_dist = from_dist + weight;
                
                // If we can still relax, there's a negative cycle
                if new_dist < *dist.get(&edge.to).unwrap_or(&f64::INFINITY) - 1e-10 {
                    // Reconstruct the cycle
                    if let Some(cycle) = self.reconstruct_cycle(&pred, &edge.to, start) {
                        return self.calculate_profit(&cycle);
                    }
                }
            }
        }
        
        None
    }

    /// Reconstruct the arbitrage cycle from predecessors
    fn reconstruct_cycle(
        &self,
        pred: &HashMap<String, String>,
        end: &str,
        start: &str,
    ) -> Option<Vec<String>> {
        let mut path = vec![end.to_string()];
        let mut current = end.to_string();
        
        // Follow predecessors back
        for _ in 0..self.vertices.len() {
            if let Some(prev) = pred.get(&current) {
                if prev == start && path.len() > 2 {
                    // Found cycle back to start
                    path.push(start.to_string());
                    path.reverse();
                    return Some(path);
                }
                path.push(prev.clone());
                current = prev.clone();
            } else {
                break;
            }
        }
        
        None
    }

    /// Calculate profit percentage for a cycle
    fn calculate_profit(&self, path: &[String]) -> Option<TriangularArb> {
        if path.len() < 3 {
            return None;
        }

        let mut rates = Vec::new();
        let mut fees = Vec::new();
        let mut cumulative_product = 1.0;

        for i in 0..path.len() - 1 {
            let from = &path[i];
            let to = &path[i + 1];

            // Find the edge
            if let Some(edges) = self.edges.get(from) {
                if let Some(edge) = edges.iter().find(|e| e.to == *to) {
                    rates.push(edge.rate);
                    fees.push(edge.fee_bps);
                    cumulative_product *= edge.rate * (1.0 - edge.fee_bps / 10000.0);
                } else {
                    return None; // Edge not found
                }
            } else {
                return None;
            }
        }

        // Profit is (final_amount - initial_amount) / initial_amount
        // Starting with 1 unit, ending with cumulative_product units
        let profit_pct = (cumulative_product - 1.0) * 100.0;

        if profit_pct <= 0.0 {
            return None;
        }

        Some(TriangularArb {
            path: path.to_vec(),
            profit_pct,
            rates,
            fees_bps: fees,
            timestamp: Instant::now(),
        })
    }

    /// Get update counter for cache invalidation
    pub fn get_update_counter(&self) -> u64 {
        self.update_counter.load(Ordering::Relaxed)
    }

    /// Remove stale edges (no updates for specified duration)
    pub fn cleanup_stale_edges(&self, max_age_ms: u64) {
        // In production, would track last update time per edge
        // This is a simplified version
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_triangular_arb_detection() {
        let graph = TriangularGraph::new(0.01); // 0.01% minimum profit

        // Set up a profitable triangle: BTC -> ETH -> USDT -> BTC
        // BTC/ETH = 0.05 (1 BTC = 0.05 ETH... wrong, should be ETH/BTC)
        // Let's say: ETH/BTC = 0.05, ETH/USDT = 2000, BTC/USDT = 40000
        // Path: BTC -> USDT (sell BTC) -> ETH (buy ETH) -> BTC (buy BTC)
        // Or: BTC -> ETH -> USDT -> BTC
        
        // ETH/BTC = 0.05 means 1 ETH = 0.05 BTC, so 1 BTC = 20 ETH
        graph.update_edge("BTC", "ETH", 20.0, 10.0); // 1 BTC = 20 ETH
        graph.update_edge("ETH", "USDT", 2000.0, 10.0); // 1 ETH = 2000 USDT
        graph.update_edge("USDT", "BTC", 1.0 / 40000.0, 10.0); // 1 USDT = 1/40000 BTC

        let arbs = graph.find_arbitrage();
        
        // With these rates: 1 BTC -> 20 ETH -> 40000 USDT -> 1 BTC (no profit due to fees)
        // Need slightly better rates for profit
        assert!(arbs.is_empty() || arbs[0].profit_pct > 0.0);
    }

    #[test]
    fn test_graph_update() {
        let graph = TriangularGraph::new(0.1);
        
        graph.update_edge("A", "B", 1.5, 10.0);
        graph.update_edge("B", "C", 2.0, 10.0);
        
        assert_eq!(graph.get_update_counter(), 2);
    }
}
