//! Graph-based lead-lag network mapping across 50+ altcoins
//! Uses cross-correlation matrices to identify sector rotation leaders
//! Enables front-running of lagging assets based on network topology

use std::collections::{HashMap, HashSet, VecDeque};

/// Node in the lead-lag network representing a cryptocurrency
#[derive(Clone, Debug)]
pub struct AssetNode {
    pub symbol: String,
    pub sector: String,
    /// Current centrality score
    pub centrality: f64,
    /// Average lead time (positive = leads others, negative = lags)
    pub avg_lead_time_ms: i64,
}

/// Edge representing lead-lag relationship between two assets
#[derive(Clone, Debug)]
pub struct LeadLagEdge {
    pub leader: String,
    pub lagger: String,
    /// Cross-correlation coefficient at optimal lag
    pub correlation: f64,
    /// Optimal lag in milliseconds (positive means leader leads by this amount)
    pub lag_ms: i64,
    /// Statistical significance (p-value)
    pub p_value: f64,
}

/// Lead-lag network for crypto assets
pub struct LeadLagNetwork {
    /// All nodes in the network
    nodes: HashMap<String, AssetNode>,
    /// Directed edges (leader -> lagger)
    edges: Vec<LeadLagEdge>,
    /// Adjacency list for efficient traversal
    adjacency: HashMap<String, Vec<String>>,
    /// Maximum number of edges to keep (memory constraint)
    max_edges: usize,
    /// Minimum correlation threshold for edge creation
    min_correlation: f64,
    /// Price history window (circular buffer)
    price_history: HashMap<String, VecDeque<f64>>,
    max_history_len: usize,
}

impl LeadLagNetwork {
    pub fn new(
        max_edges: usize,
        min_correlation: f64,
        max_history_len: usize,
    ) -> Self {
        Self {
            nodes: HashMap::new(),
            edges: Vec::new(),
            adjacency: HashMap::new(),
            max_edges,
            min_correlation,
            price_history: HashMap::new(),
            max_history_len,
        }
    }

    /// Add or update a node in the network
    pub fn add_node(&mut self, symbol: &str, sector: &str) {
        if !self.nodes.contains_key(symbol) {
            let node = AssetNode {
                symbol: symbol.to_string(),
                sector: sector.to_string(),
                centrality: 0.0,
                avg_lead_time_ms: 0,
            };
            self.nodes.insert(symbol.to_string(), node);
            self.adjacency.insert(symbol.to_string(), Vec::new());
            self.price_history.insert(symbol.to_string(), VecDeque::with_capacity(self.max_history_len));
        }
    }

    /// Update price for an asset and maintain rolling window
    #[inline]
    pub fn update_price(&mut self, symbol: &str, price: f64, timestamp_us: u64) {
        if let Some(history) = self.price_history.get_mut(symbol) {
            history.push_back(price);
            while history.len() > self.max_history_len {
                history.pop_front();
            }
        }
    }

    /// Compute cross-correlation between two assets at various lags
    pub fn compute_cross_correlation(
        &self,
        asset_a: &str,
        asset_b: &str,
        max_lag_ms: i64,
        sample_interval_ms: i64,
    ) -> Option<(f64, i64)> {
        let history_a = self.price_history.get(asset_a)?;
        let history_b = self.price_history.get(asset_b)?;

        if history_a.len() < 20 || history_b.len() < 20 {
            return None;
        }

        let n = history_a.len().min(history_b.len());
        let max_lag_samples = (max_lag_ms / sample_interval_ms) as usize;

        let mut best_corr = 0.0;
        let mut best_lag = 0i64;

        // Convert to Vec for easier indexing
        let vec_a: Vec<f64> = history_a.iter().copied().collect();
        let vec_b: Vec<f64> = history_b.iter().copied().collect();

        // Standardize
        let mean_a = vec_a.iter().sum::<f64>() / n as f64;
        let mean_b = vec_b.iter().sum::<f64>() / n as f64;
        let std_a = ((vec_a.iter().map(|x| (x - mean_a).powi(2)).sum::<f64>() / n as f64).sqrt()).max(1e-10);
        let std_b = ((vec_b.iter().map(|x| (x - mean_b).powi(2)).sum::<f64>() / n as f64).sqrt()).max(1e-10);

        let norm_a: Vec<f64> = vec_a.iter().map(|x| (x - mean_a) / std_a).collect();
        let norm_b: Vec<f64> = vec_b.iter().map(|x| (x - mean_b) / std_b).collect();

        // Search for optimal lag
        for lag in -max_lag_samples..=max_lag_samples {
            let mut corr = 0.0;
            let mut count = 0;

            for i in 0..n {
                let j = i as i64 + lag;
                if j >= 0 && (j as usize) < n {
                    corr += norm_a[i] * norm_b[j as usize];
                    count += 1;
                }
            }

            if count > 0 {
                corr /= count as f64;
                if corr.abs() > best_corr.abs() {
                    best_corr = corr;
                    best_lag = lag * sample_interval_ms;
                }
            }
        }

        Some((best_corr, best_lag))
    }

    /// Update the network structure based on recent correlations
    pub fn rebuild_network(&mut self, max_lag_ms: i64, sample_interval_ms: i64) {
        let symbols: Vec<String> = self.nodes.keys().cloned().collect();
        let mut new_edges: Vec<LeadLagEdge> = Vec::new();

        // Compute pairwise correlations
        for i in 0..symbols.len() {
            for j in (i + 1)..symbols.len() {
                let symbol_a = &symbols[i];
                let symbol_b = &symbols[j];

                if let Some((corr, lag)) = self.compute_cross_correlation(
                    symbol_a,
                    symbol_b,
                    max_lag_ms,
                    sample_interval_ms,
                ) {
                    if corr.abs() >= self.min_correlation {
                        // Determine leader and lagger based on lag sign
                        let (leader, lagger) = if lag > 0 {
                            (symbol_a.clone(), symbol_b.clone())
                        } else {
                            (symbol_b.clone(), symbol_a.clone())
                        };

                        let edge = LeadLagEdge {
                            leader,
                            lagger,
                            correlation: corr.abs(),
                            lag_ms: lag.abs(),
                            p_value: self._estimate_p_value(corr, self.max_history_len),
                        };

                        new_edges.push(edge);
                    }
                }
            }
        }

        // Sort by correlation strength and keep top edges
        new_edges.sort_by(|a, b| {
            b.correlation.partial_cmp(&a.correlation).unwrap_or(std::cmp::Ordering::Equal)
        });
        
        if new_edges.len() > self.max_edges {
            new_edges.truncate(self.max_edges);
        }

        // Update adjacency list
        self.edges = new_edges;
        self._rebuild_adjacency();
        
        // Recalculate centralities
        self._calculate_centralities();
    }

    /// Get the top leaders in the network
    pub fn get_top_leaders(&self, n: usize) -> Vec<&AssetNode> {
        let mut nodes: Vec<&AssetNode> = self.nodes.values().collect();
        nodes.sort_by(|a, b| {
            b.avg_lead_time_ms.cmp(&a.avg_lead_time_ms)
        });
        nodes.into_iter().take(n).collect()
    }

    /// Get the top laggards in the network
    pub fn get_top_laggards(&self, n: usize) -> Vec<&AssetNode> {
        let mut nodes: Vec<&AssetNode> = self.nodes.values().collect();
        nodes.sort_by(|a, b| {
            a.avg_lead_time_ms.cmp(&b.avg_lead_time_ms)
        });
        nodes.into_iter().take(n).collect()
    }

    /// Find assets that typically lag behind a given leader
    pub fn get_followers(&self, leader: &str) -> Vec<&str> {
        self.edges
            .iter()
            .filter(|e| e.leader == leader)
            .map(|e| e.lagger.as_str())
            .collect()
    }

    /// Find assets that typically lead a given lagger
    pub fn get_leaders(&self, lagger: &str) -> Vec<&str> {
        self.edges
            .iter()
            .filter(|e| e.lagger == lagger)
            .map(|e| e.leader.as_str())
            .collect()
    }

    /// Detect sector rotation: which sector is leading
    pub fn detect_sector_rotation(&self) -> HashMap<String, f64> {
        let mut sector_scores: HashMap<String, f64> = HashMap::new();

        // Calculate average centrality per sector
        for node in self.nodes.values() {
            let entry = sector_scores.entry(node.sector.clone()).or_insert(0.0);
            *entry += node.centrality;
        }

        // Normalize by sector size
        let sector_counts: HashMap<String, usize> = self.nodes
            .values()
            .fold(HashMap::new(), |mut acc, node| {
                *acc.entry(node.sector.clone()).or_insert(0) += 1;
                acc
            });

        for (sector, score) in sector_scores.iter_mut() {
            if let Some(&count) = sector_counts.get(sector) {
                *score /= count as f64;
            }
        }

        sector_scores
    }

    /// Generate trading signal: go long laggards when leader moves
    pub fn generate_front_run_signal(
        &self,
        moving_asset: &str,
        price_change_pct: f64,
    ) -> Vec<FrontRunSignal> {
        let mut signals = Vec::new();

        // Find assets that lag behind the moving asset
        let followers = self.get_followers(moving_asset);

        for follower in followers {
            if let Some(edge) = self.edges.iter().find(|e| {
                e.leader == moving_asset && e.lagger == follower
            }) {
                // Signal strength based on correlation and lag time
                let signal_strength = edge.correlation * price_change_pct.abs();
                
                signals.push(FrontRunSignal {
                    target: follower.to_string(),
                    direction: if price_change_pct > 0.0 { "long" } else { "short" }.to_string(),
                    expected_lag_ms: edge.lag_ms,
                    confidence: edge.correlation,
                    signal_strength,
                });
            }
        }

        signals
    }

    fn _rebuild_adjacency(&mut self) {
        self.adjacency.clear();
        for node in self.nodes.keys() {
            self.adjacency.insert(node.clone(), Vec::new());
        }

        for edge in &self.edges {
            if let Some(neighbors) = self.adjacency.get_mut(&edge.leader) {
                neighbors.push(edge.lagger.clone());
            }
        }
    }

    fn _calculate_centralities(&mut self) {
        // Calculate out-degree centrality (number of followers)
        let mut out_degree: HashMap<String, usize> = HashMap::new();
        let mut total_lag_time: HashMap<String, i64> = HashMap::new();

        for edge in &self.edges {
            *out_degree.entry(edge.leader.clone()).or_insert(0) += 1;
            *total_lag_time.entry(edge.leader.clone()).or_insert(0) += edge.lag_ms;
            
            // Laggards get negative contribution
            *total_lag_time.entry(edge.lagger.clone()).or_insert(0) -= edge.lag_ms;
        }

        let n = self.nodes.len().max(1);
        
        for (symbol, node) in self.nodes.iter_mut() {
            let degree = out_degree.get(symbol).copied().unwrap_or(0);
            node.centrality = degree as f64 / (n - 1).max(1) as f64;

            let total_lag = total_lag_time.get(symbol).copied().unwrap_or(0);
            let connections = out_degree.get(symbol).copied().unwrap_or(0).max(1);
            node.avg_lead_time_ms = total_lag / connections as i64;
        }
    }

    fn _estimate_p_value(&self, correlation: f64, n: usize) -> f64 {
        // Simplified p-value estimation using Fisher transformation
        if n < 3 {
            return 1.0;
        }

        let z = 0.5 * ((1.0 + correlation.min(0.999)).ln() - (1.0 - correlation.max(-0.999)).ln());
        let se = 1.0 / (n - 3).sqrt();
        let z_stat = z.abs() / se;

        // Approximate p-value using normal distribution
        // Using approximation: p ≈ 2 * (1 - Φ(|z|))
        let p = 2.0 * (1.0 - self._normal_cdf(z_stat));
        p
    }

    fn _normal_cdf(&self, x: f64) -> f64 {
        // Approximation of standard normal CDF
        let t = 1.0 / (1.0 + 0.2316419 * x.abs());
        let d = 0.3989423 * (-x * x / 2.0).exp();
        let prob = d * t * (0.3193815 + t * (-0.3565638 + t * (1.781478 + t * (-1.821256 + t * 1.330274))));
        
        if x > 0.0 {
            1.0 - prob
        } else {
            prob
        }
    }
}

/// Trading signal generated from lead-lag analysis
#[derive(Clone, Debug)]
pub struct FrontRunSignal {
    pub target: String,
    pub direction: String,
    pub expected_lag_ms: i64,
    pub confidence: f64,
    pub signal_strength: f64,
}

/// Sector-level aggregation for rotation detection
#[derive(Clone, Debug)]
pub struct SectorRotationInfo {
    pub sector: String,
    pub avg_centrality: f64,
    pub n_assets: usize,
    pub is_leading: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_network_construction() {
        let mut network = LeadLagNetwork::new(100, 0.3, 500);
        
        network.add_node("BTC", "layer1");
        network.add_node("ETH", "layer1");
        network.add_node("SOL", "layer1");
        
        // Add some synthetic price history
        for i in 0..100 {
            network.update_price("BTC", 50000.0 + i as f64, i * 1000);
            network.update_price("ETH", 3000.0 + i as f64 * 0.5, i * 1000 + 100);
            network.update_price("SOL", 100.0 + i as f64 * 0.1, i * 1000 + 200);
        }
        
        network.rebuild_network(500, 100);
        
        assert!(network.nodes.len() >= 3);
    }

    #[test]
    fn test_cross_correlation() {
        let mut network = LeadLagNetwork::new(100, 0.3, 500);
        
        network.add_node("A", "test");
        network.add_node("B", "test");
        
        // Create correlated series with lag
        for i in 0..100 {
            let price_a = 100.0 + (i as f64 * 0.1).sin() * 10.0;
            let price_b = 100.0 + ((i as f64 - 5.0) * 0.1).sin() * 10.0;
            
            network.update_price("A", price_a, i * 1000);
            network.update_price("B", price_b, i * 1000);
        }
        
        let result = network.compute_cross_correlation("A", "B", 1000, 100);
        assert!(result.is_some());
    }
}
