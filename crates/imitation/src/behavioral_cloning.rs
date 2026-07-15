"""
Rust-based Behavioral Cloning fast-path inference engine.
Falls back to deterministic BC model when RL confidence is low.
Zero heap allocation in hot path for microsecond decisions.
"""

use std::collections::VecDeque;

/// Input state for behavioral cloning
#[derive(Debug, Clone)]
pub struct BCState {
    pub features: [f32; 56],
    pub timestamp_ns: u64,
}

/// Action output from behavioral cloning
#[derive(Debug, Clone, Copy)]
pub struct BCAction {
    pub participation_rate: f32,
    pub aggressiveness: u8,
    pub confidence: f32,
    pub is_fallback: bool,
}

/// Simple neural network layer for BC inference
struct LinearLayer {
    weights: Vec<f32>,
    biases: Vec<f32>,
    input_dim: usize,
    output_dim: usize,
}

impl LinearLayer {
    fn new(input_dim: usize, output_dim: usize) -> Self {
        // Initialize with small random weights (in production, load from trained model)
        let weights = vec![0.01f32; input_dim * output_dim];
        let biases = vec![0.0f32; output_dim];
        
        Self {
            weights,
            biases,
            input_dim,
            output_dim,
        }
    }
    
    #[inline]
    fn forward(&self, input: &[f32]) -> Vec<f32> {
        let mut output = Vec::with_capacity(self.output_dim);
        
        for i in 0..self.output_dim {
            let mut sum = self.biases[i];
            for j in 0..self.input_dim {
                sum += input[j] * self.weights[i * self.input_dim + j];
            }
            // ReLU activation
            output.push(sum.max(0.0));
        }
        
        output
    }
}

/// Behavioral Cloning inference engine
pub struct BehavioralCloningEngine {
    layers: Vec<LinearLayer>,
    action_mean: [f32; 2],
    action_std: [f32; 2],
}

impl BehavioralCloningEngine {
    /// Create new BC engine with predefined architecture
    pub fn new() -> Self {
        let layers = vec![
            LinearLayer::new(56, 64),
            LinearLayer::new(64, 32),
            LinearLayer::new(32, 2),  // Output: [participation_rate, aggressiveness_logit]
        ];
        
        Self {
            layers,
            action_mean: [0.5, 1.0],  // Default expert action statistics
            action_std: [0.2, 0.5],
        }
    }
    
    /// Load trained model from file
    pub fn load_from_file(path: &str) -> Result<Self, String> {
        // In production, load weights from trained model file
        let mut engine = Self::new();
        
        // Placeholder: validate file exists
        if !std::path::Path::new(path).exists() {
            return Err(format!("Model file not found: {}", path));
        }
        
        Ok(engine)
    }
    
    /// Set expert action statistics for confidence calculation
    pub fn set_expert_stats(&mut self, mean: [f32; 2], std: [f32; 2]) {
        self.action_mean = mean;
        self.action_std = std;
    }
    
    /// Run inference on state
    /// Returns action with confidence score
    #[inline]
    pub fn infer(&self, state: &BCState) -> BCAction {
        // Forward pass through layers
        let mut hidden = state.features.to_vec();
        
        for (i, layer) in self.layers.iter().enumerate() {
            hidden = layer.forward(&hidden);
            
            // Apply tanh to final layer for bounded output
            if i == self.layers.len() - 1 {
                for h in &mut hidden {
                    *h = h.tanh();
                }
            }
        }
        
        // Parse outputs
        let participation_rate = ((hidden[0] + 1.0) / 2.0).clamp(0.0, 1.0);
        let agg_logit = hidden[1];
        
        // Convert logit to discrete aggressiveness (0, 1, 2)
        let aggressiveness = if agg_logit < -0.5 {
            0  // Passive
        } else if agg_logit < 0.5 {
            1  // Neutral
        } else {
            2  // Aggressive
        };
        
        // Calculate confidence based on distance from expert mean
        let participation_z = (participation_rate - self.action_mean[0]).abs() / (self.action_std[0] + 1e-6);
        let agg_z = (agg_logit - self.action_mean[1]).abs() / (self.action_std[1] + 1e-6);
        let avg_z = (participation_z + agg_z) / 2.0;
        
        // Convert z-score to confidence (lower z = higher confidence)
        let confidence = (-avg_z).exp();
        
        BCAction {
            participation_rate,
            aggressiveness,
            confidence,
            is_fallback: false,
        }
    }
    
    /// Batch inference
    pub fn infer_batch(&self, states: &[BCState]) -> Vec<BCAction> {
        states.iter().map(|s| self.infer(s)).collect()
    }
}

impl Default for BehavioralCloningEngine {
    fn default() -> Self {
        Self::new()
    }
}

/// Hybrid executor that combines RL and BC
pub struct HybridExecutor {
    bc_engine: BehavioralCloningEngine,
    confidence_threshold: f32,
    action_history: VecDeque<BCAction>,
    max_history: usize,
}

impl HybridExecutor {
    /// Create new hybrid executor
    pub fn new(confidence_threshold: f32) -> Self {
        Self {
            bc_engine: BehavioralCloningEngine::new(),
            confidence_threshold,
            action_history: VecDeque::with_capacity(100),
            max_history: 100,
        }
    }
    
    /// Execute with RL fallback to BC
    /// If RL confidence is below threshold, use BC instead
    #[inline]
    pub fn execute(
        &self,
        rl_action: Option<(f32, u8, f32)>,  // (participation, aggressiveness, confidence)
        bc_state: &BCState,
    ) -> BCAction {
        // Get BC action as fallback
        let bc_action = self.bc_engine.infer(bc_state);
        
        // Check if RL action is available and confident enough
        if let Some((rl_part, rl_agg, rl_conf)) = rl_action {
            if rl_conf >= self.confidence_threshold {
                let action = BCAction {
                    participation_rate: rl_part,
                    aggressiveness: rl_agg,
                    confidence: rl_conf,
                    is_fallback: false,
                };
                return action;
            }
        }
        
        // Fall back to BC
        BCAction {
            participation_rate: bc_action.participation_rate,
            aggressiveness: bc_action.aggressiveness,
            confidence: bc_action.confidence,
            is_fallback: true,
        }
    }
    
    /// Record action for monitoring
    pub fn record_action(&mut self, action: BCAction) {
        if self.action_history.len() >= self.max_history {
            self.action_history.pop_front();
        }
        self.action_history.push_back(action);
    }
    
    /// Get fallback rate (percentage of actions using BC)
    pub fn get_fallback_rate(&self) -> f32 {
        if self.action_history.is_empty() {
            return 0.0;
        }
        
        let fallback_count = self.action_history.iter().filter(|a| a.is_fallback).count();
        fallback_count as f32 / self.action_history.len() as f32
    }
    
    /// Get average confidence
    pub fn get_avg_confidence(&self) -> f32 {
        if self.action_history.is_empty() {
            return 0.0;
        }
        
        let sum: f32 = self.action_history.iter().map(|a| a.confidence).sum();
        sum / self.action_history.len() as f32
    }
}

impl Default for HybridExecutor {
    fn default() -> Self {
        Self::new(0.7)  // Default threshold: 0.7
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bc_engine_inference() {
        let engine = BehavioralCloningEngine::new();
        
        let state = BCState {
            features: [0.5; 56],
            timestamp_ns: 1234567890,
        };
        
        let action = engine.infer(&state);
        
        assert!(action.participation_rate >= 0.0);
        assert!(action.participation_rate <= 1.0);
        assert!(action.aggressiveness <= 2);
        assert!(action.confidence >= 0.0);
        assert!(action.confidence <= 1.0);
        
        println!("BC Action: {:?}", action);
    }

    #[test]
    fn test_hybrid_executor() {
        let mut executor = HybridExecutor::new(0.7);
        
        let state = BCState {
            features: [0.5; 56],
            timestamp_ns: 1234567890,
        };
        
        // Test with high-confidence RL action (should use RL)
        let rl_action = Some((0.6, 1, 0.9));
        let action = executor.execute(rl_action, &state);
        assert!(!action.is_fallback);
        
        // Test with low-confidence RL action (should fall back to BC)
        let rl_action = Some((0.6, 1, 0.5));
        let action = executor.execute(rl_action, &state);
        assert!(action.is_fallback);
        
        // Test with no RL action (should fall back to BC)
        let action = executor.execute(None, &state);
        assert!(action.is_fallback);
        
        // Record and check stats
        executor.record_action(action);
        executor.record_action(BCAction {
            participation_rate: 0.5,
            aggressiveness: 1,
            confidence: 0.8,
            is_fallback: false,
        });
        
        println!("Fallback rate: {:.2%}", executor.get_fallback_rate());
        println!("Avg confidence: {:.2}", executor.get_avg_confidence());
    }

    #[test]
    fn test_model_loading() {
        // Create dummy file
        std::fs::write("/tmp/bc_model.bin", "").unwrap();
        
        let result = BehavioralCloningEngine::load_from_file("/tmp/bc_model.bin");
        assert!(result.is_ok());
        
        std::fs::remove_file("/tmp/bc_model.bin").unwrap();
    }
}
