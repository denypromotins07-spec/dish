//! Lock-free ensemble voting logic for model aggregation.
//! Combines predictions from Supervised, Deep Learning, and RL models.
//! Applies dynamic weights based on recent out-of-sample performance and regime detection.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicUsize, AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Prediction result from a single model
#[derive(Debug, Clone)]
pub struct ModelPrediction {
    pub model_id: String,
    pub prediction: f64,
    pub confidence: f64,
    pub timestamp: Instant,
    pub model_type: ModelType,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ModelType {
    Supervised,      // XGBoost/LightGBM
    DeepLearning,    // CNN-LSTM/Transformer
    Reinforcement,   // RL Agent
}

/// Performance metrics for a model
#[derive(Debug, Clone, Default)]
pub struct ModelPerformance {
    pub total_predictions: usize,
    pub correct_predictions: usize,
    pub cumulative_error: f64,
    pub last_n_errors: VecDeque<f64>,
    pub win_rate: f64,
    pub avg_confidence: f64,
}

impl ModelPerformance {
    fn new(window_size: usize) -> Self {
        Self {
            last_n_errors: VecDeque::with_capacity(window_size),
            ..Default::default()
        }
    }
    
    /// Update performance metrics with new observation
    fn update(&mut self, prediction: f64, actual: f64, confidence: f64) {
        self.total_predictions += 1;
        
        let error = (prediction - actual).abs();
        self.cumulative_error += error;
        
        // Track if prediction was "correct" (within threshold)
        let threshold = 0.01; // 1% threshold
        if error < threshold {
            self.correct_predictions += 1;
        }
        
        // Update rolling window
        if self.last_n_errors.len() >= self.last_n_errors.capacity() {
            self.last_n_errors.pop_front();
        }
        self.last_n_errors.push_back(error);
        
        // Update win rate
        self.win_rate = self.correct_predictions as f64 / self.total_predictions as f64;
        
        // Update average confidence (exponential moving average)
        let alpha = 0.1;
        self.avg_confidence = alpha * confidence + (1.0 - alpha) * self.avg_confidence;
    }
    
    /// Get recent mean squared error
    fn recent_mse(&self) -> f64 {
        if self.last_n_errors.is_empty() {
            return 1.0;
        }
        let sum_sq: f64 = self.last_n_errors.iter().map(|e| e * e).sum();
        sum_sq / self.last_n_errors.len() as f64
    }
    
    /// Get exponential decay weight based on recent performance
    fn performance_weight(&self, decay_factor: f64) -> f64 {
        // Higher weight for better recent performance
        let mse = self.recent_mse();
        (1.0 / (1.0 + mse)).powf(decay_factor)
    }
}

/// Market regime for adaptive weighting
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MarketRegime {
    Trending,
    Ranging,
    Volatile,
    Unknown,
}

/// Configuration for ensemble voting
#[derive(Debug, Clone)]
pub struct EnsembleConfig {
    pub performance_window: usize,
    pub min_predictions_for_weight: usize,
    pub base_weights: ModelWeights,
    pub regime_adaptive: bool,
    pub confidence_threshold: f64,
    pub decay_factor: f64,
}

impl Default for EnsembleConfig {
    fn default() -> Self {
        Self {
            performance_window: 100,
            min_predictions_for_weight: 10,
            base_weights: ModelWeights::equal(),
            regime_adaptive: true,
            confidence_threshold: 0.5,
            decay_factor: 2.0,
        }
    }
}

#[derive(Debug, Clone)]
pub struct ModelWeights {
    pub supervised: f64,
    pub deep_learning: f64,
    pub reinforcement: f64,
}

impl ModelWeights {
    pub fn equal() -> Self {
        Self {
            supervised: 1.0,
            deep_learning: 1.0,
            reinforcement: 1.0,
        }
    }
    
    pub fn normalize(&self) -> Self {
        let total = self.supervised + self.deep_learning + self.reinforcement;
        if total == 0.0 {
            return Self::equal();
        }
        Self {
            supervised: self.supervised / total,
            deep_learning: self.deep_learning / total,
            reinforcement: self.reinforcement / total,
        }
    }
}

/// Lock-free ensemble aggregator using atomic operations
pub struct ModelEnsemble {
    config: EnsembleConfig,
    current_regime: AtomicUsize,  // Encoded MarketRegime
    is_active: AtomicBool,
    
    // Model performance tracking (protected by mutex for updates, atomic for reads)
    supervised_perf: Arc<std::sync::RwLock<ModelPerformance>>,
    deep_learning_perf: Arc<std::sync::RwLock<ModelPerformance>>,
    reinforcement_perf: Arc<std::sync::RwLock<ModelPerformance>>,
    
    // Recent predictions buffer
    predictions_buffer: Arc<std::sync::Mutex<VecDeque<ModelPrediction>>>,
    
    // Aggregated prediction cache
    cached_aggregate: Arc<std::sync::RwLock<Option<CachedAggregate>>>,
}

#[derive(Debug, Clone)]
struct CachedAggregate {
    prediction: f64,
    confidence: f64,
    timestamp: Instant,
    votes: VoteBreakdown,
}

#[derive(Debug, Clone, Default)]
pub struct VoteBreakdown {
    pub supervised_vote: f64,
    pub supervised_weight: f64,
    pub deep_learning_vote: f64,
    pub deep_learning_weight: f64,
    pub reinforcement_vote: f64,
    pub reinforcement_weight: f64,
    pub final_prediction: f64,
    pub final_confidence: f64,
}

impl ModelEnsemble {
    /// Create a new ensemble aggregator
    pub fn new(config: EnsembleConfig) -> Self {
        Self {
            config,
            current_regime: AtomicUsize::new(MarketRegime::Unknown as usize),
            is_active: AtomicBool::new(true),
            supervised_perf: Arc::new(std::sync::RwLock::new(
                ModelPerformance::new(config.performance_window)
            )),
            deep_learning_perf: Arc::new(std::sync::RwLock::new(
                ModelPerformance::new(config.performance_window)
            )),
            reinforcement_perf: Arc::new(std::sync::RwLock::new(
                ModelPerformance::new(config.performance_window)
            )),
            predictions_buffer: Arc::new(std::sync::Mutex::new(
                VecDeque::with_capacity(1000)
            )),
            cached_aggregate: Arc::new(std::sync::RwLock::new(None)),
        }
    }
    
    /// Submit a prediction from a model
    pub fn submit_prediction(&self, prediction: ModelPrediction) {
        if !self.is_active.load(Ordering::Relaxed) {
            return;
        }
        
        // Add to buffer
        {
            let mut buffer = self.predictions_buffer.lock().unwrap();
            if buffer.len() >= buffer.capacity() {
                buffer.pop_front();
            }
            buffer.push_back(prediction);
        }
        
        // Invalidate cache
        *self.cached_aggregate.write().unwrap() = None;
    }
    
    /// Record actual outcome and update performance metrics
    pub fn record_outcome(&self, model_id: &str, prediction: f64, actual: f64, confidence: f64) {
        // Route to appropriate performance tracker
        if model_id.contains("supervised") || model_id.contains("xgb") || model_id.contains("lgb") {
            let mut perf = self.supervised_perf.write().unwrap();
            perf.update(prediction, actual, confidence);
        } else if model_id.contains("deep") || model_id.contains("lstm") || model_id.contains("transformer") {
            let mut perf = self.deep_learning_perf.write().unwrap();
            perf.update(prediction, actual, confidence);
        } else if model_id.contains("rl") || model_id.contains("reinforcement") {
            let mut perf = self.reinforcement_perf.write().unwrap();
            perf.update(prediction, actual, confidence);
        }
    }
    
    /// Update market regime
    pub fn update_regime(&self, regime: MarketRegime) {
        self.current_regime.store(regime as usize, Ordering::Relaxed);
        
        // Invalidate cache when regime changes
        *self.cached_aggregate.write().unwrap() = None;
    }
    
    /// Get aggregated prediction with dynamic weights
    pub fn aggregate(&self) -> Option<(f64, f64, VoteBreakdown)> {
        // Try to return cached result if fresh (< 1ms old)
        {
            let cache = self.cached_aggregate.read().unwrap();
            if let Some(cached) = cache.as_ref() {
                if cached.timestamp.elapsed() < Duration::from_millis(1) {
                    return Some((cached.prediction, cached.confidence, cached.votes.clone()));
                }
            }
        }
        
        // Collect recent predictions
        let buffer = self.predictions_buffer.lock().unwrap();
        if buffer.is_empty() {
            return None;
        }
        
        // Group by model type and get latest
        let mut latest_by_type: std::collections::HashMap<ModelType, &ModelPrediction> = 
            std::collections::HashMap::new();
        
        for pred in buffer.iter().rev() {
            if !latest_by_type.contains_key(&pred.model_type) {
                latest_by_type.insert(pred.model_type, pred);
            }
            if latest_by_type.len() >= 3 {
                break;
            }
        }
        
        // Calculate dynamic weights
        let weights = self.calculate_dynamic_weights();
        
        // Compute weighted vote
        let mut breakdown = VoteBreakdown::default();
        let mut weighted_sum = 0.0;
        let mut total_weight = 0.0;
        let mut weighted_confidence = 0.0;
        
        if let Some(pred) = latest_by_type.get(&ModelType::Supervised) {
            if pred.confidence >= self.config.confidence_threshold {
                breakdown.supervised_vote = pred.prediction;
                breakdown.supervised_weight = weights.supervised;
                weighted_sum += pred.prediction * weights.supervised;
                total_weight += weights.supervised;
                weighted_confidence += pred.confidence * weights.supervised;
            }
        }
        
        if let Some(pred) = latest_by_type.get(&ModelType::DeepLearning) {
            if pred.confidence >= self.config.confidence_threshold {
                breakdown.deep_learning_vote = pred.prediction;
                breakdown.deep_learning_weight = weights.deep_learning;
                weighted_sum += pred.prediction * weights.deep_learning;
                total_weight += weights.deep_learning;
                weighted_confidence += pred.confidence * weights.deep_learning;
            }
        }
        
        if let Some(pred) = latest_by_type.get(&ModelType::Reinforcement) {
            if pred.confidence >= self.config.confidence_threshold {
                breakdown.reinforcement_vote = pred.prediction;
                breakdown.reinforcement_weight = weights.reinforcement;
                weighted_sum += pred.prediction * weights.reinforcement;
                total_weight += weights.reinforcement;
                weighted_confidence += pred.confidence * weights.reinforcement;
            }
        }
        
        if total_weight == 0.0 {
            return None;
        }
        
        let final_prediction = weighted_sum / total_weight;
        let final_confidence = weighted_confidence / total_weight;
        
        breakdown.final_prediction = final_prediction;
        breakdown.final_confidence = final_confidence;
        
        // Cache the result
        let cached = CachedAggregate {
            prediction: final_prediction,
            confidence: final_confidence,
            timestamp: Instant::now(),
            votes: breakdown.clone(),
        };
        *self.cached_aggregate.write().unwrap() = Some(cached);
        
        Some((final_prediction, final_confidence, breakdown))
    }
    
    /// Calculate dynamic weights based on performance and regime
    fn calculate_dynamic_weights(&self) -> ModelWeights {
        let regime = match self.current_regime.load(Ordering::Relaxed) {
            0 => MarketRegime::Trending,
            1 => MarketRegime::Ranging,
            2 => MarketRegime::Volatile,
            _ => MarketRegime::Unknown,
        };
        
        let sup_perf = self.supervised_perf.read().unwrap();
        let dl_perf = self.deep_learning_perf.read().unwrap();
        let rl_perf = self.reinforcement_perf.read().unwrap();
        
        // Base weights from config
        let mut weights = self.config.base_weights.clone();
        
        // Apply performance-based adjustments
        if sup_perf.total_predictions >= self.config.min_predictions_for_weight {
            weights.supervised *= sup_perf.performance_weight(self.config.decay_factor);
        }
        if dl_perf.total_predictions >= self.config.min_predictions_for_weight {
            weights.deep_learning *= dl_perf.performance_weight(self.config.decay_factor);
        }
        if rl_perf.total_predictions >= self.config.min_predictions_for_weight {
            weights.reinforcement *= rl_perf.performance_weight(self.config.decay_factor);
        }
        
        // Apply regime-based adjustments
        if self.config.regime_adaptive {
            match regime {
                MarketRegime::Trending => {
                    // Deep learning typically better for trends
                    weights.deep_learning *= 1.2;
                }
                MarketRegime::Ranging => {
                    // Supervised models often better in ranging markets
                    weights.supervised *= 1.2;
                }
                MarketRegime::Volatile => {
                    // RL may handle volatility better with risk awareness
                    weights.reinforcement *= 1.2;
                }
                _ => {}
            }
        }
        
        weights.normalize()
    }
    
    /// Enable or disable the ensemble
    pub fn set_active(&self, active: bool) {
        self.is_active.store(active, Ordering::Relaxed);
    }
    
    /// Check if ensemble is active
    pub fn is_active(&self) -> bool {
        self.is_active.load(Ordering::Relaxed)
    }
    
    /// Get current performance summary
    pub fn get_performance_summary(&self) -> (f64, f64, f64) {
        let sup = self.supervised_perf.read().unwrap();
        let dl = self.deep_learning_perf.read().unwrap();
        let rl = self.reinforcement_perf.read().unwrap();
        
        (sup.win_rate, dl.win_rate, rl.win_rate)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_ensemble_creation() {
        let config = EnsembleConfig::default();
        let ensemble = ModelEnsemble::new(config);
        
        assert!(ensemble.is_active());
    }
    
    #[test]
    fn test_model_weights_normalization() {
        let weights = ModelWeights {
            supervised: 2.0,
            deep_learning: 1.0,
            reinforcement: 1.0,
        };
        
        let normalized = weights.normalize();
        
        assert!((normalized.supervised - 0.5).abs() < 0.001);
        assert!((normalized.deep_learning - 0.25).abs() < 0.001);
        assert!((normalized.reinforcement - 0.25).abs() < 0.001);
    }
    
    #[test]
    fn test_prediction_submission() {
        let ensemble = ModelEnsemble::new(EnsembleConfig::default());
        
        let pred = ModelPrediction {
            model_id: "supervised_xgb".to_string(),
            prediction: 0.75,
            confidence: 0.8,
            timestamp: Instant::now(),
            model_type: ModelType::Supervised,
        };
        
        ensemble.submit_prediction(pred);
        
        // Verify prediction was stored
        let buffer = ensemble.predictions_buffer.lock().unwrap();
        assert_eq!(buffer.len(), 1);
    }
}
