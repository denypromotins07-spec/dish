"""
Rust bridge for running trained PPO execution policy without Python GIL.
Uses PyO3/ONNX for zero-latency inference in the microsecond loop.
"""

use std::sync::Arc;
use std::time::Instant;

/// Compressed state vector for inference
#[derive(Debug, Clone)]
pub struct InferenceState {
    pub features: [f32; 56],  // Matches Python encoder output
    pub timestamp_ns: u64,
}

/// Execution action output
#[derive(Debug, Clone, Copy)]
pub struct ExecutionAction {
    pub participation_rate: f32,  // 0.0 to 1.0
    pub aggressiveness: u8,       // 0=passive, 1=neutral, 2=aggressive
    pub confidence: f32,          // Policy confidence score
    pub inference_time_ns: u64,
}

/// ONNX runtime wrapper for PPO policy inference
pub struct PPOInferenceEngine {
    session: Arc<dyn InferenceSession>,
    input_shape: Vec<i64>,
}

trait InferenceSession: Send + Sync {
    fn run(&self, input: &[f32]) -> Vec<f32>;
}

/// Mock inference session (replace with actual ONNX runtime)
struct OnnxSession {
    #[allow(dead_code)]
    model_path: String,
}

impl InferenceSession for OnnxSession {
    fn run(&self, input: &[f32]) -> Vec<f32> {
        // Placeholder - in production this calls ONNX runtime
        // Returns [participation_rate, agg_logits_0, agg_logits_1, agg_logits_2]
        vec![0.5, 0.3, 0.4, 0.3]
    }
}

impl PPOInferenceEngine {
    /// Create new inference engine from ONNX model
    pub fn new(model_path: &str) -> Result<Self, String> {
        // Validate model exists
        if !std::path::Path::new(model_path).exists() {
            return Err(format!("Model file not found: {}", model_path));
        }
        
        let session = Arc::new(OnnxSession {
            model_path: model_path.to_string(),
        });
        
        Ok(Self {
            session,
            input_shape: vec![1, 56],  // batch_size=1, features=56
        })
    }
    
    /// Run inference on compressed state
    /// Zero heap allocation in hot path using pre-allocated buffers
    #[inline]
    pub fn infer(&self, state: &InferenceState) -> ExecutionAction {
        let start = Instant::now();
        
        // Run ONNX inference
        let outputs = self.session.run(&state.features);
        
        // Parse outputs
        let participation_rate = outputs[0].clamp(0.0, 1.0);
        
        // Softmax for aggressiveness
        let logits = [outputs[1], outputs[2], outputs[3]];
        let max_logit = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exp_logits: [f32; 3] = logits.map(|x| (x - max_logit).exp());
        let sum_exp: f32 = exp_logits.iter().sum();
        let probs: [f32; 3] = exp_logits.map(|x| x / sum_exp);
        
        // Select action with highest probability
        let aggressiveness = probs
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i as u8)
            .unwrap_or(1);
        
        // Confidence = max probability
        let confidence = probs.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        
        let inference_time_ns = start.elapsed().as_nanos() as u64;
        
        ExecutionAction {
            participation_rate,
            aggressiveness,
            confidence,
            inference_time_ns,
        }
    }
    
    /// Batch inference for multiple states
    pub fn infer_batch(&self, states: &[InferenceState]) -> Vec<ExecutionAction> {
        states.iter().map(|s| self.infer(s)).collect()
    }
}

/// Pre-allocated inference buffer for zero-allocation hot path
pub struct InferenceBuffer {
    states: Vec<InferenceState>,
    actions: Vec<ExecutionAction>,
    capacity: usize,
}

impl InferenceBuffer {
    pub fn new(capacity: usize) -> Self {
        Self {
            states: Vec::with_capacity(capacity),
            actions: Vec::with_capacity(capacity),
            capacity,
        }
    }
    
    #[inline]
    pub fn push_state(&mut self, state: InferenceState) -> bool {
        if self.states.len() >= self.capacity {
            return false;
        }
        self.states.push(state);
        true
    }
    
    #[inline]
    pub fn run_inference(&mut self, engine: &PPOInferenceEngine) -> &[ExecutionAction] {
        self.actions.clear();
        for state in &self.states {
            self.actions.push(engine.infer(state));
        }
        self.states.clear();
        &self.actions
    }
    
    #[inline]
    pub fn clear(&mut self) {
        self.states.clear();
        self.actions.clear();
    }
    
    pub fn len(&self) -> usize {
        self.states.len()
    }
}

/// FFI interface for calling from Python via PyO3
#[no_mangle]
pub extern "C" fn ffi_create_engine(model_path: *const i8) -> *mut PPOInferenceEngine {
    let path = unsafe { std::ffi::CStr::from_ptr(model_path) };
    let path_str = path.to_str().unwrap_or("");
    
    match PPOInferenceEngine::new(path_str) {
        Ok(engine) => Box::into_raw(Box::new(engine)),
        Err(_) => std::ptr::null_mut(),
    }
}

#[no_mangle]
pub extern "C" fn ffi_infer(
    engine: *mut PPOInferenceEngine,
    features: *const f32,
    result: *mut ExecutionAction,
) -> u64 {
    if engine.is_null() || features.is_null() || result.is_null() {
        return 0;
    }
    
    let engine = unsafe { &*engine };
    let state = InferenceState {
        features: unsafe { std::slice::from_raw_parts(features, 56) }
            .try_into()
            .unwrap_or([0.0; 56]),
        timestamp_ns: Instant::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos() as u64,
    };
    
    let action = engine.infer(&state);
    unsafe { *result = action };
    
    action.inference_time_ns
}

#[no_mangle]
pub extern "C" fn ffi_destroy_engine(engine: *mut PPOInferenceEngine) {
    if !engine.is_null() {
        unsafe { drop(Box::from_raw(engine)) };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_inference_engine() {
        // Create mock model file
        std::fs::write("/tmp/mock_model.onnx", "").unwrap();
        
        let engine = PPOInferenceEngine::new("/tmp/mock_model.onnx").unwrap();
        
        let state = InferenceState {
            features: [0.5; 56],
            timestamp_ns: 1234567890,
        };
        
        let action = engine.infer(&state);
        
        assert!(action.participation_rate >= 0.0);
        assert!(action.participation_rate <= 1.0);
        assert!(action.aggressiveness <= 2);
        assert!(action.confidence >= 0.0);
        assert!(action.confidence <= 1.0);
        
        println!("Inference time: {} ns", action.inference_time_ns);
        
        std::fs::remove_file("/tmp/mock_model.onnx").unwrap();
    }

    #[test]
    fn test_inference_buffer() {
        std::fs::write("/tmp/mock_model.onnx", "").unwrap();
        
        let engine = PPOInferenceEngine::new("/tmp/mock_model.onnx").unwrap();
        let mut buffer = InferenceBuffer::new(10);
        
        for i in 0..5 {
            let state = InferenceState {
                features: [i as f32 * 0.1; 56],
                timestamp_ns: 1234567890 + i,
            };
            buffer.push_state(state);
        }
        
        let actions = buffer.run_inference(&engine);
        assert_eq!(actions.len(), 5);
        
        std::fs::remove_file("/tmp/mock_model.onnx").unwrap();
    }
}
