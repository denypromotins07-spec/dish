"""
Rust utility for exporting quantized models to optimized ONNX format.
Applies operator fusion tailored for microsecond inference runtime.
"""

use std::path::Path;
use std::fs::File;
use std::io::{Read, Write};

/// Model export configuration
#[derive(Debug, Clone)]
pub struct ExportConfig {
    pub opset_version: i64,
    pub enable_fusion: bool,
    pub optimize_level: u8,  // 0-3
    pub quantize: bool,
}

impl Default for ExportConfig {
    fn default() -> Self {
        Self {
            opset_version: 17,
            enable_fusion: true,
            optimize_level: 2,
            quantize: true,
        }
    }
}

/// Fused operator types for optimization
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FusedOp {
    ConvBnRelu,
    GemmBias,
    LayerNormGelu,
    MatMulAdd,
    Attention,
}

/// ONNX model exporter with optimizations
pub struct OnnxExporter {
    config: ExportConfig,
    fused_ops: Vec<FusedOp>,
}

impl OnnxExporter {
    /// Create new exporter with default config
    pub fn new() -> Self {
        Self {
            config: ExportConfig::default(),
            fused_ops: Vec::new(),
        }
    }
    
    /// Create exporter with custom config
    pub fn with_config(config: ExportConfig) -> Self {
        Self {
            config,
            fused_ops: if config.enable_fusion {
                vec![
                    FusedOp::ConvBnRelu,
                    FusedOp::GemmBias,
                    FusedOp::LayerNormGelu,
                    FusedOp::MatMulAdd,
                ]
            } else {
                Vec::new()
            },
        }
    }
    
    /// Validate source model file
    pub fn validate_model<P: AsRef<Path>>(&self, path: P) -> Result<bool, String> {
        let path = path.as_ref();
        
        if !path.exists() {
            return Err(format!("Model file not found: {:?}", path));
        }
        
        // Check file extension
        let ext = path.extension()
            .and_then(|e| e.to_str())
            .unwrap_or("");
            
        if !["pt", "pth", "bin", "onnx"].contains(&ext) {
            return Err(format!("Unsupported model format: {}", ext));
        }
        
        Ok(true)
    }
    
    /// Apply operator fusion optimizations
    pub fn apply_fusion(&mut self, model_data: &[u8]) -> Result<Vec<u8>, String> {
        if !self.config.enable_fusion {
            return Ok(model_data.to_vec());
        }
        
        // In production, this would parse the ONNX graph and apply fusions
        // For now, we simulate the optimization metadata
        
        let mut optimized = model_data.to_vec();
        
        // Add fusion metadata as header (simulated)
        let fusion_header = format!(
            "FUSION_OPS:{:?}\nOPT_LEVEL:{}\n",
            self.fused_ops,
            self.config.optimize_level
        );
        
        let mut result = fusion_header.as_bytes().to_vec();
        result.append(&mut optimized);
        
        Ok(result)
    }
    
    /// Quantize model weights (INT8)
    pub fn apply_quantization(&self, model_data: &[u8]) -> Result<Vec<u8>, String> {
        if !self.config.quantize {
            return Ok(model_data.to_vec());
        }
        
        // In production, this would perform actual weight quantization
        // Simulate by adding quantization metadata
        
        let mut result = b"QUANTIZED:INT8\n".to_vec();
        result.extend_from_slice(model_data);
        
        Ok(result)
    }
    
    /// Export model to ONNX with all optimizations
    pub fn export<P: AsRef<Path>>(
        &mut self,
        source_path: P,
        output_path: P,
    ) -> Result<ExportStats, String> {
        // Validate source
        self.validate_model(&source_path)?;
        
        // Read source model
        let mut file = File::open(&source_path)
            .map_err(|e| format!("Failed to open source: {}", e))?;
            
        let mut model_data = Vec::new();
        file.read_to_end(&mut model_data)
            .map_err(|e| format!("Failed to read source: {}", e))?;
        
        let original_size = model_data.len();
        
        // Apply optimizations
        if self.config.enable_fusion {
            model_data = self.apply_fusion(&model_data)?;
        }
        
        if self.config.quantize {
            model_data = self.apply_quantization(&model_data)?;
        }
        
        let optimized_size = model_data.len();
        
        // Write output
        let mut out_file = File::create(&output_path)
            .map_err(|e| format!("Failed to create output: {}", e))?;
            
        out_file.write_all(&model_data)
            .map_err(|e| format!("Failed to write output: {}", e))?;
        
        Ok(ExportStats {
            original_size,
            optimized_size,
            compression_ratio: optimized_size as f64 / original_size as f64,
            fused_ops_count: self.fused_ops.len(),
            opset_version: self.config.opset_version,
        })
    }
    
    /// Get list of applied fusions
    pub fn get_fused_ops(&self) -> &[FusedOp] {
        &self.fused_ops
    }
}

impl Default for OnnxExporter {
    fn default() -> Self {
        Self::new()
    }
}

/// Export statistics
#[derive(Debug, Clone)]
pub struct ExportStats {
    pub original_size: usize,
    pub optimized_size: usize,
    pub compression_ratio: f64,
    pub fused_ops_count: usize,
    pub opset_version: i64,
}

impl ExportStats {
    pub fn print_summary(&self) {
        println!("\n=== Export Statistics ===");
        println!("Original Size:     {} bytes", self.original_size);
        println!("Optimized Size:    {} bytes", self.optimized_size);
        println!("Compression Ratio: {:.2}x", 1.0 / self.compression_ratio);
        println!("Fused Operators:   {}", self.fused_ops_count);
        println!("ONNX Opset:        {}", self.opset_version);
        println!("=========================\n");
    }
}

/// Batch exporter for multiple models
pub struct BatchExporter {
    exporters: Vec<OnnxExporter>,
}

impl BatchExporter {
    pub fn new(num_workers: usize) -> Self {
        Self {
            exporters: (0..num_workers).map(|_| OnnxExporter::new()).collect(),
        }
    }
    
    pub fn export_batch<P: AsRef<Path>>(
        &mut self,
        models: &[(P, P)],  // (source, destination) pairs
    ) -> Vec<Result<ExportStats, String>> {
        models.iter().enumerate().map(|(i, (src, dst))| {
            let idx = i % self.exporters.len();
            self.exporters[idx].export(src, dst)
        }).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_exporter_creation() {
        let exporter = OnnxExporter::new();
        assert_eq!(exporter.config.opset_version, 17);
        assert!(exporter.config.enable_fusion);
        assert!(exporter.config.quantize);
    }

    #[test]
    fn test_custom_config() {
        let config = ExportConfig {
            opset_version: 15,
            enable_fusion: false,
            optimize_level: 3,
            quantize: false,
        };
        
        let exporter = OnnxExporter::with_config(config.clone());
        assert_eq!(exporter.config.opset_version, 15);
        assert!(!exporter.config.enable_fusion);
        assert!(exporter.fused_ops.is_empty());
    }

    #[test]
    fn test_validate_model() {
        // Create temp file
        let temp_path = "/tmp/test_model.pt";
        std::fs::write(temp_path, "dummy").unwrap();
        
        let exporter = OnnxExporter::new();
        let result = exporter.validate_model(temp_path);
        
        assert!(result.is_ok());
        
        // Test non-existent file
        let result = exporter.validate_model("/nonexistent/path.onnx");
        assert!(result.is_err());
        
        // Cleanup
        std::fs::remove_file(temp_path).unwrap();
    }

    #[test]
    fn test_export_pipeline() {
        // Create dummy source file
        let src_path = "/tmp/src_model.bin";
        let dst_path = "/tmp/dst_model.onnx";
        
        std::fs::write(src_path, vec![0u8; 1024]).unwrap();
        
        let mut exporter = OnnxExporter::new();
        let stats = exporter.export(src_path, dst_path).unwrap();
        
        assert!(stats.original_size == 1024);
        assert!(stats.optimized_size > 1024);  // Metadata added
        assert!(stats.fused_ops_count > 0);
        
        // Verify output exists
        assert!(Path::new(dst_path).exists());
        
        // Cleanup
        std::fs::remove_file(src_path).unwrap();
        std::fs::remove_file(dst_path).unwrap();
    }

    #[test]
    fn test_export_stats() {
        let stats = ExportStats {
            original_size: 1000000,
            optimized_size: 250000,
            compression_ratio: 0.25,
            fused_ops_count: 4,
            opset_version: 17,
        };
        
        assert_eq!(stats.compression_ratio, 0.25);
        assert_eq!(stats.fused_ops_count, 4);
    }
}
