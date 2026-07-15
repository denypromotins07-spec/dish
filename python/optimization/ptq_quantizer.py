"""
Post-Training Quantization (PTQ) script for PyTorch models.
Converts RL and Deep Hedging models to INT8 for reduced memory and faster inference.
Optimized for AMD Radeon GPU execution.
"""

import torch
import torch.nn as nn
import torch.quantization as quantization
from typing import Dict, Optional, Callable
import copy


class QuantizationConfig:
    """Configuration for PTQ process."""
    
    def __init__(
        self,
        dtype: torch.dtype = torch.qint8,
        granularity: str = 'per_tensor',
        reduce_range: bool = False,
        inplace: bool = False,
    ):
        self.dtype = dtype
        self.granularity = granularity
        self.reduce_range = reduce_range
        self.inplace = inplace


def prepare_model_for_quantization(
    model: nn.Module,
    example_inputs: torch.Tensor,
    config: Optional[QuantizationConfig] = None,
) -> nn.Module:
    """
    Prepare model for post-training quantization.
    Inserts observers into the model.
    """
    config = config or QuantizationConfig()
    
    # Create a copy to avoid modifying original
    model_copy = copy.deepcopy(model)
    model_copy.eval()
    
    # Set quantization config
    model_copy.qconfig = quantization.get_default_qconfig('fbgemm')
    if config.dtype == torch.qint8:
        model_copy.qconfig = quantization.QConfig(
            activation=quantization.default_observer.with_args(
                dtype=torch.quint8,
                qscheme=torch.per_tensor_affine,
            ),
            weight=quantization.default_weight_observer.with_args(
                dtype=config.dtype,
            )
        )
    
    # Prepare for quantization
    prepared_model = quantization.prepare(model_copy, inplace=config.inplace)
    
    # Run calibration with example inputs
    with torch.no_grad():
        _ = prepared_model(example_inputs)
        
    return prepared_model


def convert_to_int8(
    prepared_model: nn.Module,
    inplace: bool = False,
) -> nn.Module:
    """
    Convert prepared model to INT8 quantized model.
    """
    return quantization.convert(prepared_model, inplace=inplace)


class PTQQuantizer:
    """
    Post-Training Quantization handler for RL/Hedging models.
    Supports calibration and validation.
    """
    
    def __init__(self, config: Optional[QuantizationConfig] = None):
        self.config = config or QuantizationConfig()
        self.calibration_data: list = []
        
    def collect_calibration_data(
        self,
        data_loader,
        num_batches: int = 10,
    ) -> None:
        """Collect calibration data from data loader."""
        self.calibration_data = []
        
        for i, batch in enumerate(data_loader):
            if i >= num_batches:
                break
            if isinstance(batch, (tuple, list)):
                self.calibration_data.append(batch[0])
            else:
                self.calibration_data.append(batch)
                
    def quantize_model(
        self,
        model: nn.Module,
        calibration_inputs: Optional[torch.Tensor] = None,
    ) -> nn.Module:
        """
        Full PTQ pipeline: prepare -> calibrate -> convert.
        """
        # Use collected calibration data if no inputs provided
        if calibration_inputs is None:
            if len(self.calibration_data) == 0:
                raise ValueError("No calibration data available")
            calibration_inputs = torch.cat(self.calibration_data[:5], dim=0)
            
        # Ensure model is in eval mode
        model.eval()
        
        # Prepare
        prepared = prepare_model_for_quantization(
            model, 
            calibration_inputs,
            self.config,
        )
        
        # Calibrate (already done during prepare with example inputs)
        with torch.no_grad():
            for data in self.calibration_data:
                if isinstance(data, torch.Tensor):
                    _ = prepared(data)
                    
        # Convert
        quantized_model = convert_to_int8(prepared, inplace=self.config.inplace)
        
        return quantized_model
    
    def validate_quantization(
        self,
        original_model: nn.Module,
        quantized_model: nn.Module,
        validation_loader,
        tolerance: float = 0.05,
    ) -> Dict[str, float]:
        """
        Validate quantized model against original.
        Returns metrics comparing outputs.
        """
        original_model.eval()
        quantized_model.eval()
        
        mse_sum = 0.0
        max_diff = 0.0
        n_samples = 0
        
        with torch.no_grad():
            for batch in validation_loader:
                if isinstance(batch, (tuple, list)):
                    inputs = batch[0]
                else:
                    inputs = batch
                    
                orig_output = original_model(inputs)
                quant_output = quantized_model(inputs)
                
                # Handle quantized tensor dequantization
                if isinstance(quant_output, torch.Tensor):
                    if quant_output.is_quantized:
                        quant_output = quant_output.dequantize()
                        
                # Ensure same shape for comparison
                if orig_output.shape != quant_output.shape:
                    continue
                    
                mse = ((orig_output - quant_output) ** 2).mean().item()
                diff = (orig_output - quant_output).abs().max().item()
                
                mse_sum += mse
                max_diff = max(max_diff, diff)
                n_samples += 1
                
        avg_mse = mse_sum / max(n_samples, 1)
        
        return {
            'avg_mse': avg_mse,
            'max_diff': max_diff,
            'passed': avg_mse < tolerance and max_diff < tolerance * 10,
        }
    
    def get_memory_savings(
        self,
        original_model: nn.Module,
        quantized_model: nn.Module,
    ) -> Dict[str, float]:
        """Calculate memory savings from quantization."""
        orig_params = sum(p.numel() for p in original_model.parameters())
        quant_params = sum(p.numel() for p in quantized_model.parameters())
        
        # Original: float32 (4 bytes), Quantized: int8 (1 byte)
        orig_bytes = orig_params * 4
        quant_bytes = quant_params * 1
        
        savings_mb = (orig_bytes - quant_bytes) / (1024 ** 2)
        savings_pct = (1 - quant_bytes / orig_bytes) * 100 if orig_bytes > 0 else 0
        
        return {
            'original_size_mb': orig_bytes / (1024 ** 2),
            'quantized_size_mb': quant_bytes / (1024 ** 2),
            'savings_mb': savings_mb,
            'savings_percent': savings_pct,
        }


def quantize_and_export(
    model: nn.Module,
    calibration_inputs: torch.Tensor,
    output_path: str,
    use_trace: bool = True,
) -> None:
    """
    Quantize model and export to TorchScript for deployment.
    """
    model.eval()
    
    # Quantize
    quantizer = PTQQuantizer()
    quantizer.calibration_data = [calibration_inputs]
    quantized_model = quantizer.quantize_model(model, calibration_inputs)
    
    # Export to TorchScript
    quantized_model.eval()
    with torch.no_grad():
        if use_trace:
            scripted = torch.jit.trace(quantized_model, calibration_inputs)
        else:
            scripted = torch.jit.script(quantized_model)
            
        scripted.save(output_path)
        
    print(f"Quantized model saved to: {output_path}")


if __name__ == "__main__":
    # Example: Quantize a simple model
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(56, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 2),
            )
            
        def forward(self, x):
            return self.net(x)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create model
    model = SimpleModel().to(device)
    model.eval()
    
    # Calibration data
    calibration_inputs = torch.randn(32, 56).to(device)
    
    # Quantize
    quantizer = PTQQuantizer()
    quantizer.calibration_data = [calibration_inputs]
    quantized_model = quantizer.quantize_model(model, calibration_inputs)
    
    # Get memory savings
    savings = quantizer.get_memory_savings(model, quantized_model)
    print(f"\nMemory Savings:")
    print(f"  Original: {savings['original_size_mb']:.2f} MB")
    print(f"  Quantized: {savings['quantized_size_mb']:.2f} MB")
    print(f"  Savings: {savings['savings_mb']:.2f} MB ({savings['savings_percent']:.1f}%)")
    
    # Test inference
    test_input = torch.randn(1, 56).to(device)
    
    with torch.no_grad():
        orig_output = model(test_input)
        quant_output = quantized_model(test_input)
        
        if isinstance(quant_output, torch.Tensor) and quant_output.is_quantized:
            quant_output = quant_output.dequantize()
            
    diff = (orig_output - quant_output).abs().mean().item()
    print(f"\nOutput difference (MSE): {diff:.6f}")
    
    # Export
    quantize_and_export(model, calibration_inputs, "/tmp/quantized_model.pt")
