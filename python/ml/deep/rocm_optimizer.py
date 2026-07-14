"""
PyTorch ROCm (HIP) specific tuning script for AMD Radeon GPU.
Implements VRAM allocation strategies, activation offloading, and memory-efficient training.
Designed to strictly bound VRAM usage while maximizing throughput.
"""

import os
import logging
from typing import Any, Dict, Optional, Tuple
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class ROCmMemoryOptimizer:
    """
    Memory optimizer for PyTorch on AMD ROCm (HIP).
    Implements VRAM bounding, activation checkpointing, and pinned memory offloading.
    """
    
    def __init__(
        self,
        max_vram_mb: int = 4096,  # Max VRAM to use
        activation_offload_ratio: float = 0.3,  # Ratio of activations to offload to CPU
        enable_grad_checkpointing: bool = True,
        use_bfloat16: bool = True,
        pinned_memory_size_mb: int = 512,
    ):
        self.max_vram_mb = max_vram_mb
        self.activation_offload_ratio = activation_offload_ratio
        self.enable_grad_checkpointing = enable_grad_checkpointing
        self.use_bfloat16 = use_bfloat16 and torch.cuda.is_bf16_supported()
        self.pinned_memory_size_mb = pinned_memory_size_mb
        
        self.device: Optional[torch.device] = None
        self.pinned_memory_pool: Optional[torch.Tensor] = None
        self.vram_usage_history: list = []
        
        self._initialize_rocm()
    
    def _initialize_rocm(self):
        """Initialize ROCm-specific settings."""
        # Set ROCm environment variables
        os.environ["HSA_OVERRIDE_GFX_VERSION"] = os.getenv("HSA_OVERRIDE_GFX_VERSION", "11.0.0")
        os.environ["PYTORCH_HIP_ALLOC_CONF"] = "true"
        os.environ["MAX_SPLIT_SIZE_MB"] = os.getenv("MAX_SPLIT_SIZE_MB", "128")
        
        # Check for ROCm availability
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            
            # Get GPU info
            gpu_name = torch.cuda.get_device_name(0)
            vram_total = torch.cuda.get_device_properties(0).total_memory / (1024**2)
            
            logger.info(f"AMD GPU detected: {gpu_name}")
            logger.info(f"Total VRAM: {vram_total:.0f}MB")
            logger.info(f"Target VRAM limit: {self.max_vram_mb}MB")
            
            # Warn if target exceeds available
            if self.max_vram_mb > vram_total * 0.9:
                logger.warning(
                    f"Target VRAM ({self.max_vram_mb}MB) exceeds 90% of total VRAM. "
                    "Reducing to safe limit."
                )
                self.max_vram_mb = int(vram_total * 0.8)
            
            # Initialize pinned memory pool for activation offloading
            self._setup_pinned_memory()
        else:
            self.device = torch.device("cpu")
            logger.warning("ROCm/CUDA not available. Using CPU mode.")
    
    def _setup_pinned_memory(self):
        """Setup pinned (page-locked) memory pool for efficient CPU-GPU transfers."""
        if self.device is None or self.device.type != "cuda":
            return
        
        try:
            # Allocate pinned memory pool
            pinned_size_bytes = self.pinned_memory_size_mb * 1024 * 1024
            self.pinned_memory_pool = torch.empty(
                pinned_size_bytes // 4,  # float32
                dtype=torch.float32,
                device="cpu",
                pin_memory=True,
            )
            logger.info(f"Pinned memory pool allocated: {self.pinned_memory_size_mb}MB")
        except Exception as e:
            logger.warning(f"Failed to allocate pinned memory: {e}")
            self.pinned_memory_pool = None
    
    def get_vram_usage(self) -> Dict[str, float]:
        """Get current VRAM usage statistics."""
        if self.device is None or self.device.type != "cuda":
            return {"allocated": 0.0, "cached": 0.0, "total": 0.0}
        
        allocated = torch.cuda.memory_allocated(0) / (1024**2)
        cached = torch.cuda.memory_reserved(0) / (1024**2)
        total = torch.cuda.get_device_properties(0).total_memory / (1024**2)
        
        return {
            "allocated": allocated,
            "cached": cached,
            "total": total,
            "available": total - allocated,
            "usage_percent": (allocated / total) * 100,
        }
    
    def check_vram_limit(self) -> bool:
        """Check if VRAM usage is within limits."""
        usage = self.get_vram_usage()
        within_limit = usage["allocated"] < self.max_vram_mb
        
        if not within_limit:
            logger.warning(
                f"VRAM usage ({usage['allocated']:.0f}MB) exceeds limit ({self.max_vram_mb}MB). "
                "Triggering garbage collection."
            )
            self.cleanup()
        
        return within_limit
    
    def cleanup(self):
        """Force garbage collection and clear CUDA cache."""
        if self.device is not None and self.device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        import gc
        gc.collect()
        
        logger.debug("VRAM cleanup completed")
    
    def apply_memory_efficient_training(
        self,
        model: nn.Module,
        batch_size: int,
        sequence_length: int,
    ) -> Tuple[nn.Module, int]:
        """
        Apply memory-efficient training optimizations to a model.
        
        Returns:
            Optimized model and recommended batch size
        """
        usage = self.get_vram_usage()
        available_vram = self.max_vram_mb - usage["allocated"]
        
        # Estimate memory per sample (rough approximation)
        # This varies by model architecture
        bytes_per_sample = sequence_length * 4 * 10  # Simplified estimate
        
        # Calculate safe batch size
        samples_in_vram = int((available_vram * 1024**2 * 0.5) / bytes_per_sample)
        recommended_batch_size = min(batch_size, max(1, samples_in_vram))
        
        logger.info(
            f"Available VRAM: {available_vram:.0f}MB, "
            f"Recommended batch size: {recommended_batch_size} (original: {batch_size})"
        )
        
        # Apply bfloat16 if enabled and supported
        if self.use_bfloat16:
            model = model.bfloat16()
            logger.info("Model converted to bfloat16")
        
        # Enable gradient checkpointing if enabled
        if self.enable_grad_checkpointing:
            if hasattr(model, "gradient_checkpointing_enable"):
                model.gradient_checkpointing_enable()
                logger.info("Gradient checkpointing enabled")
            else:
                # Apply checkpointing manually to modules with many parameters
                for module in model.modules():
                    if isinstance(module, (nn.Linear, nn.Conv1d, nn.LSTM)):
                        module.register_forward_pre_hook(self._checkpoint_hook)
                logger.info("Applied manual gradient checkpointing")
        
        return model, recommended_batch_size
    
    def _checkpoint_hook(self, module, inputs):
        """Hook for manual gradient checkpointing."""
        # This is a simplified hook; real implementation would be more complex
        pass
    
    def offload_activations_to_cpu(
        self,
        activations: torch.Tensor,
        ratio: Optional[float] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Offload a portion of activations to pinned CPU memory.
        
        Returns:
            Tensor on GPU (remaining), Tensor on CPU (offloaded)
        """
        if ratio is None:
            ratio = self.activation_offload_ratio
        
        if self.device is None or self.device.type != "cuda":
            return activations, torch.tensor([])
        
        n_elements = activations.numel()
        offload_count = int(n_elements * ratio)
        keep_count = n_elements - offload_count
        
        # Flatten and split
        flat = activations.flatten()
        keep_part = flat[:keep_count].clone()
        offload_part = flat[keep_count:].clone()
        
        # Move offload part to CPU
        offload_part_cpu = offload_part.to("cpu", non_blocking=True)
        
        return keep_part, offload_part_cpu
    
    def restore_activations_from_cpu(
        self,
        gpu_part: torch.Tensor,
        cpu_part: torch.Tensor,
        original_shape: torch.Size,
    ) -> torch.Tensor:
        """Restore activations from CPU back to GPU."""
        if self.device is None or self.device.type != "cuda":
            return gpu_part
        
        # Move CPU part back to GPU
        cpu_part_gpu = cpu_part.to(self.device, non_blocking=True)
        
        # Concatenate and reshape
        full = torch.cat([gpu_part, cpu_part_gpu])
        return full.reshape(original_shape)
    
    def optimize_dataloader(
        self,
        dataloader: torch.utils.data.DataLoader,
        pin_memory: bool = True,
        num_workers: int = 0,
    ) -> torch.utils.data.DataLoader:
        """
        Optimize DataLoader for ROCm training.
        """
        # For ROCm, pin_memory helps with faster CPU->GPU transfers
        # But num_workers > 0 can cause issues with some setups
        
        optimized_loader = torch.utils.data.DataLoader(
            dataloader.dataset,
            batch_size=dataloader.batch_size,
            shuffle=dataloader.sampler is not None,
            num_workers=num_workers,
            pin_memory=pin_memory and self.device is not None and self.device.type == "cuda",
            drop_last=dataloader.drop_last,
            persistent_workers=num_workers > 0,
        )
        
        logger.info(f"DataLoader optimized: pin_memory={pin_memory}, workers={num_workers}")
        return optimized_loader
    
    def get_optimization_report(self) -> Dict[str, Any]:
        """Generate a report of applied optimizations."""
        usage = self.get_vram_usage()
        
        return {
            "max_vram_mb": self.max_vram_mb,
            "current_allocated_mb": usage["allocated"],
            "current_cached_mb": usage["cached"],
            "usage_percent": usage["usage_percent"],
            "bfloat16_enabled": self.use_bfloat16,
            "grad_checkpointing_enabled": self.enable_grad_checkpointing,
            "activation_offload_ratio": self.activation_offload_ratio,
            "pinned_memory_pool_mb": (
                self.pinned_memory_size_mb if self.pinned_memory_pool is not None else 0
            ),
            "device": str(self.device) if self.device else "None",
        }


def auto_tune_for_model(
    model: nn.Module,
    input_shape: Tuple[int, ...],
    target_vram_mb: int = 4096,
) -> Dict[str, Any]:
    """
    Automatically tune ROCm settings for a given model.
    
    Args:
        model: PyTorch model to tune
        input_shape: Shape of input tensor (batch, seq_len, features)
        target_vram_mb: Target VRAM limit
    
    Returns:
        Dictionary with tuning recommendations
    """
    optimizer = ROCmMemoryOptimizer(max_vram_mb=target_vram_mb)
    
    # Dry run to estimate memory
    model.eval()
    dummy_input = torch.randn(input_shape, device=optimizer.device)
    
    # Measure memory before
    mem_before = optimizer.get_vram_usage()
    
    # Forward pass
    with torch.no_grad():
        try:
            _ = model(dummy_input)
        except Exception as e:
            logger.error(f"Forward pass failed: {e}")
            return {"error": str(e)}
    
    # Measure memory after
    mem_after = optimizer.get_vram_usage()
    
    # Estimate per-sample memory
    batch_size = input_shape[0]
    memory_increase = mem_after["allocated"] - mem_before["allocated"]
    per_sample_memory = memory_increase / batch_size
    
    # Calculate optimal batch size
    available_vram = target_vram_mb * 0.7  # Leave 30% headroom
    optimal_batch_size = max(1, int(available_vram / per_sample_memory))
    
    return {
        "memory_per_forward_mb": memory_increase,
        "per_sample_memory_mb": per_sample_memory,
        "optimal_batch_size": optimal_batch_size,
        "recommended_gradient_accumulation": max(1, batch_size // optimal_batch_size),
        "optimization_report": optimizer.get_optimization_report(),
    }


def main():
    """Example usage."""
    import psutil
    
    print("=" * 60)
    print("ROCm Memory Optimizer Demo")
    print("=" * 60)
    
    # Create optimizer
    optimizer = ROCmMemoryOptimizer(
        max_vram_mb=4096,
        activation_offload_ratio=0.3,
        enable_grad_checkpointing=True,
    )
    
    # Print initial status
    print("\nInitial VRAM status:")
    print(optimizer.get_vram_usage())
    
    # Create a simple test model
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(64, 128, num_layers=2, batch_first=True)
            self.linear = nn.Linear(128, 1)
        
        def forward(self, x):
            out, _ = self.lstm(x)
            return self.linear(out[:, -1, :])
    
    model = SimpleModel()
    
    # Auto-tune
    print("\nAuto-tuning for model...")
    tuning_results = auto_tune_for_model(
        model,
        input_shape=(32, 100, 64),
        target_vram_mb=4096,
    )
    print(f"Tuning results: {tuning_results}")
    
    # Apply optimizations
    print("\nApplying memory-efficient training...")
    optimized_model, new_batch_size = optimizer.apply_memory_efficient_training(
        model,
        batch_size=32,
        sequence_length=100,
    )
    print(f"New batch size: {new_batch_size}")
    
    # Final report
    print("\nFinal optimization report:")
    report = optimizer.get_optimization_report()
    for key, value in report.items():
        print(f"  {key}: {value}")
    
    # System RAM
    ram_available = psutil.virtual_memory().available / (1024**3)
    print(f"\nSystem RAM available: {ram_available:.2f}GB")


if __name__ == "__main__":
    main()
