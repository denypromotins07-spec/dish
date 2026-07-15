"""
ROCm-specific memory profiler for PyTorch models.
Tracks tensor allocations and forces GC to enforce VRAM/RAM ceiling.
Optimized for AMD Radeon GPU training workloads.
"""

import torch
import gc
import os
from typing import Dict, List, Optional, Callable
from contextlib import contextmanager
import warnings


class ROCmMemoryProfiler:
    """
    Memory profiler specifically for AMD ROCm devices.
    Tracks allocations, triggers GC, and enforces memory limits.
    """
    
    def __init__(
        self,
        max_vram_gb: float = 6.0,  # Conservative limit for 8GB GPU
        max_ram_gb: float = 8.0,   # RAM limit for system memory
        gc_threshold: float = 0.8,  # Trigger GC at 80% usage
    ):
        self.max_vram_bytes = int(max_vram_gb * 1024 ** 3)
        self.max_ram_bytes = int(max_ram_gb * 1024 ** 3)
        self.gc_threshold = gc_threshold
        
        self.allocation_history: List[Dict] = []
        self.peak_vram = 0
        self.peak_ram = 0
        
        # Check if ROCm is available
        self.has_rocm = (
            torch.cuda.is_available() and 
            torch.version.hip is not None
        )
        
    def get_vram_usage(self) -> int:
        """Get current VRAM usage in bytes."""
        if self.has_rocm and torch.cuda.is_available():
            return torch.cuda.memory_allocated(0)
        return 0
    
    def get_vram_reserved(self) -> int:
        """Get reserved VRAM in bytes."""
        if self.has_rocm and torch.cuda.is_available():
            return torch.cuda.memory_reserved(0)
        return 0
    
    def get_ram_usage(self) -> int:
        """Get current RAM usage in bytes (estimate)."""
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    
    def get_memory_stats(self) -> Dict[str, float]:
        """Get comprehensive memory statistics."""
        vram_used = self.get_vram_usage()
        vram_reserved = self.get_vram_reserved()
        ram_used = self.get_ram_usage()
        
        self.peak_vram = max(self.peak_vram, vram_used)
        self.peak_ram = max(self.peak_ram, ram_used)
        
        return {
            'vram_used_mb': vram_used / (1024 ** 2),
            'vram_reserved_mb': vram_reserved / (1024 ** 2),
            'vram_max_mb': self.max_vram_bytes / (1024 ** 2),
            'vram_utilization': vram_used / self.max_vram_bytes,
            'ram_used_mb': ram_used / (1024 ** 2),
            'ram_max_mb': self.max_ram_bytes / (1024 ** 2),
            'ram_utilization': ram_used / self.max_ram_bytes,
            'peak_vram_mb': self.peak_vram / (1024 ** 2),
            'peak_ram_mb': self.peak_ram / (1024 ** 2),
        }
    
    def check_limits(self) -> Dict[str, bool]:
        """Check if memory usage is within limits."""
        stats = self.get_memory_stats()
        
        return {
            'vram_ok': stats['vram_utilization'] < 1.0,
            'ram_ok': stats['ram_utilization'] < 1.0,
            'vram_warning': stats['vram_utilization'] > self.gc_threshold,
            'ram_warning': stats['ram_utilization'] > self.gc_threshold,
        }
    
    def force_gc(self, aggressive: bool = False) -> None:
        """
        Force garbage collection and empty CUDA cache.
        """
        # Python GC
        gc.collect()
        
        # Empty CUDA/ROCm cache
        if self.has_rocm and torch.cuda.is_available():
            torch.cuda.empty_cache()
            
            if aggressive:
                # Synchronize and reset accumulation counters
                torch.cuda.synchronize()
                
    def enforce_limit(self) -> bool:
        """
        Enforce memory limits by triggering GC if needed.
        Returns True if action was taken.
        """
        checks = self.check_limits()
        
        if checks['vram_warning'] or checks['ram_warning']:
            self.force_gc(aggressive=True)
            return True
            
        if not checks['vram_ok'] or not checks['ram_ok']:
            # Critical: aggressive cleanup
            self.force_gc(aggressive=True)
            
            # Clear any cached tensors
            if hasattr(torch, 'autograd') and hasattr(torch.autograd, 'grad'):
                torch.autograd.grad().clear() if hasattr(torch.autograd.grad(), 'clear') else None
                
            warnings.warn("Memory limit exceeded! Aggressive GC triggered.")
            return True
            
        return False
    
    def record_allocation(
        self,
        tensor_name: str,
        tensor_size_bytes: int,
        location: str = 'unknown',
    ) -> None:
        """Record a tensor allocation for tracking."""
        self.allocation_history.append({
            'name': tensor_name,
            'size_bytes': tensor_size_bytes,
            'location': location,
            'timestamp': torch.cuda.Event() if self.has_rocm else None,
        })
        
        # Keep history bounded
        if len(self.allocation_history) > 1000:
            self.allocation_history = self.allocation_history[-500:]
    
    def get_top_allocations(self, n: int = 10) -> List[Dict]:
        """Get top N largest allocations."""
        sorted_allocs = sorted(
            self.allocation_history,
            key=lambda x: x['size_bytes'],
            reverse=True,
        )
        return sorted_allocs[:n]
    
    def print_summary(self) -> None:
        """Print memory usage summary."""
        stats = self.get_memory_stats()
        checks = self.check_limits()
        
        print("\n" + "="*50)
        print("ROCm Memory Profiler Summary")
        print("="*50)
        print(f"VRAM: {stats['vram_used_mb']:.1f} / {stats['vram_max_mb']:.1f} MB ({stats['vram_utilization']*100:.1f}%)")
        print(f"  Reserved: {stats['vram_reserved_mb']:.1f} MB")
        print(f"  Peak: {stats['peak_vram_mb']:.1f} MB")
        print(f"RAM: {stats['ram_used_mb']:.1f} / {stats['ram_max_mb']:.1f} MB ({stats['ram_utilization']*100:.1f}%)")
        print(f"  Peak: {stats['peak_ram_mb']:.1f} MB")
        print("-"*50)
        print(f"Status: VRAM {'OK' if checks['vram_ok'] else 'EXCEEDED'} | RAM {'OK' if checks['ram_ok'] else 'EXCEEDED'}")
        
        if checks['vram_warning'] or checks['ram_warning']:
            print("WARNING: Memory usage above threshold!")
        print("="*50 + "\n")


@contextmanager
def track_memory(profiler: ROCmMemoryProfiler, operation_name: str):
    """Context manager to track memory during an operation."""
    start_stats = profiler.get_memory_stats()
    
    try:
        yield
    finally:
        end_stats = profiler.get_memory_stats()
        
        vram_delta = end_stats['vram_used_mb'] - start_stats['vram_used_mb']
        ram_delta = end_stats['ram_used_mb'] - start_stats['ram_used_mb']
        
        if abs(vram_delta) > 10 or abs(ram_delta) > 50:
            print(f"[{operation_name}] VRAM Δ: {vram_delta:+.1f} MB, RAM Δ: {ram_delta:+.1f} MB")


def gradient_checkpointing_wrapper(
    model: torch.nn.Module,
    enable: bool = True,
) -> torch.nn.Module:
    """
    Apply gradient checkpointing to reduce memory during training.
    """
    if not enable:
        return model
        
    # Enable checkpointing for all submodules
    if hasattr(model, 'gradient_checkpointing_enable'):
        model.gradient_checkpointing_enable()
        
    return model


def mixed_precision_setup(
    model: torch.nn.Module,
    use_amp: bool = True,
) -> torch.amp.autocast_mode:
    """
    Setup mixed precision training for ROCm.
    """
    if use_amp and torch.cuda.is_available():
        return torch.autocast(device_type='cuda', dtype=torch.bfloat16)
    return torch.autocast(device_type='cpu', dtype=torch.float32)


if __name__ == "__main__":
    # Example usage
    profiler = ROCmMemoryProfiler(max_vram_gb=6.0, max_ram_gb=8.0)
    
    print(f"ROCm Available: {profiler.has_rocm}")
    
    # Initial stats
    profiler.print_summary()
    
    # Simulate training with memory tracking
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Create some tensors
    print("\nAllocating tensors...")
    tensors = []
    
    for i in range(5):
        t = torch.randn(1000, 1000, device=device)
        profiler.record_allocation(f"tensor_{i}", t.numel() * t.element_size())
        tensors.append(t)
        
        # Check and enforce limits
        if profiler.enforce_limit():
            print(f"GC triggered after allocating tensor_{i}")
    
    # Stats after allocation
    profiler.print_summary()
    
    # Top allocations
    print("\nTop Allocations:")
    for alloc in profiler.get_top_allocations(5):
        print(f"  {alloc['name']}: {alloc['size_bytes'] / 1024**2:.2f} MB")
    
    # Cleanup
    print("\nCleaning up...")
    del tensors
    profiler.force_gc()
    profiler.print_summary()
    
    # Test context manager
    print("\nTesting context manager...")
    with track_memory(profiler, "test_operation"):
        _ = torch.randn(500, 500, device=device)
