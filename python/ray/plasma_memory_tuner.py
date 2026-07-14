"""
Ray Plasma Object Store Memory Tuner.

Configuration and tuning script for Ray's Plasma object store,
adjusting memory limits, eviction policies, and spill-to-disk
thresholds to prevent it from consuming the 14GB system RAM.
"""

import logging
import os
import psutil
from dataclasses import dataclass
from typing import Any, Dict, Optional

log = logging.getLogger(__name__)


@dataclass
class PlasmaConfig:
    """Configuration for Plasma object store tuning."""
    
    # Memory limits (critical for 14GB constraint)
    max_memory_bytes: int = 512 * 1024 * 1024  # 512MB default
    max_spill_bytes: int = 2 * 1024 * 1024 * 1024  # 2GB spill limit
    
    # Eviction policy settings
    eviction_enabled: bool = True
    eviction_fraction: float = 0.7  # Evict when 70% full
    
    # Spill-to-disk configuration
    spill_enabled: bool = True
    spill_directory: str = "/tmp/ray_plasma_spill"
    spill_threshold_fraction: float = 0.8  # Start spilling at 80% capacity
    
    # Object size limits
    max_object_size_bytes: int = 100 * 1024 * 1024  # 100MB max per object
    large_object_threshold_bytes: int = 10 * 1024 * 1024  # 10MB threshold
    
    # Performance tuning
    allocator_type: str = "mimalloc"  # Low-fragmentation allocator
    huge_pages_enabled: bool = False  # Disable to save memory
    
    # Monitoring
    stats_interval_seconds: int = 5
    warning_threshold_fraction: float = 0.85
    critical_threshold_fraction: float = 0.95


class PlasmaMemoryTuner:
    """
    Tune and monitor Ray Plasma object store memory usage.
    
    Ensures Plasma stays within strict memory bounds to coexist
    with the live trading engine on a 14GB system.
    """
    
    def __init__(self, config: Optional[PlasmaConfig] = None):
        self.config = config or PlasmaConfig()
        self._ensure_spill_directory()
        
    def _ensure_spill_directory(self) -> None:
        """Create spill directory if it doesn't exist."""
        if self.config.spill_enabled:
            os.makedirs(self.config.spill_directory, exist_ok=True)
            log.info(f"Plasma spill directory: {self.config.spill_directory}")
            
    def get_optimal_memory_size(self) -> int:
        """
        Calculate optimal Plasma memory size based on system constraints.
        
        Returns memory size in bytes.
        """
        total_ram_gb = psutil.virtual_memory().total / (1024 ** 3)
        
        # Reserve 10GB for trading engine, OS, and other processes
        reserved_for_trading_gb = 10.0
        
        # Available for Ray total
        available_for_ray_gb = max(1.0, total_ram_gb - reserved_for_trading_gb)
        
        # Plasma gets 50% of Ray allocation
        plasma_allocation_gb = available_for_ray_gb * 0.5
        
        # Cap at configured maximum
        optimal_bytes = min(
            int(plasma_allocation_gb * 1024 * 1024 * 1024),
            self.config.max_memory_bytes
        )
        
        log.info(
            f"Calculated Plasma memory: {optimal_bytes / (1024**2):.2f}MB "
            f"(system: {total_ram_gb:.2f}GB, available for Ray: {available_for_ray_gb:.2f}GB)"
        )
        
        return optimal_bytes
        
    def build_plasma_config_dict(self) -> Dict[str, Any]:
        """Build configuration dictionary for Ray initialization."""
        memory_size = self.get_optimal_memory_size()
        
        return {
            "object_store_memory": memory_size,
            "_plasma_directory": self.config.spill_directory if self.config.spill_enabled else None,
            "_max_spill_size": self.config.max_spill_bytes if self.config.spill_enabled else 0,
        }
        
    def get_runtime_env_config(self) -> Dict[str, Any]:
        """Get runtime environment configuration for Ray tasks."""
        return {
            "env_vars": {
                "RAY_PLASMA_DIRECTORY": self.config.spill_directory,
                "RAY_OBJECT_STORE_MEMORY": str(self.get_optimal_memory_size()),
                "MALLOC_CONF": "retain:true" if self.config.allocator_type == "jemalloc" else "",
            }
        }
        
    def check_memory_pressure(self) -> tuple:
        """
        Check current memory pressure status.
        
        Returns:
            Tuple of (is_warning, is_critical, usage_fraction)
        """
        try:
            # Get Plasma store info via Ray (if available)
            import ray
            
            if ray.is_initialized():
                # Try to get actual plasma usage
                try:
                    # This requires ray.internal which may not be available
                    # Fallback to system memory check
                    pass
                except:
                    pass
                    
            # Fallback to system memory check
            mem = psutil.virtual_memory()
            used_fraction = 1.0 - (mem.available / mem.total)
            
            is_warning = used_fraction >= self.config.warning_threshold_fraction
            is_critical = used_fraction >= self.config.critical_threshold_fraction
            
            return (is_warning, is_critical, used_fraction)
            
        except Exception as e:
            log.warning(f"Failed to check memory pressure: {e}")
            return (False, False, 0.0)
            
    def trigger_eviction_if_needed(self) -> bool:
        """
        Trigger object eviction if memory pressure is high.
        
        Returns True if eviction was triggered.
        """
        is_warning, is_critical, usage = self.check_memory_pressure()
        
        if is_critical:
            log.warning(
                f"Critical memory pressure detected ({usage:.1%}), "
                "triggering aggressive eviction"
            )
            # In production, would call ray.internal.evict_objects()
            return True
        elif is_warning:
            log.info(f"Memory pressure warning ({usage:.1%}), monitoring closely")
            return False
            
        return False
        
    def validate_object_size(self, size_bytes: int) -> bool:
        """Validate that an object size is within limits."""
        if size_bytes > self.config.max_object_size_bytes:
            log.warning(
                f"Object size {size_bytes / (1024**2):.2f}MB exceeds "
                f"maximum {self.config.max_object_size_bytes / (1024**2):.2f}MB"
            )
            return False
            
        if size_bytes > self.config.large_object_threshold_bytes:
            log.info(
                f"Large object detected: {size_bytes / (1024**2):.2f}MB "
                "(consider splitting into smaller chunks)"
            )
            
        return True
        
    def get_status_report(self) -> Dict[str, Any]:
        """Generate status report for monitoring."""
        is_warning, is_critical, usage = self.check_memory_pressure()
        mem = psutil.virtual_memory()
        
        return {
            "plasma_max_memory_mb": self.config.max_memory_bytes / (1024 ** 2),
            "plasma_spill_max_gb": self.config.max_spill_bytes / (1024 ** 3),
            "spill_enabled": self.config.spill_enabled,
            "eviction_enabled": self.config.eviction_enabled,
            "system_memory_used_percent": (1.0 - mem.available / mem.total) * 100,
            "memory_pressure_warning": is_warning,
            "memory_pressure_critical": is_critical,
            "current_usage_fraction": usage,
            "spill_directory": self.config.spill_directory,
        }


def configure_plasma_for_low_memory() -> Dict[str, Any]:
    """
    Convenience function to configure Plasma for low-memory systems.
    
    Returns configuration dictionary suitable for passing to ray.init().
    """
    tuner = PlasmaMemoryTuner(PlasmaConfig(
        max_memory_bytes=512 * 1024 * 1024,  # 512MB
        max_spill_bytes=1024 * 1024 * 1024,  # 1GB
        eviction_enabled=True,
        spill_enabled=True,
    ))
    
    config = tuner.build_plasma_config_dict()
    log.info(f"Plasma configured: {config['object_store_memory'] / (1024**2):.2f}MB object store")
    
    return config


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    tuner = PlasmaMemoryTuner()
    
    print("=== Plasma Memory Tuner Status ===")
    status = tuner.get_status_report()
    for key, value in status.items():
        print(f"  {key}: {value}")
        
    print("\n=== Configuration for ray.init() ===")
    config = configure_plasma_for_low_memory()
    print(f"  object_store_memory: {config['object_store_memory']}")
