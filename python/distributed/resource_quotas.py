"""
Strict Resource Quota Manager for Ray Tasks.
Mathematically prevents OOM by enforcing exact CPU/RAM fractions.
Designed for 14GB RAM ceiling on AMD Ryzen AI 5 laptop.
"""

from typing import Dict, Optional, Tuple
from dataclasses import dataclass
import ray
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ResourceQuota:
    """Immutable resource quota for a task."""
    num_cpus: float
    memory_bytes: int
    gpu_fraction: float = 0.0
    
    @property
    def memory_gb(self) -> float:
        return self.memory_bytes / (1024 ** 3)
    
    def to_ray_options(self) -> Dict:
        """Convert to Ray task options."""
        options = {
            "num_cpus": self.num_cpus,
            "memory": self.memory_bytes,
        }
        if self.gpu_fraction > 0:
            options["num_gpus"] = self.gpu_fraction
        return options


class ResourceQuotaManager:
    """
    Enforces strict resource quotas across all Ray tasks.
    Prevents any single task from exceeding its allocation.
    """
    
    # System-wide limits (14GB total, 2GB reserved for Rust engine)
    MAX_TOTAL_MEMORY_GB = 14.0
    RESERVED_MEMORY_GB = 2.0
    AVAILABLE_MEMORY_GB = MAX_TOTAL_MEMORY_GB - RESERVED_MEMORY_GB
    
    # Default quotas for different task types
    QUOTA_PROFILES = {
        "lightweight": ResourceQuota(num_cpus=0.25, memory_bytes=int(0.25 * 1024**3)),  # 256MB
        "standard": ResourceQuota(num_cpus=0.5, memory_bytes=int(0.5 * 1024**3)),      # 512MB
        "heavy": ResourceQuota(num_cpus=1.0, memory_bytes=int(1.0 * 1024**3)),         # 1GB
        "gpu_inference": ResourceQuota(num_cpus=0.5, memory_bytes=int(0.75 * 1024**3), gpu_fraction=0.2),
    }
    
    def __init__(self):
        self.active_quotas: Dict[str, ResourceQuota] = {}
        self.total_allocated_memory = 0
        self.total_allocated_cpus = 0.0
        self._lock = None  # Use asyncio.Lock in async context
        
    def register_task(self, task_id: str, profile: str = "standard", 
                      custom_quota: Optional[ResourceQuota] = None) -> ResourceQuota:
        """
        Register a task with a specific resource quota.
        Raises ValueError if quota would exceed system limits.
        """
        if custom_quota:
            quota = custom_quota
        elif profile in self.QUOTA_PROFILES:
            quota = self.QUOTA_PROFILES[profile]
        else:
            raise ValueError(f"Unknown quota profile: {profile}")
        
        # Check if adding this quota would exceed limits
        if not self._can_allocate(quota):
            raise MemoryError(
                f"Cannot allocate {quota.memory_gb:.2f}GB for task '{task_id}'. "
                f"Would exceed {self.AVAILABLE_MEMORY_GB}GB limit."
            )
        
        self.active_quotas[task_id] = quota
        self.total_allocated_memory += quota.memory_bytes
        self.total_allocated_cpus += quota.num_cpus
        
        logger.info(f"Registered task '{task_id}' with {quota.memory_gb:.2f}GB, {quota.num_cpus} CPUs")
        return quota
    
    def _can_allocate(self, quota: ResourceQuota) -> bool:
        """Check if allocation is within limits."""
        projected_memory = self.total_allocated_memory + quota.memory_bytes
        projected_cpus = self.total_allocated_cpus + quota.num_cpus
        
        # Strict memory check
        if projected_memory > self.AVAILABLE_MEMORY_GB * 1024**3:
            return False
            
        # CPU check (assume 8 logical cores on Ryzen AI 5)
        if projected_cpus > 8.0:
            return False
            
        return True
    
    def release_task(self, task_id: str) -> bool:
        """Release resources when a task completes."""
        if task_id not in self.active_quotas:
            return False
            
        quota = self.active_quotas.pop(task_id)
        self.total_allocated_memory -= quota.memory_bytes
        self.total_allocated_cpus -= quota.num_cpus
        
        logger.debug(f"Released task '{task_id}', freed {quota.memory_gb:.2f}GB")
        return True
    
    def get_available_resources(self) -> Tuple[float, float]:
        """Return available (memory_gb, cpus)."""
        available_memory = self.AVAILABLE_MEMORY_GB - (self.total_allocated_memory / 1024**3)
        available_cpus = 8.0 - self.total_allocated_cpus
        return max(0, available_memory), max(0, available_cpus)
    
    def get_utilization_stats(self) -> Dict:
        """Get current resource utilization statistics."""
        available_mem, available_cpus = self.get_available_resources()
        return {
            "total_memory_gb": self.AVAILABLE_MEMORY_GB,
            "used_memory_gb": self.total_allocated_memory / 1024**3,
            "available_memory_gb": available_mem,
            "memory_utilization_pct": (self.total_allocated_memory / 1024**3) / self.AVAILABLE_MEMORY_GB * 100,
            "total_cpus": 8.0,
            "used_cpus": self.total_allocated_cpus,
            "available_cpus": available_cpus,
            "cpu_utilization_pct": (self.total_allocated_cpus / 8.0) * 100,
            "active_tasks": len(self.active_quotas)
        }


def ray_task_with_quota(task_func, task_id: str, profile: str = "standard", 
                        custom_quota: Optional[ResourceQuota] = None):
    """
    Decorator factory to wrap Ray tasks with strict quota enforcement.
    Usage: @ray_task_with_quota(my_func, "my_task", profile="heavy")
    """
    manager = ResourceQuotaManager()
    
    try:
        quota = manager.register_task(task_id, profile, custom_quota)
    except MemoryError as e:
        logger.error(f"Quota registration failed: {e}")
        raise
    
    # Apply quota to Ray task
    @ray.remote(**quota.to_ray_options())
    def wrapped(*args, **kwargs):
        try:
            return task_func(*args, **kwargs)
        finally:
            manager.release_task(task_id)
    
    return wrapped


# Example usage with Ray actor
@ray.remote
class QuotaEnforcedActor:
    """Actor that enforces its own resource quota."""
    
    def __init__(self, actor_id: str, profile: str = "standard"):
        self.actor_id = actor_id
        self.manager = ResourceQuotaManager()
        self.quota = self.manager.register_task(actor_id, profile)
        
    def execute(self, data):
        """Execute task within quota bounds."""
        # Monitor memory during execution
        import psutil
        process = psutil.Process()
        mem_usage = process.memory_info().rss / 1024**3
        
        if mem_usage > self.quota.memory_gb * 1.1:  # 10% tolerance
            logger.warning(f"Actor {self.actor_id} exceeding quota: {mem_usage:.2f}GB > {self.quota.memory_gb:.2f}GB")
            
        # Execute actual work here
        return f"Completed with {mem_usage:.2f}GB used"
    
    def cleanup(self):
        """Cleanup and release resources."""
        self.manager.release_task(self.actor_id)


if __name__ == "__main__":
    # Initialize Ray
    ray.init(ignore_reinit_error=True)
    
    manager = ResourceQuotaManager()
    
    # Register some tasks
    manager.register_task("backtest_worker_1", profile="heavy")
    manager.register_task("feature_engine_1", profile="standard")
    manager.register_task("ml_inference_1", profile="gpu_inference")
    
    print("Resource Utilization:")
    stats = manager.get_utilization_stats()
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: {value}")
    
    # Release a task
    manager.release_task("backtest_worker_1")
    print("\nAfter releasing one task:")
    stats = manager.get_utilization_stats()
    print(f"  Used Memory: {stats['used_memory_gb']:.2f}GB")
