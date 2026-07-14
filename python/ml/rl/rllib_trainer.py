"""
Ray RLlib configuration for PPO and SAC training with strict memory bounds.
Implements worker limits, replay buffer constraints, and RAM-aware scheduling.
Designed to maintain the 14GB system RAM ceiling.
"""

import os
import logging
from typing import Any, Dict, Optional
import numpy as np

try:
    import ray
    from ray import tune
    from ray.rllib.algorithms.ppo import PPOConfig
    from ray.rllib.algorithms.sac import SACConfig
    from ray.rllib.utils.exploration import GaussianNoise
except ImportError:
    raise ImportError("ray[rllib] required. Install with: pip install 'ray[rllib]'")

logger = logging.getLogger(__name__)


class MemoryBoundedRLTrainer:
    """
    Ray RLlib trainer with strict memory management.
    Bounds worker count, replay buffer size, and batch sizes to stay under 14GB RAM.
    """
    
    def __init__(
        self,
        algorithm: str = "PPO",  # PPO or SAC
        max_system_ram_gb: float = 14.0,
        trading_engine_ram_gb: float = 8.0,  # Reserve for live trading engine
        max_ray_cluster_ram_gb: float = 2.0,  # Max RAM for Ray cluster
        num_gpus: int = 0,  # GPUs for training (0 for CPU-only)
        enable_gpu_training: bool = False,
    ):
        self.algorithm = algorithm.upper()
        self.max_system_ram_gb = max_system_ram_gb
        self.trading_engine_ram_gb = trading_engine_ram_gb
        self.max_ray_cluster_ram_gb = max_ray_cluster_ram_gb
        self.num_gpus = num_gpus if enable_gpu_training else 0
        
        # Calculate available resources
        self._calculate_resource_limits()
        
        # Ray cluster state
        self.ray_initialized = False
        self.current_config: Optional[Dict] = None
    
    def _calculate_resource_limits(self):
        """Calculate resource limits based on system constraints."""
        import psutil
        
        total_ram_gb = psutil.virtual_memory().total / (1024**3)
        available_ram_gb = psutil.virtual_memory().available / (1024**3)
        
        logger.info(f"Total System RAM: {total_ram_gb:.2f}GB")
        logger.info(f"Available RAM: {available_ram_gb:.2f}GB")
        logger.info(f"Reserved for Trading Engine: {self.trading_engine_ram_gb}GB")
        
        # Adjust Ray cluster limit if system has less RAM than expected
        if total_ram_gb < self.max_system_ram_gb:
            self.max_ray_cluster_ram_gb = min(
                self.max_ray_cluster_ram_gb,
                (total_ram_gb - self.trading_engine_ram_gb) * 0.5
            )
            logger.warning(
                f"Reduced Ray cluster RAM limit to {self.max_ray_cluster_ram_gb:.2f}GB "
                f"(system has only {total_ram_gb:.2f}GB total)"
            )
        
        # Calculate worker limits
        # Each worker needs ~200-500MB depending on environment complexity
        ram_per_worker_gb = 0.3
        self.max_workers = max(
            1,
            int(self.max_ray_cluster_ram_gb / ram_per_worker_gb)
        )
        
        # Cap workers to avoid excessive context switching
        self.max_workers = min(self.max_workers, 4)
        
        logger.info(f"Max Ray Workers: {self.max_workers}")
        
        # Replay buffer limits (for SAC)
        # Target: keep replay buffer under 500MB
        self.max_replay_buffer_size = int(500e6 / 32)  # ~500MB / 32 bytes per transition
        self.replay_buffer_batch_size = 256
    
    def initialize_ray(self):
        """Initialize Ray cluster with memory limits."""
        if self.ray_initialized:
            logger.info("Ray already initialized")
            return
        
        # Check if Ray is already running
        if ray.is_initialized():
            logger.info("Ray already running externally")
            self.ray_initialized = True
            return
        
        # Configure Ray with memory limits
        ray_init_kwargs = {
            "num_cpus": min(4, os.cpu_count() or 4),  # Limit CPU usage
            "num_gpus": self.num_gpus,
            "_memory": int(self.max_ray_cluster_ram_gb * 1024**3),  # Object store memory
            "object_store_memory": int(self.max_ray_cluster_ram_gb * 1024**3 * 0.6),
            "log_to_driver": True,
            "logging_level": logging.INFO,
        }
        
        logger.info(f"Initializing Ray with kwargs: {ray_init_kwargs}")
        
        try:
            ray.init(**ray_init_kwargs)
            self.ray_initialized = True
            logger.info("Ray cluster initialized successfully")
        except Exception as e:
            logger.warning(f"Ray initialization failed: {e}. Trying with minimal config.")
            
            # Fallback with minimal config
            ray.init(
                num_cpus=2,
                num_gpus=0,
                log_to_driver=True,
            )
            self.ray_initialized = True
    
    def shutdown_ray(self):
        """Shutdown Ray cluster and free resources."""
        if self.ray_initialized and ray.is_initialized():
            ray.shutdown()
            self.ray_initialized = False
            logger.info("Ray cluster shut down")
    
    def get_ppo_config(
        self,
        env_class: Any,
        horizon: int = 1000,
        clip_param: float = 0.2,
        lr: float = 3e-4,
        train_batch_size: int = 2048,
        sgd_minibatch_size: int = 128,
        num_sgd_iter: int = 10,
    ) -> PPOConfig:
        """
        Get PPO configuration with memory-efficient settings.
        """
        self.initialize_ray()
        
        config = (
            PPOConfig()
            .environment(env_class)
            .rollouts(
                rollout_fragment_length=horizon // 4,
                batch_mode="truncate_episodes",
            )
            .training(
                clip_param=clip_param,
                lr=lr,
                train_batch_size=train_batch_size,
                sgd_minibatch_size=sgd_minibatch_size,
                num_sgd_iter=num_sgd_iter,
                vf_loss_coeff=0.5,
                entropy_coeff=0.01,
                grad_clip=None,  # No gradient clipping for speed
                use_kl_loss=False,  # Disable KL loss for memory efficiency
            )
            .framework("torch")
            .resources(
                num_gpus_per_learner=self.num_gpus,
                num_gpus_per_worker=0,  # Workers on CPU
                num_cpus_per_learner=1,
                num_cpus_per_worker=1,
            )
            .workers(
                num_rollout_workers=min(2, self.max_workers - 1),  # Leave room for learner
                rollout_fragment_length=horizon // 4,
                batch_mode="truncate_episodes",
            )
            .evaluation(
                evaluation_interval=0,  # Disable evaluation for memory savings
            )
        )
        
        # Store config
        self.current_config = {"algorithm": "PPO", "config": config}
        
        logger.info("PPO Config created with memory-efficient settings")
        
        return config
    
    def get_sac_config(
        self,
        env_class: Any,
        horizon: int = 1000,
        target_entropy: Optional[float] = None,
        lr: float = 3e-4,
        train_batch_size: int = 256,
    ) -> SACConfig:
        """
        Get SAC configuration with bounded replay buffer.
        """
        self.initialize_ray()
        
        config = (
            SACConfig()
            .environment(env_class)
            .rollouts(
                rollout_fragment_length=horizon // 4,
                batch_mode="truncate_episodes",
            )
            .training(
                learning_rate=lr,
                train_batch_size=train_batch_size,
                target_entropy=target_entropy,
                tau=0.005,
                prioritized_replay=True,
            )
            .replay_buffer(
                capacity=self.max_replay_buffer_size,
                learning_starts=1000,  # Start learning after 1000 steps
            )
            .framework("torch")
            .resources(
                num_gpus_per_learner=self.num_gpus,
                num_gpus_per_worker=0,
                num_cpus_per_learner=1,
                num_cpus_per_worker=1,
            )
            .workers(
                num_rollout_workers=min(2, self.max_workers - 1),
                rollout_fragment_length=horizon // 4,
            )
            .evaluation(
                evaluation_interval=0,
            )
        )
        
        self.current_config = {"algorithm": "SAC", "config": config}
        
        logger.info("SAC Config created with bounded replay buffer")
        
        return config
    
    def tune_hyperparameters(
        self,
        config: Any,
        metric: str = "episode_reward_mean",
        mode: str = "max",
        num_samples: int = 10,
        max_training_iters: int = 100,
    ) -> Dict[str, Any]:
        """
        Run Hyperopt-style tuning with memory constraints.
        """
        if not self.ray_initialized:
            self.initialize_ray()
        
        # Define search space (conservative to save memory)
        if self.algorithm == "PPO":
            search_space = {
                "lr": tune.loguniform(1e-4, 1e-3),
                "clip_param": tune.choice([0.1, 0.2, 0.3]),
                "train_batch_size": tune.choice([1024, 2048, 4096]),
            }
        else:  # SAC
            search_space = {
                "learning_rate": tune.loguniform(1e-4, 1e-3),
                "tau": tune.choice([0.005, 0.01]),
                "batch_size": tune.choice([128, 256, 512]),
            }
        
        # Run tuning
        tuner = tune.Tuner(
            self.algorithm,
            param_space={**config.to_dict(), **search_space},
            tune_config=tune.TuneConfig(
                metric=metric,
                mode=mode,
                num_samples=num_samples,
                scheduler=ray.tune.schedulers.ASHAScheduler(
                    max_t=max_training_iters,
                    grace_period=10,
                    reduction_factor=2,
                ),
            ),
            run_config=ray.train.RunConfig(
                storage_path="/tmp/ray_results",
                name=f"{self.algorithm}_tuning",
            ),
        )
        
        results = tuner.fit()
        
        # Get best result
        best_result = results.get_best_result(metric=metric, mode=mode)
        
        logger.info(f"Best hyperparameters: {best_result.config}")
        
        return best_result.config
    
    def get_memory_status(self) -> Dict[str, Any]:
        """Get current memory status of Ray cluster."""
        import psutil
        
        status = {
            "ray_initialized": self.ray_initialized,
            "system_ram_total_gb": psutil.virtual_memory().total / (1024**3),
            "system_ram_available_gb": psutil.virtual_memory().available / (1024**3),
            "system_ram_used_gb": psutil.virtual_memory().used / (1024**3),
            "max_ray_cluster_gb": self.max_ray_cluster_ram_gb,
            "max_workers": self.max_workers,
            "max_replay_buffer_size": self.max_replay_buffer_size if self.algorithm == "SAC" else None,
        }
        
        if self.ray_initialized and ray.is_initialized():
            try:
                ray_state = ray.state.state()
                status["ray_nodes"] = len(ray_state.nodes())
            except Exception:
                status["ray_nodes"] = "unknown"
        
        return status


def create_trainer_with_memory_bounds(
    algorithm: str = "PPO",
    env_class: Any = None,
    max_training_steps: int = 100000,
) -> Dict[str, Any]:
    """
    Factory function to create a memory-bounded RL trainer.
    """
    trainer = MemoryBoundedRLTrainer(
        algorithm=algorithm,
        max_system_ram_gb=14.0,
        trading_engine_ram_gb=8.0,
        max_ray_cluster_ram_gb=2.0,
    )
    
    if algorithm.upper() == "PPO":
        config = trainer.get_ppo_config(env_class)
    elif algorithm.upper() == "SAC":
        config = trainer.get_sac_config(env_class)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")
    
    return {
        "trainer": trainer,
        "config": config,
        "memory_status": trainer.get_memory_status(),
    }


def main():
    """Example usage."""
    import psutil
    
    print("=" * 60)
    print("Memory-Bounded RLlib Trainer Demo")
    print("=" * 60)
    
    print(f"\nInitial System RAM: {psutil.virtual_memory().available / (1024**3):.2f}GB")
    
    # Create trainer for PPO
    print("\n--- Creating PPO Trainer ---")
    ppo_result = create_trainer_with_memory_bounds(
        algorithm="PPO",
        max_training_steps=50000,
    )
    
    print(f"PPO Memory Status: {ppo_result['memory_status']}")
    
    # Create trainer for SAC
    print("\n--- Creating SAC Trainer ---")
    sac_result = create_trainer_with_memory_bounds(
        algorithm="SAC",
        max_training_steps=50000,
    )
    
    print(f"SAC Memory Status: {sac_result['memory_status']}")
    
    # Cleanup
    print("\n--- Shutting down Ray ---")
    ppo_result['trainer'].shutdown_ray()
    
    print(f"\nFinal System RAM: {psutil.virtual_memory().available / (1024**3):.2f}GB")


if __name__ == "__main__":
    main()
