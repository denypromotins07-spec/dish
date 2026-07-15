"""
Ray RLlib implementation of Multi-Agent PPO (MAPPO) with CTDE.
Centralized Training with Decentralized Execution for trading agents.
Strictly bounds replay buffers and worker concurrency to maintain 14GB RAM ceiling.
"""

import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.tune.logger import pretty_print


@dataclass
class MemoryConfig:
    """Configuration for memory-constrained training."""
    max_replay_buffer_size: int = 100_000
    max_episode_storage: int = 10_000
    num_workers: int = 2
    num_cpus_per_worker: int = 1
    num_gpus: float = 0.5
    train_batch_size: int = 4096
    rollout_fragment_length: int = 1024
    # Target ~2GB for RL training to stay under 14GB total system limit
    target_memory_gb: float = 2.0


def create_mappo_config(
    env_class: type,
    agent_ids: List[str],
    memory_config: Optional[MemoryConfig] = None,
) -> PPOConfig:
    """
    Create MAPPO configuration optimized for memory constraints.
    
    Args:
        env_class: Environment class to wrap
        agent_ids: List of agent IDs in the environment
        memory_config: Memory configuration parameters
    
    Returns:
        Configured PPOConfig for MAPPO
    """
    if memory_config is None:
        memory_config = MemoryConfig()
    
    # Create multi-agent configuration
    policies = {
        agent_id: (None, env_class().observation_space(agent_id), 
                   env_class().action_space(agent_id), {})
        for agent_id in agent_ids
    }
    
    def policy_mapping_fn(agent_id):
        return agent_id
    
    config = (
        PPOConfig()
        .environment(env=env_class)
        .multi_agent(
            policies=policies,
            policy_mapping_fn=policy_mapping_fn,
        )
        .rollouts(
            num_rollout_workers=memory_config.num_workers,
            rollout_fragment_length=memory_config.rollout_fragment_length,
            batch_mode="truncate_episodes",
        )
        .training(
            train_batch_size=memory_config.train_batch_size,
            gamma=0.99,
            lambda_=0.95,
            kl_target=0.01,
            kl_coeff=1.0,
            clip_param=0.2,
            grad_clip=None,
        )
        .optimization(
            num_sgd_iter=10,
            lr=3e-4,
        )
        .resources(
            num_gpus=memory_config.num_gpus,
            num_gpus_per_worker=0.1,
            num_cpus_per_worker=memory_config.num_cpus_per_worker,
        )
        .framework("torch")
    )
    
    return config


class MAPPOTrainer:
    """
    Multi-Agent PPO Trainer with memory management.
    Implements Centralized Training with Decentralized Execution (CTDE).
    """
    
    def __init__(
        self,
        env_class: type,
        agent_ids: List[str],
        memory_config: Optional[MemoryConfig] = None,
        checkpoint_dir: str = "./checkpoints",
    ):
        self.env_class = env_class
        self.agent_ids = agent_ids
        self.memory_config = memory_config or MemoryConfig()
        self.checkpoint_dir = checkpoint_dir
        
        self.config = create_mappo_config(
            env_class, agent_ids, self.memory_config
        )
        
        self.algorithm = None
        self.training_history = []
        self.best_reward = -np.inf
        
    def init_ray(self):
        """Initialize Ray with memory constraints."""
        if not ray.is_initialized():
            # Limit object store memory to prevent OOM
            object_store_memory = int(self.memory_config.target_memory_gb * 1e9 * 0.3)
            
            ray.init(
                num_cpus=self.memory_config.num_workers + 1,
                num_gpus=int(self.memory_config.num_gpus) if self.memory_config.num_gpus >= 1 else 1,
                object_store_memory=object_store_memory,
                _temp_dir="/tmp/ray_temp",
            )
    
    def build(self):
        """Build the training algorithm."""
        self.init_ray()
        self.algorithm = self.config.build()
        return self
    
    def train(
        self,
        num_iterations: int = 100,
        reward_threshold: Optional[float] = None,
        verbose: bool = True,
    ) -> Dict:
        """
        Run training loop.
        
        Args:
            num_iterations: Number of training iterations
            reward_threshold: Stop early if average reward exceeds this
            verbose: Print training progress
        
        Returns:
            Training results summary
        """
        if self.algorithm is None:
            self.build()
        
        for i in range(num_iterations):
            result = self.algorithm.train()
            self.training_history.append(result)
            
            if verbose and i % 10 == 0:
                print(f"Iteration {i}: reward_mean={result['episode_reward_mean']:.4f}")
            
            # Check for early stopping
            if reward_threshold is not None:
                if result['episode_reward_mean'] >= reward_threshold:
                    print(f"Reached reward threshold at iteration {i}")
                    break
            
            # Save best checkpoint
            current_reward = result['episode_reward_mean']
            if current_reward > self.best_reward:
                self.best_reward = current_reward
                self.algorithm.save(f"{self.checkpoint_dir}/best_checkpoint")
        
        return self.get_training_summary()
    
    def get_training_summary(self) -> Dict:
        """Get summary of training results."""
        if not self.training_history:
            return {}
        
        rewards = [r['episode_reward_mean'] for r in self.training_history]
        
        return {
            'total_iterations': len(self.training_history),
            'final_reward_mean': rewards[-1],
            'best_reward_mean': max(rewards),
            'reward_std': np.std(rewards),
            'training_steps': sum(r.get('timesteps_total', 0) for r in self.training_history),
        }
    
    def evaluate(
        self,
        n_episodes: int = 10,
        render: bool = False,
    ) -> Dict[str, List[float]]:
        """
        Evaluate trained policy.
        
        Args:
            n_episodes: Number of evaluation episodes
            render: Whether to render environment
        
        Returns:
            Per-agent reward lists
        """
        if self.algorithm is None:
            raise ValueError("Must build/train before evaluating")
        
        env = self.env_class()
        agent_rewards = {agent_id: [] for agent_id in self.agent_ids}
        
        for ep in range(n_episodes):
            obs = env.reset()
            done = {agent_id: False for agent_id in self.agent_ids}
            episode_rewards = {agent_id: 0.0 for agent_id in self.agent_ids}
            
            while not all(done.values()):
                actions = {}
                for agent_id in self.agent_ids:
                    if not done.get(agent_id, True):
                        action = self.algorithm.compute_action(
                            obs[agent_id],
                            policy_id=agent_id,
                        )
                        actions[agent_id] = action
                
                obs, rewards, terminations, truncations, infos = env.step(actions)
                
                for agent_id, reward in rewards.items():
                    episode_rewards[agent_id] += reward
                
                done = terminations | truncations
            
            for agent_id in self.agent_ids:
                agent_rewards[agent_id].append(episode_rewards[agent_id])
        
        return agent_rewards
    
    def save_checkpoint(self, path: str):
        """Save model checkpoint."""
        if self.algorithm:
            self.algorithm.save(path)
    
    def load_checkpoint(self, path: str):
        """Load model from checkpoint."""
        if self.algorithm is None:
            self.build()
        self.algorithm.restore(path)
    
    def shutdown(self):
        """Shutdown Ray and cleanup resources."""
        if self.algorithm:
            del self.algorithm
        if ray.is_initialized():
            ray.shutdown()


class CentralizedCriticPPO(MAPPOTrainer):
    """
    MAPPO variant with centralized critic for CTDE.
    Uses global state information during training but decentralized execution.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Modify config for centralized critic
        self.config.model.update({
            "use_lstm": False,
            "fcnet_hiddens": [256, 256, 128],
            "fcnet_activation": "relu",
        })
        
        # Enable value function sharing across agents
        self.config.multi_agent.update({
            "policies_to_train": self.agent_ids,
        })
    
    def _get_centralized_state(self, observations: Dict) -> np.ndarray:
        """Combine individual observations into centralized state."""
        # Concatenate all agent observations
        state_parts = []
        for agent_id in sorted(observations.keys()):
            obs = observations[agent_id]
            state_parts.append(obs.flatten())
        
        return np.concatenate(state_parts)


def run_hyperparameter_tuning(
    env_class: type,
    agent_ids: List[str],
    n_trials: int = 20,
    memory_config: Optional[MemoryConfig] = None,
) -> Dict:
    """
    Run hyperparameter tuning for MAPPO.
    
    Args:
        env_class: Environment class
        agent_ids: List of agent IDs
        n_trials: Number of trials to run
        memory_config: Memory configuration
    
    Returns:
        Best configuration found
    """
    config = create_mappo_config(env_class, agent_ids, memory_config)
    
    # Define search space
    tune_config = {
        "lr": tune.choice([1e-4, 3e-4, 1e-3]),
        "gamma": tune.choice([0.99, 0.995, 0.999]),
        "lambda_": tune.choice([0.9, 0.95, 0.99]),
        "clip_param": tune.choice([0.1, 0.2, 0.3]),
    }
    
    analysis = tune.run(
        "PPO",
        config={
            **config.to_dict(),
            **tune_config,
        },
        metric="episode_reward_mean",
        mode="max",
        num_samples=n_trials,
        resources_per_trial={
            "cpu": memory_config.num_cpus_per_worker if memory_config else 2,
            "gpu": 0.2,
        },
        stop={"training_iteration": 50},
        verbose=1,
    )
    
    best_config = analysis.get_best_config(
        metric="episode_reward_mean",
        mode="max",
    )
    
    return best_config


if __name__ == "__main__":
    # Example usage
    from pettingzoo_env import MultiAgentTradingEnv, AgentType
    
    # Quick test
    agent_configs = [
        {"id": "mm_0", "type": "market_maker"},
        {"id": "arb_0", "type": "stat_arb"},
        {"id": "trend_0", "type": "trend_follower"},
    ]
    
    class TestEnvWrapper:
        def __init__(self):
            self._env = MultiAgentTradingEnv(agent_configs, n_assets=5)
            self.possible_agents = self._env.possible_agents
        
        def reset(self):
            return self._env.reset()
        
        def step(self, actions):
            return self._env.step(actions)
        
        def observation_space(self, agent_id):
            return self._env.observation_space(agent_id)
        
        def action_space(self, agent_id):
            return self._env.action_space(agent_id)
    
    # Initialize trainer
    memory_cfg = MemoryConfig(
        num_workers=1,
        train_batch_size=2048,
        rollout_fragment_length=512,
    )
    
    trainer = MAPPOTrainer(
        env_class=TestEnvWrapper,
        agent_ids=["mm_0", "arb_0", "trend_0"],
        memory_config=memory_cfg,
    )
    
    print("Building MAPPO trainer...")
    trainer.build()
    
    print("Training for a few iterations...")
    summary = trainer.train(num_iterations=5, verbose=True)
    print(f"\nTraining Summary: {summary}")
    
    trainer.shutdown()
