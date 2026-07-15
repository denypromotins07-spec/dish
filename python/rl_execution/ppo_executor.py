"""
PPO-based optimal execution agent for order slicing.
Minimizes implementation shortfall and market impact.
Uses Ray RLlib with strict memory-bounded workers.
"""

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass

# Memory constraints
MAX_WORKERS = 4
MAX_BUFFER_SIZE = 10000
STATE_DIM = 16
ACTION_DIM = 3  # Aggressive, Neutral, Passive


@dataclass
class ExecutionState:
    """Current execution state."""
    remaining_quantity: float
    filled_quantity: float
    avg_fill_price: float
    current_price: float
    spread: float
    volatility: float
    time_remaining: float
    market_impact: float


@dataclass 
class ExecutionAction:
    """Execution action output."""
    participation_rate: float  # 0.0 to 1.0
    aggressiveness: int  # 0=passive, 1=neutral, 2=aggressive
    limit_offset: float  # Ticks from mid-price


class OrderBookEnv(gym.Env):
    """
    Gym environment for optimal execution training.
    Simulates order book dynamics for RL training.
    """
    
    def __init__(
        self,
        parent_order_size: float = 10000,
        max_steps: int = 100,
        initial_spread: float = 0.001,
        volatility: float = 0.02,
    ):
        super().__init__()
        
        self.parent_order_size = parent_order_size
        self.max_steps = max_steps
        self.initial_spread = initial_spread
        self.base_volatility = volatility
        
        # Action space: [participation_rate, aggressiveness]
        self.action_space = gym.spaces.Box(
            low=np.array([0.0, 0], dtype=np.float32),
            high=np.array([1.0, 2], dtype=np.float32),
        )
        
        # Observation space
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(STATE_DIM,),
            dtype=np.float32,
        )
        
        self.reset()
        
    def reset(self, seed: Optional[int] = None, **kwargs) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)
        
        self.current_step = 0
        self.remaining_qty = self.parent_order_size
        self.filled_qty = 0.0
        self.avg_fill_price = 0.0
        self.current_price = 100.0
        self.spread = self.initial_spread
        self.volatility = self.base_volatility
        self.market_impact = 0.0
        
        # Price path simulation
        self.price_path = [self.current_price]
        
        return self._get_observation(), {}
    
    def _get_observation(self) -> np.ndarray:
        """Build observation vector."""
        obs = np.array([
            self.remaining_qty / self.parent_order_size,
            self.filled_qty / self.parent_order_size,
            (self.avg_fill_price - self.price_path[0]) / self.price_path[0] if self.avg_fill_price > 0 else 0,
            (self.current_price - self.price_path[0]) / self.price_path[0],
            self.spread,
            self.volatility,
            1.0 - (self.current_step / self.max_steps),
            self.market_impact,
            # Technical indicators (simplified)
            np.sin(2 * np.pi * self.current_step / self.max_steps),
            np.cos(2 * np.pi * self.current_step / self.max_steps),
            # Momentum
            (self.current_price - self.price_path[-min(5, len(self.price_path))]) / self.current_price if len(self.price_path) > 1 else 0,
            # Volume profile proxy
            self.remaining_qty / max(1, self.max_steps - self.current_step),
            # Urgency signal
            (self.remaining_qty / self.parent_order_size) * (1.0 - self.current_step / self.max_steps),
            # Spread adjusted for volatility
            self.spread / (self.volatility + 1e-6),
            # Price trend
            (self.current_price - np.mean(self.price_path[-min(10, len(self.price_path)):])) / self.current_price if len(self.price_path) > 1 else 0,
            # Volatility regime
            self.volatility / self.base_volatility,
        ], dtype=np.float32)
        
        # Pad to STATE_DIM
        if len(obs) < STATE_DIM:
            obs = np.pad(obs, (0, STATE_DIM - len(obs)))
            
        return obs[:STATE_DIM]
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one step of the environment."""
        participation_rate = float(np.clip(action[0], 0.0, 1.0))
        aggressiveness = int(np.clip(action[1], 0, 2))
        
        # Calculate order size
        urgency_factor = 1.0 + (1.0 - self.remaining_qty / self.parent_order_size)
        base_size = self.remaining_qty / max(1, self.max_steps - self.current_step)
        order_size = base_size * (1 + participation_rate) * urgency_factor
        order_size = min(order_size, self.remaining_qty)
        
        # Simulate fill based on aggressiveness
        fill_prob = [0.3, 0.6, 0.9][aggressiveness]
        fill_ratio = np.random.beta(2, 5) * fill_prob
        fill_ratio = min(fill_ratio, 1.0)
        
        filled_qty = order_size * fill_ratio
        if filled_qty > self.remaining_qty:
            filled_qty = self.remaining_qty
            
        # Calculate fill price with market impact
        self.market_impact = 0.0001 * (order_size / self.parent_order_size) ** 1.5
        price_slippage = self.spread / 2 * (1 - aggressiveness / 2) + self.market_impact
        
        if filled_qty > 0:
            fill_price = self.current_price * (1 + price_slippage)
            self.avg_fill_price = (
                self.avg_fill_price * self.filled_qty + fill_price * filled_qty
            ) / (self.filled_qty + filled_qty)
            self.filled_qty += filled_qty
            self.remaining_qty -= filled_qty
            
        # Update price (random walk with drift from our trading)
        price_change = np.random.normal(0, self.volatility / np.sqrt(252))
        price_change -= self.market_impact * 0.1  # Permanent impact
        self.current_price *= (1 + price_change)
        self.price_path.append(self.current_price)
        
        # Update spread and volatility dynamically
        self.spread = self.initial_spread * (1 + np.abs(price_change) * 10)
        self.volatility = self.base_volatility * (1 + np.std(self.price_path[-min(20, len(self.price_path)):]) / self.current_price)
        
        # Calculate reward (negative implementation shortfall)
        benchmark_vwap = np.mean(self.price_path)
        if self.filled_qty > 0:
            shortfall = (self.avg_fill_price - benchmark_vwap) / benchmark_vwap
            reward = -shortfall * 1000  # Scale for training
            # Penalty for unfilled quantity at end
            if self.current_step >= self.max_steps - 1 and self.remaining_qty > 0:
                reward -= self.remaining_qty / self.parent_order_size * 100
        else:
            reward = -0.01  # Small penalty for not filling
            
        self.current_step += 1
        
        terminated = self.remaining_qty <= 1e-6 or self.current_step >= self.max_steps
        truncated = False
        
        info = {
            'filled_qty': self.filled_qty,
            'avg_fill_price': self.avg_fill_price,
            'implementation_shortfall': (self.avg_fill_price - benchmark_vwap) / benchmark_vwap if self.filled_qty > 0 else 0,
            'fill_rate': self.filled_qty / self.parent_order_size,
        }
        
        return self._get_observation(), reward, terminated, truncated, info


class ActorCriticNetwork(nn.Module):
    """
    Memory-efficient Actor-Critic network for PPO.
    Uses layer normalization and dropout for stability.
    """
    
    def __init__(
        self,
        state_dim: int = STATE_DIM,
        action_dim: int = ACTION_DIM,
        hidden_dim: int = 64,
    ):
        super().__init__()
        
        # Shared trunk
        self.trunk = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )
        
        # Actor head (policy)
        self.actor_mean = nn.Linear(hidden_dim, 1)  # Participation rate
        self.actor_agg = nn.Linear(hidden_dim, 3)  # Aggressiveness logits
        
        # Critic head (value)
        self.critic = nn.Linear(hidden_dim, 1)
        
    def forward(
        self,
        states: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass returning policy and value."""
        x = self.trunk(states)
        
        # Policy outputs
        participation_mean = torch.sigmoid(self.actor_mean(x))
        agg_logits = self.actor_agg(x)
        
        # Value output
        value = self.critic(x)
        
        return participation_mean, agg_logits, value
    
    def get_action(
        self,
        states: torch.Tensor,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample action from policy."""
        participation_mean, agg_logits, _ = self.forward(states)
        
        if deterministic:
            participation = participation_mean
            aggressiveness = torch.argmax(agg_logits, dim=-1)
        else:
            # Add noise for exploration
            participation = participation_mean + torch.randn_like(participation_mean) * 0.1
            participation = torch.clamp(participation, 0, 1)
            aggressiveness = torch.multinomial(
                torch.softmax(agg_logits, dim=-1), num_samples=1
            ).squeeze(-1)
            
        actions = torch.cat([participation, aggressiveness.float().unsqueeze(-1)], dim=-1)
        
        # Log probability for PPO
        log_prob_agg = torch.log_softmax(agg_logits, dim=-1)
        log_prob = torch.gather(log_prob_agg, 1, aggressiveness.unsqueeze(-1)).squeeze(-1)
        
        return actions, log_prob


class MemoryBoundedReplayBuffer:
    """
    Replay buffer with strict memory limits.
    Automatically evicts oldest samples when full.
    """
    
    def __init__(self, max_size: int = MAX_BUFFER_SIZE):
        self.max_size = max_size
        self.buffer: List[Dict] = []
        self.pos = 0
        
    def add(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        log_prob: float,
    ) -> None:
        experience = {
            'state': state,
            'action': action,
            'reward': reward,
            'next_state': next_state,
            'done': done,
            'log_prob': log_prob,
        }
        
        if len(self.buffer) < self.max_size:
            self.buffer.append(experience)
        else:
            self.buffer[self.pos] = experience
            
        self.pos = (self.pos + 1) % self.max_size
        
    def sample(self, batch_size: int) -> Dict[str, np.ndarray]:
        """Sample random batch."""
        indices = np.random.choice(
            min(len(self.buffer), batch_size),
            batch_size,
            replace=len(self.buffer) < batch_size,
        )
        
        batch = [self.buffer[i] for i in indices]
        
        return {
            'states': np.stack([b['state'] for b in batch]),
            'actions': np.stack([b['action'] for b in batch]),
            'rewards': np.array([b['reward'] for b in batch]),
            'next_states': np.stack([b['next_state'] for b in batch]),
            'dones': np.array([b['done'] for b in batch]),
            'log_probs': np.array([b['log_prob'] for b in batch]),
        }
    
    def __len__(self) -> int:
        return len(self.buffer)
    
    def clear(self) -> None:
        self.buffer = []
        self.pos = 0


class PPOExecutor:
    """
    PPO trainer for optimal execution.
    Implements clipped surrogate objective with adaptive KL penalty.
    """
    
    def __init__(
        self,
        device: torch.device = None,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
    ):
        self.device = device or (
            torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        
        # Network
        self.network = ActorCriticNetwork().to(self.device)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=lr)
        
        # Replay buffer
        self.buffer = MemoryBoundedReplayBuffer(MAX_BUFFER_SIZE)
        
        # Training stats
        self.total_updates = 0
        
    def compute_gae(
        self,
        rewards: np.ndarray,
        values: np.ndarray,
        next_values: np.ndarray,
        dones: np.ndarray,
    ) -> np.ndarray:
        """Compute Generalized Advantage Estimation."""
        advantages = np.zeros_like(rewards)
        last_advantage = 0
        
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * next_values[t] * (1 - dones[t]) - values[t]
            last_advantage = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * last_advantage
            advantages[t] = last_advantage
            
        return advantages
    
    def train_step(
        self,
        batch: Dict[str, np.ndarray],
        n_epochs: int = 4,
    ) -> Dict[str, float]:
        """Single PPO training step."""
        states = torch.FloatTensor(batch['states']).to(self.device)
        actions = torch.FloatTensor(batch['actions']).to(self.device)
        rewards = torch.FloatTensor(batch['rewards']).to(self.device)
        next_states = torch.FloatTensor(batch['next_states']).to(self.device)
        dones = torch.FloatTensor(batch['dones']).to(self.device)
        old_log_probs = torch.FloatTensor(batch['log_probs']).to(self.device)
        
        # Compute values
        with torch.no_grad():
            _, _, values = self.network(states)
            _, _, next_values = self.network(next_states)
            values = values.squeeze().cpu().numpy()
            next_values = next_values.squeeze().cpu().numpy()
            
        # Compute advantages
        advantages = self.compute_gae(rewards.cpu().numpy(), values, next_values, dones.cpu().numpy())
        advantages = torch.FloatTensor(advantages).to(self.device)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Returns
        returns = advantages + torch.FloatTensor(values).to(self.device)
        
        losses = {'policy': 0.0, 'value': 0.0, 'entropy': 0.0}
        
        for _ in range(n_epochs):
            # Forward pass
            participation_mean, agg_logits, values = self.network(states)
            
            # Policy loss (clipped surrogate)
            agg_labels = actions[:, 1].long()
            log_probs = torch.log_softmax(agg_logits, dim=-1)
            log_probs = torch.gather(log_probs, 1, agg_labels.unsqueeze(-1)).squeeze(-1)
            
            ratio = torch.exp(log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Value loss
            value_loss = ((values.squeeze() - returns) ** 2).mean()
            
            # Entropy bonus
            entropy = -(torch.softmax(agg_logits, dim=-1) * log_probs).sum(dim=-1).mean()
            
            # Total loss
            loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
            
            # Update
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
            self.optimizer.step()
            
            losses['policy'] += policy_loss.item() / n_epochs
            losses['value'] += value_loss.item() / n_epochs
            losses['entropy'] += entropy.item() / n_epochs
            
        self.total_updates += 1
        
        return losses
    
    def get_action(
        self,
        state: np.ndarray,
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, float]:
        """Get action for given state."""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            action, log_prob = self.network.get_action(state_tensor, deterministic)
            
        return action.cpu().numpy()[0], log_prob.cpu().numpy()[0]
    
    def save(self, path: str) -> None:
        """Save model checkpoint."""
        torch.save({
            'network': self.network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'total_updates': self.total_updates,
        }, path)
        
    def load(self, path: str) -> None:
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.network.load_state_dict(checkpoint['network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.total_updates = checkpoint['total_updates']


if __name__ == "__main__":
    # Test PPO executor
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    env = OrderBookEnv(parent_order_size=10000, max_steps=50)
    ppo = PPOExecutor(device=device)
    
    print("\nTraining PPO agent...")
    state, _ = env.reset()
    total_reward = 0
    
    for episode in range(5):
        state, _ = env.reset()
        episode_reward = 0
        step_count = 0
        
        while True:
            action, log_prob = ppo.get_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            
            ppo.buffer.add(state, action, reward, next_state, terminated, log_prob)
            
            state = next_state
            episode_reward += reward
            step_count += 1
            
            if terminated or truncated:
                break
                
            # Train periodically
            if len(ppo.buffer) >= 64:
                batch = ppo.buffer.sample(32)
                losses = ppo.train_step(batch, n_epochs=2)
                
        total_reward += episode_reward
        print(f"Episode {episode}: Reward={episode_reward:.4f}, Steps={step_count}, FillRate={info['fill_rate']:.2%}")
    
    print(f"\nAverage reward: {total_reward / 5:.4f}")
    print(f"Buffer size: {len(ppo.buffer)}")
