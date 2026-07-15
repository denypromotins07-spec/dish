"""
Generative Adversarial Imitation Learning (GAIL) trainer.
Learns to mimic execution patterns of top-tier market makers.
Bypasses complex manual reward shaping.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import deque


# Memory constraints
MAX_EXPERT_BUFFER = 5000
MAX_GENERATOR_BUFFER = 5000
STATE_DIM = 16
ACTION_DIM = 2


class Discriminator(nn.Module):
    """
    Discriminator that distinguishes expert from agent trajectories.
    Also serves as reward function for the agent.
    """
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )
        
    def forward(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        x = torch.cat([states, actions], dim=-1)
        return self.net(x)
    
    def get_reward(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Get reward from discriminator (higher = more expert-like)."""
        with torch.no_grad():
            prob = self.forward(states, actions)
            # Reward = -log(1 - prob) for numerical stability
            return -torch.log(1 - prob + 1e-8)


class GAILTrainer:
    """
    GAIL trainer for imitation learning.
    Uses PPO-style updates for the generator (agent).
    """
    
    def __init__(
        self,
        device: torch.device = None,
        lr_disc: float = 1e-3,
        lr_gen: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
    ):
        self.device = device or (
            torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        
        # Networks
        self.discriminator = Discriminator(STATE_DIM, ACTION_DIM).to(self.device)
        self.optimizer_d = optim.Adam(self.discriminator.parameters(), lr=lr_disc)
        
        # Buffers
        self.expert_buffer = deque(maxlen=MAX_EXPERT_BUFFER)
        self.agent_buffer = deque(maxlen=MAX_GENERATOR_BUFFER)
        
        # Stats
        self.disc_updates = 0
        self.gen_updates = 0
        
    def add_expert_transition(
        self,
        state: np.ndarray,
        action: np.ndarray,
        next_state: np.ndarray,
    ) -> None:
        """Add expert transition to buffer."""
        self.expert_buffer.append({
            'state': state,
            'action': action,
            'next_state': next_state,
        })
        
    def add_agent_transition(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        log_prob: float,
    ) -> None:
        """Add agent transition to buffer."""
        self.agent_buffer.append({
            'state': state,
            'action': action,
            'reward': reward,
            'next_state': next_state,
            'done': done,
            'log_prob': log_prob,
        })
        
    def train_discriminator(self, batch_size: int = 64, n_epochs: int = 3) -> float:
        """Train discriminator to distinguish expert from agent."""
        if len(self.expert_buffer) < batch_size // 2:
            return 0.0
            
        disc_loss_total = 0.0
        
        for _ in range(n_epochs):
            # Sample expert data
            expert_indices = np.random.choice(
                len(self.expert_buffer),
                batch_size // 2,
                replace=False,
            )
            expert_states = torch.FloatTensor(
                np.stack([self.expert_buffer[i]['state'] for i in expert_indices])
            ).to(self.device)
            expert_actions = torch.FloatTensor(
                np.stack([self.expert_buffer[i]['action'] for i in expert_indices])
            ).to(self.device)
            
            # Sample agent data
            agent_indices = np.random.choice(
                len(self.agent_buffer),
                batch_size // 2,
                replace=len(self.agent_buffer) < batch_size // 2,
            )
            agent_states = torch.FloatTensor(
                np.stack([self.agent_buffer[i]['state'] for i in agent_indices])
            ).to(self.device)
            agent_actions = torch.FloatTensor(
                np.stack([self.agent_buffer[i]['action'] for i in agent_indices])
            ).to(self.device)
            
            # Discriminator loss
            self.optimizer_d.zero_grad()
            
            prob_expert = self.discriminator(expert_states, expert_actions)
            prob_agent = self.discriminator(agent_states, agent_actions)
            
            # Binary cross entropy
            loss_expert = -torch.log(prob_expert + 1e-8).mean()
            loss_agent = -torch.log(1 - prob_agent + 1e-8).mean()
            disc_loss = loss_expert + loss_agent
            
            disc_loss.backward()
            self.optimizer_d.step()
            
            disc_loss_total += disc_loss.item()
            
        self.disc_updates += 1
        return disc_loss_total / n_epochs
    
    def compute_gail_rewards(self) -> Dict[int, float]:
        """Compute GAIL rewards for all agent transitions."""
        rewards = {}
        
        if len(self.agent_buffer) == 0:
            return rewards
            
        states = torch.FloatTensor(
            np.stack([b['state'] for b in self.agent_buffer])
        ).to(self.device)
        actions = torch.FloatTensor(
            np.stack([b['action'] for b in self.agent_buffer])
        ).to(self.device)
        
        gail_rewards = self.discriminator.get_reward(states, actions)
        
        for i, reward in enumerate(gail_rewards.cpu().numpy()):
            rewards[i] = float(reward)
            
        return rewards
    
    def get_expert_statistics(self) -> Dict[str, float]:
        """Compute statistics of expert demonstrations."""
        if len(self.expert_buffer) == 0:
            return {}
            
        actions = np.stack([b['action'] for b in self.expert_buffer])
        
        return {
            'action_mean': float(np.mean(actions)),
            'action_std': float(np.std(actions)),
            'action_min': float(np.min(actions)),
            'action_max': float(np.max(actions)),
            'buffer_size': len(self.expert_buffer),
        }


class BehavioralCloningLoss(nn.Module):
    """
    Simple behavioral cloning loss for pre-training.
    MSE between agent actions and expert actions.
    """
    
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        
    def forward(
        self,
        predicted_actions: torch.Tensor,
        expert_actions: torch.Tensor,
    ) -> torch.Tensor:
        return self.mse(predicted_actions, expert_actions)


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    trainer = GAILTrainer(device=device)
    
    # Add dummy expert data
    print("\nAdding expert demonstrations...")
    for _ in range(100):
        state = np.random.randn(STATE_DIM).astype(np.float32)
        action = np.random.randn(ACTION_DIM).astype(np.float32) * 0.5
        next_state = state + np.random.randn(STATE_DIM).astype(np.float32) * 0.1
        trainer.add_expert_transition(state, action, next_state)
        
    stats = trainer.get_expert_statistics()
    print(f"Expert statistics: {stats}")
    
    # Add dummy agent data
    print("\nAdding agent experiences...")
    for _ in range(100):
        state = np.random.randn(STATE_DIM).astype(np.float32)
        action = np.random.randn(ACTION_DIM).astype(np.float32) * 0.5
        reward = np.random.randn()
        next_state = state + np.random.randn(STATE_DIM).astype(np.float32) * 0.1
        done = np.random.rand() > 0.95
        log_prob = np.random.randn()
        trainer.add_agent_transition(state, action, reward, next_state, done, log_prob)
        
    # Train discriminator
    print("\nTraining discriminator...")
    for epoch in range(10):
        disc_loss = trainer.train_discriminator(batch_size=32, n_epochs=2)
        if epoch % 5 == 0:
            print(f"Epoch {epoch}: Discriminator Loss = {disc_loss:.4f}")
            
    # Compute GAIL rewards
    rewards = trainer.compute_gail_rewards()
    print(f"\nComputed {len(rewards)} GAIL rewards")
    print(f"Reward mean: {np.mean(list(rewards.values())):.4f}")
    print(f"Reward std: {np.std(list(rewards.values())):.4f}")
