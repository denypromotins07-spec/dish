"""
Custom PettingZoo multi-agent environment wrapping NautilusTrader.
Agents represent different sub-strategies competing/cooperating for capital allocation.
Memory-efficient design to stay within 14GB RAM constraint.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import gymnasium as gym
from gymnasium import spaces


class AgentType(Enum):
    """Types of trading agents in the system."""
    MARKET_MAKER = "market_maker"
    STAT_ARB = "stat_arb"
    TREND_FOLLOWER = "trend_follower"
    MEAN_REVERSION = "mean_reversion"
    MOMENTUM = "momentum"


@dataclass
class AgentState:
    """Internal state representation for a single agent."""
    agent_id: str
    agent_type: AgentType
    position: float = 0.0
    cash: float = 1_000_000.0
    inventory: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    trades_today: int = 0
    last_action: int = 0
    signal_strength: float = 0.0


@dataclass
class MarketState:
    """Global market state shared by all agents."""
    timestamp: int = 0
    prices: Dict[str, float] = field(default_factory=dict)
    spreads: Dict[str, float] = field(default_factory=dict)
    volumes: Dict[str, float] = field(default_factory=dict)
    volatility: Dict[str, float] = field(default_factory=dict)
    order_book_imbalance: Dict[str, float] = field(default_factory=dict)


class TradingEnv(gym.Env):
    """
    Single-agent trading environment for base functionality.
    Used as building block for multi-agent environment.
    """
    
    metadata = {"render_modes": ["human", "ansi"]}
    
    def __init__(
        self,
        n_assets: int = 10,
        initial_cash: float = 1_000_000.0,
        max_position: float = 100.0,
        transaction_cost_bps: float = 5.0,
    ):
        super().__init__()
        
        self.n_assets = n_assets
        self.initial_cash = initial_cash
        self.max_position = max_position
        self.transaction_cost_bps = transaction_cost_bps
        
        # Action space: [-1, 1] for each asset (sell to buy)
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(n_assets,),
            dtype=np.float32,
        )
        
        # Observation space: [price, spread, volume, volatility, position, inventory] per asset
        obs_dim = n_assets * 6
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32,
        )
        
        self.state: Optional[AgentState] = None
        self.market: Optional[MarketState] = None
        self.step_count = 0
        self.max_steps = 10000
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.state = AgentState(
            agent_id="trader_0",
            agent_type=AgentType.TREND_FOLLOWER,
            cash=self.initial_cash,
        )
        
        self.market = MarketState()
        self._initialize_market_state()
        
        self.step_count = 0
        
        return self._get_observation(), {}
    
    def step(self, action: np.ndarray):
        self.step_count += 1
        
        # Execute actions
        self._execute_actions(action)
        
        # Update market state
        self._update_market()
        
        # Calculate reward (PnL change)
        reward = self._calculate_reward()
        
        # Check termination
        terminated = self.step_count >= self.max_steps or self.state.cash <= 0
        
        truncated = False
        
        obs = self._get_observation()
        info = self._get_info()
        
        return obs, reward, terminated, truncated, info
    
    def _initialize_market_state(self):
        """Initialize random market state."""
        for i in range(self.n_assets):
            symbol = f"ASSET_{i}"
            self.market.prices[symbol] = 100.0 + np.random.randn() * 10
            self.market.spreads[symbol] = 0.01 + np.random.rand() * 0.01
            self.market.volumes[symbol] = 1000 + np.random.rand() * 500
            self.market.volatility[symbol] = 0.01 + np.random.rand() * 0.02
            self.market.order_book_imbalance[symbol] = np.random.randn() * 0.5
    
    def _update_market(self):
        """Simulate market movement."""
        for symbol in self.market.prices:
            # Random walk with mean reversion
            drift = -0.001 * (self.market.prices[symbol] - 100.0)
            shock = np.random.randn() * self.market.volatility[symbol]
            self.market.prices[symbol] *= (1 + drift + shock)
            
            # Update other market variables
            self.market.volumes[symbol] *= (0.95 + 0.1 * np.random.rand())
            self.market.order_book_imbalance[symbol] = np.clip(
                self.market.order_book_imbalance[symbol] * 0.9 + np.random.randn() * 0.1,
                -1, 1
            )
    
    def _execute_actions(self, action: np.ndarray):
        """Execute trading actions."""
        for i in range(self.n_assets):
            symbol = f"ASSET_{i}"
            action_val = action[i]
            
            if abs(action_val) < 0.1:
                continue
            
            # Determine trade size
            trade_size = action_val * self.max_position
            price = self.market.prices[symbol]
            
            # Calculate transaction cost
            cost = abs(trade_size) * price * self.transaction_cost_bps / 10000
            
            # Update state
            self.state.inventory += trade_size
            self.state.cash -= trade_size * price + cost
            self.state.trades_today += 1
        
        # Update unrealized PnL
        self._update_unrealized_pnl()
    
    def _update_unrealized_pnl(self):
        """Calculate unrealized PnL from current positions."""
        total_value = 0.0
        for i in range(self.n_assets):
            symbol = f"ASSET_{i}"
            # Simplified: assume equal position across assets
            position_per_asset = self.state.inventory / self.n_assets
            total_value += position_per_asset * self.market.prices[symbol]
        
        # Cost basis
        cost_basis = abs(self.state.inventory) * 100.0  # Simplified
        self.state.unrealized_pnl = total_value - cost_basis
    
    def _calculate_reward(self) -> float:
        """Calculate reward based on PnL and risk metrics."""
        # Realized PnL change (simplified)
        pnl_reward = self.state.unrealized_pnl / self.initial_cash
        
        # Penalty for high inventory
        inventory_penalty = -0.001 * (self.state.inventory / self.max_position) ** 2
        
        # Penalty for transaction costs
        cost_penalty = -0.0001 * self.state.trades_today
        
        return pnl_reward + inventory_penalty + cost_penalty
    
    def _get_observation(self) -> np.ndarray:
        """Construct observation vector."""
        obs = []
        
        for i in range(self.n_assets):
            symbol = f"ASSET_{i}"
            obs.extend([
                self.market.prices.get(symbol, 0) / 100.0,
                self.market.spreads.get(symbol, 0),
                np.log(self.market.volumes.get(symbol, 1) + 1) / 10.0,
                self.market.volatility.get(symbol, 0),
                self.market.order_book_imbalance.get(symbol, 0),
                self.state.inventory / self.max_position if self.state else 0,
            ])
        
        return np.array(obs, dtype=np.float32)
    
    def _get_info(self) -> Dict:
        """Return additional info for logging/debugging."""
        return {
            "cash": self.state.cash if self.state else 0,
            "inventory": self.state.inventory if self.state else 0,
            "unrealized_pnl": self.state.unrealized_pnl if self.state else 0,
            "trades_today": self.state.trades_today if self.state else 0,
        }


class MultiAgentTradingEnv:
    """
    Multi-agent trading environment using PettingZoo-style API.
    Supports multiple strategy types competing for centralized capital.
    """
    
    metadata = {"render_modes": ["human"]}
    
    def __init__(
        self,
        agent_configs: List[Dict],
        n_assets: int = 10,
        total_capital: float = 10_000_000.0,
        max_total_position: float = 500.0,
        transaction_cost_bps: float = 5.0,
    ):
        self.agent_configs = agent_configs
        self.n_assets = n_assets
        self.total_capital = total_capital
        self.max_total_position = max_total_position
        self.transaction_cost_bps = transaction_cost_bps
        
        # Initialize agents
        self.agents: Dict[str, AgentState] = {}
        self.agent_types: Dict[str, AgentType] = {}
        
        for config in agent_configs:
            agent_id = config["id"]
            agent_type = AgentType(config.get("type", "trend_follower"))
            
            self.agents[agent_id] = AgentState(
                agent_id=agent_id,
                agent_type=agent_type,
                cash=total_capital / len(agent_configs),
            )
            self.agent_types[agent_id] = agent_type
        
        self.possible_agents = list(self.agents.keys())
        self.market = MarketState()
        self.step_count = 0
        self.max_steps = 10000
        self.agents_done: set = set()
        
        # Capital allocation state
        self.available_capital = total_capital
        self.total_inventory = 0.0
        
    def reset(self):
        """Reset the environment."""
        self.step_count = 0
        self.agents_done = set()
        self.available_capital = self.total_capital
        self.total_inventory = 0.0
        
        # Reset agent states
        for agent_id in self.agents:
            self.agents[agent_id].cash = self.total_capital / len(self.agents)
            self.agents[agent_id].position = 0.0
            self.agents[agent_id].inventory = 0.0
            self.agents[agent_id].unrealized_pnl = 0.0
            self.agents[agent_id].realized_pnl = 0.0
            self.agents[agent_id].trades_today = 0
        
        self._initialize_market_state()
        
        return {agent_id: self._get_observation(agent_id) for agent_id in self.possible_agents}
    
    def step(self, actions: Dict[str, np.ndarray]) -> Tuple[Dict, Dict, Dict, Dict, Dict]:
        """
        Execute one step in the environment.
        
        Args:
            actions: Dict mapping agent_id to action array
            
        Returns:
            observations, rewards, terminations, truncations, infos
        """
        self.step_count += 1
        
        # Process each agent's actions
        rewards = {}
        infos = {}
        
        for agent_id, action in actions.items():
            if agent_id in self.agents_done:
                continue
            
            reward = self._execute_agent_action(agent_id, action)
            rewards[agent_id] = reward
            infos[agent_id] = self._get_agent_info(agent_id)
            
            # Check if agent is done
            if self.agents[agent_id].cash <= 0:
                self.agents_done.add(agent_id)
        
        # Update market
        self._update_market()
        
        # Update all agent observations
        observations = {
            agent_id: self._get_observation(agent_id)
            for agent_id in self.possible_agents
            if agent_id not in self.agents_done
        }
        
        # Terminations and truncations
        terminations = {
            agent_id: agent_id in self.agents_done
            for agent_id in self.possible_agents
        }
        truncations = {
            agent_id: self.step_count >= self.max_steps
            for agent_id in self.possible_agents
        }
        
        return observations, rewards, terminations, truncations, infos
    
    def _initialize_market_state(self):
        """Initialize market state."""
        for i in range(self.n_assets):
            symbol = f"ASSET_{i}"
            self.market.prices[symbol] = 100.0 + np.random.randn() * 10
            self.market.spreads[symbol] = 0.01 + np.random.rand() * 0.01
            self.market.volumes[symbol] = 1000 + np.random.rand() * 500
            self.market.volatility[symbol] = 0.01 + np.random.rand() * 0.02
            self.market.order_book_imbalance[symbol] = np.random.randn() * 0.5
    
    def _update_market(self):
        """Update market state."""
        for symbol in self.market.prices:
            drift = -0.001 * (self.market.prices[symbol] - 100.0)
            shock = np.random.randn() * self.market.volatility[symbol]
            self.market.prices[symbol] *= (1 + drift + shock)
            self.market.volumes[symbol] *= (0.95 + 0.1 * np.random.rand())
    
    def _execute_agent_action(self, agent_id: str, action: np.ndarray) -> float:
        """Execute action for a single agent and return reward."""
        agent = self.agents[agent_id]
        
        for i in range(self.n_assets):
            symbol = f"ASSET_{i}"
            action_val = action[i] if len(action) > i else 0
            
            if abs(action_val) < 0.1:
                continue
            
            trade_size = action_val * (self.max_total_position / len(self.agents))
            price = self.market.prices[symbol]
            cost = abs(trade_size) * price * self.transaction_cost_bps / 10000
            
            agent.inventory += trade_size
            agent.cash -= trade_size * price + cost
            agent.trades_today += 1
        
        # Update PnL
        agent.unrealized_pnl = agent.inventory * 0.1  # Simplified
        
        # Reward based on PnL
        reward = agent.unrealized_pnl / agent.cash
        
        return reward
    
    def _get_observation(self, agent_id: str) -> np.ndarray:
        """Get observation for a specific agent."""
        obs = []
        
        for i in range(self.n_assets):
            symbol = f"ASSET_{i}"
            obs.extend([
                self.market.prices.get(symbol, 0) / 100.0,
                self.market.spreads.get(symbol, 0),
                np.log(self.market.volumes.get(symbol, 1) + 1) / 10.0,
                self.market.volatility.get(symbol, 0),
                self.market.order_book_imbalance.get(symbol, 0),
            ])
        
        # Add agent-specific state
        agent = self.agents.get(agent_id)
        if agent:
            obs.extend([
                agent.cash / self.total_capital,
                agent.inventory / self.max_total_position,
                agent.unrealized_pnl / self.total_capital,
            ])
        
        return np.array(obs, dtype=np.float32)
    
    def _get_agent_info(self, agent_id: str) -> Dict:
        """Get info dict for an agent."""
        agent = self.agents.get(agent_id)
        if not agent:
            return {}
        
        return {
            "cash": agent.cash,
            "inventory": agent.inventory,
            "unrealized_pnl": agent.unrealized_pnl,
            "realized_pnl": agent.realized_pnl,
            "trades_today": agent.trades_today,
        }
    
    def observation_space(self, agent_id: str) -> gym.Space:
        """Get observation space for an agent."""
        obs_dim = self.n_assets * 5 + 3  # market features + agent features
        return spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
    
    def action_space(self, agent_id: str) -> gym.Space:
        """Get action space for an agent."""
        return spaces.Box(low=-1.0, high=1.0, shape=(self.n_assets,), dtype=np.float32)


if __name__ == "__main__":
    # Example usage
    agent_configs = [
        {"id": "mm_0", "type": "market_maker"},
        {"id": "arb_0", "type": "stat_arb"},
        {"id": "trend_0", "type": "trend_follower"},
    ]
    
    env = MultiAgentTradingEnv(agent_configs, n_assets=5)
    obs = env.reset()
    
    print(f"Agents: {env.possible_agents}")
    print(f"Observation shapes: {[(k, v.shape) for k, v in obs.items()]}")
    
    # Run a few steps
    for step in range(5):
        actions = {
            agent_id: np.random.randn(env.n_assets).astype(np.float32)
            for agent_id in env.possible_agents
        }
        result = env.step(actions)
        print(f"Step {step}: rewards = {result[1]}")
