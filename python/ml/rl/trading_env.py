"""
Custom Gymnasium environment for NautilusTrader integration.
Maps order book state, features, and portfolio to trading actions.
Implements realistic transaction costs, slippage, and latency modeling.
"""

import os
import logging
from typing import Any, Dict, Optional, Tuple, List
import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    raise ImportError("gymnasium required. Install with: pip install gymnasium")

logger = logging.getLogger(__name__)


class TradingActionSpace:
    """Defines the action space for trading."""
    
    HOLD = 0
    BUY_MARKET = 1
    SELL_MARKET = 2
    BUY_LIMIT = 3
    SELL_LIMIT = 4
    CANCEL_ORDERS = 5
    
    # Continuous action components (for position sizing)
    # Action format: [discrete_action, position_size_pct, limit_price_offset]


class TradingEnv(gym.Env):
    """
    Custom Gymnasium environment for crypto trading.
    
    State Space:
    - Order book data (bid/ask levels, spreads)
    - Technical indicators (RSI, MACD, etc.)
    - Portfolio state (position, PnL, cash)
    - Market regime indicators
    
    Action Space:
    - Discrete: Hold, Buy/Sell Market, Buy/Sell Limit, Cancel
    - Continuous: Position size (0-100%), Limit price offset
    
    Reward:
    - Realized PnL
    - Risk-adjusted returns (Sharpe/Sortino)
    - Penalty for drawdown, excessive trading
    """
    
    metadata = {"render_modes": ["human", "ansi"]}
    
    def __init__(
        self,
        initial_cash: float = 100000.0,
        max_position_pct: float = 0.95,
        transaction_cost_bps: float = 10.0,  # Basis points
        slippage_model: str = "linear",  # linear, quadratic, fixed
        slippage_factor: float = 0.0001,
        commission_rate: float = 0.0004,  # 0.04%
        max_steps: int = 10000,
        reward_shaping: bool = True,
        include_order_book: bool = True,
        order_book_levels: int = 10,
        n_features: int = 50,
    ):
        super().__init__()
        
        self.initial_cash = initial_cash
        self.max_position_pct = max_position_pct
        self.transaction_cost_bps = transaction_cost_bps
        self.slippage_model = slippage_model
        self.slippage_factor = slippage_factor
        self.commission_rate = commission_rate
        self.max_steps = max_steps
        self.reward_shaping = reward_shaping
        
        # Order book configuration
        self.include_order_book = include_order_book
        self.order_book_levels = order_book_levels
        self.n_features = n_features
        
        # Calculate observation space dimension
        obs_dim = n_features  # Technical features
        
        if include_order_book:
            # Bid/ask prices and sizes for each level
            obs_dim += order_book_levels * 4  # bid_price, bid_size, ask_price, ask_size
        
        # Portfolio state
        obs_dim += 5  # position, entry_price, unrealized_pnl, realized_pnl, cash_ratio
        
        # Action space
        # Discrete: 6 actions (HOLD, BUY_MKT, SELL_MKT, BUY_LMT, SELL_LMT, CANCEL)
        # Continuous: position_size (0-1), limit_price_offset (-0.01 to 0.01)
        self.action_space = spaces.Tuple((
            spaces.Discrete(6),  # Discrete action
            spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),  # Position size
            spaces.Box(low=-0.01, high=0.01, shape=(1,), dtype=np.float32),  # Limit offset
        ))
        
        # Observation space
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32,
        )
        
        # State variables
        self.current_step = 0
        self.cash = initial_cash
        self.position = 0.0  # Positive = long, negative = short
        self.entry_price = 0.0
        self.realized_pnl = 0.0
        self.unrealized_pnl = 0.0
        
        # Data buffers
        self.order_book_data: Optional[np.ndarray] = None
        self.feature_data: Optional[np.ndarray] = None
        self.price_series: Optional[np.ndarray] = None
        
        # Trade history
        self.trade_history: List[Dict] = []
        self.episode_rewards: List[float] = []
        
        # Latency simulation (microseconds)
        self.latency_mean_us = 100  # 100 microseconds average
        self.latency_std_us = 50
    
    def _calculate_slippage(self, quantity: float, side: str) -> float:
        """Calculate slippage based on order size and side."""
        if self.order_book_data is None:
            return 0.0
        
        mid_price = (self.order_book_data[0] + self.order_book_data[2]) / 2
        
        if self.slippage_model == "linear":
            slippage = self.slippage_factor * abs(quantity) * mid_price
        elif self.slippage_model == "quadratic":
            slippage = self.slippage_factor * (quantity ** 2) * mid_price
        else:  # fixed
            slippage = self.slippage_factor * mid_price
        
        return slippage
    
    def _execute_trade(
        self,
        action: int,
        position_size: float,
        limit_offset: float,
        current_price: float,
    ) -> float:
        """Execute a trade and return the cost/slippage."""
        if action == TradingActionSpace.HOLD or action == TradingActionSpace.CANCEL_ORDERS:
            return 0.0
        
        # Determine direction
        is_buy = action in [TradingActionSpace.BUY_MARKET, TradingActionSpace.BUY_LIMIT]
        side = "buy" if is_buy else "sell"
        
        # Calculate target quantity
        max_quantity = (self.cash * self.max_position_pct) / current_price
        quantity = position_size * max_quantity
        
        if quantity <= 0:
            return 0.0
        
        # Calculate execution price
        if action in [TradingActionSpace.BUY_MARKET, TradingActionSpace.SELL_MARKET]:
            # Market order: immediate execution with slippage
            slippage = self._calculate_slippage(quantity, side)
            
            if is_buy:
                exec_price = current_price * (1 + self.slippage_factor) + slippage / quantity
            else:
                exec_price = current_price * (1 - self.slippage_factor) - slippage / quantity
        else:
            # Limit order: better price but may not fill
            exec_price = current_price * (1 + limit_offset)
            
            # Simulate partial fill probability based on limit offset
            fill_probability = min(1.0, abs(limit_offset) * 100)
            if np.random.random() > fill_probability:
                logger.debug("Limit order not filled")
                return 0.0
        
        # Calculate commission
        notional = quantity * exec_price
        commission = notional * self.commission_rate
        
        # Execute trade
        if is_buy:
            cost = notional + commission
            if cost <= self.cash:
                # Update position (average entry)
                if self.position >= 0:
                    # Adding to long or opening long
                    total_value = self.position * self.entry_price + notional
                    self.position += quantity
                    if self.position > 0:
                        self.entry_price = total_value / self.position
                else:
                    # Closing short
                    pnl = (self.entry_price - exec_price) * min(quantity, abs(self.position))
                    self.realized_pnl += pnl
                    self.position += quantity  # position becomes less negative
                
                self.cash -= cost
                self.trade_history.append({
                    "step": self.current_step,
                    "action": "BUY",
                    "quantity": quantity,
                    "price": exec_price,
                    "commission": commission,
                })
        else:
            # Sell
            if self.position > 0:
                # Closing long
                sell_qty = min(quantity, self.position)
                pnl = (exec_price - self.entry_price) * sell_qty
                self.realized_pnl += pnl
                self.position -= sell_qty
                self.cash += sell_qty * exec_price - commission
                
                if self.position < 0:
                    # Flipped to short
                    self.entry_price = exec_price
            else:
                # Adding to short or opening short
                self.position -= quantity
                if self.position < 0:
                    total_value = abs(self.position) * exec_price
                    if abs(self.position) == quantity:
                        self.entry_price = exec_price
                    else:
                        # Average into existing short
                        pass
                
                self.cash += quantity * exec_price - commission
            
            self.trade_history.append({
                "step": self.current_step,
                "action": "SELL",
                "quantity": quantity,
                "price": exec_price,
                "commission": commission,
            })
        
        # Update unrealized PnL
        self._update_unrealized_pnl(current_price)
        
        return commission + (slippage if 'slippage' in locals() else 0)
    
    def _update_unrealized_pnl(self, current_price: float):
        """Update unrealized PnL based on current price."""
        if self.position > 0:
            self.unrealized_pnl = (current_price - self.entry_price) * self.position
        elif self.position < 0:
            self.unrealized_pnl = (self.entry_price - current_price) * abs(self.position)
        else:
            self.unrealized_pnl = 0.0
    
    def _get_observation(self) -> np.ndarray:
        """Construct observation vector from current state."""
        obs_parts = []
        
        # Technical features
        if self.feature_data is not None:
            features = self.feature_data[self.current_step]
            obs_parts.append(features)
        else:
            obs_parts.append(np.zeros(self.n_features, dtype=np.float32))
        
        # Order book data
        if self.include_order_book and self.order_book_data is not None:
            obs_parts.append(self.order_book_data.flatten())
        elif self.include_order_book:
            obs_parts.append(np.zeros(self.order_book_levels * 4, dtype=np.float32))
        
        # Portfolio state
        portfolio_state = np.array([
            self.position,
            self.entry_price,
            self.unrealized_pnl,
            self.realized_pnl,
            self.cash / self.initial_cash,
        ], dtype=np.float32)
        obs_parts.append(portfolio_state)
        
        return np.concatenate(obs_parts)
    
    def _calculate_reward(self, action: int, trade_cost: float) -> float:
        """Calculate reward for the current step."""
        # Base reward: change in portfolio value
        total_pnl = self.realized_pnl + self.unrealized_pnl
        
        if self.reward_shaping:
            # Risk-adjusted reward
            reward = total_pnl / self.initial_cash
            
            # Penalty for drawdown
            peak_pnl = max([t.get("cumulative_pnl", 0) for t in self.trade_history] + [0])
            drawdown = peak_pnl - total_pnl
            if drawdown > 0:
                reward -= 0.1 * (drawdown / self.initial_cash)
            
            # Penalty for excessive trading
            if action != TradingActionSpace.HOLD:
                reward -= 0.001 * trade_cost  # Small penalty for transaction costs
            
            # Bonus for holding profitable positions
            if self.unrealized_pnl > 0:
                reward += 0.0001 * (self.unrealized_pnl / self.initial_cash)
        else:
            # Simple PnL-based reward
            reward = total_pnl / self.initial_cash
        
        return reward
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None,
    ) -> Tuple[np.ndarray, Dict]:
        """Reset the environment."""
        super().reset(seed=seed)
        
        self.current_step = 0
        self.cash = self.initial_cash
        self.position = 0.0
        self.entry_price = 0.0
        self.realized_pnl = 0.0
        self.unrealized_pnl = 0.0
        self.trade_history = []
        self.episode_rewards = []
        
        # Initialize data if provided
        if options and "order_book" in options:
            self.order_book_data = options["order_book"]
        if options and "features" in options:
            self.feature_data = options["features"]
        if options and "prices" in options:
            self.price_series = options["prices"]
        
        obs = self._get_observation()
        info = {
            "cash": self.cash,
            "position": self.position,
            "pnl": self.realized_pnl + self.unrealized_pnl,
        }
        
        return obs, info
    
    def step(
        self,
        action_tuple: Tuple[int, np.ndarray, np.ndarray],
    ) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one step in the environment."""
        action_discrete, position_size, limit_offset = action_tuple
        
        # Get current price
        if self.price_series is not None:
            current_price = self.price_series[self.current_step]
        elif self.order_book_data is not None:
            current_price = (self.order_book_data[0] + self.order_book_data[2]) / 2
        else:
            current_price = 100.0  # Default
        
        # Simulate latency
        simulated_latency = np.random.normal(self.latency_mean_us, self.latency_std_us)
        
        # Execute trade
        trade_cost = self._execute_trade(
            action_discrete,
            position_size[0],
            limit_offset[0],
            current_price,
        )
        
        # Calculate reward
        reward = self._calculate_reward(action_discrete, trade_cost)
        self.episode_rewards.append(reward)
        
        # Advance step
        self.current_step += 1
        
        # Check termination
        terminated = (
            self.current_step >= self.max_steps or
            self.cash <= 0 or
            (self.realized_pnl + self.unrealized_pnl) < -self.initial_cash * 0.5  # 50% drawdown limit
        )
        truncated = False
        
        # Get new observation
        obs = self._get_observation()
        
        # Info dict
        info = {
            "cash": self.cash,
            "position": self.position,
            "entry_price": self.entry_price,
            "realized_pnl": self.realized_pnl,
            "unrealized_pnl": self.unrealized_pnl,
            "total_pnl": self.realized_pnl + self.unrealized_pnl,
            "step": self.current_step,
            "trade_cost": trade_cost,
            "simulated_latency_us": simulated_latency,
        }
        
        return obs, reward, terminated, truncated, info
    
    def render(self, mode: str = "human"):
        """Render the environment."""
        if mode == "human":
            print(f"Step: {self.current_step}")
            print(f"Cash: ${self.cash:,.2f}")
            print(f"Position: {self.position:.6f} @ ${self.entry_price:.2f}")
            print(f"Realized PnL: ${self.realized_pnl:,.2f}")
            print(f"Unrealized PnL: ${self.unrealized_pnl:,.2f}")
            print(f"Total PnL: ${self.realized_pnl + self.unrealized_pnl:,.2f}")
            print("-" * 40)


def main():
    """Test the trading environment."""
    import psutil
    
    print(f"Initial RAM: {psutil.virtual_memory().available / (1024**3):.2f}GB")
    
    # Create environment
    env = TradingEnv(
        initial_cash=100000,
        max_position_pct=0.9,
        transaction_cost_bps=10,
        slippage_factor=0.0001,
        commission_rate=0.0004,
        max_steps=1000,
        n_features=20,
        order_book_levels=5,
    )
    
    # Reset
    obs, info = env.reset(seed=42)
    print(f"\nObservation shape: {obs.shape}")
    print(f"Initial info: {info}")
    
    # Run random actions
    np.random.seed(42)
    total_reward = 0.0
    
    for i in range(100):
        # Random action
        discrete = np.random.randint(0, 6)
        pos_size = np.random.uniform(0.1, 1.0, 1)
        limit_offset = np.random.uniform(-0.005, 0.005, 1)
        
        action = (discrete, pos_size, limit_offset)
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        if terminated or truncated:
            print(f"Episode terminated at step {i}")
            break
    
    print(f"\nTotal reward: {total_reward:.6f}")
    print(f"Final info: {info}")
    print(f"\nFinal RAM: {psutil.virtual_memory().available / (1024**3):.2f}GB")


if __name__ == "__main__":
    main()
