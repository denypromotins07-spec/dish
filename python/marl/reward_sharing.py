"""
Reward shaping and profit-sharing mechanisms for multi-agent trading.
Implements Shapley values and other cooperative game theory concepts.
Prevents agents from cannibalizing liquidity or taking opposing positions.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from itertools import combinations
from collections import defaultdict


@dataclass
class AgentMetrics:
    """Performance metrics for a single agent."""
    agent_id: str
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    trades_count: int = 0
    win_rate: float = 0.0
    avg_trade_pnl: float = 0.0
    # Metrics for cooperation tracking
    liquidity_impact: float = 0.0
    cross_agent_interference: float = 0.0


@dataclass
class RewardConfig:
    """Configuration for reward shaping."""
    # Base reward weights
    pnl_weight: float = 1.0
    sharpe_weight: float = 0.5
    drawdown_penalty: float = 0.3
    
    # Cooperation incentives
    cooperation_bonus: float = 0.2
    interference_penalty: float = 0.5
    
    # Risk adjustments
    risk_adjustment: bool = True
    max_position_penalty: float = 0.1
    
    # Shapley value computation
    use_shapley: bool = True
    shapley_samples: int = 50  # Number of permutations to sample


class RewardShaper:
    """
    Shapes rewards for multi-agent trading environment.
    Balances individual performance with team cooperation.
    """
    
    def __init__(self, config: Optional[RewardConfig] = None):
        self.config = config or RewardConfig()
        self.agent_metrics: Dict[str, AgentMetrics] = {}
        
    def register_agent(self, agent_id: str):
        """Register a new agent for tracking."""
        if agent_id not in self.agent_metrics:
            self.agent_metrics[agent_id] = AgentMetrics(agent_id=agent_id)
    
    def update_metrics(
        self,
        agent_id: str,
        realized_pnl: float,
        trades: List[Dict],
        portfolio_value: float,
    ):
        """Update metrics for an agent after a trading period."""
        if agent_id not in self.agent_metrics:
            self.register_agent(agent_id)
        
        metrics = self.agent_metrics[agent_id]
        metrics.realized_pnl += realized_pnl
        metrics.trades_count += len(trades)
        
        # Update trade statistics
        if trades:
            trade_pnls = [t.get('pnl', 0) for t in trades]
            wins = sum(1 for p in trade_pnls if p > 0)
            metrics.win_rate = wins / len(trades)
            metrics.avg_trade_pnl = np.mean(trade_pnls)
    
    def calculate_base_reward(self, agent_id: str) -> float:
        """Calculate base reward from individual performance."""
        if agent_id not in self.agent_metrics:
            return 0.0
        
        metrics = self.agent_metrics[agent_id]
        
        # PnL component
        pnl_component = self.config.pnl_weight * metrics.realized_pnl
        
        # Sharpe component (if available)
        sharpe_component = self.config.sharpe_weight * metrics.sharpe_ratio
        
        # Drawdown penalty
        drawdown_penalty = -self.config.drawdown_penalty * abs(metrics.max_drawdown)
        
        base_reward = pnl_component + sharpe_component + drawdown_penalty
        
        return base_reward
    
    def calculate_cooperation_bonus(
        self,
        agent_id: str,
        all_positions: Dict[str, Dict[str, float]],
    ) -> float:
        """
        Calculate bonus for cooperative behavior.
        
        Args:
            agent_id: The agent to calculate bonus for
            all_positions: Dict mapping agent_id to their positions by asset
        
        Returns:
            Cooperation bonus (positive or negative)
        """
        bonus = 0.0
        agent_positions = all_positions.get(agent_id, {})
        
        for other_id, other_positions in all_positions.items():
            if other_id == agent_id:
                continue
            
            # Check for position conflicts (opposing trades on same asset)
            for asset, pos in agent_positions.items():
                other_pos = other_positions.get(asset, 0)
                
                # Detect opposing positions
                if pos * other_pos < 0:
                    # Penalize conflicting positions
                    conflict_size = min(abs(pos), abs(other_pos))
                    bonus -= self.config.interference_penalty * conflict_size
                
                # Reward aligned positions
                elif pos * other_pos > 0:
                    alignment = min(abs(pos), abs(other_pos))
                    bonus += self.config.cooperation_bonus * alignment * 0.1
        
        return bonus
    
    def calculate_liquidity_impact_reward(
        self,
        agent_id: str,
        market_impact: Dict[str, float],
    ) -> float:
        """
        Calculate reward based on liquidity impact.
        Rewards agents that provide liquidity, penalizes those that take it.
        """
        if agent_id not in self.agent_metrics:
            return 0.0
        
        metrics = self.agent_metrics[agent_id]
        
        # Simple model: negative impact means providing liquidity
        total_impact = sum(market_impact.values())
        
        if total_impact < 0:
            # Provided liquidity
            return abs(total_impact) * self.config.cooperation_bonus
        else:
            # Took liquidity
            return -total_impact * self.config.interference_penalty
    
    def shape_reward(
        self,
        agent_id: str,
        raw_reward: float,
        all_positions: Optional[Dict[str, Dict[str, float]]] = None,
        market_impact: Optional[Dict[str, float]] = None,
    ) -> float:
        """
        Apply full reward shaping to raw reward.
        
        Args:
            agent_id: Agent receiving the reward
            raw_reward: Original reward from environment
            all_positions: Current positions of all agents
            market_impact: Market impact by asset
        
        Returns:
            Shaped reward
        """
        shaped_reward = raw_reward
        
        # Add base reward components
        base = self.calculate_base_reward(agent_id)
        shaped_reward += base * 0.1  # Scale to not overwhelm raw reward
        
        # Add cooperation bonus
        if all_positions is not None:
            coop_bonus = self.calculate_cooperation_bonus(agent_id, all_positions)
            shaped_reward += coop_bonus * 0.05
        
        # Add liquidity impact reward
        if market_impact is not None:
            liq_reward = self.calculate_liquidity_impact_reward(agent_id, market_impact)
            shaped_reward += liq_reward * 0.05
        
        return shaped_reward


class ShapleyValueCalculator:
    """
    Calculates Shapley values for fair profit distribution among agents.
    Uses sampling for efficiency with many agents.
    """
    
    def __init__(self, n_samples: int = 50):
        self.n_samples = n_samples
        self._cache: Dict[Tuple, float] = {}
    
    def calculate_shapley_values(
        self,
        agent_ids: List[str],
        coalition_value_fn,
    ) -> Dict[str, float]:
        """
        Calculate Shapley values using sampled permutations.
        
        Args:
            agent_ids: List of agent IDs
            coalition_value_fn: Function that takes a set of agents and returns coalition value
        
        Returns:
            Dict mapping agent_id to Shapley value
        """
        n = len(agent_ids)
        shapley_values = {agent_id: 0.0 for agent_id in agent_ids}
        
        # Sample permutations
        rng = np.random.default_rng()
        
        for _ in range(self.n_samples):
            perm = rng.permutation(agent_ids).tolist()
            
            # Calculate marginal contributions
            cumulative = set()
            prev_value = coalition_value_fn(cumulative)
            
            for agent_id in perm:
                cumulative.add(agent_id)
                curr_value = coalition_value_fn(cumulative)
                marginal = curr_value - prev_value
                shapley_values[agent_id] += marginal
                prev_value = curr_value
        
        # Average over samples
        for agent_id in agent_ids:
            shapley_values[agent_id] /= self.n_samples
        
        return shapley_values
    
    def calculate_profit_sharing(
        self,
        agent_ids: List[str],
        total_profit: float,
        individual_contributions: Dict[str, float],
    ) -> Dict[str, float]:
        """
        Calculate profit shares based on Shapley values.
        
        Args:
            agent_ids: List of agent IDs
            total_profit: Total profit to distribute
            individual_contributions: Individual PnL contributions
        
        Returns:
            Dict mapping agent_id to profit share
        """
        def coalition_value(coalition: set) -> float:
            """Value of a coalition = sum of individual contributions."""
            return sum(individual_contributions.get(a, 0) for a in coalition)
        
        shapley = self.calculate_shapley_values(agent_ids, coalition_value)
        
        # Normalize Shapley values to sum to 1
        total_shapley = sum(shapley.values())
        if abs(total_shapley) < 1e-10:
            # Equal split if no meaningful contributions
            equal_share = total_profit / len(agent_ids)
            return {a: equal_share for a in agent_ids}
        
        # Distribute profit according to Shapley values
        profit_shares = {}
        for agent_id in agent_ids:
            share_ratio = shapley[agent_id] / total_shapley
            profit_shares[agent_id] = total_profit * share_ratio
        
        return profit_shares


class ProfitSharingManager:
    """
    Manages profit sharing and reward distribution across agents.
    Prevents internal competition and encourages cooperation.
    """
    
    def __init__(
        self,
        agent_ids: List[str],
        sharing_method: str = "shapley",
    ):
        self.agent_ids = agent_ids
        self.sharing_method = sharing_method
        self.shapley_calculator = ShapleyValueCalculator()
        self.reward_shaper = RewardShaper()
        
        # Track cumulative profits
        self.cumulative_profits: Dict[str, float] = {a: 0.0 for a in agent_ids}
        self.total_pool: float = 0.0
        
        # History for analysis
        self.distribution_history: List[Dict[str, float]] = []
    
    def add_profit(self, agent_id: str, profit: float):
        """Record profit contribution from an agent."""
        if agent_id in self.cumulative_profits:
            self.cumulative_profits[agent_id] += profit
            self.total_pool += profit
    
    def distribute_profits(self) -> Dict[str, float]:
        """
        Distribute accumulated profits according to sharing method.
        
        Returns:
            Dict mapping agent_id to distributed amount
        """
        if abs(self.total_pool) < 1e-10:
            return {a: 0.0 for a in self.agent_ids}
        
        if self.sharing_method == "shapley":
            distribution = self._distribute_shapley()
        elif self.sharing_method == "equal":
            distribution = self._distribute_equal()
        elif self.sharing_method == "proportional":
            distribution = self._distribute_proportional()
        else:
            distribution = self._distribute_equal()
        
        self.distribution_history.append(distribution.copy())
        
        # Reset pool
        self.total_pool = 0.0
        
        return distribution
    
    def _distribute_shapley(self) -> Dict[str, float]:
        """Distribute using Shapley values."""
        return self.shapley_calculator.calculate_profit_sharing(
            self.agent_ids,
            self.total_pool,
            self.cumulative_profits,
        )
    
    def _distribute_equal(self) -> Dict[str, float]:
        """Equal distribution among all agents."""
        share = self.total_pool / len(self.agent_ids)
        return {a: share for a in self.agent_ids}
    
    def _distribute_proportional(self) -> Dict[str, float]:
        """Distribute proportional to individual contributions."""
        total_contrib = sum(self.cumulative_profits.values())
        
        if abs(total_contrib) < 1e-10:
            return self._distribute_equal()
        
        distribution = {}
        for agent_id in self.agent_ids:
            ratio = self.cumulative_profits[agent_id] / total_contrib
            distribution[agent_id] = self.total_pool * ratio
        
        return distribution
    
    def get_agent_aligned_reward(
        self,
        agent_id: str,
        raw_reward: float,
        all_positions: Dict[str, Dict[str, float]],
    ) -> float:
        """
        Get reward aligned with team objectives.
        
        Args:
            agent_id: Agent receiving reward
            raw_reward: Original reward
            all_positions: All agents' positions
        
        Returns:
            Aligned reward combining individual and team performance
        """
        # Shape individual reward
        shaped = self.reward_shaper.shape_reward(
            agent_id, raw_reward, all_positions
        )
        
        # Add team component (fraction of distributed profits)
        if self.distribution_history:
            last_distribution = self.distribution_history[-1]
            team_bonus = last_distribution.get(agent_id, 0) * 0.01
            shaped += team_bonus
        
        return shaped


class CrossAgentConflictDetector:
    """
    Detects and prevents agents from taking opposing positions.
    Implements internal crossing prevention.
    """
    
    def __init__(self, threshold: float = 0.8):
        self.threshold = threshold
        self.position_history: Dict[str, List[Dict[str, float]]] = defaultdict(list)
    
    def record_positions(self, agent_id: str, positions: Dict[str, float]):
        """Record current positions for an agent."""
        self.position_history[agent_id].append(positions.copy())
        
        # Keep limited history
        if len(self.position_history[agent_id]) > 100:
            self.position_history[agent_id].pop(0)
    
    def detect_conflicts(
        self,
        all_positions: Dict[str, Dict[str, float]],
    ) -> List[Dict]:
        """
        Detect conflicting positions between agents.
        
        Returns:
            List of conflict descriptions
        """
        conflicts = []
        agent_ids = list(all_positions.keys())
        
        for i, agent_a in enumerate(agent_ids):
            for agent_b in agent_ids[i+1:]:
                pos_a = all_positions.get(agent_a, {})
                pos_b = all_positions.get(agent_b, {})
                
                for asset in set(pos_a.keys()) | set(pos_b.keys()):
                    pa = pos_a.get(asset, 0)
                    pb = pos_b.get(asset, 0)
                    
                    # Check for opposing positions
                    if pa * pb < 0:
                        conflict_severity = min(abs(pa), abs(pb)) / max(abs(pa), abs(pb), 1)
                        
                        if conflict_severity > self.threshold:
                            conflicts.append({
                                'agent_a': agent_a,
                                'agent_b': agent_b,
                                'asset': asset,
                                'position_a': pa,
                                'position_b': pb,
                                'severity': conflict_severity,
                            })
        
        return conflicts
    
    def suggest_resolution(
        self,
        conflicts: List[Dict],
        agent_metrics: Dict[str, AgentMetrics],
    ) -> List[Dict]:
        """
        Suggest resolutions for detected conflicts.
        Prioritizes agents with better track records.
        """
        resolutions = []
        
        for conflict in conflicts:
            agent_a = conflict['agent_a']
            agent_b = conflict['agent_b']
            
            metrics_a = agent_metrics.get(agent_a)
            metrics_b = agent_metrics.get(agent_b)
            
            # Determine which agent should yield
            score_a = metrics_a.sharpe_ratio if metrics_a else 0
            score_b = metrics_b.sharpe_ratio if metrics_b else 0
            
            if score_a > score_b:
                resolutions.append({
                    'conflict': conflict,
                    'action': 'reduce_b',
                    'reason': f'{agent_a} has better Sharpe ({score_a:.2f} vs {score_b:.2f})',
                })
            else:
                resolutions.append({
                    'conflict': conflict,
                    'action': 'reduce_a',
                    'reason': f'{agent_b} has better Sharpe ({score_b:.2f} vs {score_a:.2f})',
                })
        
        return resolutions


if __name__ == "__main__":
    # Example usage
    agent_ids = ["mm_0", "arb_0", "trend_0"]
    
    # Test Shapley value calculation
    shapley_calc = ShapleyValueCalculator(n_samples=30)
    
    contributions = {
        "mm_0": 10000,
        "arb_0": 15000,
        "trend_0": 5000,
    }
    
    shares = shapley_calc.calculate_profit_sharing(
        agent_ids,
        total_profit=30000,
        individual_contributions=contributions,
    )
    
    print("Shapley-based profit sharing:")
    for agent, share in shares.items():
        print(f"  {agent}: ${share:,.2f}")
    
    # Test conflict detection
    detector = CrossAgentConflictDetector()
    
    positions = {
        "mm_0": {"BTC": 10.0, "ETH": -5.0},
        "arb_0": {"BTC": -8.0, "SOL": 20.0},
        "trend_0": {"BTC": 5.0, "ETH": 10.0},
    }
    
    conflicts = detector.detect_conflicts(positions)
    print(f"\nDetected {len(conflicts)} conflicts")
    for c in conflicts:
        print(f"  {c['agent_a']} vs {c['agent_b']} on {c['asset']}")
