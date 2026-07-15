"""
Expert trajectory extractor from Nautilus backtest logs.
Filters sub-optimal trades to create pristine demonstration datasets.
Memory-efficient design for large historical datasets.
"""

import json
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass
class Trade:
    """Single trade record."""
    timestamp: int
    symbol: str
    side: str  # 'buy' or 'sell'
    quantity: float
    price: float
    fee: float
    pnl: float = 0.0
    slippage: float = 0.0


@dataclass
class Trajectory:
    """Complete execution trajectory."""
    trades: List[Trade]
    total_pnl: float
    total_slippage: float
    total_fees: float
    sharpe_ratio: float
    fill_rate: float
    is_expert: bool


class PerformanceMetrics:
    """Calculate performance metrics for trajectory evaluation."""
    
    @staticmethod
    def calculate_sharpe(returns: np.ndarray, risk_free_rate: float = 0.0) -> float:
        """Calculate annualized Sharpe ratio."""
        if len(returns) < 2 or np.std(returns) == 0:
            return 0.0
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        return np.sqrt(252) * (mean_return - risk_free_rate) / std_return
    
    @staticmethod
    def calculate_fill_rate(filled_qty: float, intended_qty: float) -> float:
        """Calculate order fill rate."""
        if intended_qty == 0:
            return 0.0
        return filled_qty / intended_qty
    
    @staticmethod
    def calculate_slippage(avg_fill_price: float, benchmark_price: float, side: str) -> float:
        """Calculate slippage relative to benchmark."""
        if benchmark_price == 0:
            return 0.0
        if side == 'buy':
            return (avg_fill_price - benchmark_price) / benchmark_price
        else:
            return (benchmark_price - avg_fill_price) / benchmark_price


class ExpertTrajectoryExtractor:
    """
    Extracts expert trajectories from Nautilus backtest logs.
    Filters based on performance thresholds.
    """
    
    def __init__(
        self,
        min_sharpe: float = 1.5,
        max_slippage_bps: float = 10.0,
        min_fill_rate: float = 0.8,
        max_drawdown: float = 0.05,
    ):
        self.min_sharpe = min_sharpe
        self.max_slippage_bps = max_slippage_bps
        self.min_fill_rate = min_fill_rate
        self.max_drawdown = max_drawdown
        
        self.metrics = PerformanceMetrics()
        
    def parse_nautilus_log(self, log_path: str) -> List[Trade]:
        """Parse Nautilus backtest log file into Trade objects."""
        trades = []
        
        with open(log_path, 'r') as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    
                    # Extract relevant fields from Nautilus format
                    trade = Trade(
                        timestamp=int(data.get('timestamp', 0)),
                        symbol=data.get('instrument_id', ''),
                        side=data.get('side', 'buy').lower(),
                        quantity=float(data.get('quantity', 0)),
                        price=float(data.get('price', 0)),
                        fee=float(data.get('fee', 0)),
                        pnl=float(data.get('realized_pnl', 0)),
                        slippage=float(data.get('slippage', 0)),
                    )
                    trades.append(trade)
                    
                except (json.JSONDecodeError, KeyError, ValueError):
                    continue
                    
        return sorted(trades, key=lambda t: t.timestamp)
    
    def segment_into_trajectories(
        self,
        trades: List[Trade],
        time_window_ns: int = 3600_000_000_000,  # 1 hour in nanoseconds
    ) -> List[Trajectory]:
        """Segment trades into non-overlapping trajectories."""
        if not trades:
            return []
            
        trajectories = []
        current_trades = [trades[0]]
        
        for i in range(1, len(trades)):
            if trades[i].timestamp - current_trades[-1].timestamp > time_window_ns:
                # Create trajectory from current trades
                traj = self._create_trajectory(current_trades)
                trajectories.append(traj)
                current_trades = []
                
            current_trades.append(trades[i])
            
        # Add final trajectory
        if current_trades:
            traj = self._create_trajectory(current_trades)
            trajectories.append(traj)
            
        return trajectories
    
    def _create_trajectory(self, trades: List[Trade]) -> Trajectory:
        """Create Trajectory object from list of trades."""
        total_pnl = sum(t.pnl for t in trades)
        total_slippage = sum(t.slippage for t in trades)
        total_fees = sum(t.fee for t in trades)
        
        # Calculate returns for Sharpe
        if len(trades) > 1:
            cumulative_pnl = np.cumsum([t.pnl for t in trades])
            returns = np.diff(cumulative_pnl) / (np.abs(cumulative_pnl[:-1]) + 1e-8)
            sharpe = self.metrics.calculate_sharpe(returns)
        else:
            sharpe = 0.0
            
        # Calculate fill rate (assuming intended quantity from first trade context)
        total_filled = sum(abs(t.quantity) for t in trades)
        fill_rate = min(1.0, total_filled / (total_filled + 1e-8))
        
        return Trajectory(
            trades=trades,
            total_pnl=total_pnl,
            total_slippage=total_slippage,
            total_fees=total_fees,
            sharpe_ratio=sharpe,
            fill_rate=fill_rate,
            is_expert=False,  # Will be set by filter
        )
    
    def filter_expert_trajectories(
        self,
        trajectories: List[Trajectory],
    ) -> List[Trajectory]:
        """Filter trajectories based on expert criteria."""
        expert_trajectories = []
        
        for traj in trajectories:
            # Check Sharpe ratio
            if traj.sharpe_ratio < self.min_sharpe:
                continue
                
            # Check slippage (in bps)
            avg_slippage_bps = (traj.total_slippage / len(traj.trades)) * 10000 if traj.trades else 0
            if avg_slippage_bps > self.max_slippage_bps:
                continue
                
            # Check fill rate
            if traj.fill_rate < self.min_fill_rate:
                continue
                
            # Mark as expert
            traj.is_expert = True
            expert_trajectories.append(traj)
            
        return expert_trajectories
    
    def extract_from_directory(
        self,
        log_directory: str,
        output_path: str,
    ) -> Dict[str, int]:
        """
        Process all logs in directory and save expert trajectories.
        Returns statistics about processed data.
        """
        log_dir = Path(log_directory)
        all_trajectories = []
        
        # Parse all log files
        for log_file in log_dir.glob("*.log"):
            trades = self.parse_nautilus_log(str(log_file))
            trajectories = self.segment_into_trajectories(trades)
            all_trajectories.extend(trajectories)
            
        # Filter experts
        expert_trajectories = self.filter_expert_trajectories(all_trajectories)
        
        # Save to file
        self.save_trajectories(expert_trajectories, output_path)
        
        return {
            'total_trajectories': len(all_trajectories),
            'expert_trajectories': len(expert_trajectories),
            'expert_ratio': len(expert_trajectories) / max(1, len(all_trajectories)),
        }
    
    def save_trajectories(
        self,
        trajectories: List[Trajectory],
        output_path: str,
    ) -> None:
        """Save trajectories to JSONL file."""
        with open(output_path, 'w') as f:
            for traj in trajectories:
                # Convert to dict format
                data = {
                    'trades': [asdict(t) for t in traj.trades],
                    'metrics': {
                        'total_pnl': traj.total_pnl,
                        'total_slippage': traj.total_slippage,
                        'total_fees': traj.total_fees,
                        'sharpe_ratio': traj.sharpe_ratio,
                        'fill_rate': traj.fill_rate,
                        'is_expert': traj.is_expert,
                    },
                }
                f.write(json.dumps(data) + '\n')
    
    def load_trajectories(self, input_path: str) -> List[Trajectory]:
        """Load trajectories from JSONL file."""
        trajectories = []
        
        with open(input_path, 'r') as f:
            for line in f:
                data = json.loads(line.strip())
                
                trades = [
                    Trade(**t) for t in data['trades']
                ]
                
                traj = Trajectory(
                    trades=trades,
                    total_pnl=data['metrics']['total_pnl'],
                    total_slippage=data['metrics']['total_slippage'],
                    total_fees=data['metrics']['total_fees'],
                    sharpe_ratio=data['metrics']['sharpe_ratio'],
                    fill_rate=data['metrics']['fill_rate'],
                    is_expert=data['metrics']['is_expert'],
                )
                trajectories.append(traj)
                
        return trajectories


if __name__ == "__main__":
    # Example usage
    extractor = ExpertTrajectoryExtractor(
        min_sharpe=1.5,
        max_slippage_bps=10.0,
        min_fill_rate=0.8,
    )
    
    # Create dummy test data
    print("Creating dummy test trajectories...")
    
    dummy_trades = [
        Trade(timestamp=i * 1_000_000_000, symbol="BTC-PERP", side="buy", 
              quantity=1.0, price=50000 + i * 10, fee=0.5, pnl=i * 5, slippage=0.0001)
        for i in range(50)
    ]
    
    trajectories = extractor.segment_into_trajectories(dummy_trades, time_window_ns=10_000_000_000)
    print(f"Created {len(trajectories)} trajectories")
    
    expert_trajectories = extractor.filter_expert_trajectories(trajectories)
    print(f"Found {len(expert_trajectories)} expert trajectories")
    
    for i, traj in enumerate(expert_trajectories):
        print(f"\nTrajectory {i}:")
        print(f"  Trades: {len(traj.trades)}")
        print(f"  Total PnL: ${traj.total_pnl:.2f}")
        print(f"  Sharpe: {traj.sharpe_ratio:.2f}")
        print(f"  Fill Rate: {traj.fill_rate:.2%}")
        print(f"  Is Expert: {traj.is_expert}")
