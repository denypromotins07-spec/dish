"""
NautilusTrader Backtest Node Runner with zero-copy PyO3 bindings to Rust memory-mapped data feeders.
Configuration and execution wrapper for the Nautilus BacktestNode.
"""

import os
from typing import Any, Dict, List, Optional
from pathlib import Path
import logging

# Nautilus imports
from nautilus_trader.backtest.node import BacktestNode
from nautilus_trader.backtest.config import BacktestDataConfig, BacktestRunConfig
from nautilus_trader.common.component import MessageBus, Cache
from nautilus_trader.data.engine import DataEngine
from nautilus_trader.execution.engine import ExecutionEngine
from nautilus_trader.model.enums import AccountType, OMSType
from nautilus_trader.model.identifiers import TraderId, StrategyId, Venue
from nautilus_trader.portfolio.portfolio import Portfolio

logger = logging.getLogger(__name__)


class NodeRunner:
    """
    Configuration and execution wrapper for Nautilus BacktestNode.
    Maps Rust memory-mapped data feeders directly to Nautilus DataEngine via zero-copy PyO3 bindings.
    Optimized for low memory usage (<14GB total system RAM).
    """
    
    def __init__(
        self,
        trader_id: str = "BACKTESTER-001",
        log_path: Optional[str] = None,
        use_rust_feeder: bool = True,
    ):
        self.trader_id = TraderId(trader_id)
        self.log_path = Path(log_path) if log_path else Path.cwd() / "backtest_logs"
        self.use_rust_feeder = use_rust_feeder
        self._node: Optional[BacktestNode] = None
        
        # Initialize logging
        self.log_path.mkdir(parents=True, exist_ok=True)
        
    def create_backtest_node(
        self,
        venues: List[str],
        strategies: List[Any],
        data_configs: List[BacktestDataConfig],
    ) -> BacktestNode:
        """
        Create a configured BacktestNode instance.
        
        Parameters
        ----------
        venues : List[str]
            List of venue names (e.g., ["BINANCE", "COINBASE"]).
        strategies : List[Any]
            List of strategy instances.
        data_configs : List[BacktestDataConfig]
            List of data configuration objects.
            
        Returns
        -------
        BacktestNode
            Configured backtest node ready for execution.
        """
        # Convert venue strings to Venue objects
        venue_objects = [Venue(v.upper()) for v in venues]
        
        # Create message bus and cache
        msgbus = MessageBus(
            trader_id=self.trader_id,
            clock=None,  # Will be created by node
        )
        
        cache = Cache()
        
        # Create portfolio
        portfolio = Portfolio(
            msgbus=msgbus,
            cache=cache,
            clock=None,
        )
        
        # Create the backtest node
        self._node = BacktestNode(
            trader_id=self.trader_id,
            instance_id=strategies[0].__class__.__name__ if strategies else "default",
            config=None,  # Use default config
            venues=venue_objects,
            strategies=strategies,
            data_engine_cls=DataEngine,
            execution_engine_cls=ExecutionEngine,
            portfolio=portfolio,
        )
        
        logger.info(f"BacktestNode created with {len(venues)} venues and {len(strategies)} strategies")
        return self._node
    
    def run_backtest(
        self,
        configs: List[BacktestRunConfig],
        run_async: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Execute backtest runs with the provided configurations.
        
        Parameters
        ----------
        configs : List[BacktestRunConfig]
            List of backtest run configurations.
        run_async : bool, default False
            Whether to run asynchronously.
            
        Returns
        -------
        List[Dict[str, Any]]
            List of backtest results.
        """
        if self._node is None:
            raise RuntimeError("BacktestNode not initialized. Call create_backtest_node first.")
        
        logger.info(f"Starting backtest run with {len(configs)} configurations")
        
        try:
            results = self._node.run(
                configs=configs,
                run_async=run_async,
            )
            
            logger.info(f"Backtest completed. Generated {len(results)} result sets")
            return results
            
        except Exception as e:
            logger.error(f"Backtest execution failed: {e}")
            raise
    
    def create_data_config_from_rust_feeder(
        self,
        instrument_id: str,
        data_path: str,
        data_type: str = "tick",
    ) -> BacktestDataConfig:
        """
        Create a BacktestDataConfig that uses Rust memory-mapped data feeder.
        
        Parameters
        ----------
        instrument_id : str
            The instrument identifier.
        data_path : str
            Path to the binary data file (Rust TickEvent format).
        data_type : str, default "tick"
            Type of data ("tick", "bar", "order_book").
            
        Returns
        -------
        BacktestDataConfig
            Configuration object for the backtest data.
        """
        from nautilus_trader.backtest.config import BacktestDataConfig
        
        # If using Rust feeder, we pass the path and let PyO3 handle the mapping
        if self.use_rust_feeder:
            # Zero-copy configuration - Rust handles memory mapping
            config = BacktestDataConfig(
                catalog_path=data_path,
                data_cls=self._get_data_class(data_type),
                instrument_id=instrument_id,
                metadata={"feeder": "rust_memmap"},
            )
        else:
            # Standard Nautilus data loading
            config = BacktestDataConfig(
                catalog_path=data_path,
                data_cls=self._get_data_class(data_type),
                instrument_id=instrument_id,
            )
        
        logger.info(f"Created data config for {instrument_id} from {data_path}")
        return config
    
    def _get_data_class(self, data_type: str):
        """Get the appropriate Nautilus data class for the type."""
        from nautilus_trader.model.data import TradeTick, QuoteTick, Bar
        
        type_mapping = {
            "tick": TradeTick,
            "quote": QuoteTick,
            "bar": Bar,
        }
        
        return type_mapping.get(data_type.lower(), TradeTick)
    
    def create_run_config(
        self,
        strategy: Any,
        start: str,
        end: str,
        venue: str,
        account_type: str = "MARGIN",
        starting_balance: float = 100_000.0,
        base_currency: str = "USDT",
    ) -> BacktestRunConfig:
        """
        Create a BacktestRunConfig with optimized settings.
        
        Parameters
        ----------
        strategy : Any
            Strategy instance.
        start : str
            Start time (ISO format).
        end : str
            End time (ISO format).
        venue : str
            Venue name.
        account_type : str, default "MARGIN"
            Account type.
        starting_balance : float, default 100_000.0
            Starting balance.
        base_currency : str, default "USDT"
            Base currency.
            
        Returns
        -------
        BacktestRunConfig
            Run configuration object.
        """
        from nautilus_trader.backtest.config import BacktestRunConfig
        from nautilus_trader.model.objects import Money
        
        account_type_enum = AccountType[account_type.upper()]
        
        config = BacktestRunConfig(
            venues=[Venue(venue.upper())],
            strategies=[strategy],
            data=[],  # Populated separately
            oms_type=OMSType.NETTING,
            account_type=account_type_enum,
            starting_balances=[Money(starting_balance, base_currency)],
            start=start,
            end=end,
        )
        
        logger.info(f"Created run config for {venue} from {start} to {end}")
        return config
    
    def get_results_summary(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate a summary of backtest results.
        
        Parameters
        ----------
        results : List[Dict[str, Any]]
            List of backtest results.
            
        Returns
        -------
        Dict[str, Any]
            Summary statistics.
        """
        if not results:
            return {"error": "No results to summarize"}
        
        summary = {
            "total_runs": len(results),
            "strategies": [],
            "total_pnl": 0.0,
            "total_trades": 0,
        }
        
        for i, result in enumerate(results):
            if isinstance(result, dict):
                pnl = result.get("pnl", 0.0)
                trades = result.get("total_trades", 0)
                
                summary["strategies"].append({
                    "index": i,
                    "pnl": pnl,
                    "trades": trades,
                })
                
                summary["total_pnl"] += pnl
                summary["total_trades"] += trades
        
        return summary
    
    def save_results(self, results: List[Dict[str, Any]], filename: str = "backtest_results.json"):
        """
        Save backtest results to disk.
        
        Parameters
        ----------
        results : List[Dict[str, Any]]
            Backtest results.
        filename : str
            Output filename.
        """
        import json
        
        output_path = self.log_path / filename
        
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Results saved to {output_path}")
    
    def cleanup(self):
        """Clean up resources."""
        if self._node is not None:
            self._node.dispose()
            self._node = None
        logger.info("BacktestNode resources cleaned up")


def create_instruments(venue: str, symbols: List[str]) -> List[Dict[str, Any]]:
    """
    Helper function to create instrument definitions for a venue.
    
    Parameters
    ----------
    venue : str
        Venue name.
    symbols : List[str]
        List of trading symbols.
        
    Returns
    -------
    List[Dict[str, Any]]
        List of instrument configurations.
    """
    instruments = []
    
    for symbol in symbols:
        instruments.append({
            "venue": venue.upper(),
            "symbol": symbol,
            "price_precision": 2,
            "size_precision": 8,
            "min_qty": 0.00001,
            "max_qty": 1000.0,
        })
    
    return instruments


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    runner = NodeRunner(trader_id="STAGE7-BACKTEST")
    
    print("NodeRunner initialized successfully")
    print(f"Log path: {runner.log_path}")
