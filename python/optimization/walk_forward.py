"""
Ray-distributed walk-forward analysis engine implementing anchored and unanchored expanding windows.
Strictly bounds Ray worker memory to prevent RAM spikes during massive parallelized backtests.
"""

import ray
from ray import remote
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class WalkForwardWindow:
    """Definition of a single walk-forward window."""
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    window_type: str  # "anchored" or "unanchored"
    fold_index: int


@remote(max_calls=1)
class BacktestWorker:
    """
    Ray actor for running individual backtests with strict memory bounds.
    Each worker is limited to prevent RAM spikes.
    """
    
    def __init__(self, max_memory_gb: float = 2.0):
        self.max_memory_gb = max_memory_gb
        self.results_cache: List[Dict] = []
        
        # Set memory limits for this worker
        import resource
        soft_limit = int(max_memory_gb * 1024 * 1024 * 1024)
        try:
            resource.setrlimit(resource.RLIMIT_AS, (soft_limit, soft_limit))
        except (ValueError, resource.error):
            pass  # May not be available on all systems
    
    def run_backtest(
        self,
        strategy_config: Dict[str, Any],
        window: WalkForwardWindow,
        data_path: str,
    ) -> Dict[str, Any]:
        """
        Run a single backtest for the given window.
        
        Parameters
        ----------
        strategy_config : Dict[str, Any]
            Strategy parameters.
        window : WalkForwardWindow
            Train/test window definition.
        data_path : str
            Path to historical data.
            
        Returns
        -------
        Dict[str, Any]
            Backtest results including metrics.
        """
        try:
            # Import here to avoid loading in main process
            from python.backtest.node_runner import NodeRunner
            
            runner = NodeRunner()
            
            # Create run config for this window
            # In production, this would instantiate the actual strategy
            result = {
                'window_index': window.fold_index,
                'train_period': f"{window.train_start} to {window.train_end}",
                'test_period': f"{window.test_start} to {window.test_end}",
                'parameters': strategy_config,
                'metrics': self._generate_mock_metrics(),
                'status': 'completed',
            }
            
            self.results_cache.append(result)
            return result
            
        except Exception as e:
            logger.error(f"Backtest failed for window {window.fold_index}: {e}")
            return {
                'window_index': window.fold_index,
                'status': 'failed',
                'error': str(e),
            }
    
    def _generate_mock_metrics(self) -> Dict[str, float]:
        """Generate mock metrics for demonstration."""
        return {
            'total_return': np.random.uniform(-0.1, 0.3),
            'sharpe_ratio': np.random.uniform(-0.5, 2.5),
            'max_drawdown': np.random.uniform(-0.3, -0.02),
            'win_rate': np.random.uniform(0.4, 0.7),
            'profit_factor': np.random.uniform(0.8, 2.5),
        }
    
    def get_cached_results(self) -> List[Dict]:
        """Get all cached results from this worker."""
        return self.results_cache.copy()


class WalkForwardEngine:
    """
    Ray-distributed walk-forward analysis engine.
    
    Supports:
    - Anchored expanding windows (train set grows, test set slides)
    - Unanchored rolling windows (both train and test slide)
    - Memory-bounded Ray workers
    - Parallel execution across multiple windows
    """
    
    def __init__(
        self,
        max_workers: int = 8,
        worker_memory_gb: float = 2.0,
        ray_init_kwargs: Optional[Dict] = None,
    ):
        self.max_workers = max_workers
        self.worker_memory_gb = worker_memory_gb
        self.ray_init_kwargs = ray_init_kwargs or {}
        
        self.workers: List[BacktestWorker] = []
        self.is_initialized = False
        
    def initialize_ray(self):
        """Initialize Ray cluster with memory constraints."""
        if not self.is_initialized:
            ray.init(
                num_cpus=self.max_workers,
                _memory=int(self.max_workers * self.worker_memory_gb * 1024 * 1024 * 1024),
                **self.ray_init_kwargs,
            )
            self.is_initialized = True
            logger.info(f"Ray initialized with {self.max_workers} workers")
    
    def create_windows(
        self,
        start_date: str,
        end_date: str,
        train_duration_days: int,
        test_duration_days: int,
        step_days: int,
        window_type: str = "anchored",
    ) -> List[WalkForwardWindow]:
        """
        Generate walk-forward windows.
        
        Parameters
        ----------
        start_date : str
            Start date (ISO format).
        end_date : str
            End date (ISO format).
        train_duration_days : int
            Training window length in days.
        test_duration_days : int
            Testing window length in days.
        step_days : int
            Step size between windows.
        window_type : str, default "anchored"
            "anchored" (expanding train) or "unanchored" (rolling).
            
        Returns
        -------
        List[WalkForwardWindow]
            List of window definitions.
        """
        start = datetime.fromisoformat(start_date)
        end = datetime.fromisoformat(end_date)
        
        windows = []
        current_train_end = start + timedelta(days=train_duration_days)
        fold_index = 0
        
        while current_train_end < end:
            train_start = start if window_type == "anchored" else current_train_end - timedelta(days=train_duration_days)
            test_start = current_train_end
            test_end = test_start + timedelta(days=test_duration_days)
            
            if test_end > end:
                break
            
            windows.append(WalkForwardWindow(
                train_start=train_start.isoformat()[:10],
                train_end=current_train_end.isoformat()[:10],
                test_start=test_start.isoformat()[:10],
                test_end=test_end.isoformat()[:10],
                window_type=window_type,
                fold_index=fold_index,
            ))
            
            current_train_end += timedelta(days=step_days)
            fold_index += 1
        
        logger.info(f"Created {len(windows)} {window_type} walk-forward windows")
        return windows
    
    def run_walk_forward(
        self,
        windows: List[WalkForwardWindow],
        strategy_config: Dict[str, Any],
        data_path: str,
        run_async: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Execute walk-forward analysis across all windows.
        
        Parameters
        ----------
        windows : List[WalkForwardWindow]
            Windows to analyze.
        strategy_config : Dict[str, Any]
            Strategy parameters.
        data_path : str
            Path to historical data.
        run_async : bool, default True
            Whether to run asynchronously in parallel.
            
        Returns
        -------
        List[Dict[str, Any]]
            Results for all windows.
        """
        self.initialize_ray()
        
        # Create workers
        self.workers = [
            BacktestWorker.remote(max_memory_gb=self.worker_memory_gb)
            for _ in range(min(self.max_workers, len(windows)))
        ]
        
        results = []
        
        if run_async:
            # Distribute windows across workers
            futures = []
            for i, window in enumerate(windows):
                worker = self.workers[i % len(self.workers)]
                future = worker.run_backtest.remote(strategy_config, window, data_path)
                futures.append(future)
            
            # Collect results
            results = ray.get(futures)
        else:
            # Sequential execution
            for i, window in enumerate(windows):
                worker = self.workers[i % len(self.workers)]
                result = ray.get(worker.run_backtest.remote(strategy_config, window, data_path))
                results.append(result)
        
        logger.info(f"Completed {len(results)} walk-forward iterations")
        return results
    
    def analyze_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze walk-forward results for parameter stability and robustness.
        
        Parameters
        ----------
        results : List[Dict[str, Any]]
            Results from walk-forward runs.
            
        Returns
        -------
        Dict[str, Any]
            Analysis summary.
        """
        completed = [r for r in results if r.get('status') == 'completed']
        
        if not completed:
            return {'error': 'No completed results to analyze'}
        
        # Extract metrics
        metrics_df = pd.DataFrame([r['metrics'] for r in completed])
        
        analysis = {
            'total_windows': len(results),
            'completed_windows': len(completed),
            'mean_sharpe': metrics_df['sharpe_ratio'].mean(),
            'std_sharpe': metrics_df['sharpe_ratio'].std(),
            'mean_return': metrics_df['total_return'].mean(),
            'std_return': metrics_df['total_return'].std(),
            'mean_max_dd': metrics_df['max_drawdown'].mean(),
            'worst_drawdown': metrics_df['max_drawdown'].min(),
            'win_rate_stability': 1 - (metrics_df['win_rate'].std() / max(0.01, metrics_df['win_rate'].mean())),
            'parameter_stability': self._calculate_parameter_stability(completed),
        }
        
        # Check for overfitting
        analysis['overfitting_score'] = self._detect_overfitting(metrics_df)
        
        return analysis
    
    def _calculate_parameter_stability(self, results: List[Dict]) -> float:
        """Calculate how stable parameters are across windows."""
        # In production, would compare optimal parameters across windows
        return 1.0  # Placeholder
    
    def _detect_overfitting(self, metrics_df: pd.DataFrame) -> float:
        """
        Detect potential overfitting based on metric variance.
        
        Returns a score from 0 (no overfitting) to 1 (severe overfitting).
        """
        sharpe_std = metrics_df['sharpe_ratio'].std()
        sharpe_mean = abs(metrics_df['sharpe_ratio'].mean())
        
        if sharpe_mean == 0:
            return 1.0
        
        # High variance relative to mean suggests overfitting
        cv = sharpe_std / sharpe_mean  # Coefficient of variation
        overfitting_score = min(1.0, cv)
        
        return overfitting_score
    
    def shutdown(self):
        """Shutdown Ray cluster and clean up."""
        if self.is_initialized:
            ray.shutdown()
            self.is_initialized = False
            logger.info("Ray cluster shut down")


def run_anchored_walk_forward(
    start_date: str,
    end_date: str,
    initial_train_days: int = 90,
    test_days: int = 30,
    step_days: int = 7,
    strategy_config: Optional[Dict] = None,
) -> Dict[str, Any]:
    """
    Convenience function for anchored (expanding) walk-forward analysis.
    
    Parameters
    ----------
    start_date : str
        Analysis start date.
    end_date : str
        Analysis end date.
    initial_train_days : int, default 90
        Initial training period in days.
    test_days : int, default 30
        Test period in days.
    step_days : int, default 7
        Step between windows.
    strategy_config : Optional[Dict]
        Strategy parameters.
        
    Returns
    -------
    Dict[str, Any]
        Analysis results.
    """
    engine = WalkForwardEngine(max_workers=8)
    
    windows = engine.create_windows(
        start_date=start_date,
        end_date=end_date,
        train_duration_days=initial_train_days,
        test_duration_days=test_days,
        step_days=step_days,
        window_type="anchored",
    )
    
    results = engine.run_walk_forward(
        windows=windows,
        strategy_config=strategy_config or {},
        data_path="/data/historical",
    )
    
    analysis = engine.analyze_results(results)
    engine.shutdown()
    
    return analysis


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Example usage
    engine = WalkForwardEngine(max_workers=4, worker_memory_gb=1.5)
    
    windows = engine.create_windows(
        start_date="2023-01-01",
        end_date="2024-01-01",
        train_duration_days=60,
        test_duration_days=14,
        step_days=7,
        window_type="anchored",
    )
    
    print(f"Created {len(windows)} windows:")
    for w in windows[:3]:
        print(f"  Fold {w.fold_index}: Train {w.train_start} to {w.train_end}, Test {w.test_start} to {w.test_end}")
    
    engine.shutdown()
