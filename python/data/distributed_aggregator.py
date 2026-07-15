"""
Distributed Aggregator for Ray Tasks.
Tree-reduction pattern to merge partial results without loading all data into driver memory.
Essential for large-scale walk-forward optimizations with strict RAM limits.
"""

import ray
from typing import List, Dict, Any, Optional, Callable, TypeVar, Generic
from dataclasses import dataclass
import logging
import polars as pl

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

T = TypeVar('T')


@ray.remote
class PartialAggregator:
    """
    Ray actor that performs partial aggregation on a subset of results.
    Used in tree-reduction pattern to minimize driver memory usage.
    """
    
    def __init__(self, reduce_fn: str = "sum"):
        self.reduce_fn = reduce_fn
        self.partial_results: List[Any] = []
        
    def add_result(self, result: Any):
        """Add a partial result."""
        self.partial_results.append(result)
        
    def get_partial_aggregate(self) -> Any:
        """Compute and return partial aggregate."""
        if not self.partial_results:
            return None
            
        if self.reduce_fn == "sum":
            return sum(self.partial_results)
        elif self.reduce_fn == "mean":
            return sum(self.partial_results) / len(self.partial_results)
        elif self.reduce_fn == "max":
            return max(self.partial_results)
        elif self.reduce_fn == "min":
            return min(self.partial_results)
        elif self.reduce_fn == "concat":
            # For DataFrames/Series
            if isinstance(self.partial_results[0], pl.DataFrame):
                return pl.concat(self.partial_results, how="vertical")
            return self.partial_results
        else:
            raise ValueError(f"Unknown reduce function: {self.reduce_fn}")
    
    def reset(self):
        """Clear partial results."""
        self.partial_results = []


@ray.remote
class DistributedAggregator:
    """
    Main aggregator that coordinates tree-reduction across workers.
    Ensures driver node never holds all raw data simultaneously.
    """
    
    def __init__(self, max_batch_size: int = 100):
        self.max_batch_size = max_batch_size
        self.num_partial_aggregators = 0
        
    def aggregate_tree_reduction(
        self,
        partial_results_refs: List[ray.ObjectRef],
        reduce_fn: str = "sum",
    ) -> ray.ObjectRef:
        """
        Perform tree-reduction on distributed results.
        Returns an ObjectRef to the final aggregated result.
        
        Tree structure:
        Level 0: [R1, R2, R3, R4, R5, R6, R7, R8]  (raw results)
        Level 1: [A1, A2, A3, A4]                   (partial aggregates)
        Level 2: [AA1, AA2]                         (second-level aggregates)
        Level 3: [FINAL]                            (final result)
        """
        if not partial_results_refs:
            return ray.put(None)
        
        # Create partial aggregators
        num_workers = min(len(partial_results_refs), self.max_batch_size)
        aggregator_actors = [
            PartialAggregator.remote(reduce_fn) 
            for _ in range(num_workers)
        ]
        
        # Distribute results to partial aggregators
        for i, ref in enumerate(partial_results_refs):
            actor_idx = i % num_workers
            aggregator_actors[actor_idx].add_result.remote(ref)
        
        # Get partial aggregates
        partial_agg_refs = [
            actor.get_partial_aggregate.remote() 
            for actor in aggregator_actors
        ]
        
        # If we still have too many, recurse
        if len(partial_agg_refs) > self.max_batch_size:
            return self.aggregate_tree_reduction.remote(partial_agg_refs, reduce_fn)
        
        # Final reduction at driver (now safe since these are already aggregated)
        return self._final_reduce.remote(partial_agg_refs, reduce_fn)
    
    @staticmethod
    def _final_reduce(partial_refs: List[ray.ObjectRef], reduce_fn: str) -> Any:
        """Final reduction step."""
        partials = ray.get(partial_refs)
        
        if reduce_fn == "sum":
            return sum(partials)
        elif reduce_fn == "mean":
            return sum(partials) / len(partials) if partials else 0
        elif reduce_fn == "max":
            return max(partials) if partials else None
        elif reduce_fn == "min":
            return min(partials) if partials else None
        elif reduce_fn == "concat":
            if partials and isinstance(partials[0], pl.DataFrame):
                return pl.concat(partials, how="vertical")
            return partials
        else:
            raise ValueError(f"Unknown reduce function: {reduce_fn}")


class BacktestResultAggregator:
    """
    Specialized aggregator for backtest optimization results.
    Handles Sharpe ratios, drawdowns, and other metrics.
    """
    
    def __init__(self):
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True)
        self.distributed_agg = DistributedAggregator.remote(max_batch_size=50)
        
    async def aggregate_walk_forward_results(
        self,
        fold_results_refs: List[ray.ObjectRef],
    ) -> Dict[str, float]:
        """
        Aggregate walk-forward optimization results from multiple folds.
        Each fold result is a dict with metrics like 'sharpe', 'drawdown', etc.
        """
        
        # Extract individual metrics using Ray tasks
        sharpe_refs = [self._extract_metric.remote(ref, "sharpe") for ref in fold_results_refs]
        drawdown_refs = [self._extract_metric.remote(ref, "drawdown") for ref in fold_results_refs]
        return_refs = [self._extract_metric.remote(ref, "total_return") for ref in fold_results_refs]
        
        # Aggregate each metric using tree reduction
        final_sharpe_ref = self.distributed_agg.aggregate_tree_reduction.remote(
            sharpe_refs, "mean"
        )
        final_drawdown_ref = self.distributed_agg.aggregate_tree_reduction.remote(
            drawdown_refs, "mean"
        )
        final_return_ref = self.distributed_agg.aggregate_tree_reduction.remote(
            return_refs, "mean"
        )
        
        # Get final values
        final_sharpe, final_drawdown, final_return = await asyncio.gather(
            final_sharpe_ref,
            final_drawdown_ref,
            final_return_ref,
        )
        
        return {
            "avg_sharpe": final_sharpe,
            "avg_drawdown": final_drawdown,
            "avg_return": final_return,
            "num_folds": len(fold_results_refs),
        }
    
    @staticmethod
    @ray.remote
    def _extract_metric(result_ref: ray.ObjectRef, metric_name: str) -> float:
        """Extract a specific metric from a result object."""
        result = ray.get(result_ref)
        return result.get(metric_name, 0.0) if isinstance(result, dict) else 0.0
    
    def cleanup(self):
        """Cleanup Ray actors."""
        ray.kill(self.distributed_agg)


# Example usage
async def example_aggregation():
    """Example: Aggregate results from 1000 walk-forward folds."""
    
    @ray.remote
    def run_fold(fold_id: int) -> Dict[str, float]:
        """Simulate running a single walk-forward fold."""
        import random
        return {
            "sharpe": random.uniform(1.0, 3.0),
            "drawdown": random.uniform(0.05, 0.15),
            "total_return": random.uniform(0.1, 0.5),
            "fold_id": fold_id,
        }
    
    # Run 1000 folds in parallel (in batches to avoid overwhelming Ray)
    fold_refs = []
    for i in range(1000):
        ref = run_fold.remote(i)
        fold_refs.append(ref)
    
    # Aggregate results
    aggregator = BacktestResultAggregator()
    final_metrics = await aggregator.aggregate_walk_forward_results(fold_refs)
    
    print("Final Walk-Forward Metrics:")
    for metric, value in final_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    aggregator.cleanup()


if __name__ == "__main__":
    import asyncio
    asyncio.run(example_aggregation())
