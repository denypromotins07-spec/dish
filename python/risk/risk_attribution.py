# python/risk/risk_attribution.py
"""
Brinson-Fachler style performance attribution engine.
Decomposes daily PnL into Allocation Effect, Selection Effect, and Interaction Effect.
Memory-efficient implementation using Polars for strict RAM constraints.
"""

from __future__ import annotations
import polars as pl
from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass
class AttributionResult:
    """Single-period attribution results."""
    allocation_effect: float
    selection_effect: float
    interaction_effect: float
    total_active_return: float
    tracking_error: Optional[float] = None
    
    def to_dict(self) -> dict:
        return {
            "allocation_effect": self.allocation_effect,
            "selection_effect": self.selection_effect,
            "interaction_effect": self.interaction_effect,
            "total_active_return": self.total_active_return,
            "tracking_error": self.tracking_error,
        }


@dataclass
class SectorAttribution:
    """Attribution breakdown by sector/asset class."""
    sector: str
    portfolio_weight: float
    benchmark_weight: float
    portfolio_return: float
    benchmark_return: float
    allocation_effect: float
    selection_effect: float
    interaction_effect: float


class BrinsonFachlerAttributor:
    """
    Brinson-Fachler performance attribution engine.
    
    Decomposes active return into three components:
    - Allocation Effect: Impact of over/underweighting sectors vs benchmark
    - Selection Effect: Impact of security selection within sectors
    - Interaction Effect: Combined effect of allocation and selection decisions
    
    Formula (Brinson-Hood-Beebower / Brinson-Fachler):
    - Allocation_i = (Wp_i - Wb_i) * (Rb_i - Rb_total)
    - Selection_i = Wb_i * (Rp_i - Rb_i)
    - Interaction_i = (Wp_i - Wb_i) * (Rp_i - Rb_i)
    
    Where:
    - Wp_i = Portfolio weight in sector i
    - Wb_i = Benchmark weight in sector i
    - Rp_i = Portfolio return in sector i
    - Rb_i = Benchmark return in sector i
    - Rb_total = Total benchmark return
    """
    
    def __init__(self, max_history_rows: int = 100_000):
        """
        Initialize the attributor with memory constraints.
        
        Args:
            max_history_rows: Maximum rows to keep in memory for rolling calculations
        """
        self.max_history_rows = max_history_rows
        self._attribution_history: list[pl.DataFrame] = []
        
    def calculate_single_period(
        self,
        portfolio_weights: dict[str, float],
        benchmark_weights: dict[str, float],
        portfolio_returns: dict[str, float],
        benchmark_returns: dict[str, float],
        total_benchmark_return: float,
    ) -> AttributionResult:
        """
        Calculate Brinson-Fachler attribution for a single period.
        
        Args:
            portfolio_weights: Dict mapping sector -> portfolio weight (0-1)
            benchmark_weights: Dict mapping sector -> benchmark weight (0-1)
            portfolio_returns: Dict mapping sector -> portfolio return (%)
            benchmark_returns: Dict mapping sector -> benchmark return (%)
            total_benchmark_return: Overall benchmark return for the period
            
        Returns:
            AttributionResult with decomposed effects
        """
        # Get all sectors present in either portfolio or benchmark
        all_sectors = set(portfolio_weights.keys()) | set(benchmark_weights.keys())
        
        total_allocation = 0.0
        total_selection = 0.0
        total_interaction = 0.0
        
        for sector in all_sectors:
            wp = portfolio_weights.get(sector, 0.0)
            wb = benchmark_weights.get(sector, 0.0)
            rp = portfolio_returns.get(sector, 0.0)
            rb = benchmark_returns.get(sector, 0.0)
            
            # Weight difference
            weight_diff = wp - wb
            
            # Return difference
            return_diff = rp - rb
            
            # Allocation effect: (Wp - Wb) * (Rb - Rb_total)
            # Positive if overweight sectors that outperformed the benchmark average
            allocation = weight_diff * (rb - total_benchmark_return)
            
            # Selection effect: Wb * (Rp - Rb)
            # Positive if stock selection within sector beat the sector benchmark
            selection = wb * return_diff
            
            # Interaction effect: (Wp - Wb) * (Rp - Rb)
            interaction = weight_diff * return_diff
            
            total_allocation += allocation
            total_selection += selection
            total_interaction += interaction
        
        total_active_return = total_allocation + total_selection + total_interaction
        
        return AttributionResult(
            allocation_effect=total_allocation,
            selection_effect=total_selection,
            interaction_effect=total_interaction,
            total_active_return=total_active_return,
        )
    
    def calculate_with_polars(
        self,
        portfolio_df: pl.DataFrame,
        benchmark_df: pl.DataFrame,
        period: str,
    ) -> AttributionResult:
        """
        Calculate attribution using Polars DataFrames for efficiency.
        
        Expected DataFrame schema:
        - sector: str
        - weight: f64 (0-1)
        - return: f64 (decimal, e.g., 0.05 for 5%)
        
        Args:
            portfolio_df: Portfolio holdings and returns
            benchmark_df: Benchmark weights and returns
            period: Time period identifier
            
        Returns:
            AttributionResult
        """
        # Join portfolio and benchmark on sector
        joined = portfolio_df.join(
            benchmark_df,
            on="sector",
            how="full",
            suffix="_bench",
        ).fill_null(0.0)
        
        # Calculate total benchmark return (weighted average)
        total_benchmark_return = (
            joined["weight_bench"] * joined["return_bench"]
        ).sum()
        
        # Calculate attribution components per sector
        joined = joined.with_columns([
            (pl.col("weight") - pl.col("weight_bench")).alias("weight_diff"),
            (pl.col("return") - pl.col("return_bench")).alias("return_diff"),
        ])
        
        # Allocation: (Wp - Wb) * (Rb - Rb_total)
        allocation = (
            joined["weight_diff"] * (joined["return_bench"] - total_benchmark_return)
        ).sum()
        
        # Selection: Wb * (Rp - Rb)
        selection = (joined["weight_bench"] * joined["return_diff"]).sum()
        
        # Interaction: (Wp - Wb) * (Rp - Rb)
        interaction = (joined["weight_diff"] * joined["return_diff"]).sum()
        
        total_active_return = allocation + selection + interaction
        
        result = AttributionResult(
            allocation_effect=float(allocation),
            selection_effect=float(selection),
            interaction_effect=float(interaction),
            total_active_return=float(total_active_return),
        )
        
        # Store for rolling analysis (memory-bounded)
        self._store_result(result, period)
        
        return result
    
    def _store_result(self, result: AttributionResult, period: str) -> None:
        """Store result with memory management."""
        row = pl.DataFrame({
            "period": [period],
            "allocation_effect": [result.allocation_effect],
            "selection_effect": [result.selection_effect],
            "interaction_effect": [result.interaction_effect],
            "total_active_return": [result.total_active_return],
        })
        
        self._attribution_history.append(row)
        
        # Enforce memory limit by keeping only recent history
        if len(self._attribution_history) > self.max_history_rows // 10:
            self._attribution_history = self._attribution_history[-100:]
    
    def get_rolling_attribution(
        self,
        window_periods: int = 20,
    ) -> pl.DataFrame:
        """
        Get rolling attribution statistics over recent periods.
        
        Args:
            window_periods: Number of periods for rolling window
            
        Returns:
            DataFrame with rolling averages of each effect
        """
        if not self._attribution_history:
            return pl.DataFrame()
        
        df = pl.concat(self._attribution_history[-window_periods:])
        
        # Calculate rolling means
        rolling_stats = df.select([
            pl.col("allocation_effect").mean().alias("avg_allocation"),
            pl.col("selection_effect").mean().alias("avg_selection"),
            pl.col("interaction_effect").mean().alias("avg_interaction"),
            pl.col("total_active_return").mean().alias("avg_active_return"),
            pl.col("total_active_return").std().alias("tracking_error"),
        ])
        
        return rolling_stats
    
    def get_sector_breakdown(
        self,
        portfolio_df: pl.DataFrame,
        benchmark_df: pl.DataFrame,
    ) -> list[SectorAttribution]:
        """
        Get attribution breakdown by individual sector.
        
        Returns:
            List of SectorAttribution objects
        """
        joined = portfolio_df.join(
            benchmark_df,
            on="sector",
            how="full",
            suffix="_bench",
        ).fill_null(0.0)
        
        total_benchmark_return = (
            joined["weight_bench"] * joined["return_bench"]
        ).sum()
        
        breakdown = []
        for row in joined.iter_rows(named=True):
            sector = row["sector"]
            wp = row["weight"]
            wb = row["weight_bench"]
            rp = row["return"]
            rb = row["return_bench"]
            
            weight_diff = wp - wb
            return_diff = rp - rb
            
            allocation = weight_diff * (rb - total_benchmark_return)
            selection = wb * return_diff
            interaction = weight_diff * return_diff
            
            breakdown.append(SectorAttribution(
                sector=sector,
                portfolio_weight=wp,
                benchmark_weight=wb,
                portfolio_return=rp,
                benchmark_return=rb,
                allocation_effect=allocation,
                selection_effect=selection,
                interaction_effect=interaction,
            ))
        
        return breakdown
    
    def explain_alpha_source(self, result: AttributionResult) -> str:
        """
        Generate human-readable explanation of alpha sources.
        
        Args:
            result: AttributionResult to explain
            
        Returns:
            Explanation string
        """
        explanations = []
        
        abs_alloc = abs(result.allocation_effect)
        abs_select = abs(result.selection_effect)
        abs_interact = abs(result.interaction_effect)
        
        # Determine primary source of alpha
        if abs_alloc >= abs_select and abs_alloc >= abs_interact:
            if result.allocation_effect > 0:
                explanations.append(
                    "Primary alpha source: Positive asset allocation decisions. "
                    "Overweighting outperforming sectors contributed significantly."
                )
            else:
                explanations.append(
                    "Primary drag: Negative asset allocation decisions. "
                    "Sector weighting choices detracted from performance."
                )
        elif abs_select >= abs_alloc and abs_select >= abs_interact:
            if result.selection_effect > 0:
                explanations.append(
                    "Primary alpha source: Security selection. "
                    "Stock picking within sectors generated excess returns."
                )
            else:
                explanations.append(
                    "Primary drag: Security selection. "
                    "Stock picks within sectors underperformed their benchmarks."
                )
        else:
            if result.interaction_effect > 0:
                explanations.append(
                    "Primary alpha source: Interaction effect. "
                    "The combination of allocation and selection decisions was synergistic."
                )
            else:
                explanations.append(
                    "Primary drag: Interaction effect. "
                    "Allocation and selection decisions worked against each other."
                )
        
        # Add magnitude context
        total_abs = abs_alloc + abs_select + abs_interact
        if total_abs > 0:
            explanations.append(
                f"Allocation contributed {abs_alloc/total_abs*100:.1f}%, "
                f"Selection {abs_select/total_abs*100:.1f}%, "
                f"Interaction {abs_interact/total_abs*100:.1f}% of total active variance."
            )
        
        return " ".join(explanations)


def create_sample_data() -> tuple[pl.DataFrame, pl.DataFrame]:
    """Create sample portfolio and benchmark data for testing."""
    portfolio_data = {
        "sector": ["Technology", "Healthcare", "Financials", "Energy", "Consumer"],
        "weight": [0.35, 0.20, 0.15, 0.10, 0.20],
        "return": [0.08, 0.04, 0.02, -0.03, 0.05],
    }
    
    benchmark_data = {
        "sector": ["Technology", "Healthcare", "Financials", "Energy", "Consumer"],
        "weight": [0.30, 0.15, 0.20, 0.15, 0.20],
        "return": [0.07, 0.03, 0.03, -0.02, 0.04],
    }
    
    return pl.DataFrame(portfolio_data), pl.DataFrame(benchmark_data)


if __name__ == "__main__":
    # Example usage
    attributor = BrinsonFachlerAttributor()
    
    portfolio_df, benchmark_df = create_sample_data()
    
    result = attributor.calculate_with_polars(
        portfolio_df,
        benchmark_df,
        period="2024-01-15",
    )
    
    print(f"Total Active Return: {result.total_active_return:.4%}")
    print(f"  Allocation Effect: {result.allocation_effect:.4%}")
    print(f"  Selection Effect: {result.selection_effect:.4%}")
    print(f"  Interaction Effect: {result.interaction_effect:.4%}")
    print()
    print(attributor.explain_alpha_source(result))
